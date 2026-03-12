# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import re
import time
import yaml
from contextlib import nullcontext
from pathlib import Path
from pkg_resources import packaging
import datetime

import functools
import hydra
import torch
import torch_npu
# import torch.npu.nccl as nccl
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import LlamaTokenizer
from typing import Any, Callable, List, Optional
from textwrap import dedent
from hydra import version
from hydra.main import _UNSPECIFIED_, _get_rerun_conf
from hydra._internal.deprecation_warning import deprecation_warning
from hydra._internal.utils import _run_hydra, get_args_parser
from hydra.types import TaskFunction
from hydra.core.utils import _flush_loggers, configure_log

# from dataset.speech_dataset_large import MultiTaskDynamicBatchDataset,MultiTaskDataset
from utils.checkpoint_handler import save_model_checkpoint_deepspeed
from utils.memory_utils import MemoryTrace, NoTrace
from torch.utils.data import IterableDataset
from utils.wenet_compute_cer import compute_wer_simple

# import wandb
import logging

logger = logging.getLogger(__name__)

# For deepspeed --local_rank argument
def deepspeed_main_wrapper(
    config_path: Optional[str] = _UNSPECIFIED_,
    config_name: Optional[str] = None,
    version_base: Optional[str] = _UNSPECIFIED_,
) -> Callable[[TaskFunction], Any]:
    """
    :param config_path: The config path, a directory where Hydra will search for
                        config files. This path is added to Hydra's searchpath.
                        Relative paths are interpreted relative to the declaring python
                        file. Alternatively, you can use the prefix `pkg://` to specify
                        a python package to add to the searchpath.
                        If config_path is None no directory is added to the Config search path.
    :param config_name: The name of the config (usually the file name without the .yaml extension)
    """

    version.setbase(version_base)

    if config_path is _UNSPECIFIED_:
        if version.base_at_least("1.2"):
            config_path = None
        elif version_base is _UNSPECIFIED_:
            url = "https://hydra.cc/docs/1.2/upgrades/1.0_to_1.1/changes_to_hydra_main_config_path"
            deprecation_warning(
                message=dedent(
                    f"""
                config_path is not specified in @hydra.main().
                See {url} for more information."""
                ),
                stacklevel=2,
            )
            config_path = "."
        else:
            config_path = "."

    def main_decorator(task_function: TaskFunction) -> Callable[[], None]:
        @functools.wraps(task_function)
        def decorated_main(cfg_passthrough: Optional[DictConfig] = None) -> Any:
            if cfg_passthrough is not None:
                return task_function(cfg_passthrough)
            else:
                args_parser = get_args_parser()
                args_parser.add_argument("--local_rank", type=int, default=-1)
                args = args_parser.parse_args()
                if args.experimental_rerun is not None:
                    cfg = _get_rerun_conf(args.experimental_rerun, args.overrides)
                    task_function(cfg)
                    _flush_loggers()
                else:
                    # no return value from run_hydra() as it may sometime actually run the task_function
                    # multiple times (--multirun)
                    _run_hydra(
                        args=args,
                        args_parser=args_parser,
                        task_function=task_function,
                        config_path=config_path,
                        config_name=config_name,
                    )

        return decorated_main

    return main_decorator


def deepspeed_join(group_join):
    """
    Copy from wenet:https://github.com/wenet-e2e/wenet/blob/main/wenet/utils/executor.py#L64
    """
    try:
        # 获取 timeout，如果没有 options，则使用默认值
        timeout = getattr(group_join, 'options', None)
        if timeout is not None:
            timeout = timeout._timeout
        else:
            timeout = datetime.timedelta(seconds=30)

        # NOTE(xcsong): Why we need a new group?
        #   Because Deepspeed has its own group where all the relevant communication
        #   operations are executed. If we add a communication operation that is not
        #   managed by Deepspeed in this group, it's highly likely to cause
        #   communication chaos, resulting in hard-to-troubleshoot hangs.
        dist.monitored_barrier(group=group_join, timeout=timeout)
    except RuntimeError as e:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        logger.info(
            "Detected uneven workload distribution. " +
            "Break current worker to manually join all workers, " +
            "world_size {}, current rank {}, current local_rank {}\n".format(
                world_size, rank, local_rank
            )
        )
        return True
    return False


def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"


# Converting Bytes to Megabytes
def byte2mb(x):
    return int(x / 2**20)


def train(
    model,
    train_dataloader,
    eval_dataloader,
    tokenizer,
    gradient_accumulation_steps,
    train_config,
    log_config,
    local_rank=None,
    rank=None,
):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        log_config: The logging configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    DO_MSPROBE_TEST = os.environ.get('MSPROBE_TEST', 'false') == 'true'
    if DO_MSPROBE_TEST:
        from msprobe.pytorch import seed_all, PrecisionDebugger
        seed_all()
        debugger = PrecisionDebugger(config_path=os.environ.get("MSPROBE_CONFIG_DIR", "msprobe/config.json"))
    # Create a gradient scaler for fp16
    # if train_config.use_fp16 and train_config.enable_fsdp:
    #     scaler = ShardedGradScaler()
    # elif train_config.use_fp16 and not train_config.enable_fsdp:
    #     scaler = torch.npu.amp.GradScaler()
    if train_config.enable_ddp:
        world_size = int(os.environ["WORLD_SIZE"])
    autocast = torch.npu.amp.autocast if train_config.use_fp16 else nullcontext
    # experimental_config = torch_npu.profiler._ExperimentalConfig(
    #     export_type=torch_npu.profiler.ExportType.Text,
    #     profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    #     msprof_tx=False,
    #     aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
    #     l2_cache=False,
    #     op_attr=False,
    #     data_simplification=False,
    #     record_op_args=False,
    #     gc_detect_threshold=None
    # )

    # prof = torch_npu.profiler.profile(
    #     activities=[
    #         torch_npu.profiler.ProfilerActivity.CPU,
    #         torch_npu.profiler.ProfilerActivity.NPU
    #         ],
    #     schedule=torch_npu.profiler.schedule(wait=0, warmup=2, active=5, repeat=1, skip_first=10),
    #     on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
    #     record_shapes=True,
    #     profile_memory=False,
    #     with_stack=False,
    #     with_modules=False,
    #     with_flops=False,
    #     experimental_config=experimental_config)
    # prof.start()

    writer = SummaryWriter(log_dir=os.path.join(train_config.output_dir, "run"))
    train_prep = []
    train_loss = []
    train_acc = []
    val_prep = []
    val_loss = []
    val_acc = []
    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    best_val_acc = 0.0
    total_step = 0
    device = f"npu:{local_rank}"
    train_len = None if isinstance(train_dataloader.dataset, IterableDataset) else len(train_dataloader)
    for epoch in range(train_config.num_epochs):
        dist.barrier()
        group_join = dist.new_group(
            backend="gloo", timeout=datetime.timedelta(seconds=40))
        epoch_start_time = time.perf_counter()
        with NoTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = torch.tensor(0.0).to(device) # 必须用float32，用float16在1967左右累加会缺失
            total_acc = torch.tensor(0.0).to(device)
            epoch_step = 0
            if rank == 0:
                if train_len is None:
                    pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", dynamic_ncols=True)
                else:
                    total_length = train_len //gradient_accumulation_steps
                    pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)

            for step, batch in enumerate(train_dataloader):
                if DO_MSPROBE_TEST:
                    debugger.start(model)
                if deepspeed_join(group_join):
                    break
                
                total_step += 1
                for key in batch.keys():
                    batch[key] = (
                        batch[key].to(device).half()
                        if isinstance(batch[key], torch.Tensor)
                        and batch[key].dtype in [torch.float32, torch.float64]
                        else (
                            batch[key].to(device)
                            if isinstance(batch[key], torch.Tensor)
                            else batch[key]
                        )
                    )
                batch['experiment_name'] = train_config.exp_name
                with autocast(dtype=torch.bfloat16):
                    """
                    input_ids: torch.int64
                    attention_mask: torch.bool
                    input_features: torch.float16
                    input_feature_length: torch.int64
                    labels: torch.int64
                    """
                    outputs, acc = model(**batch)
                loss = outputs.loss

                loss = loss / gradient_accumulation_steps
                acc = acc / gradient_accumulation_steps

                acc_val = acc.detach().float().item()
                loss_val = loss.detach().float().item()

                # acc_val, loss_val = 0, 0
                epoch_step += 1
                total_loss += loss_val
                total_acc += acc_val

                # deepspeed should handle gradient accumulate
                model.backward(loss)
                model.step()
                grad_norm = model.get_global_grad_norm()
                # prof.step()
                if rank == 0:
                    writer.add_scalar("train/step_lr", model.get_lr()[0], total_step)
                    writer.add_scalar("train/step_loss", loss_val, total_step)
                    writer.add_scalar("train/grad_norm", grad_norm, total_step)
                    if (step + 1) % gradient_accumulation_steps == 0 :
                        pbar.update(1)
                    
                    desc = f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{train_len if train_len is not None else ''} completed (loss: {loss_val : .4f}, acc: {acc_val : .4f})"
                    
                    if train_config.exp_name == "wavprompt":
                        desc += f" embed_loss: {outputs.embed_loss.detach().item() : .2f} quantity_loss: {outputs.quantity_loss.detach().item() : .2f}"
                    elif train_config.exp_name == 'ctc':
                        desc += f" ctc_loss: {outputs.ctc_loss.detach().item() : .2f}"
                        writer.add_scalar("train/ctc_loss", outputs.ctc_loss.detach().item(), total_step)

                    pbar.set_description(desc)

                if total_step % train_config.validation_interval == 0 and train_config.run_validation:
                    eval_ppl, eval_epoch_loss, *rest = evaluation(
                        model, train_config, eval_dataloader, local_rank, tokenizer
                    )
                    eval_epoch_acc = rest[0] if rest else -1
                    checkpoint_start_time = time.perf_counter()

                    
                    # if train_config.save_model and (eval_epoch_loss < best_val_loss or eval_epoch_acc > best_val_acc):
                    checkpoint_name = f"{train_config.model_name}_epoch_{str(epoch+1)}_total_step_{total_step}"
                    save_model_checkpoint_deepspeed(
                        model, train_config, checkpoint_name
                    )
                    dist.barrier()
                    checkpoint_end_time = time.perf_counter() - checkpoint_start_time
                    checkpoint_times.append(checkpoint_end_time)
                    if eval_epoch_loss < best_val_loss:
                        best_val_loss = eval_epoch_loss
                        if rank == 0:
                            logger.info(
                                f"best eval loss on epoch {epoch+1} is {best_val_loss}"
                            )
                    val_loss.append(eval_epoch_loss)
                    val_prep.append(eval_ppl)
                    if rest:
                        if eval_epoch_acc > best_val_acc:
                            best_val_acc = eval_epoch_acc
                            if rank == 0:
                                logger.info(
                                    f"best eval acc on epoch {epoch+1} is {best_val_acc}"
                                )
                        val_acc.append(rest[0])
                    else:
                        val_acc.append(-1)

                    if log_config.use_wandb:
                        if rank == 0:
                            wandb.log(
                                {
                                    "valid/val_epoch_loss": eval_epoch_loss,
                                    "valid/val_perplexity": eval_ppl,
                                    "valid/best_val_loss": best_val_loss,
                                    "valid/val_accuracy": val_acc[-1],
                                    "valid/val_best_accuracy": best_val_acc,
                                }
                            )

                    dist.barrier()
                if DO_MSPROBE_TEST:
                    debugger.stop()
                    debugger.step()
            if rank == 0:
                pbar.close()
        # prof.stop()
        dist.destroy_process_group(group_join)
        epoch_end_time = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one npu device
        if torch.npu.device_count() > 1 and (
            train_config.enable_fsdp or train_config.enable_ddp
        ):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_acc, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / epoch_step
        train_epoch_acc = total_acc / epoch_step
        if train_config.enable_fsdp or train_config.enable_ddp:
            train_epoch_loss = train_epoch_loss / world_size
            train_epoch_acc = train_epoch_acc / world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(train_perplexity)
        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)

        if log_config.use_wandb:
            if train_config.enable_fsdp or train_config.enable_ddp:
                if rank == 0:
                    wandb.log(
                        {
                            "train/train_perplexity": train_perplexity,
                            "train/train_epoch_loss": train_epoch_loss,
                            "train/train_epoch_acc": train_epoch_acc,
                        }
                    )
            else:
                wandb.log(
                    {
                        "train/train_perplexity": train_perplexity,
                        "train/train_epoch_loss": train_epoch_loss,
                        "train/train_epoch_acc": train_epoch_acc,
                    }
                )
        if rank == 0:
            writer.add_scalar("train/perplexity", train_perplexity, epoch)
            writer.add_scalar("train/epoch_loss", train_epoch_loss, epoch)
            writer.add_scalar("train/epoch_acc", train_epoch_acc, epoch)
            writer.add_scalar("train/epoch_lr", model.get_lr()[0], epoch)

        if rank == 0:
            logger.info(
                f"Epoch {epoch+1}: train_perplexity={train_perplexity.detach().item():.4f}, train_epoch_loss={train_epoch_loss.detach().item():.4f}, epoch time {epoch_end_time}s"
            )

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_checkpoint_time = (
        sum(checkpoint_times) / len(checkpoint_times)
        if len(checkpoint_times) > 0
        else 0
    )
    avg_train_prep = sum(train_prep) / len(train_prep)
    avg_train_loss = sum(train_loss) / len(train_loss)
    avg_train_acc = sum(train_acc) / len(train_acc)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep) / len(val_prep)
        avg_eval_loss = sum(val_loss) / len(val_loss)
        avg_eval_acc = sum(val_acc) / len(val_acc)

    results["avg_train_prep"] = avg_train_prep
    results["avg_train_loss"] = avg_train_loss
    results["avg_train_acc"] = avg_train_acc
    if train_config.run_validation:
        results["avg_eval_prep"] = avg_eval_prep
        results["avg_eval_loss"] = avg_eval_loss
        results["avg_eval_acc"] = avg_eval_acc
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time

    # saving the training params including fsdp setting for reference.
    # if (train_config.enable_fsdp or train_config.enable_ddp)and not train_config.use_peft:
    #     save_train_params(train_config, fsdp_config, rank)

    return results

# TODO: fix
def evaluation(model, train_config, eval_dataloader, local_rank, tokenizer):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    world_size = int(os.environ["WORLD_SIZE"])
    model.eval()
    eval_preds = []
    autocast = (
        torch.npu.amp.autocast if train_config.use_fp16 else nullcontext
    )  # (Fix:MZY): fix expected scalar type mismatch in norm
    device = f"npu:{local_rank}"
    eval_loss = torch.zeros(1, device=device, dtype=torch.bfloat16)  # Initialize evaluation loss
    eval_acc = torch.zeros(1, device=device, dtype=torch.bfloat16)

    eval_len = None if isinstance(eval_dataloader.dataset, IterableDataset) else len(eval_dataloader)
    with NoTrace():
        if eval_len is not None:
            pbar = tqdm(colour="green", desc=f"Evaluating Epoch", total=eval_len, dynamic_ncols=True)
        else:
            pbar = tqdm(colour="green", desc=f"Evaluating Epoch",  dynamic_ncols=True)
        for step, batch in enumerate(eval_dataloader):
            for key in batch.keys():
                batch[key] = (
                    batch[key].to(device).half()
                    if isinstance(batch[key], torch.Tensor) and batch[key].dtype in [torch.float32, torch.float64]
                    else (
                        batch[key].to(device) if isinstance(batch[key], torch.Tensor) else batch[key]
                    )
                )
            # Ensure no gradients are computed for this scope to save memory
            if train_config.exp_name == 'cot':
                with torch.no_grad():
                    with autocast(dtype=torch.bfloat16):
                        model_outputs = model.generate(**batch)
                    output_text = model.tokenizer.batch_decode(model_outputs, add_special_tokens=False, skip_special_tokens=True)
                    labels_text = model.tokenizer.batch_decode(batch['labels'].clamp(min=0), add_special_tokens=False, skip_special_tokens=True)
                    def clean_text(text):
                        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
                        text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower().strip()
                        return text
                    wers = [compute_wer_simple(clean_text(o), clean_text(l))[0] for o, l in zip(output_text, labels_text)]
                    acc = 1.0 - sum(wers) / len(wers) / 100
                    eval_acc += acc
                eval_preds.extend(output_text)
            else:
                with torch.no_grad():
                    # Forward pass and compute loss
                    with autocast(dtype=torch.bfloat16):  # (Fix:MZY): fix expected scalar type mismatch in norm
                        outputs, *rest = model.module(**batch)
                    acc = rest[0] if rest else -1
                    loss = outputs.loss

                    eval_loss += loss.detach()
                    eval_acc += acc.detach()
                # Decode predictions and add to evaluation predictions list
                preds = torch.argmax(outputs.logits, -1)
                eval_preds.extend(
                    tokenizer.batch_decode(
                        preds.detach().cpu().numpy(), skip_special_tokens=True
                    )
                )
            pbar.update(1)
            pbar.set_description(
                f"step: {step+1}/{eval_len if eval_len is not None else '' }, eval_loss: {eval_loss.detach().item()/(step+1):.4f}, eval_acc: {eval_acc.detach().item()/(step+1):.4f}"
            )
    dist.barrier()
    # If there's more than one npu device, reduce evaluation loss across all devices
    if (
        torch.npu.device_count() > 1
    ):
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(eval_acc, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / (eval_len if eval_len is not None else step + 1)
    eval_epoch_acc = eval_acc / (eval_len if eval_len is not None else step + 1)
    eval_epoch_loss = eval_epoch_loss / world_size
    eval_epoch_acc = eval_epoch_acc / world_size
    eval_ppl = torch.exp(eval_epoch_loss)

    # Print evaluation metrics
    if local_rank == 0:
        logger.info(f" {eval_ppl=} {eval_epoch_loss=} {eval_epoch_acc=}")

    model.train()
    dist.barrier()
    return eval_ppl, eval_epoch_loss, eval_epoch_acc


def freeze_transformer_layers(model, num_layer):
    for i, layer in enumerate(model.model.layers):
        if i < num_layer:
            for param in layer.parameters():
                param.requires_grad = False


def check_frozen_layers_peft_model(model):
    for i, layer in enumerate(model.base_model.model.model.layers):
        for name, param in layer.named_parameters():
            logger.info(
                f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}"
            )


def setup():
    """Initialize the process group for distributed training"""
    dist.init_process_group("hccl")


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["HCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with npu memory fragmentations that can lead into OOM in some cases.
    # Note this is only availble in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_npu_ALLOC_CONF']='expandable_segments:True'
    if rank == 0:
        logger.info(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        logger.info(f"Clearing GPU cache for all ranks")
    torch.npu.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes


def save_train_params(train_config, fsdp_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be hepful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries,
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {
        k: str(v) for k, v in vars(train_config).items() if not k.startswith("__")
    }
    fsdp_config_dict = {
        k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith("__")
    }
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (follwoing FSDP checkpointing style) using properties of the train_config object
    folder_name = (
        train_config.dist_checkpoint_root_folder
        + "/"
        + train_config.dist_checkpoint_folder
        + "-"
        + train_config.model_name
    )

    save_dir = Path.cwd() / folder_name
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir, "train_params.yaml")

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        logger.info(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, "w") as f:
            f.write(config_yaml)
        if rank == 0:
            logger.info(f"training params are saved in {file_name}")
