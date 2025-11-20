import torch
from wavlm.WavLM import WavLM, WavLMConfig
import kaldiio
from soundfile import read

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# load the pre-trained checkpoints
checkpoint = torch.load('/aistor/sjtu/hpc_stor01/home/guoyiwei/remote/code/AudioFeatExtraction/wavlm/pretrained/WavLM-Large.pt')
cfg = WavLMConfig(checkpoint['cfg'])
model = WavLM(cfg)

model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

import pdb;pdb.set_trace()
sample_rate, wav_np = kaldiio.load_mat("/aistor/sjtu/hpc_stor01/home/yangyi/data/asr/train/data/data_wav.1.ark:50")

# extract the representation of last layer
# wav_input_16khz = torch.randn(1,10000)
wav_input_16khz, sr = read("/aistor/sjtu/hpc_stor01/home/guxiaoyu/workspace/multi-words-kws-context-asr/SLAM-LLM-FireRedASR/2003-143255-0059.wav")
wav_input_16khz = torch.from_numpy(wav_input_16khz).unsqueeze(0).float().to(device)
# print(wav_input_16khz)
if cfg.normalize:
    wav_input_16khz = torch.nn.functional.layer_norm(wav_input_16khz , wav_input_16khz.shape)

with torch.no_grad():
    output_layer=6
    rep = model.extract_features(wav_input_16khz, output_layer=output_layer)[0]
    # rep now is the output after output_layer
    print(rep)
    
    # extract the representation of each layer
    # wav_input_16khz = torch.randn(1,10000)
    print(model.cfg.encoder_layers, "layers in total")
    rep, layer_results = model.extract_features(wav_input_16khz, output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]
    layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
    print(len(layer_reps))
    print(layer_reps[output_layer])
    print([x.shape for x in layer_reps])


