import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from matplotlib import font_manager

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['axes.unicode_minus'] = False 

def save_scalar_withstr(
    x_labels,      # List[str]
    y_values,      # List[float] or List[int]
    save_path,
    xlabel="X",
    ylabel="Y",
    title="Line Plot",
    rotate_xticks=45,
    marker_size=4,       # 数据点大小
    x_tick_step=1        # x 轴标签间隔
):
    assert len(x_labels) == len(y_values), "x_labels and y_values must have same length"

    x = range(len(x_labels))  # 数值索引作为真实 x

    plt.figure(figsize=(max(10, len(x_labels)*0.2), 4))  # 宽度自适应，最小 10 inch
    plt.plot(x, y_values, marker="o", markersize=marker_size)

    if x_tick_step > 1:
        plt.xticks(x[::x_tick_step], x_labels[::x_tick_step], rotation=rotate_xticks, ha="right", fontproperties=font_prop)
    else:
        plt.xticks(x, x_labels, rotation=rotate_xticks, ha="right", fontproperties=font_prop)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def save_heatmap(
    matrix, #[H, W]
    save_path,
    x_labels=None,       # List[str], len = W
    y_labels=None,       # List[str], len = H
    xlabel="X",
    ylabel="Y",
    title="Heatmap",
    cmap="magma"
):
    H, W = matrix.shape

    plt.figure(figsize=(max(8, W * 0.15), max(4, H * 0.15)))
    im = plt.imshow(matrix, aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    if x_labels is not None:
        assert len(x_labels) == W, f"Given labels size {len(x_labels)} is not equal to matrix length {W}"
        plt.xticks(np.arange(W), x_labels, rotation=45, ha="right", fontproperties=font_prop)
    else:
        plt.xticks(np.arange(W))

    if y_labels is not None:
        assert len(y_labels) == H
        plt.yticks(np.arange(H), y_labels, fontproperties=font_prop)
    else:
        plt.yticks(np.arange(H))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def save_scalar(
    root_dir,
    values,
    metric,  
):
    writer = SummaryWriter(root_dir)

    for step, value in enumerate(values):
        writer.add_scalar(metric, value, step)

    writer.close()
