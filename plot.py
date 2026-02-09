import numpy as np
import matplotlib.pyplot as plt

# 映射半径
R = 1.3

# 生成 x 和 y 值，这里固定 y=0，x 从 -R 到 R
x = np.linspace(-1, 1, 500)
y = 1.0

# 精确点积
def f_dot_exact(x, y, R):
    return (x*y)/R**2 + np.sqrt(1 - (x**2)/R**2) * np.sqrt(1 - (y**2)/R**2)

# 一次近似点积
def f_dot_approx(x, y, R):
    return 1 - ((x-y)**2) / (2*R**2)

# 计算点积
dot_exact = f_dot_exact(x, y, R)
dot_approx = f_dot_approx(x, y, R)

# 误差
error = dot_exact - dot_approx

# 绘图
plt.figure(figsize=(8,5))
plt.plot(x, dot_exact, label="Exact f(x).f(y)")
plt.plot(x, dot_approx, '--', label="Exact L2(f(x), f(y))")
plt.plot(x, error, '-.', label="Error (Exact - Approx)")
label=f"x (y={y})"
plt.xlabel(label)
plt.ylabel("Dot product / Error")
plt.title("1D -> 2D Sphere Mapping: Dot Product Approximation Error")
plt.legend()
plt.grid(True)
plt.show()

plt.savefig("dot_product_error.png", dpi=300)
plt.close()  # 关闭图表，防止在某些环境弹窗
print("图片已保存为 dot_product_error.png")