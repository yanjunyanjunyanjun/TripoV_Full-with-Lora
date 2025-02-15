
import numpy as np

# 示例 Loss 数据
loss_values = [0.77, 0.68, 0.69, 0.5,0.688,0.677,0.675,0.674,0.673,0.672,0.671,0.670,0.669,0.668,0.669]

# 计算进步率
improvement_rate = [(loss_values[i - 1] - loss_values[i]) / loss_values[i - 1] * 100 
                    for i in range(1, len(loss_values))]

# 输出结果
for epoch, rate in enumerate(improvement_rate, start=2):
    print(f"Epoch {epoch}: Improvement Rate = {rate:.2f}%")

import matplotlib.pyplot as plt

# 绘制进步率曲线
epochs = range(2, len(loss_values) + 1)  # 从第 2 个 Epoch 开始
plt.plot(epochs, improvement_rate, marker='o', label="Improvement Rate (%)")
plt.xlabel("Epoch")
plt.ylabel("Improvement Rate (%)")
plt.title("Loss Improvement Rate Over Epochs")
plt.legend()
plt.grid()
plt.show()