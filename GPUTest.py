# # import tensorflow as tf
# # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# print("PyTorch version:", torch.__version__)
# print("CUDA version:", torch.version.cuda)
# print("Is CUDA available:", torch.cuda.is_available())
# print("Number of GPUs:", torch.cuda.device_count())
#
# # 检查是否有可用的 GPU 设备
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("Using device:", device)
#
# # 设置 GPU 内存增长（PyTorch 自动管理显存，无需手动设置）
#
# # 加载 MNIST 数据集
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
#
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
#
# # 定义模型
# class SimpleNN(nn.Module):
#     def __init__(self):
#         super(SimpleNN, self).__init__()
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(28 * 28, 128)
#         self.dropout = nn.Dropout(0.2)
#         self.fc2 = nn.Linear(128, 10)
#
#     def forward(self, x):
#         x = self.flatten(x)
#         x = torch.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x
#
# model = SimpleNN().to(device)
#
# # 定义损失函数和优化器
# loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters())
#
# # 训练模型
# epochs = 5
# for epoch in range(epochs):
#     model.train()
#     for data, target in train_loader:
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = loss_fn(output, target)
#         loss.backward()
#         optimizer.step()
#
#     print(f"Epoch {epoch+1}/{epochs} finished")
#
# # 评估模型
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for data, target in test_loader:
#         data, target = data.to(device), target.to(device)
#         output = model(data)
#         _, predicted = torch.max(output, 1)
#         total += target.size(0)
#         correct += (predicted == target).sum().item()
#
# accuracy = 100 * correct / total
# print(f"Test Accuracy: {accuracy:.2f}%")
#
#
#
import os

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

import tensorflow as tf
from tensorflow.keras import backend as K

# 创建一个全零的张量
tensor = K.zeros((3, 3))
print(tensor)

# 计算一个张量的和
sum_tensor = K.sum(tensor)
print(sum_tensor)

print(tf.__version__)

# 列出可用的 GPU 设备
print(tf.config.list_physical_devices('GPU'))

# 设置 TensorFlow 使用 GPU 进行计算
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

# 加载数据集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 定义模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# 编译模型
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=5)

# 评估模型
model.evaluate(x_test, y_test, verbose=2)
