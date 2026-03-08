import torch
import torchvision
import torchvision.transforms as transforms
import struct
import sys
import numpy as np

from model import Net

def generate_test_data(float_pt_file, index=0, output_input='input.bin', output_ref='output_ref.bin'):
    # 1. 加载浮点模型权重
    model = Net()
    model.load_state_dict(torch.load(float_pt_file, map_location='cpu'))
    model.eval()

    # 2. 创建量化层
    model.quantize(num_bits=8)

    # 3. 准备训练数据加载器用于校准
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ])),
        batch_size=64, shuffle=True, num_workers=1, pin_memory=True
    )

    # 4. 校准
    for i, (data, target) in enumerate(train_loader):
        model.quantize_forward(data)

    # 5. 冻结
    model.freeze()

    # 6. 获取一个测试样本
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    img, label = testset[index]
    img = img.unsqueeze(0)  # [1,1,28,28]

    # 保存输入
    input_np = img.numpy().astype(np.float32)
    with open(output_input, 'wb') as f:
        f.write(struct.pack('I', input_np.size))
        f.write(input_np.tobytes())

    # 运行量化推理
    with torch.no_grad():
        output = model.quantize_inference(img)
    output_np = output.numpy().astype(np.float32)

    # 保存参考输出
    with open(output_ref, 'wb') as f:
        f.write(struct.pack('I', output_np.size))
        f.write(output_np.tobytes())

    print(f"输入已保存到 {output_input}, 参考输出保存到 {output_ref}")
    print(f"标签: {label}")

if __name__ == '__main__':

    generate_test_data("ckpt/mnist_cnn.pt")