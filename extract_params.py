# coding:utf-8
import torch
import sys
import struct
import numpy as np
from torchvision import datasets, transforms

# 假设 model.py 和 module.py 在同一目录
from model import Net

def direct_quantize(model, data_loader):
    # 使用整个数据加载器进行校准
    for i, (data, target) in enumerate(data_loader, 1):
        model.quantize_forward(data)
        # 可以设置一个上限，但为了完整校准，跑完整个数据集
    print('direct quantization finish')

def extract_params(float_pt_file, output_bin, batch_size=64):
    # 1. 创建原始模型，加载浮点权重（键为 'conv1.weight', 'conv2.weight', 'fc.weight'）
    model = Net()
    # model.quantize(num_bits=8)
    # 关键：将浮点权重复制到量化层内部的原始卷积层（conv_module）
    # 由于 quantize() 创建了 qconv1，其 conv_module 指向原始的 self.conv1，
    # 因此我们只需将浮点权重加载到原始卷积层即可（即加载到 model 本身）
    model.load_state_dict(torch.load(float_pt_file, map_location='cpu'))
    model.eval()

    # 2. 创建量化层（此时原始层已有权重，量化层的 conv_module 会自动引用它们）
    model.quantize(num_bits=8)

    # 3. 准备训练数据加载器用于校准
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
    )

    # 校准（更新 qi, qo 的 min/max，以及 qw 的 min/max）
    direct_quantize(model, train_loader)

    # 冻结模型，将权重量化为整数并替换，同时计算 M
    model.freeze()

    # 现在提取参数
    params = {}

    # qconv1
    qconv1 = model.qconv1
    params['qconv1_qi_scale'] = qconv1.qi.scale.item()
    params['qconv1_qi_zero'] = qconv1.qi.zero_point.item()
    params['qconv1_qo_scale'] = qconv1.qo.scale.item()
    params['qconv1_qo_zero'] = qconv1.qo.zero_point.item()
    params['qconv1_qw_scale'] = qconv1.qw.scale.item()
    params['qconv1_qw_zero'] = qconv1.qw.zero_point.item()
    params['qconv1_weight'] = qconv1.conv_module.weight.detach().cpu().numpy()  # 量化后的权重（减去 zero_point）
    params['qconv1_bias'] = qconv1.conv_module.bias.detach().cpu().numpy()      # 量化后的偏置
    # 计算 M
    params['qconv1_M'] = (params['qconv1_qw_scale'] * params['qconv1_qi_scale'] / params['qconv1_qo_scale'])

    # qconv2
    qconv2 = model.qconv2
    params['qconv2_qo_scale'] = qconv2.qo.scale.item()
    params['qconv2_qo_zero'] = qconv2.qo.zero_point.item()
    params['qconv2_qw_scale'] = qconv2.qw.scale.item()
    params['qconv2_qw_zero'] = qconv2.qw.zero_point.item()
    params['qconv2_weight'] = qconv2.conv_module.weight.detach().cpu().numpy()
    params['qconv2_bias'] = qconv2.conv_module.bias.detach().cpu().numpy()
    # qconv2 的 qi 是 qconv1.qo，已经在推理时传递，不需要保存，但 M 需要用到 qi 即 qconv1.qo.scale
    params['qconv2_M'] = (params['qconv2_qw_scale'] * params['qconv1_qo_scale'] / params['qconv2_qo_scale'])

    # qfc
    qfc = model.qfc
    params['qfc_qo_scale'] = qfc.qo.scale.item()
    params['qfc_qo_zero'] = qfc.qo.zero_point.item()
    params['qfc_qw_scale'] = qfc.qw.scale.item()
    params['qfc_qw_zero'] = qfc.qw.zero_point.item()
    params['qfc_weight'] = qfc.fc_module.weight.detach().cpu().numpy()
    params['qfc_bias'] = qfc.fc_module.bias.detach().cpu().numpy()
    # qfc 的 qi 是 qconv2.qo
    params['qfc_M'] = (params['qfc_qw_scale'] * params['qconv2_qo_scale'] / params['qfc_qo_scale'])

    # 写入二进制文件
    with open(output_bin, 'wb') as f:
        # 写入魔数/版本（可选）
        # 依次写入每个参数，先写入形状和数值
        # 简单起见，按固定顺序写入所有标量和数组

        # 标量
        scalar_keys = [
            'qconv1_qi_scale', 'qconv1_qi_zero', 'qconv1_qo_scale', 'qconv1_qo_zero',
            'qconv1_qw_scale', 'qconv1_qw_zero', 'qconv1_M',
            'qconv2_qo_scale', 'qconv2_qo_zero', 'qconv2_qw_scale', 'qconv2_qw_zero', 'qconv2_M',
            'qfc_qo_scale', 'qfc_qo_zero', 'qfc_qw_scale', 'qfc_qw_zero', 'qfc_M'
        ]
        for key in scalar_keys:
            f.write(struct.pack('f', params[key]))

        # 权重和偏置数组
        # qconv1 weight: [40,1,3,3]
        arr = params['qconv1_weight'].astype(np.float32)
        f.write(struct.pack('I', arr.size))          # 元素个数
        f.write(arr.tobytes())

        # qconv1 bias: [40]
        arr = params['qconv1_bias'].astype(np.float32)
        f.write(struct.pack('I', arr.size))
        f.write(arr.tobytes())

        # qconv2 weight: [40,2,3,3]
        arr = params['qconv2_weight'].astype(np.float32)
        f.write(struct.pack('I', arr.size))
        f.write(arr.tobytes())

        # qconv2 bias: [40]
        arr = params['qconv2_bias'].astype(np.float32)
        f.write(struct.pack('I', arr.size))
        f.write(arr.tobytes())

        # qfc weight: [10,1000]
        arr = params['qfc_weight'].astype(np.float32)
        f.write(struct.pack('I', arr.size))
        f.write(arr.tobytes())

        # qfc bias: [10]
        arr = params['qfc_bias'].astype(np.float32)
        f.write(struct.pack('I', arr.size))
        f.write(arr.tobytes())

    print(f"参数已保存到 {output_bin}")

if __name__ == '__main__':

    extract_params("ckpt/mnist_cnn.pt", "mnist_cnn_ptq.bin")