#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// 定义结构体存储网络参数
typedef struct
{
    // 标量参数
    float qconv1_qi_scale;
    float qconv1_qi_zero;
    float qconv1_qo_scale;
    float qconv1_qo_zero;
    float qconv1_qw_scale;
    float qconv1_qw_zero;
    float qconv1_M;

    float qconv2_qo_scale;
    float qconv2_qo_zero;
    float qconv2_qw_scale;
    float qconv2_qw_zero;
    float qconv2_M;

    float qfc_qo_scale;
    float qfc_qo_zero;
    float qfc_qw_scale;
    float qfc_qw_zero;
    float qfc_M;

    // 数组数据
    float *qconv1_weight; // shape [40,1,3,3]
    int qconv1_weight_size;
    float *qconv1_bias; // [40]
    int qconv1_bias_size;

    float *qconv2_weight; // [40,2,3,3]
    int qconv2_weight_size;
    float *qconv2_bias; // [40]
    int qconv2_bias_size;

    float *qfc_weight; // [10,1000]
    int qfc_weight_size;
    float *qfc_bias; // [10]
    int qfc_bias_size;
} QuantModel;

// 加载二进制参数文件
QuantModel *load_model(const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        perror("无法打开参数文件");
        return NULL;
    }

    QuantModel *model = (QuantModel *)malloc(sizeof(QuantModel));
    if (!model)
        return NULL;

    // 读取标量
    fread(&model->qconv1_qi_scale, sizeof(float), 1, fp);
    fread(&model->qconv1_qi_zero, sizeof(float), 1, fp);
    fread(&model->qconv1_qo_scale, sizeof(float), 1, fp);
    fread(&model->qconv1_qo_zero, sizeof(float), 1, fp);
    fread(&model->qconv1_qw_scale, sizeof(float), 1, fp);
    fread(&model->qconv1_qw_zero, sizeof(float), 1, fp);
    fread(&model->qconv1_M, sizeof(float), 1, fp);

    fread(&model->qconv2_qo_scale, sizeof(float), 1, fp);
    fread(&model->qconv2_qo_zero, sizeof(float), 1, fp);
    fread(&model->qconv2_qw_scale, sizeof(float), 1, fp);
    fread(&model->qconv2_qw_zero, sizeof(float), 1, fp);
    fread(&model->qconv2_M, sizeof(float), 1, fp);

    fread(&model->qfc_qo_scale, sizeof(float), 1, fp);
    fread(&model->qfc_qo_zero, sizeof(float), 1, fp);
    fread(&model->qfc_qw_scale, sizeof(float), 1, fp);
    fread(&model->qfc_qw_zero, sizeof(float), 1, fp);
    fread(&model->qfc_M, sizeof(float), 1, fp);

    // 读取 qconv1 weight
    fread(&model->qconv1_weight_size, sizeof(int), 1, fp);
    model->qconv1_weight = (float *)malloc(model->qconv1_weight_size * sizeof(float));
    fread(model->qconv1_weight, sizeof(float), model->qconv1_weight_size, fp);

    // qconv1 bias
    fread(&model->qconv1_bias_size, sizeof(int), 1, fp);
    model->qconv1_bias = (float *)malloc(model->qconv1_bias_size * sizeof(float));
    fread(model->qconv1_bias, sizeof(float), model->qconv1_bias_size, fp);

    // qconv2 weight
    fread(&model->qconv2_weight_size, sizeof(int), 1, fp);
    model->qconv2_weight = (float *)malloc(model->qconv2_weight_size * sizeof(float));
    fread(model->qconv2_weight, sizeof(float), model->qconv2_weight_size, fp);

    // qconv2 bias
    fread(&model->qconv2_bias_size, sizeof(int), 1, fp);
    model->qconv2_bias = (float *)malloc(model->qconv2_bias_size * sizeof(float));
    fread(model->qconv2_bias, sizeof(float), model->qconv2_bias_size, fp);

    // qfc weight
    fread(&model->qfc_weight_size, sizeof(int), 1, fp);
    model->qfc_weight = (float *)malloc(model->qfc_weight_size * sizeof(float));
    fread(model->qfc_weight, sizeof(float), model->qfc_weight_size, fp);

    // qfc bias
    fread(&model->qfc_bias_size, sizeof(int), 1, fp);
    model->qfc_bias = (float *)malloc(model->qfc_bias_size * sizeof(float));
    fread(model->qfc_bias, sizeof(float), model->qfc_bias_size, fp);

    fclose(fp);
    return model;
}

void free_model(QuantModel *model)
{
    if (model)
    {
        free(model->qconv1_weight);
        free(model->qconv1_bias);
        free(model->qconv2_weight);
        free(model->qconv2_bias);
        free(model->qfc_weight);
        free(model->qfc_bias);
        free(model);
    }
}

// 卷积函数 (2D, 单输入通道)
void conv2d(const float *input, int in_h, int in_w, int in_ch,
            const float *weight, const float *bias,
            int out_ch, int kernel_size, int stride, int padding,
            int groups,
            float *output)
{
    // 简化实现：假设 groups=1 或 groups=in_ch 等，这里我们针对具体网络
    // conv1: in_ch=1, out_ch=40, groups=1
    // conv2: in_ch=40, out_ch=40, groups=20, kernel=3, stride=1, padding=0
    // 对于分组卷积，每个组的输入通道为 in_ch/groups，输出通道为 out_ch/groups
    // 权重 shape: [out_ch, in_ch/groups, kh, kw]
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    for (int oc = 0; oc < out_ch; oc++)
    {
        int group = oc / (out_ch / groups); // 当前输出通道所属组
        int in_ch_per_group = in_ch / groups;
        int start_in_c = group * in_ch_per_group;

        for (int oh = 0; oh < out_h; oh++)
        {
            for (int ow = 0; ow < out_w; ow++)
            {
                float sum = bias ? bias[oc] : 0.0f;
                for (int ic = 0; ic < in_ch_per_group; ic++)
                {
                    for (int kh = 0; kh < kernel_size; kh++)
                    {
                        for (int kw = 0; kw < kernel_size; kw++)
                        {
                            int ih = oh * stride + kh - padding;
                            int iw = ow * stride + kw - padding;
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w)
                            {
                                int in_idx = ((start_in_c + ic) * in_h + ih) * in_w + iw;
                                int w_idx = ((oc * in_ch_per_group + ic) * kernel_size + kh) * kernel_size + kw;
                                sum += input[in_idx] * weight[w_idx];
                            }
                        }
                    }
                }
                int out_idx = (oc * out_h + oh) * out_w + ow;
                output[out_idx] = sum;
            }
        }
    }
}

// 全连接层
void linear(const float *input, int in_features,
            const float *weight, const float *bias,
            int out_features,
            float *output)
{
    for (int oc = 0; oc < out_features; oc++)
    {
        float sum = bias ? bias[oc] : 0.0f;
        for (int ic = 0; ic < in_features; ic++)
        {
            sum += input[ic] * weight[oc * in_features + ic];
        }
        output[oc] = sum;
    }
}

// ReLU 量化版本
void relu_quant(float *x, int size, float zero_point)
{
    for (int i = 0; i < size; i++)
    {
        if (x[i] < zero_point)
            x[i] = zero_point;
    }
}

// MaxPooling 2D
void maxpool2d(const float *input, int in_h, int in_w, int channels,
               int kernel_size, int stride, int padding,
               float *output)
{
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    for (int c = 0; c < channels; c++)
    {
        for (int oh = 0; oh < out_h; oh++)
        {
            for (int ow = 0; ow < out_w; ow++)
            {
                float max_val = -INFINITY;
                for (int kh = 0; kh < kernel_size; kh++)
                {
                    for (int kw = 0; kw < kernel_size; kw++)
                    {
                        int ih = oh * stride + kh - padding;
                        int iw = ow * stride + kw - padding;
                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w)
                        {
                            int in_idx = (c * in_h + ih) * in_w + iw;
                            if (input[in_idx] > max_val)
                                max_val = input[in_idx];
                        }
                    }
                }
                int out_idx = (c * out_h + oh) * out_w + ow;
                output[out_idx] = max_val;
            }
        }
    }
}

// 主推理函数
void inference(QuantModel *model, const float *input_img, float *output)
{
    // 输入图像 [1,28,28]
    int in_h = 28, in_w = 28, in_c = 1;

    // ---- 量化输入 ----
    int total_pixels = in_c * in_h * in_w;
    float *qx = (float *)malloc(total_pixels * sizeof(float));
    for (int i = 0; i < total_pixels; i++)
    {
        // q_x = zero_point + x / scale, 然后取整 (round)
        float val = model->qconv1_qi_zero + input_img[i] / model->qconv1_qi_scale;
        qx[i] = roundf(val); // Python 中使用了 round_()
    }

    // ---- qconv1 ----
    // 先减去 qi.zero_point
    for (int i = 0; i < total_pixels; i++)
    {
        qx[i] -= model->qconv1_qi_zero;
    }
    // conv1 参数
    int conv1_out_ch = 40;
    int conv1_out_h = (in_h - 3) / 1 + 1; // 26
    int conv1_out_w = (in_w - 3) / 1 + 1; // 26
    float *conv1_out = (float *)malloc(conv1_out_ch * conv1_out_h * conv1_out_w * sizeof(float));
    conv2d(qx, in_h, in_w, in_c,
           model->qconv1_weight, model->qconv1_bias,
           conv1_out_ch, 3, 1, 0, 1, // groups=1
           conv1_out);
    // 乘以 M 并加上 qo.zero_point
    int conv1_out_size = conv1_out_ch * conv1_out_h * conv1_out_w;
    for (int i = 0; i < conv1_out_size; i++)
    {
        conv1_out[i] = model->qconv1_M * conv1_out[i] + model->qconv1_qo_zero;
    }

    // ---- qrelu1 ----
    relu_quant(conv1_out, conv1_out_size, model->qconv1_qo_zero);

    // ---- qmaxpool2d_1 (2x2, stride=2) ----
    int pool1_out_h = (conv1_out_h - 2) / 2 + 1; // 13
    int pool1_out_w = (conv1_out_w - 2) / 2 + 1; // 13
    float *pool1_out = (float *)malloc(conv1_out_ch * pool1_out_h * pool1_out_w * sizeof(float));
    maxpool2d(conv1_out, conv1_out_h, conv1_out_w, conv1_out_ch,
              2, 2, 0,
              pool1_out);
    free(conv1_out);

    // ---- qconv2 ----
    // 输入已经是量化值，直接减去 qi.zero_point? 在 qconv2.quantize_inference 中第一行是 x - self.qi.zero_point
    // qi 是 qconv1.qo，所以需要减去 qconv1_qo_zero
    int conv2_in_size = conv1_out_ch * pool1_out_h * pool1_out_w;
    for (int i = 0; i < conv2_in_size; i++)
    {
        pool1_out[i] -= model->qconv1_qo_zero;
    }
    // conv2 参数
    int conv2_out_ch = 40;
    int conv2_out_h = (pool1_out_h - 3) / 1 + 1; // 11
    int conv2_out_w = (pool1_out_w - 3) / 1 + 1; // 11
    float *conv2_out = (float *)malloc(conv2_out_ch * conv2_out_h * conv2_out_w * sizeof(float));
    conv2d(pool1_out, pool1_out_h, pool1_out_w, conv1_out_ch,
           model->qconv2_weight, model->qconv2_bias,
           conv2_out_ch, 3, 1, 0, 20, // groups=20
           conv2_out);
    free(pool1_out);
    // 乘以 M 并加上 qo.zero_point
    int conv2_out_size = conv2_out_ch * conv2_out_h * conv2_out_w;
    for (int i = 0; i < conv2_out_size; i++)
    {
        conv2_out[i] = model->qconv2_M * conv2_out[i] + model->qconv2_qo_zero;
    }

    // ---- qrelu2 ----
    relu_quant(conv2_out, conv2_out_size, model->qconv2_qo_zero);

    // ---- qmaxpool2d_2 (2x2, stride=2) ----
    int pool2_out_h = (conv2_out_h - 2) / 2 + 1; // 5
    int pool2_out_w = (conv2_out_w - 2) / 2 + 1; // 5
    float *pool2_out = (float *)malloc(conv2_out_ch * pool2_out_h * pool2_out_w * sizeof(float));
    maxpool2d(conv2_out, conv2_out_h, conv2_out_w, conv2_out_ch,
              2, 2, 0,
              pool2_out);
    free(conv2_out);

    // ---- view (flatten) ----
    int flat_size = conv2_out_ch * pool2_out_h * pool2_out_w; // 40*5*5 = 1000
    // pool2_out 已经是连续存储，可以直接作为 flat 输入

    // ---- qfc ----
    // 减去 qi.zero_point (qconv2.qo)
    for (int i = 0; i < flat_size; i++)
    {
        pool2_out[i] -= model->qconv2_qo_zero;
    }
    float *fc_out = (float *)malloc(10 * sizeof(float));
    linear(pool2_out, flat_size,
           model->qfc_weight, model->qfc_bias,
           10,
           fc_out);
    free(pool2_out);
    // 乘以 M
    for (int i = 0; i < 10; i++)
    {
        fc_out[i] = model->qfc_M * fc_out[i];
    }
    // 加上 qo.zero_point
    for (int i = 0; i < 10; i++)
    {
        fc_out[i] += model->qfc_qo_zero;
    }

    // ---- dequantize ----
    for (int i = 0; i < 10; i++)
    {
        output[i] = model->qfc_qo_scale * (fc_out[i] - model->qfc_qo_zero);
    }

    free(fc_out);
    free(qx);
}

// 读取输入文件
float *read_input(const char *filename, int *size)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
        return NULL;
    fread(size, sizeof(int), 1, fp);
    float *data = (float *)malloc(*size * sizeof(float));
    fread(data, sizeof(float), *size, fp);
    fclose(fp);
    return data;
}

int main(int argc, char **argv)
{
    SetConsoleOutputCP(65001);
    if (argc != 4)
    {
        printf("用法: %s <参数文件> <输入文件> <参考输出文件>\n", argv[0]);
        return 1;
    }

    QuantModel *model = load_model(argv[1]);
    if (!model)
    {
        printf("加载模型失败\n");
        return 1;
    }

    int input_size;
    float *input = read_input(argv[2], &input_size);
    if (!input)
    {
        printf("读取输入失败\n");
        free_model(model);
        return 1;
    }
    // 输入应为 [1,1,28,28] 共 784 个元素
    if (input_size != 784)
    {
        printf("输入大小错误: 期望 784, 得到 %d\n", input_size);
        free(input);
        free_model(model);
        return 1;
    }

    float output[10];
    inference(model, input, output);

    // 读取参考输出并比较
    int ref_size;
    float *ref = read_input(argv[3], &ref_size);
    if (ref && ref_size == 10)
    {
        printf("推理结果 vs 参考输出:\n");
        float max_diff = 0.0f;
        for (int i = 0; i < 10; i++)
        {
            printf("  %d: %f  %f  diff=%f\n", i, output[i], ref[i], fabs(output[i] - ref[i]));
            if (fabs(output[i] - ref[i]) > max_diff)
                max_diff = fabs(output[i] - ref[i]);
        }
        printf("最大绝对误差: %f\n", max_diff);
        if (max_diff < 1e-3)
        {
            printf("结果一致！\n");
        }
        else
        {
            printf("结果差异较大！\n");
        }
        free(ref);
    }
    else
    {
        printf("无法读取参考输出或大小不匹配\n");
    }

    free(input);
    free_model(model);
    return 0;
}