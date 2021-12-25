# 3D-ResNet-Paddle:用于动作识别的3D-ResNet

复现论文：《Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?》


## 论文摘要

本研究的目的是确定当前的视频数据集是否有足够的数据来训练具有时空三维核的非常深卷积神经网络(cnn)。
近年来，3D-cnn在动作识别领域的性能水平有了显著提高。
然而，到目前为止，传统的研究只对相对浅层的3D结构进行了探索。
我们在当前的视频数据集上，从相对较浅到非常深的各种三维cnn的架构进行了研究。
根据这些实验的结果，可以得到以下结论:

(i) ResNet-18训练对UCF-101、hmd -51和ActivityNet有显著的过拟合，但对Kinetics没有显著的过拟合。

(ii) Kinetics数据集有足够的数据用于训练深度3D-cnn，并可以训练多达152个ResNets层，有趣的是，类似于ImageNet上的2D-ResNets。
ResNeXt-101在Kinetics测试集上的平均准确率达到了78.4%。

(iii) Kinetics预处理简单的3D结构优于复杂的2D结构，在UCF-101和hmd -51上，预处理后的resnet-101分别达到94.5%和70.2%。

使用在ImageNet上训练的2D-cnn在图像的各种任务中取得了显著的进展。
我们相信，将深度3D-cnn与Kinetics结合使用，将重现2D-cnn和ImageNet的成功历史，并促进视频计算机视觉的发展。

## 复现指标

| name                 | Accuracies |
|:---------------------|:----------:|
| target               |    42.4    |
| result               |    42.8    |



## 数据集介绍

###  下载

1.通过aistudio平台下载数据集：[UCF-101](https://aistudio.baidu.com/aistudio/datasetdetail/48916)

2.通过aistudio平台下载数据集标签：[UCF101-for-3D_ResNet](https://aistudio.baidu.com/aistudio/datasetdetail/122966)

### 数据集结构目录树

## 训练模型及训练日志

可在aiztudio平台上下载[训练日志和模型](https://aistudio.baidu.com/aistudio/datasetdetail/122966)

## 训练准备
### 构建数据集目录树

```bash
import os
os.makedirs(r'./3D-ResNets-Paddle/data/UCF101/jpg',exist_ok=True)
```

### 解压数据集到指定目录

```bash
!unzip /home/aistudio/data/data122966/UCF101TrainTestSplits-RecognitionTask.zip -d /home/aistudio/3D-ResNets-Paddle/data
!unzip /home/aistudio/data/data48916/UCF-101.zip -d /home/aistudio/3D-ResNets-Paddle/data
```

### 制作数据集

1. 将 **avi**文件 转换为 **jpg** 文件
```bash
python utils/video_jpg_ucf101_hmdb51.py /home/aistudio/3D-ResNets-Paddle/data/UCF-101 /home/aistudio/3D-ResNets-Paddle/data/UCF101/jpg
```
2. 生成 n_frames 文件
```bash
python utils/n_frames_ucf101_hmdb51.py /home/aistudio/3D-ResNets-Paddle/data/UCF101/jpg
```
3. 生成json格式的注释文件，存放到指定位置
```bash
python utils/ucf101_json.py annotation_dir_path /home/aistudio/3D-ResNets-Paddle/data/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist /home/aistudio/3D-ResNets-Paddle/data/UCF101
```


## 测试启动

```bash
python main.py --root_path /home/aistudio/3D-ResNets-Paddle/data --video_path UCF101/jpg --annotation_path UCF101/ucf101_01.json \
--result_path results_2 --dataset ucf101 --model resnet \
--model_depth 18 --n_classes 101 --batch_size 128 --n_threads 4 --checkpoint 5 --std_norm
```

## 训练启动

```bash
python test.py --root_path /home/aistudio/3D-ResNets-Paddle/data --video_path UCF101/jpg --annotation_path UCF101/ucf101_01.json \
--result_path results_val --dataset ucf101 --n_classes 101 \
--pretrain_path /home/aistudio/3D-ResNets-Paddle/data/results_4/save_55.pdparams \
--model resnet --model_depth 18 --batch_size 128 --n_threads 4 --checkpoint 5
```
