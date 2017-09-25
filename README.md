# 机器学习纳米学位毕业项目
## 侦测走神司机

## FloydHub
本项目主要在[FloydHub](https://www.floydhub.com)的云计算服务器上完成。服务器配置Nvidia Tesla K80 GPU。环境为`tensorflow`。相对于 AWS 的 p2.xlarge 服务器，FloydHub 的 CPU 速度较慢，但 GPU 速度和内存应该比 p2.xlarge 大。相同的文本和代码情况下，FloydHub上batch_size=32毫无压力，而 p2.xlarge 只能 batch_size=8。

原始图片数据通过终端上传至服务器，名称为 `mingyi/datasets/drivers_original/2`。原始计算文件保存于项目`mingyi/projects/keras_test`中。在Keras文件夹下进行`floyd init`命令后，用以下命令载入数据和文本
```shell
floyd run --data mingyi/datasets/drivers_original/2:dataset_dir --mode jupyter --gpu --env tensorflow
```

## 库
主项目：
1. [Keras 2.0.6](Keras.io), TensorFlow 后台。
2. `sklearn`, `numpy`, `pandas`
3. `matplotlib.pyplot`, `cv2`, `skimage.io`
4. `tqdm`

基准模型：[TensorFlow-Slim image classification model library](https://github.com/tensorflow/models/tree/master/slim)

## 系统
本地：Windows 10,  python 3.6.2

## 运行时间
训练每一个fold大约两个半小时。总共10-fold， 25小时。
KNN计算大约8小时。

## 文件目录
#### Keras: 主项目文件夹
5-fold cross-validation

| 基础模型 | 运行文件                                     | 输出文件                          | LB-Private | LB-Public |
|----------|----------------------------------------------|-----------------------------------|------------|-----------|
| VGG16    | `Keras_fine_tuning_aug_fold0.ipynb`          | `submission_vgg16_ft0_aug.csv`    | 0.46152    | 0.54683   |
| VGG16    | `Keras_fine_tuning_aug_fold1.ipynb`          | `submission_vgg16_ft1_aug.csv`    | 0.34059    | 0.34422   |
| VGG16    | `Keras_fine_tuning_aug_fold2.ipynb`          | `submission_vgg16_ft2_aug.csv`    | 0.44347    | 0.63316   |
| VGG16    | `Keras_fine_tuning_aug_fold3.ipynb`          | `submission_vgg16_ft3_aug.csv`    | 0.44602    | 0.36146   |
| VGG16    | `Keras_fine_tuning_aug_fold4.ipynb`          | `submission_vgg16_ft4_aug.csv`    | 0.55130    | 0.52167   |
| ResNet50 | `Keras_fine_tuning_ResNet50_aug_fold0.ipynb` | `submission_resnet50_ft0_aug.csv` | 0.35491    | 0.48216   |
| ResNet50 | `Keras_fine_tuning_ResNet50_aug_fold1.ipynb` | `submission_resnet50_ft1_aug.csv` | 0.45376    | 0.42659   |
| ResNet50 | `Keras_fine_tuning_ResNet50_aug_fold2.ipynb` | `submission_resnet50_ft2_aug.csv` | 0.26854    | 0.26618   |
| ResNet50 | `Keras_fine_tuning_ResNet50_aug_fold3.ipynb` | `submission_resnet50_ft3_aug.csv` | 0.41992    | 0.49777   |
| ResNet50 | `Keras_fine_tuning_ResNet50_aug_fold4.ipynb` | `submission_resnet50_ft4_aug.csv` | 0.37796    | 0.36269   |

集成学习</br>
`Keras_5fold.ipynb`

KNN</br>
`KNN.ipynb`


#### baseline：基准模型
`Resnet50_baseline_train_validation.ipynb`</br>
`Resnet50_baseline_test.ipynb`

#### submissions：主要上传预测</br>
