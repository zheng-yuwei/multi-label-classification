# multi-label-classification

基于tf.keras，实现多标签分类CNN模型。

## 如何使用

### 快速上手
1. run.py同目录下新建 `logs`文件夹，存放日志文件；训练完毕会出现`models`文件夹，存放模型；
2. 查看`configs.py`并进行修改，此为参数配置文件；
3. 实际用自己的数据训练时，可能需要执行以下`utils/check_label_file.py`，确保标签文件中的图片真实可用；
4. 执行`python run.py`，会根据配置文件`configs.py`进行训练/测试/模型转换等。

### 学习掌握
1. 先看`README.md`;
2. 再看`1_learning_note`下的note；
3. 看`multi_label`下的`trainer.py`里的`__init__`函数，把整体模型串起来；
4. 看`run.py`文件，结合着看`configs.py`。

## 目录结构

- `A_learning_notes`: README后，**先查看本部分**了解本项目大致结构；
- `backbone`: 模型的骨干网络脚本；
- `dataset`: 数据集构造脚本；
    - `dataset_util.py`: 使用tf.image API进行图像数据增强，然后用tf.data进行数据集构建；
    - `file_util.py`: 以txt标签文件的形式，构造tf.data数据集用于训练；
    - `tfrecord_util.py`: 读取txt标签文件，写tfrecord，然后读取tfrecord为数据集用于训练；
- `images`: 项目图片；
- `logs`: 存放训练过程中的日志文件和tensorboard文件（当前可能不存在）；
- `models`: 存放训练好的模型文件（当前可能不存在）；
- `multi_label`: 多标签分类模型构建脚本；
    - `classifier_loss.py`: 多标签分类的损失函数，包含多种损失函数：`focal loss`、`GHM`等；
    - `classifier_model.py`: 多标签分类模型，负责调用`backbone`里的骨干网络和本脚本中的多标签`head`组成整体模型；
    - `train.py`: 模型训练接口，集成模型构建/编译/训练/debug/预测、数据集构建等功能；
- `utils`: 一些工具脚本；
    - `generate_txt`: 扫描指定路径下的图片数据，生成训练、测试等label.txt（根据实际项目而定，当前可能不存在）；
    - `check_label_file.py`: 在训练前检查训练集，确保标签文件中的图片真实可用；
    - `draw_tools.py`: 模型训练完进行测试时，绘制每个类别的混淆图；
    - `logger_callback.py`: 日志打印的keras回调函数；
    - `radam.py`: RAdam算法的tf.keras优化器实现；
- `configs.py`: 配置文件；
- `run.py`: 启动脚本；


## 算法说明

在**多标签多分类模型**基础上，添加功能：
- loss函数改造：
    - `label smoothing`: 标签平滑。
    - `focal loss`: 给每个样本的分类loss增加一个因子项，降低分类误差小的样本的影响，解决难易样本问题。
    > ![focal loss类别概率和损失关系图](https://github.com/zheng-yuwei/multi-label-classification/blob/master/images/focal-loss.jpg)
    - `gradient harmonizing mechanism (GHM)`: 
    根据样本梯度密度曲线（这里的梯度是梯度范数，并且不是所有网络参数的梯度，而是最后一层的回传梯度），
    取反得到梯度密度调和参数（和平衡多类别数据集一个意思，只不过这里不是按类别来平衡，而是按梯度区间来平衡），
    再乘以梯度以**调整梯度贡献曲线**，从而降低高密度区域的梯度贡献比例，提升低密度区域的梯度贡献比例。
    > ![GHM论文梯度分布与贡献图](https://github.com/zheng-yuwei/multi-label-classification/blob/master/images/GHM-insight.jpg)
    >
    > 原论文insight： 对网络训练而言，梯度是最重要的东西，而网络训练不好，也是因为梯度没调节好。
    focal loss认为前背景不平衡问题，本质为难易样本不平衡问题，从而调节样本的梯度贡献，一定程度上解决了背景问题。
    作者认为，类别不平衡、难易样本不平衡，造成的本质驱动是梯度不平衡。
    > 然后通过绘制训练好的模型在样本空间上的梯度分布曲线，发现小梯度和大梯度都是高密度区域，
    （作者认为小梯度对应易学习样本，大密度对应异常样本）；
    然后绘制正常loss和focal loss梯度贡献曲线，发现正常loss中，高密度区域的梯度贡献度很高，
    而focal loss中，小梯度的高密度区域被因子项惩罚而降低梯度贡献度，
    但大梯度的高密度区域的梯度贡献度依然很高。
    作者认为focal loss平衡了一部分梯度贡献度，所以使得训练低密度的中间梯度的梯度贡献度影响提升，
    提升了算法性能；同时，认为focal loss并没有从本质出发，所以还有残留问题（异常样本大梯度的高密度区域）。
    然后提出了GHM，从梯度分布和梯度贡献角度出发，提升网络训练效果。
    
- 分离conv层的权重衰减项$\lambda_{conv}$ 和 BN层gamma的权重衰减项$\lambda_{gamma}$  


## 缓解过拟合/标注错误/样本错误（稍微按效果分先后，按实际数据来）

1. 一定程度提高BN层中gamma的L2权重衰减，conv层的L2权重衰减可以维持不变，去掉bias；[1,2,3]
1. 加大batch，然后要用warmup（我一开始用adam+warmup,后面用radam+warmup, radam中用动态学习率）；[4,5,6]
1. 白化预处理；
1. 修改网络结构，resnext18相比resnet18多了结构正则的作用，效果好些；
1. 剪枝，其实和修改网络结构一个道理，只不过剪枝可以类似NAS自动找到更好的sub-network(网络结构)；[3,9,10]
1. GHM损失函数；[8]
1. 数据增强（增加数据量）；
1. label smoothing:；[7]

TIPS：其他试过但基本无效的手段包括：
继续加大weight decay权重，BN层的gamma不加weight decay，BN层的beta加weight decay，
全连接层加dropout，focal loss，从Adam训练改为SGDM，加warmup。

[1] L2 Regularization versus Batch and Weight Normalization  
[2] Towards Understanding Regularization in Batch Normalization  
[3] Learning Efficient Convolutional Networks through Network Slimming  
[4] Accurate, Large Minibatch SGD：Training ImageNet in 1 Hour  
[5] Large Batch Training of Convolutional Networks  
[6] On the Variance of the Adaptive Learning Rate and Beyond  
[7] Rethinking the inception architecture for computer vision  
[8] Gradient Harmonized Single-stage Detector  
[9] Data-Driven Sparse Structure Selection for Deep Neural Networks  
[10] Rethinking the Value of Network Pruning

## TODO
1. 解决类别不平衡的做法：
    - reweighted sample从而实现self-balance（参考sklearn）；
    - 先用训练一个网络然后采样平衡数据集做finetune。
1. 使用GAN生成数据，进行数据增强；
1. Handwriting Recognition in Low-resource Scripts Using Adversarial Learning。

