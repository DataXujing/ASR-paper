## QuartzNet:Deep Automatic Speech Recognition with 1D Time-Channel Separable Convolutions

### 1.Motivation

端到端的语音识别表现SOTA,但是往往这种模型需要大计算量和大内存的支撑，QuartzNet使用少量的参数和少量的算力实现接近SOTA的结果，并且具有训练快速，部署方便，推断有高的throughput的特点。可以在[NeMo](https://github.com/NVIDIA/NeMo)中使用QuartzNet。

### 2.Related Work

QuartzNet主要受视觉中的小网络的影响比如SqueezeNet, MobileNet, ShuffleNet, EfficientNet,Xception等这里我们将细致的说明MobileNet v1和Xception中使用的深度可分卷积，和ShuffleNet中的分组卷积和Channel Shuffle的概念。关于各种卷积的介绍可以参考笔者的[CNN-Paper中的计算机视觉教程](https://dataxujing.github.io/CNN-paper2/#/zh-cn/chapter11)

**深度可分卷积:**

普通2D卷积如下图所示：


<div align=center>
    <img src="zh-cn/img/ch13/p1.png"   /> 
</div>

可分卷积分为深度可分卷积和空间可分卷积，我们这里着重复习深度可分卷积，深度可分卷积最早出现在MobileNet v1和Xception网络中，其可以使用极少的餐数量实现和普通卷积相似的效果。深度可分卷积分为2步：

第一步（deepwise conv)：我们不使用 2D 卷积中大小为 `3×3×3` 的单个过滤器，而是分开使用 `3` 个核。每个过滤器的大小为 `3×3×1`。每个核与输入层的一个通道卷积（仅一个通道，而非所有通道！）。每个这样的卷积都能提供大小为 `5×5×1`的映射图。然后我们将这些映射图堆叠在一起，创建一个 `5×5×3` 的图像。经过这个操作之后，我们得到大小为 `5×5×3` 的输出。

<div align=center>
    <img src="zh-cn/img/ch13/p2.png"   /> 
</div>

第二步（pointwise conv):为了扩展深度，我们应用一个核大小为 `1×1×3` 的 `1×1` 卷积。将 `5×5×3` 的输入图像与每个 `1×1×3` 的核卷积，可得到大小为 `5×5×1` 的映射图。

<div align=center>
    <img src="zh-cn/img/ch13/p3.png"   /> 
</div>

因此，在应用了 128 个 `1×1` 卷积之后，我们得到大小为 `5×5×128` 的层:

<div align=center>
    <img src="zh-cn/img/ch13/p4.png"   /> 
</div>

通过这两个步骤，深度可分卷积也会将输入层`7×7×3`变换到输出层`5×5×128`:

<div align=center>
    <img src="zh-cn/img/ch13/p5.png"   /> 
</div>



**分组卷积:**

AlexNet最早使用分组卷积的思想在ImageNet上达到了惊人的精度，同时又一次点燃了深度学习在计算机视觉上的应用。首先，典型的 2D 卷积的步骤如下图所示。在这个例子中，通过应用 128 个大小为 `3×3×3` 的过滤器将输入层`7×7×3`变换到输出层`5×5×128`。推广而言，即通过应用 `Dout` 个大小为 `h x w x Din` 的核将输入层`Hin x Win x Din`变换到输出层`Hout x Wout x Dout`。

<div align=center>
    <img src="zh-cn/img/ch13/p6.png"   /> 
</div>

在分组卷积中，卷积核会被分为不同的组。每一组都负责特定深度的2D卷积。下面的例子能让你更清楚地理解。

<div align=center>
    <img src="zh-cn/img/ch13/p7.png"   /> 
</div>


**ShuffleNet v1:**

由于使用$1 \times 1$卷积核进行操作时的复杂度较高，因为需要和每个像素点做互相关运算，作者关注到ResNeXt的设计中，$1 \times 1$卷积操作的那一层需要消耗大量的计算资源，因此提出将这一层也设计为分组卷积的形式。然而，分组卷积只会在组内进行卷积，因此组和组之间不存在信息的交互，为了使得信息在组之间流动，作者提出将每次分组卷积后的结果进行组内分组，再互相交换各自的组内的子组。

<div align=center>
    <img src="zh-cn/img/ch13/p8.png"   /> 
</div>
<div align=center>
    <img src="zh-cn/img/ch13/p9.png"   /> 
</div>

上图`c`就是一个shufflenet块，图`a`是一个简单的残差连接块，区别在于，shufflenet将残差连接改为了一个平均池化的操作与卷积操作之后做cancat，并且将$1 \times 1$卷积改为了分组卷积，并且在分组之后进行了channel shuffle操作。


**Time-Depth Separable Convolutions for ASR:**

除此之外，Hannun等人在《Sequence-to-Sequence Speech Recognition with Time-Depth Separable Convolutions》中使用time-depth seqarable(TDS)卷积实现了Encoder-Decoder的ASR模型，首先input的数据$X$的维度为$T \times w \times c$(time-frequency-channels format)的，这里的$T$为input的时间步，$w$为input的宽，$c$是input的channel。 TDS block包含一个2D卷积模块卷积核大小为$k \times 1$作用在$T \times w$维度上后接一个全连接模块，然后两个$1\times1$的逐点卷积作用在$w\times c$维度上，最后接Layer norm操作。在QuartzNet中我们的卷积首先作用在$T\times c$维度上（time-channel format)


### 3.Model Architecture

#### 3.1 Basic Model

QuartzNet基于Jasper(笔者将在下一节介绍Jasper)的卷积网络训练基于CTC loss。其结构如下图所示：

<div align=center>
    <img src="zh-cn/img/ch13/p13.png"   /> 
</div>

下面我们将深入的解释QuartzNet的网络结构即参数：

+ 表1中的R表示$B_i$结构重复的次数，K表示1为卷积的卷积核大小（这里都是在时间维度T上的1D卷积），C表示输出channel的大小，S表示block的重复次数。
+ C1模块是1D卷积，接Batch Norm和Relu激活
+ C2,C3,C4均是1D`Conv-BN-Relu`结构，C1中的stride是2，C4的dilation为2
+ 每一个$B_i$ block有相同的结构包括1)卷积核是K的depthwise Conv 输出channel为$c_{out}$,2)pointwise Conv,3)normalization layer,4)Relu并重复$R_i$次上述结构
+ 每个$B_i$ block重复$S_i$次

注意表1中的QuartzNet-5x5,QuartzNet-10x5,QuartzNet-15x5表示：
+ QuartzNet-5x5: $B_1-B_2-B_3-B_4-B_5$重复1次
+ QuartzNet-10x5： $B_1-B_1-B_2-B_2-B_3-B_3-B_4-B_4-B_5-B_5$重复2次
+ QuartzNet-15x5：$B_1-B_1-B_1-B_2-B_2-B_2-B_3-B_3-B_3-B_4-B_4-B_4-B_5-B_5-B_5$重复3次


<div align=center>
    <img src="zh-cn/img/ch13/p17.png"   /> 
</div>

标准的1D卷积的卷积核$K$,$c_{in}$为input channel, $c_{out}$为output channel,权重参数个数为$K \times c_{in} \times c_{out}$。time-channel separable Conv有$K\times c_{in} + c_{in} \times c_{out}$个参数，分别对应depthwise layer:$K \times c_{in}$,pointwize layer: $c_{in} \times c_{out}$

论文中作者尝试了4中不同的normalization: batch normalization,layer normalization,group normalization,instance normalization。发现batch normalization更稳定。

#### 3.2 Pointwise convolutions with groups

<div align=center>
    <img src="zh-cn/img/ch13/p15.png"   /> 
</div>

使用group Conv会在损失一定精度的情况下减少参数量，上图表3展示了参数量和精度的trade-off。


### 4.Experiments

QuartzNet在LibriSpeech和WSJ数据上进行了测试，表现均接近SOTA.下图展示了QuartzNet的对比结果：

<div align=center>
    <img src="zh-cn/img/ch13/p10.png"   /> 
</div>

上述结果训练了QuartzNet15x5,400个epochs,在8块V100上训练了大约5天，每个V100上的batch size为32.我们也对比了不同训练超参数下的WER的变化，如下图所示：

<div align=center>
    <img src="zh-cn/img/ch13/p11.png"   /> 
</div>

在WSJ数据上有相似的结果：

<div align=center>
    <img src="zh-cn/img/ch13/p12.png"   /> 
</div>



### 5.Conclusions

QuartzNet是一个全新的端到端的ASR模型，它基于1D time-channel seqarable卷积层，在LibriSpeech和Wall Street Journal(WSJ)数据集上表现SOTA。QuartzNet是一个CTC-based model采用CTC损失。未来我们将探索QuartzNet作为Encoder和attention-based Decoder融合的新方法。