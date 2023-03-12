## ContextNet:Improving Convolutional Neural Networks for Automatic Speech Recognition with Global Context

<!-- https://zhuanlan.zhihu.com/p/415981398 -->


### 1.Abstract

CNN在ASR方向上，已经展示了其牛叉的一面，但是仍然和基于RNN，基于Transformer的模型的表现有差距。本论文就是填这个坑的，即构建了一个名称为CNN-RNN-Transducer的架构，命名为ContextNet。

在ContextNet中，使用一个全卷积编码器，该编码器通过使用squeeze-and-excitation（挤压+激发）模块来捕捉全局相关性。

此外，ContextNet中还有个简单的比例方法，其可以对ContextNet的”宽度“进行缩放，从而实现计算量和精度之间的”平衡“。

在效果上，使用Librispeech数据集合，ContextNet可以达到的精度如下：
+ 不使用外部语言模型的前提下，WER在test-clean上是2.1%；在test-other上是4.6%
+ 使用外部语言模型的话，效果分别为test-clean上的1.9%，和test-other上的4.1%
+ 如果模型只有10M-千万级别参数，则效果分别为test-clean上的2.9%和test-other上的7.0%,这个效果，可以匹敌先行研究中的20M参数规模下的精度。
+ Google内部数据集合上，也验证了ContextNet的卓越性能

### 2.Introduction

针对卷积神经网络在ASR上的应用，作者提到了Nvidia的JasperNet(我们在之前章节中已经做了细致的介绍），其在Librispeech test-clean上可以达到2.95%的WER，已经和基于RNNt-T和Transformer的效果很接近了，不过还有进一步提高的空间。

CNN和RNN/Transformer的最大区别在于”感受野“，即对于整体上下文（wav的）的特征的表示能力的问题。如果window size 或者kernel size开的太小，则只有有限的局部时域特征被表示出来，缺乏对全局信息的表示能力。

而本文作者强调，这种”全局信息“，全局上下文，是CNN略逊于RNN-T/Transformer的重要的原因。这也构成了本论文的ContextNet的motivation。

为了让CNN拥有表示全局上下文信息的能力，该论文引入了squeeze-and-excitation（挤压+激发）(SE):

+ 一个SE层，它先是把一个局部特征向量的序列”压榨“成一个单个的全局上下文向量
+ 然后把这个全局上下文向量反向传回给每个局部特征向量
+ 然后通过乘积，把被改造后的局部特征向量和全局上下文向量综合起来

在关于Hybrid ASR的先行研究中，曾经有如下一些经典的ideas：

+ 堆砌更多的CNN层
+ 独立地训练一个全局向量，该向量表示的是speaker和环境相关的信息
+ SE之前被用于RNN中，来做无监督adaptation，（例如domain迁移，从一般领域到一些垂直领域，例如金融ASR，医学ASR等等）；
+ 本论文中，作者认为SE也可以用来强化CNN(这个思想来源于SENet,不是其首创，关于SENet请参开CitriNet中笔者对SENet的介绍)

此外，本论文的ContextNet，一定程度上也借鉴了Nvidia的QuartzNet，使用depthwise separable 1D卷积。当然也有很多不同的地方：

+ 本文使用的是RNN-T解码器而不是CTC解码器
+ 本文使用swish激活函数其对WER略有贡献
+ **相比QuartzNet有更好的效果,而且ContextNet超越了基于LSTM，Transformer的模型**

本文还介绍了一个加速解码的方法，”逐步下采样和模型收缩方案“(a progressive downsampling and model scaling scheme)后续详细介绍。效果图如下：

<div align=center>
    <img src="zh-cn/img/ch16/p1.png"   /> 
</div>

上图中的红色五角星，表示的就是ContextNet的WER效果。可以简单认为，其结果都在帕累托最优曲线上。
或者可以简单理解为，要达到同样的WER效果，ContextNet需要更少的参数,或者在同等参数规模下ContextNet可以达到更小的WER。


### 3.Model

#### 3.1 端到端的网络: CNN-RNN-Transducer

基本的RNN-T其包括三个部分：
1. audio encoder，对输入wav的语音编码器
2. 对输入label/text的label encoder(文本方向)
3. 一个连接上面两者wav和text的表示的网络，并且负责解码

而本论文的最主要的创新点在于1，对wav的更好的使用CNN的编码方法。

#### 3.2 编码器设计

输入wav序列，表示为:$x=(x_1,...,x_T)$，时长为$T$,而编码器的工作，就是把输入$x$，用高维向量集合表示：
$$h=(h_1, ..., h_{T^{'}})$$其中$T^{'} <= T$。

从而，audioencoder(.)就被定义为：

$$h = AudioEncoder(x) = C_k (C_{k-1} (... C_1(x)))$$

这里面的每个$C_k$表示的是一个卷积块，包括若干卷积层，每个卷积层后边接的是batch normalization，以及一个激活函数。卷积块中也包括了squeeze-and-exciation 模块，和skip connection-残差求和操作。

下面就是细化地看$C_k$，一个卷积块中都有啥了。

**squeeze-and-excitation:**

<div align=center>
    <img src="zh-cn/img/ch16/p2.jpg"   /> 
</div>

而上图对应的公式如下：


<div align=center>
    <img src="zh-cn/img/ch16/p3.png"   /> 
</div>

从$x$开始，先经过平均池化，然后是两个线性层，之间有个非线性激活函数Act，然后再接一个sigmoid函数，把每个维度的值映射到(0,1)区间。之后，就是$\theta(x)$和$x$的逐点乘积了，这样就得到了”挤压-激发“的输出`SE(x)`。

**Depthwise separable convolution:**

这一部分参考之前关于QuartzNet中的关于深度可分卷积的介绍。

**Swish 激活函数：**

<div align=center>
    <img src="zh-cn/img/ch16/p4.png"   /> 
</div>

需要注意到的是，在”挤压-激发“SE网络中，使用了一个Act非线性激活函数，上面就是这个激活函数的具体选择。实验证明，Swish比ReLU效果更好一些。

**卷积块:**

据此，就可以图示出来一个卷积块内部的构造：

<div align=center>
    <img src="zh-cn/img/ch16/p5.png"   /> 
</div>

上图绘制了n个$Conv_i, BN, Act$这样的”子块“，其可以被表示为：

$$f(x) = Act(BN(Conv(c)))$$

这样的化，就相当于在SE之前，我们有了n个这样的函数，（论文中是m个），即：

<div align=center>
    <img src="zh-cn/img/ch16/p6.png"   /> 
</div>

m个f函数，然后是送给SE这个模块，即”挤压-激发“；同时，$x$会经过上面的弧，即`conv->bn`，然后通过残差连接和`SE()`结合起来，这样操作之后，再最后扔给Act激活函数。

这里，又涉及到了downsampling-下采样的操作，即允许上面的m个函数中，第一层和最后一层和其他的中间层不一样

1. 如果一个卷积块，需要把输入的句子下采样为2倍，（稀疏化两倍），那么，最后一层的步长为2，而其他层的步长为1
2. 一般地，如果卷积块的输入为D-in维度，输出为D-out维度，则，第一层卷积负责把D-in转换为D-out，其他m-1层函数都是从D-out到D-out的。
3. 基于如此的规定，上面的映射函数P，和f1的使用的stride步长相同，就可以了

**渐进式下采样：**

上面也提到了，使用滑块步长来控制下采样的”精度“，但是过度的下采样可能会损害到编码器的表示能力。这样的话，基于实验，我们发现，一个渐进式的`8x的下采样的方案`可以很好地平衡速度和精度。(参考下面的消融实验)

**ContextNet的配置细节：**

所有的卷积块都有五层卷积，m=5除了`C0`和`C22`，他们都是只有一层卷积。下面的表格中罗列的是具体的配置：

<div align=center>
    <img src="zh-cn/img/ch16/p7.png"   /> 
</div>


这里面的$\alpha$就是控制缩放的重要的超参数了。当$\alpha>1$的时候，则是说明整个ContextNet更加宽了，从而模型参数更多，可能会导致“模型的表达能力”更好。

特别需要注意的一点是，这里一共有三组`stride=2`，也就意味着一共进行了`2*2*2=8`倍的下采样！（`8x`）记住这个很重要，因为后续需要和`2x`的下采样进行对比。


### 4.Experiments

所有实验都是在LibriSpeech这个（几乎被刷爆了的）数据集上进行,970小时。抽取的梅尔谱是80维度的，使用的window是25ms，以及步长为10ms。解码器是单层`LSTM，dimension=640`。

语言模型方面，是3层`LSTM LM，width=4096`，使用的文本是LibriSpeech Lanuage Model Corpus，并追加了LibriSpeech960h的transcripts文本。所有的模型都是使用`lingvo tooklit`来实现的。

<div align=center>
    <img src="zh-cn/img/ch16/p8.png"   /> 
</div>

上图可以看到，在比QuartzNet更小的参数规模下(S），达到了更好的效果！

还记得SE（挤压-激发网络）中的pooling吗，根据window size的大小，其可以控制“全局上下文相关性”信息有多少被采纳

<div align=center>
    <img src="zh-cn/img/ch16/p9.png"   /> 
</div>

而基于这个pooling的window size，做的消融实验的效果如下：

<div align=center>
    <img src="zh-cn/img/ch16/p10.png"   /> 
</div>

可以看到，window size越大，引入的全局信息越多，效果越好。

<div align=center>
    <img src="zh-cn/img/ch16/p11.png"   /> 
</div>

这里的8倍下采样，是基于Table1中的三次stride=2的设定而来的。而只有2倍的下采样，则是只对`C3`进行一次`stride=2`，其他的如`C7`和`C14`都不变，这样就得到了2倍下采样的结果。

可以看到，8倍下采样，不但速度是2倍下采样的`1/2`还少，而且精度上没有损失（甚至相同kernel size）下更好。这也说明了，多倍速下采样对于ContextNet的效果是有效的！

<div align=center>
    <img src="zh-cn/img/ch16/p12.png"   /> 
</div>

上面的表格，调研了contextNet的“宽度”-width对于效果的影响，这是通过$\alpha$的取值来控制的，而可以看到，当`alpha=2`的时候，相比`alpha=1`的时候，模型增大了接近四倍，然后精度上WER上，有一定的提升。这也刻画了，模型越大，具有相对更好的表达能力。

最后，是在超大规模语料上的结果了.[https://arxiv.org/pdf/1911.02242.pdf]上说，用了125K小时的训练数据，数据来源于youtube。其结果为：

<div align=center>
    <img src="zh-cn/img/ch16/p13.png"   /> 
</div>

ASR模型可以这么小啊！112M，这还没有Bert-large的350M的`1/3`.效果上，无论是速度还是youtube test set（24.12小时）的WER的精度都不错。

### 5.Conclusion

ContextNet，基于CNN，以及SE block。效果不错，一则是使用了SE，二则是是用了渐进式的下采样方法。这两个方法，贡献了大部分的收益。这个工作简介而容易理解。