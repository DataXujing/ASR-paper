## Citrinet:Closing the Gap between Non-Autoregressive and Autoregressive End-to-End Models for Automatic Speech Recognition


### 1.Motivation

目前端到端的ASR模型主要分为3类：
+ CTC model
+ seq2seq模型+attention比如LAS,Transducer,Transformer with Seq2seq loss
+ RNN-Transducer(RNN-T)

CTC模型比自回归的模型更容易训练，但是seq2seq,RNN-T精度往往比CTC高（因为CTC有强条件独立性的假设）如下表所示：

<div align=center>
    <img src="zh-cn/img/ch15/p1.png"   /> 
</div>

Citrinet是一个基于1D time-channel separable convolutions的CTC based的ASR模型，其构造基于QuartzNet和ContextNet的squeeze-and-excitation结构。在没有语言模型的情况下，Citrinet-1024在LibriSpeech test-other数据上WER为6.22%，Multi-lingual LibriSpeech(MLS) English WER为8.46%，TEDLIUM2 WER为8.9%，AISHELL-1 test-set WER为6.8%。

### 2.Model architecture

在介绍Citrinet之前，我们将介绍SENet,因为Citrinet和下一章中ContextNet均用到了Squeeze-and-Excitation Networks。

**SENet结构详解：**

<!--SENet https://blog.csdn.net/Evan123mg/article/details/80058077 -->
SENet的核心思想在于通过网络根据loss去学习特征权重，使得有效的feature map权重大，无效或效果小的feature map权重小的方式训练模型达到更好的结果。SE block嵌在原有的一些分类网络中不可避免地增加了一些参数和计算量，但是在效果面前还是可以接受的 。Sequeeze-and-Excitation(SE) block并不是一个完整的网络结构，而是一个子结构，可以嵌到其他分类或检测模型中。

<div align=center>
    <img src="zh-cn/img/ch15/p2.png"   /> 
</div>

上图是SE模块的示意图。给定一个输入$x$，其特征通道数为$c_1$，通过一系列卷积等一般变换后得到一个特征通道数为$c_2$的特征。与传统的 CNN 不一样的是，接下来通过三个操作来重标定前面得到的特征。

+ 首先是 Squeeze 操作，顺着空间维度来进行特征压缩，将每个二维的特征通道变成一个实数，这个实数某种程度上具有全局的感受野，并且输出的维度和输入的特征通道数相匹配。它表征着在特征通道上响应的全局分布，而且使得靠近输入的层也可以获得全局的感受野，这一点在很多任务中都是非常有用的。
+ 其次是 Excitation 操作，它是一个类似于循环神经网络中门的机制。通过参数 $w$ 来为每个特征通道生成权重，其中参数 $w$被学习用来显式地建模特征通道间的相关性。
+ 最后是一个 Reweight 的操作，将 Excitation 的输出的权重看做是进过特征选择后的每个特征通道的重要性，然后通过乘法逐通道加权到先前的特征上，完成在通道维度上的对原始特征的重标定。

首先Ftr这一步是转换操作（严格讲并不属于SENet，而是属于原网络，可以看后面SENet和Inception及ResNet网络的结合），在文中就是一个标准的卷积操作而已，输入输出的定义如下表示。

<div align=center>
    <img src="zh-cn/img/ch15/p3.png"   /> 
</div>

那么这个Ftr的公式就是下面的公式（卷积操作，$v_c$表示第$c$个卷积核，$x^s$表示第$s$个输入）。

<div align=center>
    <img src="zh-cn/img/ch15/p4.png"   /> 
</div>

Ftr得到的$U$就是SENet结构图中的左边第二个三维矩阵，也叫Tensor，或者叫$C$个大小为$H\times W$的feature map。而$u_c$表示$U$中第$c$个二维矩阵，下标$c$表示channel。

接下来就是 Squeeze操作，公式非常简单，就是一个global average pooling:

<div align=center>
    <img src="zh-cn/img/ch15/p5.png"   /> 
</div>

因此公式(2)就将$H\times W\times C$的输入转换成$1\times 1\times C$的输出，对应结构图中的Fsq操作。 为什么会有这一步呢？这一步的结果相当于表明该层$C$个feature map的数值分布情况，或者叫全局信息。

再接下来就是Excitation操作，如公式3。直接看最后一个等号，前面squeeze得到的结果是$z$，这里先用$W_1$乘以$z$，就是一个全连接层操作， $W_1$的维度是$\frac{C}{r} \times C$，这个$r$是一个缩放参数，在文中取的是16，这个参数的目的是为了减少channel个数从而降低计算量。又因为$z$的维度是$1\times 1\times C$，所以$W_1z$的结果就是$1\times 1\times \frac{C}{r}$；然后再经过一个ReLU层，输出的维度不变；然后再和$W_2$相乘，和$W_2$相乘也是一个全连接层的过程， $W_2$的维度是$C\times \frac{C}{r}$，因此输出的维度就是$1\times 1\times C$；最后再经过sigmoid函数，得到$s$。

<div align=center>
    <img src="zh-cn/img/ch15/p6.png"   /> 
</div>

也就是说最后得到的这个$s$的维度是$1\times 1\times C$，$C$表示channel数目。 这个$s$其实是本文的核心，它是用来刻画Tensor $U$中$C$个feature map的权重。而且这个权重是通过前面这些全连接层和非线性层学习得到的，因此可以end-to-end训练。这两个全连接层的作用就是融合各通道的feature map信息，因为前面的squeeze都是在某个channel的feature map里面操作。
在得到$s$之后，就可以对原来的Tensor $U$操作了，就是下面的公式(4)。也很简单，就是channel-wise multiplication，什么意思呢？$u_c$是一个二维矩阵，$s_c$是一个数，也就是权重，因此相当于把$u_c$矩阵中的每个值都乘以$s_c$。对应结构图中的Fscale。

<div align=center>
    <img src="zh-cn/img/ch15/p7.png"   /> 
</div>

**Citrinet：**

Crtrinet是一个由1D time-channel separable conv构成的CTC-based的模型，其结构类似于QuartzNet,并增加了1D Squeeeze-and-Excitation（SE) context 模块，整体结构如下图所示：

<div align=center>
    <img src="zh-cn/img/ch15/p8.png"   /> 
</div>

+ Citrinet-BxRxC:B表示block的个数，R表示每个block中sub-block的个数，C表示每个block中卷积核的个数
+ Citrinet输入的声学特征：80维的fbank,每一帧是25ms,帧与帧之间有10ms的重叠
+ Citrinet结构开始于$B_0$ block,随后接三组大的block:$B_1...B_6$,$B_7...B_{13}$,$B_{14}...B_{21}$,最后接$B_{22}$
+ 每一个大的block由stride为2的1D time-channel separable Conv
+ 每一个大的block都与上一个block有一个残差连接结构 
+ 每个残差结构由基本的QuartzNet block 重复了R次和一个SE模块添加在最后
+ QuartzNet:一个QuartzNet block由1D time-channel separbale Conv，其卷积核的大小为$K$,batch norm,ReLU,和dropout layer.除了开始的block$B_0$所有的卷积层有相同的channel个数$C$
+ Citrinet与QuartzNet不同的一点是1D卷积层支持不同的kernel大小，如下图所示：

<div align=center>
    <img src="zh-cn/img/ch15/p11.png"   /> 
</div>

小的$K_1$适合流式识别，宽的$K_4$适合离线识别。
+ Citrinet使用Squeeze-and-Excitation(SE)模式(参考ContextNet),SE block 计算以一个通道的权重$\theta(x)$
$$\theta(x)=\sigma(W_2(ReLU(W_1\bar{x}+b_1))+b_2)$$
这里的$W_1,W_2,b_1,b_2$是可学习的参数，$\sigma$是sigmoid函数，$\bar{x}=\frac{1}{T}\sum_tx_t$.权重$\theta(x)$和input feature map进行逐点矩阵乘
$$SE(x)=\theta(x)\odot x$$


### 3.Experiments

下图是Citrinet在不同数据集上的测试结果：

<div align=center>
    <img src="zh-cn/img/ch15/p9.png"   /> 
</div>

<div align=center>
    <img src="zh-cn/img/ch15/p10.png"   /> 
</div>

### 4.Conclusions

在这篇paper中，介绍了Citrinet,一种端到端的非自回归CTC-based的模型，Citrinet增强了前面章节介绍的QuartzNet结构,使用ContextNet中的Squeeze-and-Excitation机制。Citrinet有效的缩短了non-autoregressive model和SOTA的 autorehressive Seq2Seq，RNN-T模型之间的gap。我们可以通过[NeMo](https://github.com/NVIDIA/NeMo)训练该模型。