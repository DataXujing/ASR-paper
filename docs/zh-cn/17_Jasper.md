## Jasper:An End-to-End Convolutional Neural Acoustic Model

<!-- https://zhuanlan.zhihu.com/p/141178215 -->
<!-- https://blog.csdn.net/qq_32458499/article/details/81513720 -->
<!-- https://zhuanlan.zhihu.com/p/524310167 -->

### 1.Motivation

Jasper的灵感来源于wav2letter的卷积方法，其中融入了TDNN(我们将在TDNN章节给出详细介绍)，GLU,GAU,DenseNet,DenseRNet。可以通过[NeMo](https://github.com/NVIDIA/NeMo)训练该模型。本文的主要贡献有4个:
+ 提供了一个计算高效的端到端的卷积神经网络声学模型
+ 通过消融实验，找到了ReLU和batch norm是比较稳健的一种正则化的组合，并且residual connection在网路训练中也是必要的
+ 介绍了一种新的优化器：NovoGrad
+ 在LibriSpeech test-clean数据集上WER表现SOTA

### 2.Related Work

我们将从Jasper参考的相关模型和技术进行介绍包括DenseNet,DenseRNet,GLU,GAU激活关于TDNN我们将在其他章节介绍。

**DenseNet:**

<div align=center>
    <img src="zh-cn/img/ch14/p1.jpg"   /> 
</div>

上图展示了ResNet的结构，挑层和输入进行Add，而对于DenseNet,则是采用concat的形式连接，如下图所示：

<div align=center>
    <img src="zh-cn/img/ch14/p2.jpg"   /> 
</div>

当CNN的层数变深时,输出到输入的路径就会变得更长,这就会出现一个问题:梯度经过这么长的路径反向传播回输入的时候很可能就会消失,那有没有一种方法可以让网络又深梯度又不会消失?DenseNet提出了一种很简单的方法,DenseNet直接通过将前面所有层与后面的层建立密集连接来对特征进行重用来解决这个问题，我们可以看到每一层的输出都连接到了后一层,这样对于一个$L$层的网络来说就会有$L(L+1)/2$个连接

<div align=center>
    <img src="zh-cn/img/ch14/p3.gif"   /> 
</div>

这里要注意,因为我们是直接跨通道直接做concat,所以这里要求不同层concat之前他们的特征图大小应当是相同的,所以DenseNet分为了好几个Dense Block,每个Dense Block内部的feature map的大小相同。而每个Dense Block之间使用一个Transition模块来进行下采样过渡连接。

<div align=center>
    <img src="zh-cn/img/ch14/p3.jpg"   /> 
</div>

Transition层就是用来连接每一个Dense Block,他的构成是`1x1`的卷积和`2x2`的`AvgPooling`,也就是上文提到的下采样,压缩模型的功能.假定上一层得到的feature map的channel大小为$m$,那经过Transition层就可以产生$\theta m$个特征,其中$\theta$是0和1之间的压缩系数。

**DenseRNet:**

了解了DenseNet和ResNet,《Tang, Jian, Song, Yan, Dai, Li-Rong and McLoughlin, Ian Vince (2018) Acoustic
Modeling with Densely Connected Residual Network for Multichannel Speech Recognition.》中构建了DenseRNet，其基本结构如下图所示：

<div align=center>
    <img src="zh-cn/img/ch14/p4.png"   /> 
</div>

**GLU:**

门控线性单元(Gated Linear Unit，GLU）出自[Dauphin et al., 2016](https://arxiv.org/abs/1612.08083),其一般的形式为：

$$h(x)=\sigma(xW+b)\otimes (xV+c)$$

或者

$$GLU(x,W,V,b,c) = \sigma(xW+b)\otimes (xV+c)$$

即$x$的两个线性映射的逐点卷积，其中一个先经过sigmoid函数，这里的$\sigma$是sigmoid函数，$\otimes$是逐帧逐点卷积又叫做Hadamard积，有时候也用$\odot$表示。关于GLU的介绍和相关变体的介绍可以参考[知乎：GLU介绍](https://zhuanlan.zhihu.com/p/524310167)

**GAU:**
<!-- https://zhuanlan.zhihu.com/p/487423771 -->

在讲解Gate Attention Unit(GAU)之前，让我们复习一下transformer中的多头自注意力机制(Multi Head Self Attention:MHSA)，首先介绍self-attention。

self-attention的本质就是将输入向量$E=[e_1,e_2,...,e_n]$通过三个权重矩阵$W_q,W_k,W_v$进行线性变换得到想用的query,key,value矩阵，也就是$Q,K,V$,通过计算$Q,K$向量矩阵的乘积得到注意力值（这里使用的就是矩阵乘积），那么可以想象的就是同一个输入在不同权重矩阵下线性变换后的值进行矩阵乘法，不就是输入的每个向量都进行互相交互吗？过程如下图所示：

<div align=center>
    <img src="zh-cn/img/ch14/p9.png"   /> 
</div>

上图就是$Q,K$计算的具体步骤，那么这里又有一个问题，为什么这个就可以代表注意力矩阵呢？我们知道，在线性代数中，两个向量$A,B$的乘积可以看做向量$A$向$B$所在的直线投影的标量大小，这个乘积扩展到多位后就是我们熟知的欧氏距离，既然是欧式距离，那么是不是就能代表两个向量之间的相似性呢！基于这个理解，就可以知道其实$Q,K$的乘积得到的注意力矩阵就是文本中每个字符与自己以及其他字符之间的相似矩阵，这样就知道为什么self-attention可以表示每个字符之间的相关性了。

下一步要做的就是Scale(缩放)了，也就是注意力矩阵除以一个Scale value,在 Multi head attention中除以$\sqrt{head size}$(head size=64),这里是为了不让注意力集中在部分数值上，起到一个分散注意力的作用，提升泛化效果。

然后加上一个shape与注意力矩阵一致的mask矩阵，处理文本的padding(这部分可选)。得到的注意力权重矩阵进过softmax函数进行归一化，缩放到$[0,1]$之间与另一个线性变换矩阵$V$进行矩阵乘。通过注意力权重矩阵对$V$进行加权得到最后输出的特征矩阵。

上面所述基本上就是一个完整的Self-Attention机制了，而Transformer中的Multi Head Self Attention就是在对输入进行线性变换时，采用多组变换权重$W_q,W_k,W_v$,来对其进行线性变换，从不同的维度对文本进行特征抽取，得到多组$q_i,k_i,v_i$  ，然后对这些值进行concat拼接后进行线性映射。这就是多头注意力的基本思想。

Gate Attention Unit和 Multi Head Attention的区别和联系如下图所示：

<div align=center>
    <img src="zh-cn/img/ch14/p10.jpg"   /> 
</div><p align=center>GLU与MHSA以及GAU</p>

基于上图，我们需要考虑一下GLU与MHSA之间最大的不同点在哪，通过上一节的分析我们可知，MHSA实现了不同特征之间的交互，而GLU并没有这种性质，所以我们这里需要保留MHSA的特征交互的结构，即$Qk^T$部分，那么，结合这二者的结构，就可以很自然的得到上图右边的部分。可表示为：

$$O=(U\otimes AV)W_0$$

这里的$A\in R^{T\times T}$  ，表示注意力权重矩阵。来到这一步，那么很自然的一个设计思路就是将多头注意力的计算方式套用过来，论文中的思路表示为：

<div align=center>
    <img src="zh-cn/img/ch14/p11.png"   /> 
</div>

那么，到这里就明了了，在GAU中我们通过计算$AV$可以得到文本的context layer(上下文向量矩阵)，通过$(U\otimes AV)W$这一整步可以替换Transformer中的FFN层，如此一来通过GLU与MHSA进行结合可以得到GAU来替换Transformer中的Multi Head Attention+FFN的结构

有了上述相关工作的介绍，下面我们将详细介绍Jasper的网络结构。

### 3.Jasper Architecture

<div align=center>
    <img src="zh-cn/img/ch14/p5.png"   /> 
</div>

上图是Jasper的模型结构示意图

+ Jasper BxR (eg.Jasper 5x3),这里的B表示block的个数，R表示一个block内部sub-block的个数。
+ 每一个sub-block由1D卷积，batch norm, ReLU,和dropout构成。
+ 一个block中所有sub-block的output channel数是相同的。
+ 每一个block有Residual Connection。residual connection通过`1x1`卷积和batch norm来处理input之间的channel不同

**Normalization and Activation：**

我们通过消融实验尝试了：
+ 3种类型的normalization:batch norm,weight norm, layer norm
+ 3种rectified linear units: ReLU,clipped ReLU,Leaky ReLU
+ 2种gated units: gated linear units(GLU),gated activation units(GAU)

实验结果如下表所示：

<div align=center>
    <img src="zh-cn/img/ch14/p12.png"   /> 
</div>

由上表2所示，我们发现layer norm+GAU在小模型上表现最好，layer norm+ReLU，batch norm+ReLU表现次之；对于大模型，batch norm + ReLU是最好的选择，综合起来我们选择batch norm + ReLU的组合方式。

对于一个batch,为了统一sequence的长度需要padding,在计算norm的均值和方差的过程中需要把padding的内容mask掉，上图表3发现，在卷积前mask会得到较低的WER,在卷积和batch norm之间使用mask表现最差。

**Residual Connections:**

<div align=center>
    <img src="zh-cn/img/ch14/p7.png"   /> 
</div>

如上图所示，我们对比了Residual, Dense Residual,DenseNet,DenseRNet四中方式的connection，我们的这种Dense Residual的效果整体表现要好一些。


**Language Model**

我们尝试了N-gram和Transformer-XL，最好的结果是使用使用word-level N-gram的语言模型，通过beam search进行解码。(语言模型和声学模型是独立训练的）下图展示了Transformer-XL的质量(measured by perplexity)（笔者不清楚质量是指的什么和WER之间的关系：

<div align=center>
    <img src="zh-cn/img/ch14/p14.png"   /> 
</div>


**NovoGrad:**

Jasper的训练过程使用了一种新的优化器，其类似于Adam。在时间步$t$,NovoGrad计算梯度$g_t^l$,随后计算二阶矩$v_t^l$:

<div align=center>
    <img src="zh-cn/img/ch14/p15.png"   /> 
</div>

二阶矩$v_t^l$用来re-scale梯度$g_t^l$,传统的计算一阶矩$m_t^l$:

<div align=center>
    <img src="zh-cn/img/ch14/p16.png"   /> 
</div>

NovoGrad使用了L2正则化的方式如AdamW优化器：

<div align=center>
    <img src="zh-cn/img/ch14/p17.png"   /> 
</div>

最后我们以学习率$\alpha_t$更新权重：

<div align=center>
    <img src="zh-cn/img/ch14/p18.png"   /> 
</div>

使用NovoGrad代替SGD在LibriSpeech的dev-clean数据集上Jasper DR 10x5将WER由4.00%降低到3.64%，精度提高了9%。

### 4.Experiments

Jasper分别在LibriSpeech,WSJ，Hub5'00数据集上做了测试，其测试结果如下图所示：

<div align=center>
    <img src="zh-cn/img/ch14/p19.png"   /> 
</div>
<div align=center>
    <img src="zh-cn/img/ch14/p20.png"   /> 
</div>


### 5.Conclusions

基于wav2letter卷积方法，Jasper结构是一个训练和推断高效的声学模型。希望Jasper被设计的更深层并应用到更大的训练数据集上。