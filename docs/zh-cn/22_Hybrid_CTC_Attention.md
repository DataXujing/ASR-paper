## Transformer-Based Online CTC/Attention End-to-End Speech Recognition Architecture

<!-- https://blog.csdn.net/wudibaba21/article/details/112364541 -->


### 1.摘要

Transformer在ASR中取得了成功，但是其端到端的流式语音识别对于Transformer-based的模型是一个挑战。本方法提供了一种Transformer-based的流式语音识别模型，包括chunk self-attention encoder(chunk-SAE),单调截断注意力monotonic truncated attention(MTA) based的self-attention decoder(SAD).首先，chunk-SAE 将语音分割为独立的chunk.为了减少计算量，本方法提供了state reuse chunk-SAE.此外，MTA based SAD截断音频特征进行注意力计算。最终将state reuse chunk-SAE和MTA based SAD融入到在线联合CTC/Attention架构。根据科大的普通话ASR基准评估了在线模型，320毫秒的延迟实现23.66%的字错误率(CER),比起离线系统，在线模型产生的CER绝对降低为0.19%


### 2.Introduction

CTC/Attention 语音识别模型很难应用到在线识别，因为全局注意力机制依托于完整的input,CTC的prefix score也依赖于完整的input。在先前的工作中提出了单调逐块注意(sMoChA)，单调截断注意力(MTA),解码方面是有在线联合解码方法，截断的CTC(T-CTC)前缀分数和动态等待联合解码(DWJD)。

Transformer的优点：可并行化，缺点:不适合在线任务，编码器需要计算整个输入帧上的注意力权重；其次，自注意力解码器计算整个输出的注意力权重。本方法意在针对Transformer-based模型额这一缺点进行改进。

下面将在section 3中回归online CTC/Attention之前优化的结构包括MTA以及回顾Transformer的技术细节；section 4将介绍本方法online Tranformer-based CTC/attention architecture；section 5提供模型训练的一些细节和最终的实验结果。

### 3.Online CTC/Attention E2E Architecture及Transformer复习介绍

1.Hybrid CTC/attention E2E ASR Architecture

损失函数：

<div align=center>
    <img src="zh-cn/img/ch20/p1.png"   /> 
</div>

这里的$\alpha$是超参数，$\mathcal{L}_ {dec}$ 和 $\mathcal{L}_{ctc}$分别指Decoder个CTC的Loss.

解码：

<div align=center>
    <img src="zh-cn/img/ch20/p2.png"   /> 
</div>

这里的$P_{dec}(Y|X)$和$P_{t-ctc}(Y|X)$表示给定声学特征$X$输出$Y$在Decoder和T-CTC分支的概率。$P_{lm}(Y)$是语言模型的概率。这里的参数$\lambda$,$\gamma$是可训练的参数。对于解码使用[DWJD算法（动态等待联合解码）](https://ieeexplore.ieee.org/abstract/document/8068205)主要是为了1)协调编码器中的前向传播和解码器中波束搜索；2)解决基于sMoChA的解码器的未同步预测，以及CTC的输出。

2.[MTA](https://github.com/HaoranMiao/streaming-attention)

MTA在截断的历史编码器输出上进行注意力，通过利用更长的历史，其性能优于sMoChA。一般的我们用$q_i$和$h_j$分别表示第$i$个decoder的state和第$j$个encoder的output。MTA定义$p_{i,j}$表示为截断的编码器在$h_j$位置的输出：

<div align=center>
    <img src="zh-cn/img/ch20/p3.png"   /> 
</div>

这里的$W_1,W_2$,向量$b$,$v$和参数$g,r$是可训练的参数。attention weight $a_{i,j}$计算如下：

<div align=center>
    <img src="zh-cn/img/ch20/p4.png"   /> 
</div>

这里的$a_{i,j}$表示在$h_j$处截断encoder的输出的概率，跳过$h_j$之前的编码器输出。

编码阶段： MTA通过下式确定第$i$个decoder步的截断终点$t_i$:

<div align=center>
    <img src="zh-cn/img/ch20/p5.png"   /> 
</div>

这里的$z_{i,j}$是一个示性函数，表示在encoder output $h_j$处是否截断。根据等式5中的条件$j>t_{i-1}$，MTA强制端点以从左到右模式移动。一旦对应一些$j$有$z_{i,j}=1$,MTA把$t_i$设置为$j$。最终，MTA截断encoder输出的注意力机制为：

<div align=center>
    <img src="zh-cn/img/ch20/p6.png"   /> 
</div>

这里的$r_i$第$i$个decoder step的letter-wise hidden vector.

训练阶段：MTA的注意力机制作用在整个encoder output:

<div align=center>
    <img src="zh-cn/img/ch20/p7.png"   /> 
</div>

这里的$T$是所有的encoder output的个数。


3.Transformer Architecture

Transformer基于Encoder-Decoder架构，在Encoder和Decoder同时使用self-attention和FFN，下面我们简单介绍Transformer。

Multi-head attention: 

<div align=center>
    <img src="zh-cn/img/ch20/p8.png"   /> 
</div>

<div align=center>
    <img src="zh-cn/img/ch20/p9.png"   /> 
</div>

因为Transformer缺少序列建模的位置信息，通过绝对位置编码或相对位置编码解决该问题。

Self-attention encoder(SAE) and Self-attention decoder(SAD):

<div align=center>
    <img src="zh-cn/img/ch20/p10.png"   /> 
</div>


### 4.Model


整体模型结构如下图所示：

<div align=center>
    <img src="zh-cn/img/ch20/p11.png"   /> 
</div>


1.Chunk-SAE

将语音分割为$N_C$中心长度的菲重叠的孤立chunk.为了获取上下文信息，我们将每个chunk之前的$N_l$个左帧拼接成为历史上下文，将其之后的$N_r$个右帧拼接为未来上下文。拼接的帧仅提供上下文没有输出。利用预定义的$N_c$,$N_l$,$N_r$,每个chunk-SAE输出的感受野被限制为$N_l+N_c+N_r$，并且chunk-SAE的延迟被限制为$N_r$.

2.Sate reuse chunk-SAE

在chunk-SAE中，为每个chunk重新计算历史上下文。为了降低计算成本，我们将计算出的隐藏状态存储在中心上下文中。然后当计算新的chunk时，我们再与历史上下文相同的位置重用来自先前块的存储的隐藏状态，其灵感来源于Transformer-XL。
下图说明了chunk-SAE和state reuse chunk-SAE的区别：

<div align=center>
    <img src="zh-cn/img/ch20/p12.png"   /> 
</div>

一般的，$s^l_{\tau} \in \mathbb{R}^{N_l\times d_m}$和$h^l_{\tau} \in \mathbb{R}^{(N_c+N_r)\times d_m}$
表示第$l$层第$\tau$个chunk存储和新计算的hidden state。则第$l$层self attention layer 第$\tau$个chunk的queries,keys和values的定义如下：

<div align=center>
    <img src="zh-cn/img/ch20/p13.png"   /> 
</div>

上式中的$SG(.)$表示stop-gradient,state reuse chunk-SAE的计算复杂度降低了$N_l/(N_l+N_C+N_r)$.

此外，reuse chunk-SAE捕获了块之外的长期依赖性。
假设reuse chunk-SAE由$L$个层组成，左侧的感受野延伸到$L\times N_l$个帧，这比chunk-SAE的感受野宽得多。

3.MTA based SAD

为了使SAD可流式，使用MTA based SAD 以单调的从左到右的方式截断感受野，并对SAE的截断输出进行attention。如Fig2所示，其表示如下：

<div align=center>
    <img src="zh-cn/img/ch20/p14.png"   /> 
</div>

这里的$W.\in \mathbb{d_m \times d_m}$,$r$是可训练的参数。$\epsilon$表示噪声。$P=\{p_{i,j}\}$表示截断的概率矩阵，$p_{i,j}$表示截断第$j$个SAE输出以预测第$i$个输出标签的概率.$cumprod(x)=[1,x_1,x_1x_2,...,\prod^{|x|-1}_ {k=1}x_k]$,$cumprod(.)$作用在矩阵$P$的行，$\odot$表示矩阵的逐元素乘积。

MTA通过可训练标量r学习等式14中激活的适当偏移。为了防止$cumprod(1-P)$消失为0，把$r$设置为1个负数比如$r=-4$,为了鼓励截断概率的离散性，我们只需在训练期间加入高斯噪声。

解码阶段，我们需要一行行的计算$P^l=\{p^{l}_ {i,j}\}$的元素，这里的$P^l$表示第$l$层的截断概率矩阵。我们定义$t^l_i$是第$l$层预测帝$i$个label的截断终点，由下式决定：

<div align=center>
    <img src="zh-cn/img/ch20/p15.png"   /> 
</div>

这里的$z^l_{i,j}$表示第$l$层第$j$个SAE的output是否被截断。一旦$z^l_{i,j}=1$对于某些$j$,那么设定$t^i_i=j$,这意味着第$l$层的感受野被限制在$t^l_i$ SAE output。假设MTA based SAD由L层构成，那么需要L个截断终点对于每一个decoder step。每一层的截断的SAE output之间是不受影响的。

### 5.模型训练及实验


HKUST普通话数据集包含200个小时的训练数据和5小时的测试数据，该模型通过ESPNet构建。模型的相关配置如下：

+ input: `80维的fbank+音高和音高的二阶差分及归一化的相关系数`；窗口宽度是25ms,窗口偏移是10ms
+ output:3655个字符包括3623个中文字符，26个英文字符，6个非语言的标识符（比如sos/eos,blank,unknown-character,...)
+ 声学模型：前端使用2层的CNN，每个CNN有256个卷积核卷积核的大小是$3\times 3$,stride是$2\times2$这样在时间维度上缩减到原来的$1/4$;SAE和SAD分别是12和6层所有子层包括embedding层output的维度均为256，multi-head attention的head数是4，FFN中结点的个数是2048
+ 语言模型：使用HKUST训练数据训练了2层1024隐层节点个数的LSTM作为语言模型
+ 训练过程的超参数设置：
	- CTC/attention joint training $\alpha=0.7$
	- Adam optimizer
	- Noam learning rate schedle(25000 warm steps)
	- 30 epochs
	- dropout rate=0.1
	- label smoothing (penalty=0.1)
	- model averageing:最后10个训练周期的参数进行平均作为最终模型
+ Decoding阶段的参数：
	- combining T-CTC prefix score($\lambda=0.5$)
	- LM score($\gamma=0.3$)
	- beam size=10
最终的实验结果如下图所示：

<div align=center>
    <img src="zh-cn/img/ch20/p16.png"   /> 
</div>

