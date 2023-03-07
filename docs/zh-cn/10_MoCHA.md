## MoChA (Monotonic Chunkwise Attention)


### 0.Abstract

具有软注意力的序列到序列模型已经成功地应用于各种问题，但是它们的解码过程产生了二次时间和空间成本，并且不适用于实时序列。为了解决这些问题，我们提出了单音纯音注意（MoChA），它将输入序列自适应地分割成小块，在小块(chunks)上计算软注意。利用MoChA的模型可以通过标准反向传播进行有效训练，同时允许在测试时进行在线和线性时间解码。MoCha在流式语音识别中获得了SOTA和离线soft attention机制的表现相当。在我们不期望单调对齐的文档摘要实验中，与基于monotonic attention-based model相比，我们显示出显著改进的性能。


### 1.Introduction

序列转录问题比如翻译，语音识别等，seq2seq是一个标准的范式，在此基础航基于attention机制的seq2seq在seq2seq的基础上增加了attention机制可以更好的保证input-output的对齐问题。传统的soft attention有弊端，其一是解码过程需要付出巨大的时间和空间成本，其二是必须等所有input进入后才可以解码，这导致无法进行流式的识别。Raffel et al.(2017)提出了一种monotonic input-output alignment(单调的input-output对齐)，但是实验表明这种方式限制了模型的表达，不如soft attention表现的好。在在线语音识别上MoCha填充了soft attention和monotonic的gap，在文本摘要上相比monotonic有20%的相对提高。

<div align=center>
    <img src="zh-cn/img/ch10/p1.png"   /> 
</div><p align=center>b.在测试时，单调注意力从左到右检查记忆条目，选择是移动到下一个记忆条目（显示为带有X的节点）还是停止并参与（显示为黑色节点）。上下文向量被硬分配给所关注的内存条目。在下一个输出时间步，它将从停止的位置重新开始。</p>


### 2.Defining MoCha

我们将从回顾seq2seq,soft attention,monotonic attention开始，逐步介绍MoCha.

#### 2.1 Sequence-to-Sequence Model

<div align=center>
    <img src="zh-cn/img/ch10/p2.gif"   /> 
</div>

+ input sequence:$x=x_1,...,x_T$,output sequence: $y=y_1,...,y_U$,
+ 一般input sequence 会经过一个隐层为$h=h_1,...,h_T$的RNN
$$h_j=EncoderRNN(x_j,h_{j-1})$$
+ decoderRNN更新状态并自回归一个output layer(往往用softmax)产生output sequence:
$$s_i=DecoderRNN(y_{i-1},s_{i-1},c_i)$$
$$y_i=Output(s_i,c_i)$$
注意，这里的$s_i$是DecoderRNN的隐层状态，$c_i$是“context” vector,他是Encoder和Decoder的连接管道。
在2014年之前我们往往使用$c_i=h_T$即Encoder的最后一个时间步骤的隐层状态作为$c_i$,2015年后$c_i$可以通过attention机制进行计算得到。


#### 2.2 Standard Soft Attention


<div align=center>
    <img src="zh-cn/img/ch10/p3.png"   /> 
</div>

关于各种attention机制笔者在NLP-paper的教程中给出了详细的介绍，这里我们仅回顾分析Bahdanau Attention机制。如上图所示：
首先计算“energy” value$e_{i,j}$
$$e_{i,j}=Energy(h_j,s_{i-1})$$
一个可行的$Energy(.)$表达如下：
$$Energy(h_j,s_{i-1}):=v^Ttanh(W_hh_j+W_ss_{i-1}+b)$$
这里的$W_h \in R^{d \times dim(h_j)}$,$W_s \in R^{d \times dim(s_i-1)}$,$b \in R^{d}$,$v \in R^d$均是可学习的参数，$d$是energy function隐层的维度。

下一步对energy scaler进行归一化得到$\alpha_{i,j}$
$$\alpha_{i,j}=\frac{exp(e_{i,j})}{\sum_{k=1}^Texp(e_{i,k})}=softmax(e_{i,:})_j$$

最后，计算context vector $c_i$
$$c_i=\sum_{j=1}^T\alpha_{i,j}h_j$$

soft attention的缺点有两个，其一：每次计算第$i$步的$c_i$都需要$h_j,j\in [1,...,T]$,无法做到online;其二:$c_i$的计算需要$T$个energy的值和相应的weights,这需要大量的时间和存储空间。 

#### 2.3 Monotonic Attention
<!-- https://zhuanlan.zhihu.com/p/99389088 -->

为了解决soft attention的缺点，Raffel et al.(2017)提出了hard monotonic attention.其描述如下：在输出时间步i，注意力机制开始从其在前一输出时间步（称为$t_{i-1}$）所关注的存储器索引开始检查存储器条目;接着计算非归一化的energy scalar $e_{i,j}$ 其中$j=t_{i-1},t_{i-1}+1,...$,并将这些energy scalar送入sigmoid函数生成“selection probabilities” $p_{i,j}$.从参数为$p_{i,j}伯努利分布生成随机变量$z_{i,j}$来决定是否attend.


<div align=center>
    <img src="zh-cn/img/ch10/p4.png"   /> 
</div>

当遇到$z_{i,j}=1$对于某一个$j$,当前注意力结束，设定$t_i=j$,$c_i=h_{t_i}$,如Figure1.b所示。因为只需要attend到$h_j$因此encoderRNN只需要input到$x_1,...,x_j$,可以用于online的识别。

因为该attention过程依赖于sampling和hard assignments，因此不能使用后向传播。为了解决这一问题，Raffel等人（2017）提出通过计算注意力过程引起的记忆的概率分布，对$c_i$的期望值进行训练。其分布如下：

<div align=center>
    <img src="zh-cn/img/ch10/p5.png"   /> 
</div>

$c_i$的计算通过（11）式计算和soft attention相似。



#### 2.4 Monotonic Chunkwise Attention

<!-- https://zhuanlan.zhihu.com/p/99389088 -->

<div align=center>
    <img src="zh-cn/img/ch10/p6.png"   /> 
</div>

Vanilla monotonic attention虽然提高了效率, 但是会导致性能的下降. 原因是其相比于soft attention加了两条额外的限制: 
1. decoder每次只attend to 一个$h_j$; 
2. attention是严格单调的.

解决思路，一句话来说就是，vanilla monotonic attention每次停下来只attend to 一个$h_j$，即$c_i=h_j$;而Mocha以停下来的位置为终点, 往前倒推`w-1`个$h_j$,在这`w`个$h_j$内部做soft attention,即：

<div align=center>
    <img src="zh-cn/img/ch10/p7.png"   /> 
</div>

这里的$ChunkEnergy(.)$和soft attention中的Energy函数相似。

这里需要注意的是不管是vanilla monotonic attention还是MoCha我们仅介绍了inference的过程，其训练过程为了使用后向传播，都采用了计算期望，均非常复杂。如果对该部分该兴趣请细心阅读其原始paper或参考知乎大佬的介绍：[Monotonic Attention](https://zhuanlan.zhihu.com/p/99389088),你也可以在Github搜索相关源码的实现。