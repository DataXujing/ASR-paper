## Neural Transducer


### 0. Abstract

序列到序列模型在各种任务上都取得了令人印象深刻的结果。然而，它们不适用于需要随着更多数据进入input而进行增量预测的任务或具有长输入序列和输出序列的任务。
这是因为它们生成以整个输入序列为条件的输出序列。
在本文中，我们提出了一种Neural Transducer，它可以在更多输入input时进行增量预测，而无需重复整个计算。
与序列到序列模型不同，Neural Transducer计算
以部分观察到的输入序列和部分生成的序列为条件的下一步分布。
在每个时间步，transducer可以决定产生零到多个输出符号。
可以使用编码器处理数据，并将其作为传感器的输入。
在每个时间步发射符号的离散决策使得传统的反向传播难以学习。
然而，可以通过使用动态编程算法来训练transducer以生成目标离散决策。
我们的实验表明，Neural Transducer在数据输入时需要产生输出预测的设置中工作良好。
我们还发现，即使在不使用注意机制的情况下，Neural Transducer也能很好地处理长序列。


### 1. Introduction

Seq2seq的模型在解决一些序列到序列的问题中非常成功比如翻译，离线语义识别，image captioning和对话模型等。然而无法解决当input作为一个流源源不断的作为input进入模型，因为seq2seq只能接收一个完整的input序列，输出一个output序列。而流式的语音识别的任务是无法用seq2seq解决的，这也是在LAS中无法实现流式语音识别的原因。

<div align=center>
    <img src="zh-cn/img/ch9/p1.png"   /> 
</div>

而Neurl Transducer也可以被认为是一个广义的seq2seq模型，如上图(b)所示：
+ Nueral Transducer input一个block可以产生一个chunk的output(当然output的长度也可以是0，即产生一个blank在语音识别中)
+ 通过使用一个transducer RNN网络产生output。
+ transducer RNN的input包括两部分一部分来源encoder RNN的output，另一部分来源于他的上一个状态。

训练阶段，output和input的对齐问题一般有两种解决方案：
+ 像CTC一样，把合法的对齐方式当做潜变量，一个output是由所有合法的映射到该output的对齐的和做成的。
+ 一种是找一个不同的算法来训练模型，该模型用来做对齐，然后在该对齐的基础上训练声学模型。

显然这两种方式都不适合Neural Transducer,因为前者在计算一个对齐的边缘概率的时候对于Nerual Transducer是复杂的，因为其依赖于encoder RNN和transducer RNN。后者需要训练一个额外的模型增加了复杂度。在本文的稍后会介绍Neural Transducer如何解决训练数据的对齐问题。

在TIMIT音素识别的任务上，Nerual Transducer使用3层LSTM作为encoder,3层LSTM作为transducer可以得到20.8%的音素错误率(PER)已经接近了当时同类模型的SOTA,如果采用GMM-HMM等网络做对齐，Neural Transducer可以得到19.8%的PER。


### 2. Related Work

之前的seq2seq的模型需要input整个sequence，然后预测output,不能实现流式的语音识别，虽然加入了一些全局或局部的注意力机制使得精度提高，但是仍然没有解决流式识别的问题。而Neural Transducer通过input一个chunk的sequence预测若干个或o个output,相当于做个一个window的attention同时解决了流式识别的问题。如下图所示：

<div align=center>
    <img src="zh-cn/img/ch9/p2.png"   /> 
</div>


### 3. Methods

下图展示了Neural Transducer的模型细节：

<div align=center>
    <img src="zh-cn/img/ch9/p3.png"   /> 
</div>

#### 3.1 Model

+ $x_{1,...,L}$是时间长度为$L$的input data，$x_i$是第$i$个time的声学特征输入
+ $W$是block(window)的长度
+ $N=[\frac{L}{W}]$表示block的个数
+ $\tilde{y}_{1...S}$ 是输出
+ $\tilde{y}_{i...(i+k)}$ 表示一个input block的输出，其中$0\leq k\leq W$
+ `<e>`表示空和CTC中的blank意义相同，当预测为`<e>`意味着transducer将跳转到下一个block
+ 一个输出：$\tilde{y}_{1...S}$ 可能有若干个对齐(Alignments)产生
+ $\mathscr{Y}$ 表示所有输出是$\tilde{y}_{1...S}$ 的对齐构成的集合
+ $y_{1...(S+B)} \in \mathscr{Y}$ 表示一个对齐序列，主要$y$比$\tilde{y}$长$B$,表示多出$B$个`<e>`

$e_b$，其中$b \in 1...N$表示第$b^{th}$个block的最后一个预测字符，注意$e_0=0$和$e_N=S+B$，即对于每一个block b,$y_{e_b}=\<e\>$,下面展示如何计算对齐序列的概率$p(y_{1...(S+B)}|x_{1...L})$

我们可以首先给出计算任意多个block的公式：

<div align=center>
    <img src="zh-cn/img/ch9/p4.png"   /> 
</div>

对于每一个block,我们可以这样计算：

<div align=center>
    <img src="zh-cn/img/ch9/p5.png"   /> 
</div>

而对于上式中的右侧的每一项：

<div align=center>
    <img src="zh-cn/img/ch9/p6.png"   /> 
</div>

我们可以通过transducer得到。


#### 3.2 Next Step Prediction


回到开始的网络结构图Figure2.在这个结构示例图中展示了Transducer有两个隐藏层用$s_m$和 $h_{m}^{'}$ 表示。这里展示了下一个block b的预测，在这个block中，第一个index是$m=e_{b-1}+1$,最后一个index是$m+2$(即$e_b=m+2$).

Transducer的网络表示为：

<div align=center>
    <img src="zh-cn/img/ch9/p7.png"   /> 
</div>

关于相关符号的说明：

+ $f_{RNN}(a_{m-1},b_m;\theta)$ 表示一个RNN网络(比如LSTM)
+ $f_{softmax}(.;a_m;\theta)$ 表示softmax层
+ $f_{context}(s_m,h_{(b-1)W+1};\theta)$ 表示的上下文向量的计算，这里用到了一些attention机制

分析发现每一个block其实包含了之前block的所有input信息，关于$f_{context}$的不同计算方式将在下一节给出。

#### 3.3 Computing <img src="zh-cn/img/ch9/p8.png"   />

我们通过一个attention模型来计算$f_{context}$。

**MLP-attention model:**

context vecter $c_m$通过如下两个步骤进行计算：

<div align=center>
    <img src="zh-cn/img/ch9/p9.png"   /> 
</div>


**Dot-attention model:**

将上式中的$f_{attention}$替换为：
$$e_j^m=s_m^Th_{(b-1)W+j}$$

这两种注意力模型都有两个缺点。首先，没有明确的机制要求注意力模型将其焦点从一个输出时间步移到下一个。其次，对于不同的输入帧j，作为softmax函数的输入而计算的能量在每个时间步长彼此独立，
并且因此除了通过softmax功能之外不能彼此调制（例如增强或抑制）。Chorowski改善了第二个问题通过使用在一个时间步骤影响注意力的卷积算子，使用在最后时间步骤的注意力。

解决上述两个缺点，本文提出了新的attention机制，**LSTM-attention**,在上式中，我们替换了将$[e_1^m;e_2^m;...e_W^m]$进行softmax.而是将其input到一个单隐层的RNN中输出softmax值。


#### 3.4 Addressing End of Blocks

Transducer每一个block输出若干个output以`<e>`输出结束当前block,而进入下一个block.本文尝试了3中方式，第一种，没有`<e>`而是直接让transducer学出何时终止block；第二种就是transducer现在的学习方式；第三种是我们使用独立的logist fuction输入attention vector输出`0,1`值代表当前是否结束block进入下一个block的学习。


#### 3.5 Training

在训练过程中我们需要计算给定$x_{1..L}$ 输出$\tilde{y}_{1...S}$ 概率。

<div align=center>
    <img src="zh-cn/img/ch9/p10.png"   /> 
</div>

理论上，我们只需对上式取对数最大化即可，其对数似然的梯度表示如下：

<div align=center>
    <img src="zh-cn/img/ch9/p11.png"   /> 
</div>

但是上述概率计算中右边的求和不好算，由于对数概率的表达式不允许分解成可以独立计算的较小项，因此该方案中的精确推断在计算上是昂贵的！本文采用了一种新颖的算法叫dynamic programming-like algorithm.其思想就是每个block我们都能找到最优的Alignments,合并起来是整个最优的Alignments。

<!-- 由于对数概率的表达式不允许分解成可以独立计算的较小项，因此该方案中的精确推断在计算上是昂贵的。相反，每个候选y都必须独立测试，并且必须发现指数级大量序列中的最佳序列。 -->

#### 3.6 Inference

推断的时候给定$x_{1...L}$ 和模型参数$\theta$ 我们可以找到一个label $y_{1...M}$ 满足下式：

<div align=center>
    <img src="zh-cn/img/ch9/p12.png"   /> 
</div>

我们可以通过Beam Search找到最优解。

paper最后我们看到消融试验中不同block大小及有无注意力机制对TIMIT开发集上的PER的影响：

<div align=center>
    <img src="zh-cn/img/ch9/p13.png"   /> 
</div>

发现，no-attention, window size越大，PER越大，这和我们的认知是一致的；加入注意力机制，window size对PER的影响就没有那么大了。
