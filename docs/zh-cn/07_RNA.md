
## RNA（Recurrent Neural Aligner: An Encoder-Decoder Neural Network Model forSequence to Sequence Mapping）


### 1.RNA解决的问题

CTC的其中的一个缺点是条件独立性即该模型假设每个输出在给定输入的情况下都与其他输出有条件独立，对于许多序列到序列问题，这是一个糟糕的假设。RNA解决了这个问题，RNA网络就是将CTC中Encoder后的多个分类器（Decoder）换成了一个RNN网络，使网络能够参考序列上下文信息。


### 2.RNA的模型结构

<div align=center>
    <img src="zh-cn/img/ch7/p1.png"   /> 
</div>

+ CTC的Decoder是输入一个vector输出一个token
+ RNA添加了依赖，通过RNN使得当前output依赖于之前的output和output的隐层单元


### 3.RNA的损失函数

RNA的Alignment和CTC是相同的，如下图所示：

<div align=center>
    <img src="zh-cn/img/ch7/p2.png"   /> 
</div>

+ 上图展示了所有可能的Alignments
+ 横轴表示input的特征
+ 纵轴表示RNA解码的label
+ 每一个点(t，n)表示RNA解码的状态
+ 对角线的箭头表示预测为下一个label的概率，横向箭头表示预测为blank的概率
+ RNA走过上所有的合法Alignments,input是声学特征，output是一个one-hot编码的label

其余CTC本质的不同如下：

<div align=center>
    <img src="zh-cn/img/ch7/p3.png"   /> 
</div>

这里的$z=(z_1,z_2,...,z_T)$表示一个长度为$T$的合法路径，去掉blank和重复后变为$y$.$P(y|x)$表示所有的和标注$y$相关的alignments $z$在给定声学特征$x$下的条件概率和。这个也是给定声学特征$x$预测label为$y$的条件分布。这是典型的CTC的做法。但是RNA的表示如下：

<div align=center>
    <img src="zh-cn/img/ch7/p4.png"   /> 
</div>

对比CTC的表示：

<div align=center>
    <img src="zh-cn/img/ch7/p5.png"   /> 
</div>

你会发现其中的不同！


最后RNA的损失是由两部分构成的一部分是`log-likelihood loss`,一部分是`Expected loss`

$$L1 = \sum_{(x,y)} - log(P(y|x))$$


$L1$表示对数似然损失，用来优化RNA模型生成目标label的概率。然而，当我们将模型应用于任务时我们通常使用序列级损失或度量来度量模型的性能，例如用于语音识别的词错误率（WER）。理想情况下，我们希望模型为具有较小序列级损失的标记序列分配更高的概率。其它研究表明经过似然损失训练的RNA模型也存在暴露偏差。在训练时间中通过对所有可能的Alignments求和来计算目标标签序列的概率，然而，RNA解码器网络总是以来自目标序列的真实标记为条件。因此，当用于Beam Search的推断时，模型只暴露于正确的标签，并且可能无法学会对其标签预测中的错误具有鲁棒性。另一个缺点是本事对数似然中的概率计算我们也是使用近似的方法得到的而非真正的穷举了所有的Alignments。

可以使用一个序列级别的loss来解决这些问题。

<div align=center>
    <img src="zh-cn/img/ch7/p6.png"   /> 
</div>

这里的$P(z|x)$表示一个aligenment $z$在给定声学特征条件下的RNA的预测概率，$loss(x,z,y)$表示序列级别的loss的估计，这里的序列级别的loss可以使用目标序列$y$与去掉blank的alignment $z$的编辑距离(Edit Distance)。

### 4.RNA的性能分析


<div align=center>
    <img src="zh-cn/img/ch7/p7.png"   /> 
</div>

个人感觉RNA和CTC的性能对比差不多，没有什么太大优势！

!> 问题：目前我们可以实现多个input的特征对应1个label，有时候我们因为特征切分的原因会出现一个特征对应多个label的情况，CTC个RNA仍未解决该问题，RNN-T将解决。