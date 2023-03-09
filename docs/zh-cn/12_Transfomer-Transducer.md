## Transfomer-Transducer:A STREAMABLE SPEECH RECOGNITION MODEL WITH TRANSFORMER ENCODERS AND RNN-T LOSS


<!-- https://blog.csdn.net/qq13269503103/article/details/105233648 -->

<!-- https://blog.csdn.net/weixin_48994423/article/details/124350435?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3-124350435-blog-105233648.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3-124350435-blog-105233648.pc_relevant_default&utm_relevant_index=4
 -->

<!-- https://www.cnblogs.com/zy230530/p/13681954.html -->


### 1.思想

论文作者借助RNN-T的整体架构，利用Transformer替换RNN结构；因为Transformer是一种非循环的attention机制，所以可以并行化计算，提升计算效率；此外，作者还对attention的上下文时序信息宽度做了限制，即仅利用有限宽度的上下文时序信息，在损失较小精度的条件下，可以满足流式语音识别的要求；还有一点是，作者表明当Transformer采用非限制的attention结构时，在librispeech数据集上能够取得state-of-the-art的识别效果

该轮文思路跟[Meta的论文](https://arxiv.org/pdf/1910.12977.pdf)基本一致，都是采用Transformer替换RNN-T中的RNN结构，且均从限制attention上下文时序信息的宽度角度考虑，降低计算和延迟；但二者在细节方面略有不同，比如输入特征维度、数据增强、模型大小、结点参数、位置编码产生方式等均有所不同；此外，该论文在解码时采样了语言模型融合的策略，提升识别效果；


### 2.RNN/Transformer Transducer architecture

<div align=center>
    <img src="zh-cn/img/ch12/p1.png"   /> 
</div>

RNN-T体系结构是一种神经网络体系结构，可以通过RNN-T损失进行端到端训练，将输入序列（如音频特征向量）映射到目标序列。给定一个长度为$T$，$x=(x_1，x_2,...,x_T)$的实值向量的输入序列，RNN-T模型试图预测长度为U的标签$y=(y_1，y_2，...，y_U)$的目标序列。RNN-T模型在每一个时间步给出了一个标签空间的概率分布,和输出标签空间包括一个额外的空标签,此时跳转到下一个时间步。

RNN-T模型对所有可能的对齐都定义了一个条件分布$P(z|x)$，其中:

<div align=center>
    <img src="zh-cn/img/ch12/p2.png"   /> 
</div>


是长度为$U$的$(z_i,t_i)$对序列，$(z_i,t_i)$表示输出标签$z_i$和编码的$t_i$特征之间的对齐。标签$z_i$也可以是空白标签（空预测）。删除空白标签将得到实际的输出标签序列$y$，长度为$U$。可以在所有可能的对齐$z$上边缘$P(z|x)$，以获得给定输入序列$x$的目标标签序列$y$的概率，

<div align=center>
    <img src="zh-cn/img/ch12/p3.png"   /> 
</div>
其中$Z(y,T)$是标签序列长度为$T$的有效对齐的集合。

对齐的概率$P(z|x)$可以分解为：

<div align=center>
    <img src="zh-cn/img/ch12/p4.png"   /> 
</div>

其中，$labels(z_{1:(i−1)})$是$z_{1:(i−1)}$中的非空白标签序列。RNN-T架构通过音频编码器、标签编码器和联合网络参数化$P(z|x)$。编码器是两个神经网络，分别编码输入序列和目标输出序列。本文用transformer代替原来的LSTM编码器。

<div align=center>
    <img src="zh-cn/img/ch12/p5.png"   /> 
</div>

<div align=center>
    <img src="zh-cn/img/ch12/p6.png"   /> 
</div>

其中每个线性函数是一个不同的单层前馈神经网络，$AudioEncoder(x)$是时间$t_i$时的音频编码器输出，$LabelEncoder(labels(z_{1:(i−1)}))$是给定之前的非空白标签序列的标签编码器输出。

其中下面式子中前向变量`α(t,u)`定义为在时间框架`t`处结束的所有路径和在标记位置`u`处结束的所有路径的概率之和。然后，使用前向算法来计算最后一个`α`变量`α(T,U)`，模型的训练损失是等式中定义的负对数概率的和：

<div align=center>
    <img src="zh-cn/img/ch12/p7.png"   /> 
</div>

其中，$T_i$和$U_i$分别为第$i$个训练示例的输入序列和输出目标标签序列的长度。

### 3.Transformer Transducer Model

Transformer Transducer模型以RNN-T为整体框架，包含Transformer Encoder网络、Transformer预测网络和Feed-Forward联合网络; 损失采用的RNN-T的损失，即最大化标签序列对应所有对齐的概率和;

<div align=center>
    <img src="zh-cn/img/ch12/p8.png"   /> 
</div>

+ Transformer Encoder：由多个block堆叠而成，每个block包含`Layer norm`、`Multi-head attention`、`Feed-forward network`和`Resnet connection`;
+ 每个block的输入都会先进行`Layer norm`进行归一化，使得训练稳定，收敛更快;
+ `Multi-head attention`有多个`self-attention(Q=K=V)`并连而成，输入特征被转换到多个子空间进行特征建模，最后再将各个子空间的输出进行合并，得到高层次的特征编码;需要说明的是，**为提升计算效率，可以对attention所关注的上下文时序信息宽度进行控制**;
+ `Feed-forward network`由多层全连接串联而成，激活函数为`ReLU`;并且训练时采用dropout防止过度拟合;
+ `Resnet connection`的采样一方面能够为上层提供更多的特征信息，另一方面也使得训练时反向传播更加稳定;
+ Transformer 预测网络：具有跟Encoder类似的结构，只不过预测网络的attention不能利用未来信息，所以网络的attention仅限定在历史的状态;此外，作者还通过限定attention的历史状态个数来降低计算复杂度，$O(T^2) \limit O(T)$
+ `Feed-forward`联合网络：Encoder的输出和预测网络的输出进行线性组合之后，输入到联合网络进行任务化学习;网络由多层全连接层组成，最后一层为softmax;网络的输出为后验概率分布

<div align=center>
    <img src="zh-cn/img/ch12/p5.png"   /> 
</div>

<div align=center>
    <img src="zh-cn/img/ch12/p6.png"   /> 
</div>

其中，$AudioEncoder$、$LabelEncoder$分别为Encoder网络和预测网络输出；$P$为联合网络输出的后验概率；$t_i$为时间序号；$Label(z_{1:i-1})$表示预测网络的历史non-blank输出序列。
+ Loss:网络的目标是最大化标签序列对应的所有对齐的概率和，取负号是可转化成最小化
 <div align=center>
    <img src="zh-cn/img/ch12/p7.png"   /> 
</div>

这部分和RNN-T相同。


### 4.Traning

+ 数据集：语音数据集：LibriSpeech 970hours;文本数据集：LibriSpeech对应10M文本＋额外800M本文
+ 输入特征：`128 log fbanks`;下采样帧率`33.3hz`;
+ 特征增强[specaugment](https://arxiv.org/pdf/1904.08779.pdf)：通过specaugment进行增强，仅采样时间掩蔽和频率掩蔽，且`frequency masking(F = 50,mF = 2)`, `time masking(T = 30,mT = 10)`
+ 模型参数：
<div align=center>
    <img src="zh-cn/img/ch12/p9.png"   /> 
</div>

	- Transformer Encoder网络：`18*block(feed-forward(layer-norm(x)+attentionlayer(layer-norm(x)))`
	- 预测网络：`2*block(feed-forward(layer-norm(x)+attentionlayer(layer-norm(x)))`
	- 联合网络：一层全连接(激活tanh)＋一层softmax

+ 学习率策略：ramp-up阶段`(0～4k steps)`:从0线性ramp-up到`2.5e−4`;hold阶段(`4k~30k` steps):保持`2.5e−4`;指数衰减阶段(`30k～200k` steps):衰减到`2.5e−6`为止
+ 高斯噪声(`10k steps~`)：训练时模型的权重参数引入高斯噪声，提升鲁棒性，高斯参数(`μ = 0, σ = 0.01`)


### 5.实验结果

<div align=center>
    <img src="zh-cn/img/ch12/p10.png"   /> 
</div>

可以看到，T-T模型显著优于基于LSTM的RNN-T基线。
为了与使用单独训练LM的浅融合系统进行比较，还使用完整的810M的数据集训练了一个基于Transformer的LM，该LM与T-T中使用的标签编码器的架构相同。
使用该LM和训练的T-T系统以及训练的双向基于LSTM的RNN-T基线进行浅融合。结果显示在“With LM”列的表2中。T-T系统的浅融合结果与高性能现有系统的相应结果具有竞争力。

<div align=center>
    <img src="zh-cn/img/ch12/p11.png"   /> 
</div>

为了使AudioEncoder的一步推断易于处理(即具有恒定的时间复杂度)，进一步通过再次掩盖注意力分数，将AudioEncoder的注意力限制在先前状态的固定窗口。由于计算资源有限，对不同的Transformer层使用相同的mask，但是对不同的层使用不同的上下文(mask)是值得探索的。前两列中的N表示模型在当前帧的左边或右边使用的状态数。使用更多的音频历史记录会带来更低的WER，但考虑到一个具有合理时间复杂度的可流模型，尝试了每层10帧的左上下文。

<div align=center>
    <img src="zh-cn/img/ch12/p12.png"   /> 
</div>

类似地，探索了使用有限的右上下文来允许模型看到一些未来的音频帧，希望能够弥合可流化的T-T模型（左=10，右=0）和全关注的T-T模型（左=512，右=512）之间的差距。由于对每个层应用相同的掩码，因此通过使用正确的上下文引入的延迟将聚合在所有层上。例如，在图3中，要从一个具有正确上下文的一帧的3层变压器中生成y7，它实际上需要等待x10到达，这是90ms的延迟。

<div align=center>
    <img src="zh-cn/img/ch12/p13.png"   /> 
</div>

为了探索建模的右上下文影响，对每层固定的512帧的左上下文进行了比较，并与全注意力T-T模型进行了比较。从表4中可以看到，每层的正确上下文为6帧（约3.2秒的延迟），性能比全注意模型差16%左右。与可流媒体化的T-T模型相比，每层2帧的右上下文（大约1秒的延迟）带来了大约30%的改进。

<div align=center>
    <img src="zh-cn/img/ch12/p14.png"   /> 
</div>

此外，还评估了在T-T标签编码器中使用的左上下文如何影响性能。在表5中，展示了限制每一层只使用三个以前的三个标签状态产生与每层使用20个状态的模型相似的精度。标签编码器的左上下文非常有限，很适合T-T模型。当使用全注意力T-T音频编码器时，当限制左标签状态时，看到了类似的趋势。最后，表6报告了使用有限的10帧左上下文时的结果，这将一步推断的时间复杂性降低到一个常数，展望未来框架，作为一种弥合左注意和全注意模型之间的差距的方法。


### 6.结论

+ 提出了一种基于Transformer的端到端的RNN-T结构，称之为Transformer Transducer；该模型，一方面借助Transformer的非循环结构，网络可并行化计算，显著提升训练效率；另一方面，在LibriSpeech数据集上取得了新的SOTA的效果。
+ Transformer Transducer还允许控制attention利用的上下文状态个数，从而有效降低延迟和计算，在精度轻微损失的条件下，满足`流式语音识别`的要求。