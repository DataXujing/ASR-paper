
## LAS: Listen, Attend and Spell

<!-- https://blog.csdn.net/u013010473/article/details/104469263 -->
<!-- https://github.com/jx1100370217/LAS_Tensorflow_jack.git -->
<!-- https://github.com/jackaduma/LAS_Mandarin_PyTorch -->
<!-- https://github.com/kaituoxu/Listen-Attend-Spell -->

### 1.简介

Listen,Attend and Spell(LAS)是一种学习将语音转换成字符的神经网络。与传统的DNN-HMM模型不同的是，LAS模型联合学习（jointly）语音识别器的所有组件。LAS系统有两个组件:一个监听器(Listener)和一个拼写器(Speller)。Listener是一个接受滤波器组谱作为输入的金字塔递归网络（pBiLSTM）编码器;speller是一种基于注意力的周期性网络译码器，它将字符作为输出来发射,该网络生成字符序列时不做任何字符间的独立性假设。这是LAS较以往端到端CTC模型的关键改进。在谷歌语音搜索任务的一个子集上，LAS在没有字典或语言模型的情况下实现了14.1%的单词错误率(WER)，而在有语言模型重评分的情况下，实现了10.3%的单词错误率（WER）。

### 2.网络结构

<div align=center>
    <img src="zh-cn/img/ch5/p1.png"  /> 
</div>

LAS在标签序列中不做独立假设，也不依赖于HMMs。LAS是基于注意力的Seq2Seq学习框架,它由编码器循环神经网络(RNN)和解码器循环神经网络(RNN)组成。Listener是一个金字塔形的RNN，它将低级的语音信号转换为高级的特征信号。speller是一个RNN，它通过注意机制指定字符序列上的概率分布，将这些高级特征转换成输出话语，Listener和Speller联合训练。


### 3.网络输入

fBank滤波器组光谱特征(filter bank spectra features)的输入序列：$x = (x_1,……,x_T )$


### 4.网络输出

字符序列：$y = (y_1,……,y_s,)$

### 5.目标函数

<div align=center>
    <img src="zh-cn/img/ch5/p2.png"  /> 
</div>

### 6.模型结构

**Listen**

采用3层512个nodes的金字塔形BiLSTM(pBiLSTM)网络结构：

<div align=center>
    <img src="zh-cn/img/ch5/p3.png"  /> 
</div>

Listen结构的好处：

1）时间分辨率（time resolution）减少8倍

2）更方便注意力机制提取信息

3）深层网络提供更好的非线性特征


**Attend & Spell**

采用基于注意力（attention-based）的LSTM Transducer。在每个输出步骤中，Transducer根据前面看到的所有字符生成下一个字符的概率分布。$y_i$的分布是解码器状态$s_i$和上下文$c_i$的函数。解码器状态$s_i$是先前的状态$s_{i-1}$，先前发出的字符$y{i-1}$和上下文$c_{i-1}$的函数。语境向量$c_i$是由注意机制产生的:

<div align=center>
    <img src="zh-cn/img/ch5/p4.png"  /> 
</div>

其中CharacterDistribution是一个softmax的MLP，RNN是一个两层的LSTM，$c_i$封装了用以生成下一个字符的声学信号信息，AttentionContext生成上下文向量$c$。


### 7.Training

无需pre-train,seq2seq模型基于前一个字符来预测下一个字符，并最大化log概率：

<div align=center>
    <img src="zh-cn/img/ch5/p5.png"  /> 
</div>


### 8.Decoding & Rescoring

在推理过程中，通过最大化log概率找出输入声学中最有可能的字符序列：

<div align=center>
    <img src="zh-cn/img/ch5/p6.png"  /> 
</div>

结合语言模型LM来消除短对话的偏差（bias）,其中语言模型的权重设置为$0.008$:

<div align=center>
    <img src="zh-cn/img/ch5/p7.png"  /> 
</div>


### 9.性能对比

<div align=center>
    <img src="zh-cn/img/ch5/p8.png"  /> 
</div>

那时的性能还不如CD-DNN-HMM!

**注意力可视化**

<div align=center>
    <img src="zh-cn/img/ch5/p9.png"  /> 
</div>

**Beam Search Width的选择**

<div align=center>
    <img src="zh-cn/img/ch5/p10.png"  /> 
</div>


### 基于普通话的模型训练和测试