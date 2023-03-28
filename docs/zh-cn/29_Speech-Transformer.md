## Speech Transformer：The SpeechTransformer for Large-scale Mandarin Chinese Speech Recognition

<!-- 1.Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition
2.The SpeechTransformer for Large-scale Mandarin Chinese Speech Recognition -->

<!-- https://www.cnblogs.com/zy230530/ -->

<!-- https://www.cnblogs.com/zy230530/p/13681892.html -->
<!-- SPEECH-TRANSFORMER: A NO-RECURRENCE SEQUENCE-TO-SEQUENCE MODELFOR SPEECH RECOGNITION -->
<!-- https://www.cnblogs.com/zy230530/p/13681774.html -->

### 1.Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition

#### 1.1.思路

+ 整体采用seq2seq的encoder和decoder架构；
+ 借助transformer对文本位置信息进行学习；
+ 相对于RNN，transformer可并行化训练，加速了训练过程；
+ 论文提出了2D-attention结构，能够对时域和频域两个维度进行建模，使得attention结构能更好的捕获时域和空间域信息
 

#### 1.2.模型

speech-transformer 整体采用encoder和decoder结构，其中encoder和decoder的主要模块都是multi-head attention和feed-forward network；此外，encoder为更好的对时域和空域不变性建模，还额外添加了conv结构和2D-attention

<div align=center>
    <img src="zh-cn/img/ch18/p1.png"   /> 
</div>

+ Conv: encoder采用了两层`3*3`，`stride=2`的conv，对时域和频域进行卷积，一方面提升模型学习时域信息能力；另一方面缩减时间片维度至跟目标输出长度相近，节约计算和缓解特征序列和目标序列长度长度不匹配问题;conv的激活为ReLU
+ Multi-Head Attention: encoder和decoder都采用了多层multi-head attention来获取区分性更强的隐层表达(不同的head采用的变换不同，最后将不同变换后输出进行拼接，思想有点类似于模型融合)；multi-head attention结构由多个并行的scaled dot-product attention组成，在训练时可并行计算

<div align=center>
    <img src="zh-cn/img/ch18/p2.png"   /> 
</div>

+ Scaled Dot-Product Attention：结构有三个输入Q1 $(t_q\times d_q)$，K1 $(t_k\times d_k)$，V1$(t_v\times d_v)$；输出维度为$t_q\times d_v$；基本思想类似于attention的注意力机制，Q跟K的运算$softmax(QK^T)$可以看作是计算相应的权重因子，用来衡量V中各个维度特征的重要性；缩放因子$\sqrt{d_k}$的作用在论文中提到是为了缓解当$d_k$过大时带来的softmax剃度过小问题；mask分为padding mask和掩蔽mask，前者主要是用于解决padding后的序列中0造成的影响，后者主要是解码阶段不能看到未来的文本信息

<div align=center>
    <img src="zh-cn/img/ch18/p3.png"   /> 
</div>

+ Multi-Head Attention:结构有三个输入Q0$(t_q\times d_{model})$、K0$(t_k\times d_{model})$、V0$(t_v\times d_{model})$，分别经过$h$次不同的线性变换（$W^Q_i(d_{model}\times d_q)$、$W^K_i(d_{model}\times d_k)$、$W^V_i(d_{model}\times d_v)$,$i=1,2,3...,h$)，输入到$h$个分支scaled dot-product attention，各个分支的输出维度为$t_q\times d_v$($d_v=d_{model}/h$),这样经过concat后维度变成$t_q\times hd_v$，再经过最后的线性层$W^O(hd_v\times d_{model})$之后就得到了最终的$t_q\times d_{model}$

<div align=center>
    <img src="zh-cn/img/ch18/p4.png"   /> 
</div>

注意这里的$Q、K、V$与scaled dot-product attention不等价，所以我这里用$Q^0、K^0、V^0$以作区分

+ Feed-Forward network:前馈网络包含一个全连接层和一个线性层，全连接层激活为ReLU

<div align=center>
    <img src="zh-cn/img/ch18/p5.png"   /> 
</div>

其中$W_1(d_{model}\times d_{ff})$,$W_2(d_{ff}\times d_{model})$,$b_1(d_{ff})$,$b_2(d_{model})$

+ Positional Encoding：因为transformer中不包含RNN和conv，所以其对序列的位置信息相对不敏感，于是在输入时引入与输入编码相同维度的位置编码，增强序列的相对位置和绝对位置信息。

<div align=center>
    <img src="zh-cn/img/ch18/p6.png"   /> 
</div>

其中，pos代表序列位置，i表示特征的第i个维度，$PE_{(pos,i)}$一方面可以衡量位置pos的绝对位置信息，另一方面因为$ sin(a+b)=sin(a)\times cos(b)+cos(a)\times sin(b)$、$cos(a+b)=sin(a)\times cos(b)+cos(a)\times sin(b)$，所以对于位置p的相对位置k，$PE_{(pos+k)}$可以表示$PE_{pos}$的线性变换；这样同时引入了序列的绝对和相对位置信息

+ Resnet-block:论文中引入了resnet中的跳跃连接，将底层的输入不经过网络直接传递到上层，减缓信息的流失和提升训练稳定性

<div align=center>
    <img src="zh-cn/img/ch18/p7.png"   /> 
</div>

+ Layer Norm：论文采用LN，对每一层的神经元输入进行归一化，加速训练

<div align=center>
    <img src="zh-cn/img/ch18/p8.png"   /> 
</div>

其中，$l$为网络的第$l$层，$H$为第$l$层的神经元节点数，$g，b$分别为学习参数，使得特征变换成归一化前的特性，$f$为激活函数，$h$为输出。

+ 2D-Attention:transformer中的attention结构仅仅针对时域的位置相关性进行建模，但是人类在预测发音时同时依赖时域和频域变化，所以作者在此基础上提出了2D-attention结构，即对时域和频域的位置相关性均进行建模，有利于增强模型对时域和频域的不变性。
    - 输入：$c$通道的输入$I$
	- 卷积层：三个卷积层分别作用于I，获得相应的$Q、K、V$，滤波器分别为$W^Q_i、W^K_i、W^V_i(i=1,2,3...,c)$
	+  两个multi-head scaled dot-product attention分别对时域和频域进行建模，获取相应的时间和频域依赖，head数为c
	+  对时域和频域的attention输出进行concat，得到2c通道的输出，并输入到卷积中得到n通道的输出

<div align=center>
    <img src="zh-cn/img/ch18/p9.png"   /> 
</div>

<div align=center>
    <img src="zh-cn/img/ch18/p10.png"   /> 
</div>


其中$W^o$为最后一层卷积，$t$为时间轴，$f$为频率轴

#### 1.3.训练

+ 数据集：WSJ(Wall Street Journal),train si284/dev dev93/ test dev92
+ 输入特征：80维fbank
+ 输出单元：输出单元数为31，包含26个小写字母以及撇号，句点，空格，噪声和序列结束标记
+ GPU型号：NVIDIA K80 GPU 100steps
+ 优化方法和学习率：Adam

<div align=center>
    <img src="zh-cn/img/ch18/p11.png"   /> 
</div>

其中，n为steps数，k为缩放因子，warmupn= 25000,k= 10

+ 训练时，在每一个resnet-block和attention中添加dropout(丢弃率为0.1)；其中，resnet-block的dropout在残差添加之前；attention的dropout在每个softmax激活之前
+ 所有用到的conv的输出通道数固定为64，且添加BN加速训练过程
+ 训练之后，对最后得到的10个模型进行平均，得到最终模型
+ 解码时采用beam width＝10 beam search，此外length归一化的权重因子为1

#### 1.4.实验

在实验中$d_{model}$固定为256，head数固定为4

+ encoder的深度相比于decoder深度更有利于模型效果

<div align=center>
    <img src="zh-cn/img/ch18/p12.png"   /> 
</div>

+ 论文提出的2D-attention相比于ResCNN, ResCNNLSTM效果更好；表现2D-attention可以更好的对时域和频域相关性进行建模
+ encoder使用较多层数的resnet-block时(比如12),额外添加2D-attention对识别效果没有提升，分析原因是当encoder达到足够深度后，对声学信息的提取和表达能力以及足够，更多是无益

<div align=center>
    <img src="zh-cn/img/ch18/p13.png"   /> 
</div>

+ 训练时间上，相比于seq2seq结构，在取得相似的识别效果的同时，训练速度提升4.25倍

<div align=center>
    <img src="zh-cn/img/ch18/p15.png"   /> 
</div>

#### 1.5.实战

!> https://github.com/ZhengkunTian/OpenTransformer

+ 环境：pytorch>=1.20;Torchaudio >= 0.3.0
+ 输入特征：40fbank，CMVN=False
+ spec-augment[3]:频率掩蔽＋时间掩蔽，忽略时间扭曲(复杂度大，提升不明显)
+ 模型结构：
	- `encoder:2*conv(3*2)->1*linear+pos embedding->6*(multi-head attention(head=4, d_model=320)+ffn(1280))`
	- `decoder:6*(multi-head attention(head=4, d_model=320)+ffn(1280))`
+ 语言模型：`4*(multi-head attention(head=4, d_model=320)+ffn(1280))`
+ 训练：
	- 优化算法：adam
	- 学习率策略：stepwise＝12000，学习率在前warmup_n迭代步数线性上升，在n_step^(-0.5)迭代次数时停止下降
	- clip_grad=5
	- label smoothing[4]:平滑参数0.1
	- batch=16
	- epoch=80
	
+ 解码：beam search(beam width=5)+长度惩罚(权重因子＝0.6)
+ 实验效果：aishell test cer：6.7%


### 2.The SpeechTransformer for Large-scale Mandarin Chinese Speech Recognition

#### 2.1.思想

在speechTransformer基础上进行三点改进，
+ 降低帧率，缩短声学特征的时序长度，在大规模语音数据训练时提升计算效率；
+ decoder输入采样策略，如果训练时，decoder部分输入全部采用label，而预测时decoder部分为前一时刻预测输出，这样训练和预测之间会存在一定的偏差，为缓解该问题，在训练decoder时，以一定的采样概率决定该时刻decoder输入是否采用前一时刻的预测输出；
+ Focal Loss，因为模型是以字符为建模单元的，在训练语料中很难保证每个字符的出现频率基本相近，相反字符之间可能存在较大的频次差异，这就会导致类别之间的不均衡问题，为缓解该问题，在计算loss时，对于分类概率大的样本进行降权，对分类概率小的样本进行升权，这样会使得模型更加关注被误分类的hard样本；

#### 2.2.模型

模型整体框架采用的即是transformer的encoder-decoder形式，主要包含几个模块

+ Multi-Head Attention模块，该模块是一种非循环的attention机制，思想有些类似于模型融合，即先将输入声学特征转换到多个不同的attention子空间分别学习特征表达，然后再将各个子空间的输出进行拼接，得到具有较高特征表达能力的encoder
+ Feed-Forward network,前馈网络由一到两层全连接组成，全连接激活函数ReLU
+ Resnet connection，在每个multi-head attention模块和feed-forward network的输出位置条件添加resnet connection，保证浅层信息传播和训练稳定性
+ Layer Norm，在每个multi-head attention模块和feed-forward network的输入之间添加Layer norm，归一化输入的分布，保证训练稳定性
+ 位置编码，对encoder和decoder的输入进行位置编码add操作，引入绝对位置和相对位置信息，缓解attention对于时间顺序和位置信息的不敏感性

#### 2.3.Tricks

+ 低帧率，对特征提取后的frames进行降采样，原始帧率为100hz，每帧10ms，降采用后的帧率为16.7hz，每帧60ms，在大规模语音识别，尤其对于长时语音输入，**降低帧率到合适的大小，在几乎不影响精度的同时，可加快计算效率**

<div align=center>
    <img src="zh-cn/img/ch18/p16.png"   /> 
</div>

上图，对应采样因子为4，那么采样后的帧率为`100/n＝25hz`，每帧`1000/25=40ms`

+ deocder输入采样，如果decoder在训练时输入完全采用label对应编码作为输入，而预测时deocder输入为上一时刻预测输出，这样造成训练和预测之间会存在一定的偏差，为缓解该问题，可以以一定的概率决定在该时刻是否采用上一时刻的预测输出作为deocder输入；此外，因为模型在训练初始阶段，预测能力较差，所以预测不够准确，以此做为decoder输入可能影响模型训练稳定性，所以，在训练初始阶段采用较小的采样概率，而在训练中后期时采用较大的采样概率；概率的变化趋势有如下三种选择：

<div align=center>
    <img src="zh-cn/img/ch18/p17.png"   /> 
</div>

其中， $0 < \epsilon_{max} ≤ 1$，Nst为采样开始的step，Ned为采样概率达到max的$\epsilon_{step}$，i为当前step，$\epsilon_{(i)}$为当前的采样概率

+ Focal Loss，模型是以字符为建模单元时，训练语料中很难保证每个字符的出现频率可能相差很大，导致类别之间的不均衡问题，为缓解该问题，在计算loss时，对于分类概率大的样本进行降权，对分类概率小的样本进行升权，这样会使得模型更加关注被误分类的hard样本

<div align=center>
    <img src="zh-cn/img/ch18/p18.png"   /> 
</div>

上式中，γ 属于 `[0, 5]`,对于$p_t$较大的类别，其在损失中的权重项越小，比如当$p_t＝0.9$时，`γ＝2`，那么其权重缩小了$(1-0.9)^2=1/100$,反之，预测概率越小，其权重越大;该策略使得模型训练更关注被误分类的样本

#### 2.4.训练

+ 训练数据集：8000小时中文普通话数据
+ 验证集：`aishell-1 dev 14326utts`
+ 测试集：`aishell-1 test 7176utts；LiveShow：5766utts；voiceComment：5998utts`
+ baseline：`TDNN-LSTM,7*TDNN(1024)+3*LSTMP(1024-512)+4-gram LM`
+ 输入特征：40 MFCCs，global CMVN,batch_size=512
+ 输出单元：`5998中文字符＋<UNK>+<PAD>+<S>+<\S>=6002`
+ speechTransformer模型参数:`6*encoder+4*decoder,dmodel=512,head=16`
+ 优化方法：`Adam，β1 = 0.9, β2 = 0.98, ε = 10−9 `
+ 学习率：n是迭代次数，k为可学习缩放因子，学习率在前warmup_n迭代步数线性上升，在$n^{-0.5}$迭代次数时停止下降


<div align=center>
    <img src="zh-cn/img/ch18/p19.png"   /> 
</div>

+ 标签平滑策略[[1]](https://arxiv.org/pdf/1512.00567.pdf) ：降低正确分类样本的置信度，提升模型的自适应能力，`ε=0.2`

<div align=center>
    <img src="zh-cn/img/ch18/p20.png"   /> 
</div>

其中，H为cross-entropy,ε为平滑参数，K为输出类别数，$\delta_{k,y}＝1$ for $k=y$ and 0 otherwise

+ 模型训练后，以最后15个保存的模型的平均参数，做为最终模型
+ 验证时，beam search的beam width＝3，字符长度惩罚`α = 0.6`

!> 注意： CMVN （倒谱均值方差归一化）

提取声学特征以后，将声学特征从一个空间转变成另一个空间，使得在这个空间下更特征参数更符合某种概率分布，压缩了特征参数值域的动态范围，减少了训练和测试环境的不匹配等
提升模型的鲁棒性，**其实就是归一化的操作**。

#### 2.5.实验结果

+ speechTransformer与TDNN-LTSTM混合系统在几个测试集上性能相近；具体地，在aishell-1 test/dev和voiceCom上略差于混合系统；在liveshow上略优于混合系统
+ 降低帧率时，识别效果呈现先上升后下降的趋势，当帧率＝17.6hz，即60ms每帧时，在提升计算效率的同时，得到最佳的识别效果

<div align=center>
    <img src="zh-cn/img/ch18/p21.png"   /> 
</div>

+ 三种decoder输入采用中，线性采样的效果最好，并且采样概率在训练初始阶段稍小，而在训练后期阶段稍大

<div align=center>
    <img src="zh-cn/img/ch18/p22.png"   /> 
</div>

+ Focal Loss的应用可以对识别效果带来进一步的提升

<div align=center>
    <img src="zh-cn/img/ch18/p23.png"   /> 
</div>

#### 2.6.结论

在speechTransformer基础上进行一系列的改进，
+ 低帧率，提升计算效率
+ decoder输入采样减少训练和预测偏差，以一定概率决定是否采样前一时刻预测输出作为输入
+ Focal Loss，缓解字符类别之间的数据不均衡问题

实验结果表明，三者均可以对模型效果带来提升，相比于speechTransformer提升幅度在`10.8％～26.1%`；相比于TDNN-LSTM混合系统提升`12.2%～19.1%`

#### 2.7.实战

!> https://github.com/kaituoxu/Speech-Transformer

环境：python3/pytorch>=0.4.1/kaldi

模型参数：

+ 特征维度80，帧率100/3=33.3hz
+ `6*encoder＋6*decoder`
+ $d_{model}＝512$
+ head nums＝8
+ feed-forward network hidden size=2048
+ decoder embedding=512

训练参数：

+ 数据集：aishell train／dev／test
+ epoch=30
+ batch size=32
+ 学习率：可调缩放因子k＝1，学习率线性上升迭代次数warmup_steps＝4000
+ 优化方法：`adam，β1 = 0.9, β2 = 0.98, ε = 10−9 `
+ beam search解码：beam width＝1

实验效果：aishell1 test cer:12.8％