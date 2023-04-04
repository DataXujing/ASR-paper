## WeNet paper 解读


### 1.WeNet 2021: WeNet: Production Oriented Streaming and Non-streaming End-to-End Speech Recognition Toolkit


#### 1.1 摘要

WeNet1.0提供了一种统一流式和非流式的E2E的ASR模型框架U2,目的是为了缩短科研和产品化之间的代沟（gap),U2是一个hybird CTC/attention的架构，基于transformer或conformer作为encoder和attention decoder用来对CTC的结果进行rescore。为了统一流式或非流式的模型架构，WeNet使用dynamic chunk-based attention strategy,这样可以使得 self-attention关注未来数据的任意随机长度，满足不同延时的流式语音识别。在AISHELL-1数据集上和非流式Transformer对比有5.03%的CER的降低，经量化后可部署到边缘设备，代码工具的开原地址：<https://github.com/wenet-e2e/wenet>

#### 1.2 Introduction

相比于传统的hybrid ASR framework, E2E ASR framework像CTC, RNN-T, attention based encoder-decoder(AED)模型具有训练简单的优点。最近的研究也表明在精度上E2E也已经超过了传统的framework。但是部署E2E ASR是不容易的需要解决很多现实问题：

+ the streaming problem: 流式识别是非常重要的，一些E2E ASR无法实现流式识别比如LAS, 原始的Transformer结构。
+ unifying streaming and non-streaming models: 一般的处理方式是流式和非流式的开发是分开的，为了统一成一个模型需要权衡牺牲模型精度性能。
+ the production problem: 这也是WeNet设计的很重要的关注原因，部署的延时和对硬件的消耗是模型部署需要考虑的音素，现在也有一些类似于ONNX,LibTorch,TensorRT,MNN，NCNN等异构计算的推理框架，同时需要语音的前处理和深度学习模型结构优化的知识。

WeNet中的"We"来源于"WeChat"，"Net"来源于"ESPNet",WeNet设计意在缩短E2E ASR模型架构在科研和产品化之间的鸿沟，其主要的优点包括：

+ Production first and production ready: WeNet模型可以方便的转TorchScript供Libtorch调用，前处理使用torchaudio实现，笔者后期会在教程中演示如何在TensorRT中调用WeNet并基于Triton进行云端服务的部署
+ Unified solution for streaming and non-streaming ASR：U2结构和Dynamic chunk training 更好的权衡流式和精度
+ Portable runtime:WeNet提供了在x86和嵌入式设备上部署ASR的方式
+ Light weight: 所有模型基于Pytorch及其生态系统(torchaudio),不依赖于Kaldi

总之WeNet适合科研和产品化，有很多大厂目前也选择WeNet作为其语音识别业务的开发引擎进行二次开发。

#### 1.3 WeNet

WeNet中采用的U2模型，如下图所示，该模型使用Joint CTC/AED的结构，训练时使用CTC和Attention Loss联合优化，并且通过dynamic chunk的训练技巧，使Shared Encoder能够处理任意大小的chunk（即任意长度的语音片段）。在识别的时候，先使用CTC Decoder产生得分最高的多个候选结果，再使用Attention Decoder对候选结果进行重打分(Rescoring)，并选择重打分后得分最高的结果作为最终识别结果。识别时，当设定chunk为无限大的时候，模型需要拿到完整的一句话才能开始做解码，该模型适用于非流式场景，可以充分利用上下文信息获得最佳识别效果；当设定chunk为有限大小（如每0.5秒语音作为1个chunk）时，Shared Encoder可以进行增量式的前向运算，同时CTC Decoder的结果作为中间结果展示，此时模型可以适用于流时场景，而在流失识别结束时，可利用低延时的重打分算法修复结果，进一步提高最终的识别率。可以看到，依靠该结构，我们同时解决了流式问题和统一模型的问题。

<div align=center>
    <img src="zh-cn/img/ch21/p1.png"   /> 
</div>

**训练：**

其损失函数如下：

<div align=center>
    <img src="zh-cn/img/ch21/p2.png"   /> 
</div>

这里的$L_{CTC}(x,y)$表示给定声学特征$x$,输出相关label $y$的CTC Loss,$L_{AED}(x,y)$表示AED Loss.这里的$\lambda$是一个超参数用来权衡两个损失的作用大小。

WeNet使用dynamic chunk training技术来统一流式和非流式的识别，首先input被一个特征的chunk size $C$分割为一些chunks:
$$[t+1,t+2,...,t+C]$$

每个chunk attend自身及其之前的chunk,因此整个CTC Decoder的延时依赖于chunk size,如果chunk size是一个有限大的数则用于流式语音识别，无穷大则是非流式的语音识别；此外，在训练过程中chunk size是动态变化的在1到最大的输入长度之间变化。经验表明chunk size越大精度越高同时带来的流式识别的延时也越大。

**解码：**

WeNet支持以下四种解码方式：

+ attention：就是基于AED通过自回归方式beam search解码
+ ctc greedy search： CTC  greedy search,优点就是解码速度快，但效果不如beam search
+ ctc prefix beam search: 在CTC部分使用CTC prefix beam search,提供最优的n-best候选输出
+ attention rescoring： CTC部分产生n-best候选output,然后对着n-best 候选输出通过AED重新打分找到最优output

在runtime部分支持 attention rescoring.


**系统设计**

为解决落地问题，同时兼顾简单、高效、易于产品化等的准则，WeNet做了如下图的三层系统设计。

<div align=center>
    <img src="zh-cn/img/ch21/p3.png"   /> 
</div>

+ 第一层为PyTorch及其生态。WeNet充分利用PyTorch及其生态以达到使用方便、设计简洁和易于产品化的目标。其中，TorchScript用于开发模型，Torchaudio用于on-the-fly的特征提取，DistributedDataParallel用于分布式训练，Torch JIT(Just In Time)用于模型导出，Pytorch Quantization用于模型量化，LibTorch用于产品模型的推理。
+ 第二层分为研发和产品化两部分。模型研发阶段，WeNet使用TorchScript做模型开发，以保证实验模型能够正确的导出为产品模型。产品落地阶段，用LibTorch Production做模型的推理。
+ 第三层是一个典型的研发模型到落地产品模型的工作流。其中：
	- Data Prepare:数据准备部分，WeNet仅需要准备kaldi风格的wav列表文件和标注文件。
	- Training:WeNet支持on-the-fly特征提取，频谱增强、CTC/AED联合训练和分布式训练。
    - Decoding:支持python环境的模型性能评估，方便在正式部署前的模型调试工作。
    - Export:研发模型直接导出为产品模型，同时支持导出float模型和量化int8模型。
    - Runtime:WeNet中基于LibTorch提供了云端X86和嵌入式端Android的落地方案，并提供相关工具做准确率、实时率RTF(real time factor)， 延时(latency)等产品级别指标的基准测试。


#### 1.4 训练参数及实验结果

使用AISHELL-1数据集，包含150h训练数据，10h的开发数据和5h的测试数据，test包含7176个样本。

模型结构及训练超参数如下：

+ input: `80 fbank` `window 25ms，10ms shfit` `SpecAugment`
+ model: input后接2层的Conv卷积核大小`3*3`,stride大小`2*2`用来对音频时间维度进行降维；使用12层的transformer layer作为encoder,6层的transformer layer作为decoder
+ 训练的超参数：
	- Adam optimizer
	- 25000 warms-up
	- top-K model的weights平均 作为最终模型（ASR好像都这么搞）

其实验结果如下：

<div align=center>
    <img src="zh-cn/img/ch21/p4.png"   /> 
</div>



### 2.WeNet2022: WeNet 2.0: More Productive End-to-End Speech Recognition Toolkit


#### 2.1 摘要

WeNet是一个以产品为导向的E2E ASR toolkit,U2将流式和非流式语音识别统一到一个识别模型。根据产品需求，更新WeNet2.0：
1. U2++,采用双向attention decoders,通过一个从右向左 的attention decoder强化shared encoder的特征表现能力，包含了未来上下文的信息增强rescore的精度;
2. 引入n-gram based的语言模型和WSFT based decoder;
3. 设计了contextual biasing framework(热词)可以使用用户提供的问额外信息比如（通讯列表）增加ASR的识别精度;
4. 工程上设计了unified IO用来支持大尺度数据集的加载。整体来说，WeNet2.o相比于WeNet1.0精度提升了10%。


#### 2.2 Introduction

E2E ASR模型不但简化了语音识别的pipeline,最近的研究表明其精度已经完全超过给hybrid ASR system,常见的E2E ASR model有 CTC, RNN-T, attention based encoder decoder(AED) model.

在生产环境中稳定高效的部署E2E ASR 模型变的非常重要，常用的工具包括ESPNet,SpeechBrrain等是以科研为导向的工具而非产品化为导向的。WeNet是以产品为导向的解决以Transformer和Conformer为基础的端到端模型部署的框架。WeNet以joint CTC/AED 架构为基础，通过U2结构解决流式识别的问题，通过dynamic chunk masking strategy统一流式和非流式模型。runtime可以方便在x86或Android设备部署。

WeNet2.0 更新了如下内容：
+ U2++: 更新U2结构到U2++,同时使用left-to-right和rght-to-left双向的上下文信息，训练阶段学习更丰富的上线文信息，实验结果表明U2++相比于U2在CER上有10%的降低
+ Production language model solution: 在CTC解码阶段，支持n-gram LM,通过WFST与E2E model统一在一起；n-gram LM可以通过大量的累积的文本数据进行快速训练，实验结果表明n-gram LM带来8%的精度提升
+ Contextual biasing(专有领域的语境偏移):是一个很重要的功能。举个例子，对于手机上的ASR，系统要能准确识别出用户说的app的名字，联系人的名字等等，而不是发音相同的其他词。更具体一点，比如读作“Yao Ming”的这个词语，在体育领域可能是我们家喻户晓的运动员“姚明”，但是在手机上，它可能是我们通讯录里面一个叫做“姚敏”的朋友。如何随着应用领域的变化，解决这种偏差问题就是我们这个系列的文章要探索的主要问题。WeNet2.0 设计了一个统一的上下文偏置框架，该框架提供了在流解码阶段利用具有或不具有LM的用户特定上下文信息的机会。
利用用户特定的上下文信息（例如，联系人列表、特定对话状态、对话主题、位置等）在提高ASR准确性和提供快速适应能力方面发挥着重要作用。
实验表明，WeNet2.0 的上下文偏置解决方案可以为有LM和没有LM的情况带来明显的显著改善。
+ Unified IO (UIO):设计了一个统一的IO系统，可以解决大数据和小数据的训练高效加载问题（本文该部分不做详细介绍）

#### 2.3 WeNet 2.0

下面将介绍WeNet2.0中涉及到的U2++,LM，Contextual biasing,UIO的具体更新细节.

**U2++**

U2++ 是一个统一的two-pass joint CTC/AED结构，使用了双向的attention decoders,如下图所示：

<div align=center>
    <img src="zh-cn/img/ch21/p5.png"   /> 
</div>

+ Shared Encoder用来处理声学特征，其由多层的Transformer或Conformer layer构成，过程中仅使用了有限的右侧的上下文信息用来权衡延时问题。
+ CTC Decoder用来对齐声学特征和tokens,其包含一个linear layer,用来将shared layer 的output映射为CTC activation。
+ Left-to-Right Attention Decoder(L2R)对从左到右的有序token序列进行建模，以表示过去上下文信息。
+ Right-to-Left Attention  Decoder(R2L)其对从右到左的反向token序列进行建模，以表示未来的上下文信息。 L2R,R2L Attention Decoder使用多层的Transformer Decoder

解码阶段，CTC Decoder流式解码，beam search多个候选结果交个L2R，R2L Attention Decdoder进行打分(rescore)(过程是non-streaming的)进而得到最优的output。

与U2对比，U2++增加了一个额外的right-to-left attention decoder,用来增强上下文的特征表达，这样上下文的信息不仅来自于历史（left-to-right decoder)还来自于未来(right-to-left decoder)。增强了shared encoder的特征表达能力。

U2++的损失函数：

<div align=center>
    <img src="zh-cn/img/ch21/p6.png"   /> 
</div>

这里的$x$表示声学特征，$y$表示输出label,$\lambda$是一个超参数用来权衡CTC损失和AED损失。对于AED损失，其构成如下：

<div align=center>
    <img src="zh-cn/img/ch21/p7.png"   /> 
</div>

与U2相同，采用dynamic chunk masking strategy策略来统一流式和非流式模型。

训练阶段，首先随机生成一个chunk size $C$
<div align=center>
    <img src="zh-cn/img/ch21/p8.png"   /> 
</div>

进而将input分割成chunk size大小的chunk。最后，当前的chunk和他自身做双向的chunk-level attention decoder,和previous的chunk做L2R attention decoder，和following chunk做R2L attention decoder。

解码阶段，CTC decoder生成n-best的结果，然后通过基于shared encoder与之相关的声学特征特征相关的R2L和L2R attention decoder进行rescoring(重新打分)，最终的输出融合了two attention decoder和CTC decoder的score。

实验表明，大的chunk size精度越好，得益于dynamic strategy,U2++支持任意chunk size用来均衡latecny和accuracy。

**Language Model**

为了使用丰富的文本数据，WeNet2.0提供了一套统一的使用和不使用LM的架构，如下图所示：

<div align=center>
    <img src="zh-cn/img/ch21/p9.png"   /> 
</div>

作为统一的LM和LM-free的系统，CTC用来产生n-best的result,当没有LM时，使用CTC prefix beam search来产生n-best的候选结果；当使用LM时，WeNet2.0将LM(G),lexicon(L),CTC Model(T)构建成WFST-based的解码图结构（TLG):

<div align=center>
    <img src="zh-cn/img/ch21/p10.png"   /> 
</div>

CTC WFST beam search用来产生n-best的候选输出，进而进过attention decoder进行重新打分找到最优的output. CTC WFST beam search服用了Kaldi的代码。

**Contextual Biasing**

利用用户特定的上下文信息(如联系人列表、驾驶员导航)在语音生成中起着至关重要的作用，它总是能显著提高准确性，并提供快速的适应能力。Contextual Biasing是一个成熟的技术可以用在传统的和E2E ASR系统。

WeNet2.0的参考<<Streaming end-to-end speech recognition for mobile devices>>实现可与行在有LM和LM-free的情形下。当提前已知可以写biasing phrases,WeNet2.0用来构建一个contextual WSFT graph。首先，将biasing phrases 分割成 biasing units,过程中如果有LM则基于词典（以词为单位）完成没有则基于 E2E modeling units(以字为单位)；进而，contextual WFST graph的构建过程如下：
+ (1) Each biasing unit with a boosted score（分数提升） is placed on a corresponding
arc（弧） sequentially to generate an acceptable chain. 
+ (2) For each intermediate state(中间状态) of the acceptable chain, a special failure arc with a negative accumulated boosted score is added.
+ (3) The failure arcs are used to remove the boosted scores when only partial biasing units are matched rather than the entire phrase.(当只匹配部分偏置单元而不是整个短语时，失败弧用于去除提升的分数。)

下图展示了LM-free case: char(E2E modeing unit) level context graph和LM case: word-level context graph

<div align=center>
    <img src="zh-cn/img/ch21/p11.png"   /> 
</div>

最后在流解码阶段，当beam sezrch结果通过contextual WFST graph 与biasing unit匹配时，立即添加增强的分数,表示如下：

<div align=center>
    <img src="zh-cn/img/ch21/p12.png"   /> 
</div>

这里的$P_C(x)$表示biasing score,$\lambda$是一个可调整的超参数，表示模型score有多少依赖于contextual LM

**UIO**

!> 如下图这里不做过多介绍

<div align=center>
    <img src="zh-cn/img/ch21/p13.png"   /> 
</div>


#### 2.4 训练参数及实验结果

测试的数据集包括：AISHELL-1,AISHELL-2,KibriSpeech,GigaSpeech,WenetSpeech。 其模型配置如下所述：
+ input: `80维fbank with 25ms window, 10ms shift` + SpecAugment
+ model: 
	- 2个卷基层用来降维卷积核大小`3*3`,stride大小`2*2`
	- 12 conformer layer encoder
	- 3 left-to-right ,3 right-to-left decoder layer 
	- model averaging
+ AISHELL-1,AISHELL-2的attention layer的attention dim是256，ffn是2048个隐含单元，head是4
+ LibriSpeech,GigaSpeech,WenetSpeech的attention layer的attention dim是512，ffn是2048个隐含单元，head是8
+ Conv的卷积核对应5个语料分别为`8/8/31/31/15`

其实验结果如下：

<div align=center>
    <img src="zh-cn/img/ch21/p14.png"   /> 
</div>

<div align=center>
    <img src="zh-cn/img/ch21/p15.png"   /> 
</div>

!> 最后让我们期待WeNet 3.0,作者介绍WeNet3.0可能关注语音识别中的无监督或自监督学习！