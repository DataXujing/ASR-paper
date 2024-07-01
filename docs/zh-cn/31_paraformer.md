## Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition

!> https://arxiv.org/abs/2206.08317

!> https://github.com/alibaba-damo-academy/FunASR

<!-- https://mp.weixin.qq.com/s/EvtK0ExOVAxfOQ0aLmv4xw -->
<!-- https://mp.weixin.qq.com/s/xQ87isj5_wxWiQs4qUXtVw -->

最近一段时间openai开源了whisper,也出现了各种xxformer的ASR解决方案，比如Conformer,Branchformer,EfficientConformer,Squeezeformer,Zipformer,Paraformer。 Paraformer是2022-2023年阿里开源的非自回归的语音识别模型，并开源了工业级的语音识别训练和部署库FunASR。

+ Conformer： 解决语音全局和局部信息的建模。提出的方案是CNN学习局部信息，Transformer学习全局信息，使用夹心饼干的方式结合两者。结果确实比transformer更好了。
+ Branchformer：提出了另一个CNN和Transformer结合的结构，Conformer是串行夹心饼干，它则是并行结合。
+ EfficientConformer：Conformer在深层的时间尺度上下采样提升效率。
+ Squeezeformer：从数据角度证实了时间维度上的冗余，使用U-Net对中间层降采样，从实验角度证明夹心饼干结构是次优。
+ Zipformer：使用了更多种采样率对transformer进行降采样。
+ Paraformer：使用非自回归方式建模。用实验证实了Transformer中的全局信息用token数量和token之间的关系就可以代替。

自回归与非自回归结构如下所示，Transformer模型属于自回归模型，也就是说后面的token的推断是基于前面的token的。不能并行，如果使用非自回归模型的话可以极大提升其速度。

<div align=center>
    <img src="zh-cn/img/ch31/p1.png"   /> 
</div>

### Abstract

Transformer最近在ASR领域占据了主导地位。虽然能够产生良好的性能，但它们涉及一个自回归 （AR） 解码器来逐个生成token，这在计算上效率低下。为了加快推理速度，设计了非自回归 （NAR） 方法，例如单步 NAR(single-step NAR)，以实现并行生成。然而，由于输出token内部的独立性假设，单步NAR的性能不如AR模型，尤其是在大规模语料库的情况下。改进单步NAR面临两个挑战：一是准确预测输出token数量，提取隐藏变量;其次，加强输出token之间相互依赖关系的建模。为了应对这两个挑战，我们提出了一种快速准确的并行Transformer，称为Paraformer。这利用基于continuous integrate-and-fire(CIF)的预测器来预测token的数量并生成隐藏变量。然后，glancing language model （GLM） 采样器生成语义嵌入，以增强 NAR 解码器对上下文相互依赖关系进行建模的能力。最后，我们设计了一种策略来生成负样本，用于最小单词错误率训练，以进一步提高性能。使用公共 AISHELL-1、AISHELL-2 基准测试和工业级 20,000 小时任务的实验表明，所提出的 Paraformer 可以达到与最先进的 AR Transformer相当的性能，加速超过 10 倍。 

### 1.引言

在过去几年中，端到端 （E2E） 模型在自动语音识别 （ASR） 任务上的性能已经超过了传统的混合系统。有三种流行的 E2E 方法：CTC[1]、RNN-T[2] 和基于注意力的编码器-解码器 （AED） [3，4]。其中，AED 模型因其卓越的识别准确性而主导了 ASR 的 seq2seq 建模。例如Transformer [4] 和 Conformer [5]。虽然性能很好，但此类 AED 模型中的自动回归 （AR） 解码器需要逐个生成token，因为每个token都以所有先前的token为条件。因此，解码器的计算效率低下，解码时间随输出序列长度线性增加。为了提高效率和加速推理，已经提出了非自回归（NAR）模型来并行生成输出序列[6,7,8]。

<div align=center>
    <img src="zh-cn/img/ch31/p2.png"   /> 
</div>
<p align=center> 图 1：分析三个系统的不同错误类型，在工业 20,000 小时任务上进行评估 </p>

根据迭代持续时间推断的次数，NAR 模型可以分为迭代模型和单步模型。在前者中，A-FMLM 是第一个尝试 [9]，旨在预测以unmasked tokens为条件的不断迭代的masked tokens。由于需要预定义目标token长度，性能会受到影响。 为了解决这个问题，Mask-CTC及其变体建议使用CTC解码来增强解码器输入[10,11,12]。 即便如此，这些迭代的NAR模型需要多次迭代才能获得有竞争力的结果，这限制了实践中的推理速度。 最近，人们提出了几种单步NAR模型来克服这一局限性[13,14,15,16,17]。它们通过消除时间依赖性同时生成输出序列。 虽然单步NAR模型可以显著提高推理速度，但其识别准确率明显不如AR模型，尤其是在大规模语料库上进行评估时。

上面提到的单步 NAR 工作主要集中在如何预测token数量以及准确提取隐藏变量。与通过预测器网络预测token数量的机器翻译相比，由于说话者的语速、沉默和噪音等各种因素，ASR 确实很困难。另一方面，根据我们的调查，与 AR 模型相比，单步 NAR 模型犯了很多替换错误（图 1 中描述为 AR 和普通 NAR）。我们认为，缺乏上下文相互依赖性会导致替换错误增加，特别是由于单步 NAR 中所需的条件独立性假设。除此之外，所有这些 NAR 模型都是在阅读场景中记录的学术基准上探索的。性能尚未在大规模的工业级语料库上进行评估。 因此，本文旨在改进单步NAR模型，使其在大规模语料库上获得与AR模型相当的识别性能。

**这项工作提出了一种快速准确的并联Transformer模型（称为Paraformer），可以解决上述两个挑战。首先，与以前基于CTC的工作不同，我们利用基于CIF[18]的预测器网络来估计目标token数量并生成隐藏变量。对于第二个挑战，我们设计了一个基于GLM 的采样器模块，以增强 NAR 解码器对token相互依赖关系进行建模的能力。这主要受到神经机器翻译工作的启发[19]。 此外，我们还设计了一种包含负样本的策略，通过利用最小单词错误率 （MWER） [20] 训练来提高性能。**

我们在公共 178 小时 AISHELL-1 和 1000 小时 AISHELL-2 基准测试以及工业 20,000 小时普通话语音识别任务上评估 Paraformer。Paraformer 在 AISHELL-1 和 AISHELL-2 上分别获得了 5.2% 和 6.19% 的 CER，这不仅优于其他最近发表的 NAR 模型，而且可与没有外部语言模型的最先进的 AR 转换器相媲美。据我们所知，Paraformer 是第一个能够达到与 AR 转换器相当的识别精度的 NAR 模型，并且在大型语料库上加速了 10 倍。

### 2.方法

#### 2.1 Overview

所提出的Paraformer模型的整体框架如图2所示。 该架构由五个模块组成，分别是**编码器encoder、预测器predictor、采样器sampler、解码器decoder和损失函数loss function**。编码器与AR编码器相同，由多个配备存储器的自注意力(memory equipped self-attention)（SAN-M）和前馈网络（FFN）[21]或Conformer[5]块组成。 预测器用于产生声学嵌入并指导解码。然后，采样器模块根据声学特征嵌入和 char token嵌入生成语义嵌入。解码器类似于 AR 解码器，只是是双向的。它由 SAN-M、FFN 和交叉多头注意力 （MHA） 的多个模块组成。除了交叉熵 （CE） 损失外，还将引导预测变量收敛的平均绝对误差 （MAE） 和 MWER 损失相结合，共同训练系统。

<div align=center>
    <img src="zh-cn/img/ch31/p3.png"   /> 
</div>
<p align=center> 图 2：Paraformer的结构 </p>


训练过程：

我们将输入表示为$(𝐗,𝐘)$，其中$X$是包含$T$帧的声学特征，$𝐘$是包含$N$个字符的目标识别文本。编码器将输入序列$X$映射 
到隐藏表示序列$𝐇$。然后，这些隐藏表示 $𝐇$被馈送到预测器，以预测token数量$N{'}$并产生声学嵌入 
$𝐄_a$。解码器采用声学嵌入$𝐄_a$和隐藏表示 $𝐇$，生成第一次的目标预测$𝐘^{'}$（不需要梯度的后向传播，代码中是可选的可以后向传播梯度，也可以关掉） 。采样器根据预测$𝐘^{'}$和目标标签$𝐘$之间的距离在声学嵌入 $𝐄_a$和目标字符嵌入 $𝐄_c$之间进行采样 得到语义嵌入$𝐄_s$。 然后，解码器接受语义嵌入 $𝐄_s$和隐藏表示 $𝐇$，以生成最终的第二次预测$𝐘^{''}$（需要梯度的后向传播）最后，对预测$𝐘^{''}$进行采样，为MWER训练生成负候选样本，并在目标token数$N$和预测token数$N^{'}$之间计算MAE。MWER 和 MAE 合并 CE 损失进行联合训练。

推理过程：

采样器模块处于非活动状态，双向并行解码器直接利用声学嵌入$𝐄_a$和隐藏表示 
$𝐇$，仅$𝐘^{'}$通过一次传递预测即可输出最终预测。尽管解码器在每个训练阶段都向前运行两次，但由于单步解码过程，计算复杂度实际上在推理过程中并没有增加。

#### 2.2 Predictor

<div align=center>
    <img src="zh-cn/img/ch31/p4.png"   /> 
</div>
<p align=center> 图 3：CIF 过程的图示（ $\beta$设置为 1） </p>

predictor由2个卷积层组成，输出的范围为0到1的浮点权重$\alpha$。累积权重$\alpha$来预测token数量，MAE损失指导学习predictor

$$ℒ_ {MAE} = |𝒩-\sum^{T}_ {t=1}\alpha_t|$$

通过CIF（Continuous Integrate-and-Fire）机制生成声学嵌入。CIF是一种柔和的单调对齐，在[18]中被提出作为AED模型的流解决方案。为了产生声学嵌入$𝐄_ a$,CIF累积权重$\alpha$并整合隐藏表示$𝐇$,直到累积的权重达到给定的阈值$\beta$,这表明已经达到了声学边界（图3显示了这一过程的说明），举个例子，$\alpha$从左到右，`0.3+0.5+0.4=1.1>1`,于是fire一个token,$𝐄_ {a1}=0.3\times 𝐇_ 1+0.5\times 𝐇_ 2+0.2\times H_ 3$。由于还剩0.1的值没有用，于是0.1用于下一个token计算。同理，$𝐄_ {a2}=0.1\times 𝐇_ 3+0.6\times 𝐇_ 4+0.3\times H_ 5$，$𝐄_ {a3}=0.1\times 𝐇_ 5+0.9\times 𝐇_ 6$，$𝐄_ {a4}=0.2\times 𝐇_ 7+0.6\times 𝐇_ 8$。共fire了4次，也就是4个$𝐄_a$。

根据[18],在训练过程中，权重$\alpha$按目标长度进行缩放$𝐄_ c$,以便将声学嵌入的数量$𝐄_ a$与目标嵌入的数量相匹配，而推断阶段权重$\alpha$则直接用于生成$𝐄_ a$，用于推理。因此，训练和推理之间可能存在不匹配，导致预测变量的精度下降。由于 NAR 模型比流模型对预测变量的准确性更敏感，因此我们建议使用动态阈值$\beta$而不是预定义的阈值来减少不匹配。动态阈值机制表述为：
$$\beta=\frac{\sum^{T}_ {t=1}\alpha_t}{\lceil \sum^{T}_ {t=1}\alpha_t \rceil}$$

#### 2.3 Sampler

在普通的单步NAR中，其优化目标可以表述为：

$$ℒ_ {NAT}=\sum^{N}_ {n=1}\log P(y_ n|X;\theta)$$

然而，如前所述，与 AR 模型相比，条件独立性假设会导致性能较差。同时，GLM(glancing language model) 损失定义为：
$$ℒ_ {GLM}=\sum_ {y^{''}_ n \in \overline{𝔾𝕃𝕄(Y,Y^{'})}}\log p[y^{''}_ {n} | 𝔾𝕃𝕄(Y,Y^{'}),X;\theta]$$

其中$𝔾𝕃𝕄(Y,Y^{'})$表示sampler module在$𝐄_ a$和$𝐄_ c$之间选择token子集,$\overline{𝔾𝕃𝕄(Y,Y^{'})}$表示目标$Y$中剩余的未选择的token子集。

$$𝔾𝕃𝕄(Y,Y^{'})=Sampler(E_s|E_a,E_c,\lceil \lambda d(Y,Y^{'}) \rceil)$$

其中$\lambda$是控制采样率的采样因子。$d(Y,Y^{'})$是采样数。当模型训练不佳时，她会变大，并且应随之训练过程而减少。为此，我们只需要使用汉明距离，定义为：
$$d(Y,Y^{'})=\sum^{N}_ {n=1}(y_n \neq y^{'}_ {n})$$

总而言之，sampler module 通过将目标嵌入$E_c$ 随机替换$\lambda d(Y,Y^{'}) \rceil$个token到声学嵌入$E_ a$,生成语义嵌入$E_ s$。Parallel decoder被训练为使用语义上下文$𝔾𝕃𝕄(Y,Y^{'})$预测目标token：$\overline{𝔾𝕃𝕄(Y,Y^{'})}$,使模型能够学习输出标记之间的相互依赖关系。

#### 2.4 Loss Function

三种损失函数：CE,MAE和MWER losses。联合训练，如下：

$$ℒ_ {total} = \gamma ℒ_ {CE} + ℒ_ {MAE} + ℒ^{N}_ {werr}(x,y^{\ast})$$

对于MWER，它可以表述为[20]:

<div align=center>
    <img src="zh-cn/img/ch31/p6.png"   /> 
</div>

由于使用greedy search decoding ,NAR模型只有一个输出路径，如上所述，我们利用负采样策略，通过在MWER训练期间随机屏蔽top1 score的token来生成多个候选路径。


### 3.实验

#### 3.1 参数

我们在公开可用的 AISHELL-1（178 小时）[26]、AISHELL-2 （1000 小时） 基准 [27] 以及 20,000 小时的工业普通话任务上评估了所提出的方法。后一项任务与 [21，28] 中的大型语料库相同。使用一组约 15 小时的远场数据和一组约 30 小时的通用数据来评估性能。其他配置可以在 [21，28，29] 中找到。 实时率（RTF） 用于测量 GPU （NVIDIA Tesla V100） 上的推理速度。代码开源在FunASR。


#### 3.2 AISHELL-1和AISHELL-2任务

AISHELL-1 和 AISHELL-2 评估结果详见表 1。为了与已发表的作品进行公平的比较，RTF在ESPNET上进行了评估[30]。表 1 中的任何实验均未使用外部语言模型 （LM） 或无监督预训练。 对于 AISHELL-1 任务，我们首先训练了一个 AR transformer 作为基线，其配置与 [ 15 ] 中的 AR 基线相匹配。基线的性能在 AR transformer中是最先进的，不包括具有大规模知识迁移的系统，例如 [ 31]，因为我们的目标是架构改进，而不是从更大的数据集中获益。vanilla NAR 与我们提出的模型 Paraformer 具有相同的结构，但没有采样器。可以看出，我们的vanilla NAR超越了最近发表的其他NAR作品， 
比如改进的CASS-NAT[15]和CTC增强的NAR [12]。 然而，由于输出token之间缺乏上下文依赖性，其性能略逊于 AR 基线。但是，当我们通过 Paraformer 中的采样器模块使用 GLM 增强原版 NAR 时，我们获得了与 AR 模型相当的性能。 虽然 Paraformer 在开发集和测试集上的识别 CER 分别为 4.6% 和 5.2%，但推理速度 （RTF） 比 AR 基线快 12 倍以上。对于 AISHELL-2 任务，模型配置与 AISHELL-1 相同。从表1可以看出，性能提升与AISHELL-1相似。 具体来说，Paraformer 在test_ios任务中实现了 6.19% 的 CER，推理速度提高了 12 倍以上。 据作者所知，这是 NAR 模型在 AISHELL-1 和 AISHELL-2 任务上的最新性能。

<div align=center>
    <img src="zh-cn/img/ch31/p5.png"   /> 
</div>
<p align=center> 表 1：ASR 系统在 AISHELL-1 和 AISHELL-2 任务 （CER%） 上的比较，没有 LM。 AR/NAR表示使用AR或NAR beamsearch, RTF 列的评估批大小为 8， 其他列的批大小为 1）。我们的代码将很快发布 </p>


#### 3.3 Industrial 20,000 hour任务

我们进行了大量的实验来评估我们提出的方法，详见表3。动态$\beta$表示第2.2节中详述的动态阈值，而CTC是指具有LM的DFSMN-CTC-sMBR系统[ 32]。RTF在OpenNMT上进行了评估[ 33]。

<div align=center>
    <img src="zh-cn/img/ch31/p7.png"   /> 
</div>
<p align=center> 表 3：三个系统在工业 20,000 小时任务 （CER%） 上的性能 </p>

首先看大小为 41M 的模型，注意力维度为 256 的 AR 基线与 [ 21 ] 相同。我们可以看到与第 3.2 节中提到的现象不同的现象。在这里，我们发现 vanilla NAR的CER与AR模型的CER相差很大。尽管如此，vanilla NAR 的表现仍然优于 CTC，后者做出了类似的条件独立性假设。 当配备 GLM 时，与普通 NAR 相比，Paraformer 在远场和常见任务上的相对改进分别为 13.5% 和 14.6%。当我们进一步添加MWER训练时，准确性略有提高。更重要的是，Paraformer 实现了与 AR 模型相当的性能（相对损失小于 2%），推理速度提高了 10 倍。 我们还评估了 CIF 的动态阈值。从表3可以看出，动态阈值有助于进一步提高精度。与 CIF 中的预定义阈值相比，动态阈值减少了推理和训练之间的不匹配，从而更准确地提取声学嵌入。

在63M的较大模型尺寸上进行评估，所看到的现象与上述现象相似。在这里，Paraformer 在远场和普通任务上的相对改进分别比普通 NAR 高 13.0% 和 11.1%。同样，Paraformer 的精度与 AR 模型相当（相对差异小于 2.8%），再次实现了 10 倍的加速。如果我们将 Paraformer-63M 与 AR transformer-41M 进行比较，尽管 Paraformer 模型尺寸更大，但其推理速度有所提高（RTF 从 0.067 提高到 0.009）。因此，Paraformer-63M 在远场任务上可以比 AR transformer-41M 实现 6.0% 的相对改进，同时推理速度提高了 7.4 倍。这表明 Paraformer 可以通过增加模型大小来实现卓越的性能，同时仍然保持比 AR transformer 更快的推理速度。

<div align=center>
    <img src="zh-cn/img/ch31/p8.png"   /> 
</div>
<p align=center> 表2：抽样率（CER%）的评估 </p>

最后，我们评估采样器中的采样因子 $\lambda$，如表2所示。正如预期的那样，由于目标提供了更好的上下文，识别准确性会随着 
$\lambda$增加而提高。但是，当采样因子过大时，会导致训练和推理之间的不匹配，我们用训练目标解码两次，在没有目标的情况下解码一次。尽管如此，Paraformer 的性能在$\lambda$在 0.5 到 1.0 的范围内是稳健的。


#### 3.4 讨论

从上述实验中，我们注意到，与AR模型相比，vanilla  NAR在AISHELL-1和AISHELL-2任务上的性能衰减较小，但对于大规模的工业级语料库而言，性能衰减要大得多。与来自阅读语料库的学术基准（例如AISHELL-1和-2）相比，工业级数据集反映了更复杂的场景，因此在评估NAR模型方面更可靠。据我们所知，这是第一个在大规模工业级语料库任务上探索NAR模型的工作。

上面的实验表明，与普通的 NAR 相比，Paraformer 获得了超过 11% 的显着改进，而 Paraformer 的性能与训练有素的 AR transformer 相似。

为了了解原因，我们进行了进一步的分析。首先，我们确定了 AR、vanilla NAR 和 Paraformer 模型在 20,000 小时任务中的误差类型统计数据，如图 1 所示。我们统计了 Far-field 和 Common 上分别插入、删除和替换错误类型的总数，并按目标token总数进行归一化。图1的纵轴是误差类型的比率。 我们可以看到，与AR系统性能相比，vanilla  NAR中的插入错误略有增加，而删除错误则略有减少。这表明在动态阈值的帮助下，预测变量的准确性更高。然而，替换误差急剧上升，这解释了它们之间在性能上的巨大差距。我们认为这是由vanilla  NAR模型中的条件独立性假设引起的。与原版NAR相比，Paraformer的替换误差显著降低，是其性能提升的主要原因。我们认为替代率的下降是因为增强的 GLM 使 NAR 模型能够更好地学习输出代币之间的相互依赖关系。 然而，与AR相比，替换错误的数量仍然存在小的差距，导致识别准确率略有不同。我们认为，究其原因，是因为与GLM相比，AR的beam search在语言模型中可以发挥重要作用。为了消除这个剩余的性能差距，**我们的目标是在未来的工作中将Paraformer与外部语言模型相结合**。


### 4.结论

本文提出了一种单步NAR模型Paraformer，以提高NAR端到端ASR系统的性能。我们首先利用基于CIF的预测器来预测token数量并生成隐藏变量。我们使用动态阈值而不是预定义阈值改进了 CIF，以减少推理和训练之间的不匹配。然后，我们设计了一个基于GLM的采样器模块来生成语义嵌入，以增强NAR解码器对上下文相互依赖性进行建模的能力。最后，我们设计了一种生成负样本的策略，以便进行最小的单词错误率训练，以进一步提高性能。 在公共AISHELL-1（178小时）和AISHELL-2（1000小时）基准测试以及工业级20,000小时语料库上进行的实验表明，所提出的Paraformer模型可以达到与最先进的AR变压器相当的性能，加速超过10倍。

------

## FunASR

### 1.FunASR训练Paraformer，静音检测模型，语言模型，热词增强模型和标点预测模型

> 这一部分我们将基于自己标注的数据微调Paraformer声学模型，并基于标注数据训练语言模型，并添加热词，关于语言模型的训练以及热词的添加，已经在下一节中做了详细的介绍，本节主要介绍训练环境的搭建，训练数据的构建,以及Paraformer模型的微调训练。

#### 1. 环境搭建

+ 需要的基础环境

```shell
python>=3.8
torch>=1.13
torchaudio
```

+ 安装funasr

```shell
git clone https://github.com/alibaba/FunASR.git && cd FunASR
pip3 install -e ./
```

+ 安装modelscope 或 huggingface_hub（下载预训练模型用的）(非必须的)

```
pip install modelscope huggingface_hub
```

+ 下载预训练的模型

[ModelZoo](https://github.com/modelscope/FunASR/tree/main/model_zoo)

<div align=center>
    <img src="zh-cn/img/ch31/p23.png"   /> 
</div>


#### 2. 训练数据构建

+ 最终需要的数据是jsonl格式的，如下所示 (`train.jsonl`)：

```json
{"key": "BAC009S0764W0121", "source": "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav", "source_len": 90, "target": "甚至出现交易几乎停滞的情况", "target_len": 13}
{"key": "BAC009S0916W0489", "source": "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0916W0489.wav", "source_len": 90, "target": "湖北一公司以员工名义贷款数十员工负债千万", "target_len": 20}
{"key": "asr_example_cn_en", "source": "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_cn_en.wav", "source_len": 91, "target": "所有只要处理 data 不管你是做 machine learning 做 deep learning 做 data analytics 做 data science 也好 scientist 也好通通都要都做的基本功啊那 again 先先对有一些也许对", "target_len": 19}
{"key": "ID0012W0014", "source": "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_en.wav", "source_len": 88, "target": "he tried to think how it could be", "target_len": 8}
```

+ 上述数来源2个数据

1. `train_text.txt`

```
# 文件名（文件ID), text
ID0012W0013 当客户风险承受能力评估依据发生变化时
ID0012W0014 所有只要处理 data 不管你是做 machine learning 做 deep learning
ID0012W0015 he tried to think how it could be
```

2. `train_wav.scp`

```
# 文件名（文件ID）， wav路径，wav是16K的采样率
BAC009S0764W0121 https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav
BAC009S0916W0489 https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0916W0489.wav
ID0012W0015 https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_cn_en.wav
```

+ 使用工具`scp2jsonl`进行转换

```shell
# generate train.jsonl and val.jsonl from wav.scp and text.txt
scp2jsonl \
++scp_file_list='["./data/list/train_wav.scp", "./data/list/train_text.txt"]' \
++data_type_list='["source", "target"]' \
++jsonl_file_out="./data/list/train.jsonl"

scp2jsonl \
++scp_file_list='["./data/list/val_wav.scp", "./data/list/val_text.txt"]' \
++data_type_list='["source", "target"]' \
++jsonl_file_out="./data/list/val.jsonl"
```

```shell
scp2jsonl \
++scp_file_list='["./train_wav.scp", "./train_text.txt"]' \
++data_type_list='["source", "target"]' \
++jsonl_file_out="./train.jsonl"

scp2jsonl \
++scp_file_list='["./val_wav.scp", "./val_text.txt"]' \
++data_type_list='["source", "target"]' \
++jsonl_file_out="./val.jsonl"
```


如果从jsonl文件返回txt和scp文件则可以

```shell
# generate wav.scp and text.txt from train.jsonl and val.jsonl
jsonl2scp \
++scp_file_list='["./data/list/train_wav.scp", "./data/list/train_text.txt"]' \
++data_type_list='["source", "target"]' \
++jsonl_file_in="./data/list/train.jsonl"
```


#### 3. 模型训练

```shell
# os.environ['PYTORCH_JIT'] = '0'
funasr/bin/train.py \
++model="/home/myuser/LLMs/asr/FunASR/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online" \
++train_data_set_list="/home/myuser/LLMs/asr/FunASR/data/list/train.jsonl" \
++valid_data_set_list="/home/myuser/LLMs/asr/FunASR/data/list/val.jsonl" \
++dataset="AudioDataset" \
++dataset_conf.index_ds="IndexDSJsonl" \
++dataset_conf.data_split_num=1 \
++dataset_conf.batch_sampler="BatchSampler" \
++dataset_conf.batch_size=1 \
++dataset_conf.batch_type="token" \
++dataset_conf.num_workers=1 \
++train_conf.max_epoch=5 \
++train_conf.log_interval=10 \
++train_conf.resume=false \
++train_conf.validate_interval=2000 \
++train_conf.save_checkpoint_interval=2000 \
++train_conf.keep_nbest_models=20 \
++train_conf.avg_nbest_model=10 \
++optim_conf.lr=0.0002 \
++output_dir="./save_model" &> train_20240626.log


funasr/bin/train.py \
++model="/workspace/FunASR/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online" \
++train_data_set_list="/workspace/FunASR/data/train.jsonl" \
++valid_data_set_list="/workspace/FunASR/data/val.jsonl" \
++dataset="AudioDataset" \
++dataset_conf.index_ds="IndexDSJsonl" \
++dataset_conf.data_split_num=1 \
++dataset_conf.batch_sampler="BatchSampler" \
++dataset_conf.batch_size=512 \
++dataset_conf.batch_type="token" \
++dataset_conf.num_workers=4 \
++train_conf.max_epoch=50 \
++train_conf.log_interval=50 \
++train_conf.resume=false \
++train_conf.validate_interval=500 \
++train_conf.save_checkpoint_interval=500 \
++train_conf.keep_nbest_models=20 \
++train_conf.avg_nbest_model=10 \
++optim_conf.lr=0.0002 \
++output_dir="./save_model" 

```

参数说明：


+ `model`（str）: The name of the model (the ID in the model repository), at which point the script will automatically download the model to local storage; alternatively, the path to a model already downloaded locally.
+ `train_data_set_list`（str）: The path to the training data, typically in jsonl format, for specific details refer to examples.
+ `valid_data_set_list`（str）：The path to the validation data, also generally in jsonl format, for specific details refer to examples](https://github.com/alibaba-damo-academy/FunASR/blob/main/data/list).
+ `dataset_conf.batch_type`（str）：example (default), the type of batch. example means batches are formed with a fixed number of batch_size samples; length or token means dynamic batching, with total length or number of tokens of the batch equalling batch_size.
+ `dataset_conf.batch_size`（int）：Used in conjunction with batch_type. When batch_type=example, it represents the number of samples; when batch_type=length, it represents the length of the samples, measured in fbank frames (1 frame = 10 ms) or the number of text tokens.
+ `train_conf.max_epoch`（int）：The total number of epochs for training.
+ `train_conf.log_interval`（int）：The number of steps between logging.
+ `train_conf.resume`（int）：Whether to enable checkpoint resuming for training.
+ `train_conf.validate_interval`（int）：The interval in steps to run validation tests during training.
+ `train_conf.save_checkpoint_interval`（int）：The interval in steps for saving the model during training.
+ `train_conf.keep_nbest_models`（int）：The maximum number of model parameters to retain, sorted by validation set accuracy, from highest to lowest.
+ `train_conf.avg_nbest_model`（int）：Average over the top n models with the highest accuracy.
+ `optim_conf.lr`（float）：The learning rate.
+ `output_dir`（str）：The path for saving the model.
+ `**kwargs`(dict): Any parameters in config.yaml can be specified directly here, for example, to filter out audio longer than 20s: dataset_conf.max_token_length=2000, measured in fbank frames (1 frame = 10 ms) or the number of text tokens.

看到下述界面应该是训练过程成功了：

<div align=center>
    <img src="zh-cn/img/ch31/p24.png"   /> 
</div>


训练数据的相关统计：

```shell
wave num: 4325
Effective hours:  9.08088696180557
```


训练的日志：

<div align=center>
    <img src="zh-cn/img/ch31/p25.png"   /> 
</div>

<div align=center>
    <img src="zh-cn/img/ch31/p26.png"   /> 
</div>


#### 4. 流式语音识别服务的搭建

<!-- https://mp.weixin.qq.com/s/8He081-FM-9IEI4D-lxZ9w -->
FunASR实时语音听写软件包，集成了实时版本的语音端点检测模型(静音检测）、语音识别、标点预测模型等。采用多模型协同，既可以实时的进行语音转文字，也可以在说话句尾用高精度转写文字修正输出，输出文字带有标点，支持多路请求。依据使用者场景不同，支持实时语音听写服务（online）、非实时一句话转写（offline）与实时与非实时一体化协同（2pass）3种服务模式。软件包提供有html、python、c++、java与c#等多种编程语言客户端，用户可以直接使用与进一步开发。

<div align=center>
    <img src="zh-cn/img/ch31/online_structure.png"   /> 
</div>

0. 流式语音识别镜像拉取

```shell
sudo docker pull \
  registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.10

mkdir -p ./funasr-runtime-resources/models

sudo docker run -p 10096:10095 -it --privileged=true \
  -v $PWD/funasr-runtime-resources/models:/workspace/models \
  registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.10
```


1. 模型转换
pytorch导出onnx：

```shell
funasr-export ++model=/workspace/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/ ++output-dir=/workspace/models ++type=onnx ++quantize=true
```

```shell
funasr-export ++model=/workspace/save_model/  ++type=onnx ++quantize=true
```


```shell
funasr-export ++model=../save_model/  ++type=onnx ++quantize=true
```

!> 导出过程可能会出现`model.export_model`的错误，注意按照错误指导进行修改。国人的开源项目去开发的时候本着不PR就不要PUA的原则任何报错只能自己动手去搞


3. 静音检测, 标点预测, 语言模型模型, 逆文本标准化

+ FSMN语音端点检测：https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-8k-common
+ CT-Transformer标点预测： https://modelscope.cn/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-onnx
+ 基于FST的Ngram语言模型：https://modelscope.cn/models/iic/speech_ngram_lm_zh-cn-ai-wesp-fst 
+ 基于FST的中文ITN: https://modelscope.cn/models/thuduj12/fst_itn_zh
+ 关于热词模型： https://github.com/modelscope/FunASR/issues/851


语音识别的后处理技术，主要是优化语音识别产品的用户体验，包括：口语顺滑(Disfluency Detection)、标点恢复(Punctuation Restoration)和逆文本标准化(Inverse Text Normalization)等。


4. 语言模型的训练

替换掉原有的语音识别模型。

参考：[FunASR部署流式或非流式加热词和语言模型的Paraformer](https://dataxujing.github.io/ASR-paper/#/zh-cn/31_paraformer?id=_2funasr%e9%83%a8%e7%bd%b2%e6%b5%81%e5%bc%8f%e6%88%96%e9%9d%9e%e6%b5%81%e5%bc%8f%e5%8a%a0%e7%83%ad%e8%af%8d%e5%92%8c%e8%af%ad%e8%a8%80%e6%a8%a1%e5%9e%8b%e7%9a%84paraformer)
中的语言模型的训练的介绍

+ https://github.com/modelscope/FunASR/blob/main/runtime/docs/lm_train_tutorial.md

注意需要安装openfst: 

+ https://www.openfst.org/twiki/bin/view/FST/FstDownload
+ https://blog.csdn.net/qq_33424313/article/details/122293358

设置环境变量

```shell
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```


5. 扩展热词

模型不变，增加热词词典

<div align=center>
    <img src="zh-cn/img/ch31/p27.png"   /> 
</div>


6. 开启流式语音识别服务


+ https://github.com/modelscope/FunASR/blob/main/runtime/docs/SDK_advanced_guide_online_zh.md#%E6%9C%8D%E5%8A%A1%E7%AB%AF%E7%94%A8%E6%B3%95%E8%AF%A6%E8%A7%A3
+ https://github.com/modelscope/FunASR/issues/883

```shell
cd FunASR/runtime
nohup bash run_server_2pass.sh \
  --download-model-dir /workspace/models \
  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --model-dir damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx  \
  --online-model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx  \
  --punc-dir damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx \
  --lm-dir damo/speech_ngram_lm_zh-cn-ai-wesp-fst \
  --itn-dir thuduj12/fst_itn_zh \
  --hotword /workspace/models/hotwords.txt > log.txt 2>&1 &

# 如果您想关闭ssl，增加参数：--certfile 0
# 如果您想使用时间戳或者nn热词模型进行部署，请设置--model-dir为对应模型：
#   damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx（时间戳）
#   damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404-onnx（nn热词）
# 如果您想在服务端加载热词，请在宿主机文件./funasr-runtime-resources/models/hotwords.txt配置热词（docker映射地址为/workspace/models/hotwords.txt）:
#   每行一个热词，格式(热词 权重)：阿里巴巴 20（注：热词理论上无限制，但
```


```shell
bash run_server_2pass.sh 
  --certfile 0 \
  --vad-dir /workspace/models/damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --model-dir /workspace/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online \
  --online-model-dir /workspace/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online  \
  --punc-dir /workspace/models/damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx \
  --lm-dir /workspace/FunASR/runtime/tools/lm/resource \
  --itn-dir  /workspace/models/thuduj12/fst_itn_zh \
  --hotword /workspace/hotwords.txt
```

```shell
bash run_server_2pass.sh --certfile 0 --port 10098 --vad-dir /workspace/models/damo/speech_fsmn_vad_zh-cn-16k-common-onnx --model-dir /workspace/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online --online-model-dir /workspace/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online  --punc-dir /workspace/models/damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx --lm-dir /workspace/FunASR/runtime/tools/lm/resource --itn-dir  /workspace/models/thuduj12/fst_itn_zh --hotword /workspace/hotwords.txt

```

```shell
bash run_server_2pass.sh --certfile 0 --port 10098 --vad-dir /workspace/models/damo/speech_fsmn_vad_zh-cn-16k-common-onnx --model-dir /workspace/save_model --online-model-dir /workspace/save_model  --punc-dir /workspace/models/damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx --lm-dir /workspace/FunASR/runtime/tools/lm/resource --itn-dir  /workspace/models/thuduj12/fst_itn_zh --hotword /workspace/hotwords.txt

```

部署的最终命令：

```shell
bash run_server_2pass.sh --certfile 0 --port 10098 --vad-dir /workspace/models/damo/speech_fsmn_vad_zh-cn-16k-common-onnx --model-dir /workspace/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx --online-model-dir /workspace/save_model  --punc-dir /workspace/models/damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx --lm-dir /workspace/FunASR/runtime/tools/lm/resource --itn-dir  /workspace/models/thuduj12/fst_itn_zh --hotword /workspace/hotwords.txt

```

```shell
ps -x | grep funasr
kill -9 ID
```


#### 5. C++流式语音识别客户端调用

TODO


### 2.FunASR部署流式或非流式加热词和语言模型的Paraformer

这里以FunASR离线文件转写服务开发为例，测试如何调用离线的预训练Paraformer和热词增强实现离线语音识别服务。

<div align=center>
    <img src="zh-cn/img/ch31/offline_structure.jpg"   /> 
</div>

+ 下载Docker镜像

```shell
sudo docker pull \
  registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-0.4.4
```

实例化容器

```shell
sudo docker run -p 10095:10095 -p 10096:10096 -p 10097:10097 -it --privileged=true \
  registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-0.4.4
```

更新安装funasr

```shell
pip install -U funasr
```

+ 准备Paraformer模型，语言模型和热词模型或热词词表

下载预训练的模型

```
https://github.com/alibaba-damo-academy/FunASR/tree/main/model_zoo

https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary
```

构建热词词表

FunASR支持训练神经网络热词模型，也支持热词词表。下面构建热词词表

```
寻腔 100
进境 100
寻腔进境 100
到达部位 100
回肠末端 100
回盲部 100
退镜观察 100
绒毛状态 100
结构规则 100
充血水肿 100
溃疡 100
肿物 100
黏膜状态 100
光滑 100
糜烂 100
清晰 100
```



copy到容器

```
sudo docker cp funasr_model/ bc0e3f4af6b1:/workspace
```


pytorch导出onnx

```
funasr-export ++model=/workspace/funasr_model/ ++export-dir=./models ++type=onnx ++quantize=true
```


训练语言模型

<!-- ```
https://zhuanlan.zhihu.com/p/465801692
https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/docs/lm_train_tutorial.md
``` -->

**>>>安装srilm**

SRILM是一个构建和应用统计语言模型的开源工具包，主要用于语音识别，统计标注和切分，以及机器翻译，可运行在UNIX及Windows平台上,SRILM的主要目标是支持语言模型的估计和评测。

srilm安装包下载：

```
#百度云盘
https://link.zhihu.com/?target=https%3A//pan.baidu.com/s/18T474NLSqlBL_xhMKEivnA

#提取码

adsl
```

TCL安装包下载：

```
#百度云盘
https://link.zhihu.com/?target=https%3A//pan.baidu.com/s/1E-0E_IrY5rLnfKAepoY5BA
#提取码
adsl
```

在此，大家肯定会疑问SRILM工具包的安装，为什么还要放一个TCL 的压缩包，这是因为我门SRILM的安装需要依赖在TCL工具上（脚本解释工具），因此在安装过程中需要先安装TCL，再安装SRILM。

TCL安装：

解压：

```shell
tar -xf tcl8.7a5-src.tar.gz
```

然后进入解压后的目录，进入unix目录。执行命令 ：

```shell
./configure

```

打开Makefile文件，将其中的`/usr/local` 替换成 `个人目录/tcl` (以`/workspace/tcl`为例)。替换完成后执行命令：

```shell
make
#（root权限可以直接运行命令，过程中会出现很多日志，等待运行完。）

```

<div align=center>
    <img src="zh-cn/img/ch31/p13.png"   /> 
</div>

运行完成并出现上图所示内容，执行命令：

```shell
make install
```

<div align=center>
    <img src="zh-cn/img/ch31/p14.png"   /> 
</div>

出现上图所示即为成功，`/workspace/tcl` 目录如下图所示：

<div align=center>
    <img src="zh-cn/img/ch31/p15.png"   /> 
</div>

SRILM安装：

在`/workspace/`目录下 创建一个srilm的文件夹，在该文件夹下解压SRILM的压缩包。

```shell
tar -xf srilm-1.7.1.tar.gz
```
如图所示:

<div align=center>
    <img src="zh-cn/img/ch31/p16.png"   /> 
</div>

打开Makefile文件，修改参数：

打开Makefile文件，修改参数：

第七行：

<div align=center>
    <img src="zh-cn/img/ch31/p17.png"   /> 
</div>

修改成：

```
SRILM = $(PWD)
```

第十三行：

<div align=center>
    <img src="zh-cn/img/ch31/p18.png"   /> 
</div>

修改成：

<div align=center>
    <img src="zh-cn/img/ch31/p19.png"   /> 
</div>

进入common文件夹，如下所示：

<div align=center>
    <img src="zh-cn/img/ch31/p20.png"   /> 
</div>

找到上述第十三行修改的文件名Makefile.machine.i686-m64 并打开：

该文件第五十四行：

```
NO_TCL = 1
```

修改成：

```
NO_TCL = X
```

回到srilm目录下：执行命令：

```shell
make World 
#（接着等待…）

```

<div align=center>
    <img src="zh-cn/img/ch31/p21.png"   /> 
</div>


显示上图即编译成功，进行测试：

环境变量：

```
export PATH=/workspace/srilm/bin/:/workspace/srilm/bin:$PATH
```

测试命令：

```
make test

```


**>>>准备训练数据集**

```
# 下载: 示例训练语料text、lexicon 和 am建模单元units.txt
wget https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/requirements/lm.tar.gz
# 如果是匹配8k的am模型，使用 https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/requirements/lm_8358.tar.gz
tar -zxvf lm.tar.gz
```

解压后，按照格式增加`text`中的数据，比如:


<div align=center>
    <img src="zh-cn/img/ch31/p22.png"   /> 
</div>


**>>>训练arpa**

修改`runtime/tools/fst/train_lms.sh`中的`ngram-count`的路径：

```shell
#第22行修改为：
/workspace/srilm/bin/i686-m64/ngram-count
```

训练模型：

```shell
# make sure that srilm is installed
# the format of the text should be:
# BAC009S0002W0122 而 对 楼市 成交 抑制 作用 最 大 的 限 购
# BAC009S0002W0123 也 成为 地方 政府 的 眼中 钉

bash fst/train_lms.sh
```


**>>>生成lexicon**

```shell
python3 fst/generate_lexicon.py lm/corpus.dict lm/lexicon.txt lm/lexicon.out
```

**>>>编译TLG.fst**

编译TLG需要依赖fst的环境

```
# Compile the lexicon and token FSTs
fst/compile_dict_token.sh  lm lm/tmp lm/lang

# Compile the language-model FST and the final decoding graph TLG.fst
fst/make_decode_graph.sh lm lm/lang || exit 1;

# Collect resource files required for decoding
fst/collect_resource_file.sh lm lm/resource

#编译后的模型资源位于 lm/resource
```


+ 启动funasr-wss-server服务

启动 funasr-wss-server服务程序：

```shell
cd FunASR/runtime
nohup bash run_server.sh \
  --download-model-dir /workspace/models \  # 在魔塔下载的模型文件
  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --model-dir damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx  \
  --punc-dir damo/punc_ct-transformer_cn-en-common-vocab471067-large-onnx \
  --lm-dir damo/speech_ngram_lm_zh-cn-ai-wesp-fst \
  --itn-dir thuduj12/fst_itn_zh \
  --hotword /workspace/models/hotwords.txt > log.txt 2>&1 &

# 如果您想关闭ssl，增加参数：--certfile 0
# 如果您想使用时间戳或者nn热词模型进行部署，请设置--model-dir为对应模型：
#   damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx（时间戳）
#   damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404-onnx（nn热词）
# 如果您想在服务端加载热词，请在宿主机文件./funasr-runtime-resources/models/hotwords.txt配置热词（docker映射地址为/workspace/models/hotwords.txt）:
#   每行一个热词，格式(热词 权重)：阿里巴巴 20（注：热词理论上无限制，但为了兼顾性能和效果，建议热词长度不超过10，个数不超过1k，权重1~100）


```

参数说明：

```shell
--download-model-dir 模型下载地址，通过设置model ID从Modelscope下载模型
--model-dir  modelscope model ID 或者 本地模型路径
--vad-dir  modelscope model ID 或者 本地模型路径
--punc-dir  modelscope model ID 或者 本地模型路径
--lm-dir modelscope model ID 或者 本地模型路径
--itn-dir modelscope model ID 或者 本地模型路径
--port  服务端监听的端口号，默认为 10095
--decoder-thread-num  服务端线程池个数(支持的最大并发路数)，
                      脚本会根据服务器线程数自动配置decoder-thread-num、io-thread-num
--io-thread-num  服务端启动的IO线程数
--model-thread-num  每路识别的内部线程数(控制ONNX模型的并行)，默认为 1，
                    其中建议 decoder-thread-num*model-thread-num 等于总线程数
--certfile  ssl的证书文件，默认为：../../../ssl_key/server.crt，如果需要关闭ssl，参数设置为0
--keyfile   ssl的密钥文件，默认为：../../../ssl_key/server.key
--hotword   热词文件路径，每行一个热词，格式：热词 权重(例如:阿里巴巴 20)，
            如果客户端提供热词，则与客户端提供的热词合并一起使用，服务端热词全局生效，客户端热词只针对对应客户端生效。

```

```shell
export PYTHONPATH=/workspace/FunASR

./run_server.sh --certfile 0\
  --model-dir /workspace/funasr_model  \
  --hotword /workspace/funasr_model/hotwords.txt 

```

<div align=center>
    <img src="zh-cn/img/ch31/p9.png"   /> 
</div>

加载自己训练的lm
```shell
export PYTHONPATH=/workspace/FunASR

./run_server.sh --certfile 0\
  --model-dir /workspace/funasr_model  \
  --hotword /workspace/funasr_model/hotwords.txt \
  --lm-dir /workspace/FunASR/runtime/tools/lm/resource

```

停止服务

```
ps -x | grep funasr-wss-server
kill -9 PID

```


+ html客户端

chrome浏览器打开：`funasr_samples\samples\html\static\index.html`，注意修改`main.js`使其支持`ws`和`http`。

<div align=center>
    <img src="zh-cn/img/ch31/p10.png"   /> 
</div>


+ 客户端测试

```
python funasr_wss_client.py --host "10.10.15.106" --port 10095 --ssl 0 --mode offline --audio_in "./long.wav" --output_dir "./results"
```

<div align=center>
    <img src="zh-cn/img/ch31/p11.png"   /> 
</div>

```
demo    富士康在印度工厂出现大规模感染，目前工厂产量已下降超50%。  [[520,700],[700,820],[820,1100],[1100,1320],[1320,1540],[1540,1860],[1860,2020],[2020,2280],[2280,2420],[2420,2700],[2700,2920],[2920,3080],[3080,3360],[3360,3560],[3560,4020],[4020,4200],[4200,4460],[4460,4620],[4620,4880],[4880,5040],[5040,5280],[5280,5500],[5500,5680],[5680,5920],[5920,6240],[6240,6651],[6651,7062],[7062,7475]]
```


在服务器上完成FunASR服务部署以后，可以通过如下的步骤来测试和使用离线文件转写服务。 目前分别支持以下几种编程语言客户端

+ [Python](https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/docs/SDK_advanced_guide_offline_zh.md#python-client)
+ [CPP](https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/docs/SDK_advanced_guide_offline_zh.md#cpp-client)
+ [html网页](https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/docs/SDK_advanced_guide_offline_zh.md#Html%E7%BD%91%E9%A1%B5%E7%89%88)
+ [Java](https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/docs/SDK_advanced_guide_offline_zh.md#Java-client)

以上我们尝试了基于html和python的websocket的调用方式，我们修改简化了python调用，其代码如下：

```python
'''
徐静
2024-03-15

'''
import os
import time
import websockets, ssl
import wave
import asyncio
import json

async def record_from_scp(chunk_begin,wav_path):
    # global voices
    # is_finished = False
    chunk_size=[5, 10, 5]
    chunk_interval = 10
    use_itn=True
    mode = "2pass"  # "offline, online, 2pass"
    # wavs = "xxx.wav"

    # wav_path = "xxx.wav"
    with wave.open(wav_path, "rb") as wav_file:
        params = wav_file.getparams()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
        audio_bytes = bytes(frames)

    stride = int(60 * chunk_size[1] / chunk_interval / 1000 * sample_rate * 2)
    chunk_num = (len(audio_bytes) - 1) // stride + 1


    # send first time
    message = json.dumps({"mode": mode, "chunk_size": chunk_size, "chunk_interval": chunk_interval, "audio_fs":sample_rate,
                          "wav_name": "demo", "wav_format": "pcm", "is_speaking": True, "hotwords":"", "itn": use_itn})

    await websocket.send(message)

    is_speaking = True
    for i in range(chunk_num):

        beg = i * stride
        data = audio_bytes[beg:beg + stride]
        message = data
        #voices.put(message)
        await websocket.send(message)
        if i == chunk_num - 1:
            is_speaking = False
            message = json.dumps({"is_speaking": is_speaking})
            #voices.put(message)
            await websocket.send(message)

        sleep_duration = 0.001
        await asyncio.sleep(sleep_duration)

async def message(id):
    while True:
        try:
            meg = await websocket.recv()
            meg = json.loads(meg)
            wav_name = meg.get("wav_name", "demo")
            text = meg["text"]

            offline_msg_done = meg.get("is_final", True)

            # print(meg)
            # print(text)

            offline_msg_done = True

            await websocket.close()
        except Exception as e:
            # print("Exce: ",e)
            # exit(0)
            break

    return text

async def ws_client(id,wav_path):
    global websocket,offline_msg_done

    offline_msg_done=False
    uri = "ws://{}:{}".format("10.10.15.106", 10095)
    ssl_context = None
    print("connect to", uri)

    async with websockets.connect(uri, subprotocols=["binary"], ping_interval=None, ssl=ssl_context) as websocket:
        task1 = asyncio.create_task(record_from_scp(id,wav_path))
        task2 = asyncio.create_task(message(str(id))) #processid+fileid
        return await asyncio.gather(task1, task2)

if __name__ == "__main__":

    loop =  asyncio.get_event_loop()
    task = loop.create_task(ws_client(0,"./long.wav"))
    loop.run_until_complete(task)
    loop.close()

    print(task.result()[1])

```


执行上述代码的调用输出结果如下:

<div align=center>
    <img src="zh-cn/img/ch31/p12.png"   /> 
</div>


+ Gradio网页版本测试

我们将FunASR的调用集成到gradio中，并且和我们的任务型对话机器人进行关联，实现类似于微信的发送语音或文本实现和对话机器人交互的目的。

!> gradio实现离线语音识别

我们实现了基于gradio的调用

<div align=center>
    <img src="zh-cn/img/ch31/funasr_gradio1.png"   /> 
</div>

其代码如下：

```python
# -*- encoding: utf-8 -*-
import os
import time
import websockets, ssl
import wave
import asyncio
import json

import gradio as gr
import numpy as np
import uuid
from scipy.io import wavfile
# from scipy.signal import resample
import librosa
import soundfile as sf

async def record_from_scp(chunk_begin,wav_path):
    # global voices
    # is_finished = False
    chunk_size=[5, 10, 5]
    chunk_interval = 10
    use_itn=True
    mode = "2pass"  # "offline, online, 2pass"
    # wav_path = "xxx.wav"
    with wave.open(wav_path, "rb") as wav_file:
        params = wav_file.getparams()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
        audio_bytes = bytes(frames)

    stride = int(60 * chunk_size[1] / chunk_interval / 1000 * sample_rate * 2)
    chunk_num = (len(audio_bytes) - 1) // stride + 1


    # send first time
    message = json.dumps({"mode": mode, "chunk_size": chunk_size, "chunk_interval": chunk_interval, "audio_fs":sample_rate,
                          "wav_name": "demo", "wav_format": "pcm", "is_speaking": True, "hotwords":"", "itn": use_itn})

    await websocket.send(message)

    is_speaking = True
    for i in range(chunk_num):

        beg = i * stride
        data = audio_bytes[beg:beg + stride]
        message = data
        #voices.put(message)
        await websocket.send(message)
        if i == chunk_num - 1:
            is_speaking = False
            message = json.dumps({"is_speaking": is_speaking})
            #voices.put(message)
            await websocket.send(message)

        sleep_duration = 0.001
        await asyncio.sleep(sleep_duration)



async def message(id):
    while True:
        try:
            meg = await websocket.recv()
            meg = json.loads(meg)
            wav_name = meg.get("wav_name", "demo")
            text = meg["text"]

            offline_msg_done = meg.get("is_final", True)
            offline_msg_done = True

            await websocket.close()
        except Exception as e:
            break

    return text


async def ws_client(id,wav_path):
    global websocket,offline_msg_done

    offline_msg_done=False
    uri = "ws://{}:{}".format("10.10.15.106", 10095)
    ssl_context = None
    print("connect to", uri)

    async with websockets.connect(uri, subprotocols=["binary"], ping_interval=None, ssl=ssl_context) as websocket:
        task1 = asyncio.create_task(record_from_scp(id,wav_path))
        task2 = asyncio.create_task(message(str(id))) #processid+fileid
        return await asyncio.gather(task1, task2)



def recognition(audio):
    sr, y = audio
    print(sr)

    audio_signal = y
    save_name = os.path.join("./audio",str(uuid.uuid1()).replace("-","_")+".wav")
    wavfile.write(save_name, sr, audio_signal)

    # 重采样音频
    if sr != 16000:
        rc_sig,sr = sf.read(save_name)
        audio_signal_16k = librosa.resample(rc_sig,sr,16000)
        sf.write(save_name,audio_signal_16k,16000)

    # loop =  asyncio.get_event_loop()
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    task = new_loop.create_task(ws_client(0,save_name))
    new_loop.run_until_complete(task)
    new_loop.close()
    # print(preds)
    preds = task.result()[1]
    return preds

text = "<h3>基于Paraformer和热词的语音识别 | Medcare Test</h3>"
# iface = gr.Interface(recognition, inputs="mic", outputs="text",description=text)
# iface = gr.Interface(recognition, inputs=gr.Audio(source="microphone", type="filepath"), outputs="text",description=text)

iface = gr.Interface(recognition, inputs=gr.Audio(sources=["microphone"],format="wav"), outputs="text",description=text)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0",server_port=8089)

```


!> gradio实现离线语音识别+任务型对话机器人关联

我们结合任务型对话机器人，实现了基于gradio的语音识别与任务型对话机器人的交互


<div align=center>
    <img src="zh-cn/img/ch31/funasr_gradio2.png"   /> 
</div>

其代码如下：

```python


"""
徐静
2023003-16
"""
import os
import time
import websockets, ssl
import wave
import asyncio
import json

import gradio as gr
import numpy as np
import uuid
from scipy.io import wavfile
# from scipy.signal import resample
import librosa
import soundfile as sf

# import random
import uuid
import requests

global user_id

async def record_from_scp(chunk_begin,wav_path):
    # global voices
    # is_finished = False
    chunk_size=[5, 10, 5]
    chunk_interval = 10
    use_itn=True
    mode = "2pass"  # "offline, online, 2pass"
    # wav_path = "xxx.wav"
    with wave.open(wav_path, "rb") as wav_file:
        params = wav_file.getparams()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
        audio_bytes = bytes(frames)

    stride = int(60 * chunk_size[1] / chunk_interval / 1000 * sample_rate * 2)
    chunk_num = (len(audio_bytes) - 1) // stride + 1


    # send first time
    message = json.dumps({"mode": mode, "chunk_size": chunk_size, "chunk_interval": chunk_interval, "audio_fs":sample_rate,
                          "wav_name": "demo", "wav_format": "pcm", "is_speaking": True, "hotwords":"", "itn": use_itn})

    await websocket.send(message)

    is_speaking = True
    for i in range(chunk_num):

        beg = i * stride
        data = audio_bytes[beg:beg + stride]
        message = data
        #voices.put(message)
        await websocket.send(message)
        if i == chunk_num - 1:
            is_speaking = False
            message = json.dumps({"is_speaking": is_speaking})
            #voices.put(message)
            await websocket.send(message)

        sleep_duration = 0.001
        await asyncio.sleep(sleep_duration)



async def message(id):
    while True:
        try:
            meg = await websocket.recv()
            meg = json.loads(meg)
            wav_name = meg.get("wav_name", "demo")
            text = meg["text"]

            offline_msg_done = meg.get("is_final", True)
            offline_msg_done = True

            await websocket.close()
        except Exception as e:
            break

    return text


async def ws_client(id,wav_path):
    global websocket,offline_msg_done

    offline_msg_done=False
    uri = "ws://{}:{}".format("10.10.15.106", 10095)
    ssl_context = None
    print("connect to", uri)

    async with websockets.connect(uri, subprotocols=["binary"], ping_interval=None, ssl=ssl_context) as websocket:
        task1 = asyncio.create_task(record_from_scp(id,wav_path))
        task2 = asyncio.create_task(message(str(id))) #processid+fileid
        return await asyncio.gather(task1, task2)

# 调用Rasa
def post(url, data=None):
    data = json.dumps(data, ensure_ascii=False)
    data = data.encode(encoding="utf-8")
    r = requests.post(url=url, data=data)
    # r = json.loads(r.text,encoding="gbk")
    r = json.loads(r.text,encoding="utf-8")
    return r


def get_robot(user_id, input):
    # sender = secrets.token_urlsafe(16)
    url = "http://10.10.15.106:5005/webhooks/rest/webhook"
    data = {"sender": user_id,"message": input}
    output = post(url, data)
    return output[0]["text"]



#gradio chat demo
# my_theme = gr.Theme.from_hub("gradio/seafoam")
with gr.Blocks() as demo:

    user_id = str(uuid.uuid1()).replace("-","_")
    # text = "<h3>基于Paraformer和热词的语音识别 | Medcare Test</h3>"
    gr.Markdown(
        f"""
        ## 语音识别+任务型对话机器人
        ### Paraformer+RASA
        """)

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(render_markdown=False, label="Medcare 智能报告",
                                bubble_full_width=False,
                                avatar_images=("./static/doctor.png", "./static/robot.png",), height=600) # height=800,
        
        with gr.Column(scale=1):
            msg = gr.Textbox(show_label=False,placeholder="请输入异常描述...",container=False)
            audi = gr.Audio(sources=["microphone"],format="wav")

            with gr.Row():  # 这个row没有作用？
                submit = gr.Button(value="提交",icon="./static/submit.png")
                clear = gr.ClearButton([msg, chatbot],value="清除",icon="./static/clear.png")
                new_user = gr.Button(value="新建用户",icon="./static/人.png")

    # 槽函数          
    def recognition(audio,chat_history):
        sr, y = audio
        print(sr)

        audio_signal = y
        save_name = os.path.join("./audio",str(uuid.uuid1()).replace("-","_")+".wav")
        wavfile.write(save_name, sr, audio_signal)

        # 重采样音频
        if sr != 16000:
            rc_sig,sr = sf.read(save_name)
            audio_signal_16k = librosa.resample(rc_sig,sr,16000)
            sf.write(save_name,audio_signal_16k,16000)

        # loop =  asyncio.get_event_loop()
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        task = new_loop.create_task(ws_client(0,save_name))
        new_loop.run_until_complete(task)
        new_loop.close()
        # print(preds)
        preds = task.result()[1]

        # bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
        bot_message = get_robot(user_id,preds)
        chat_history.append((preds, bot_message))
        return chat_history

    def respond(message, chat_history):
        # bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
        bot_message = get_robot(user_id,message)
        chat_history.append((message, bot_message))
        # time.sleep(1)
        return "", chat_history
    
    def create_use():
        user_id = str(uuid.uuid1()).replace("-","_")
        gr.Info(f"用户切换为： {user_id}")

        
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    # audi.submit(recognition, [msg, chatbot], [chatbot])
    submit.click(fn=recognition, inputs=[audi,chatbot], outputs=[chatbot])
    new_user.click(fn=create_use)



if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",server_port=8089)

```

下一步我们将对基于FunASR的Paraformer语音识别加热词增强加语言模型的方式进行更广泛和细致的测试！