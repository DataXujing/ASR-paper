## Tandem

<!-- 该部分内容来与原 <<语音识别实践>> p144-p149-->

<div align=center>
    <img src="zh-cn/img/ch3/p1.png" /> 
</div>

我们将在下一章介绍（DNN-HMM)的混合模型，深度神经网络同时学习了非线性的特征变换和对数线性分类器。更重要的是，通过神经网络学习到的特征表示比原始特征在说话人和环境变量方面更加稳健。一个很自然的想法是将神经网络的隐层和输出层视为更好的特征，并且将他们用于传统的GMM-HMM系统中。

### 1.使用Tandem和瓶颈层特征的GMM-HMM

在使用浅层的多层感知机时期，文献`[1]`中提出了Tandem的方法，这是最早的将隐藏层和输出层视为更好的特征的方法。Tandem方法通过使用从一个或者多个神经网络中衍生出的特征来扩展GMM-HMM系统中的输入向量。因为神经网络输出层的维度和训练的维度是一样的，Tandem特征通常以单音素分布为训练目标以控制所增加的特征的维度。

另外，文献`[2,3]`提出了使用瓶颈隐藏层（隐藏层节点个数比其他隐层的少）的输出作为特征的方法来代替直接使用神经网络的输出。因为隐藏层大小的选择是独立于输出层大小的。这个方法提供了训练目标维度和扩展的特征维度之间的灵活性。瓶颈层在网络中建立了一个限制，将用于分类的相关信息压缩为一个低维度的表示。注意在自编码器中也同样可以使用瓶颈层。因为在瓶颈层的激活函数是一个关于输入特征的低维的非线性函数，所以一个自编码器也可以视为一种非线性的维度下降的方法。然而，因为从自编码器中学习到的平静特征对识别任务没有针对性，这些特征通常不如从那些用于进行识别的神经网络中的瓶颈层提取出的特征有区分性。

文献`[4,5,6]`中使用了类似的方法将神经网络特征用于大词汇语音识别任务中，包括使用神经网络的输出层或者更早的隐层来扩展GMM-HMM系统中的特征。更新的工作中，深度神经网络代替了浅层的多层感知机来提取更稳健的特征。这些深度神经网络的识别目标通常采用聚类后的状态来替代单音素。基于这个原因，通常使用隐层特征，而不是输出层特征用于后续的GMM-HMM系统。

图1展示了DNN中典型的用于提取特征的隐层。图1a展现了一个所有的隐层都拥有相同的隐层节点的DNN,最后一个隐层被用于提取深度特征。这个特征通常会链接上MFCC或PLP等原始特征。然而在这样一个结构中，生成出的特征维度通常非常高。为了使其易于管理，我们可以使用主成分分析PCA来减少特征的维度。另一种方法是我们可以直接减少最后一个隐层的大小，将其改造为一个瓶颈层，如图1b所示。因为所有的隐层都可以被视为原始特征的一种非线性变换。我们可以使用任意瓶颈层的输出作为GMM的特征，如图1c所示。

<div align=center>
    <img src="zh-cn/img/ch3/p2.png" /> 
</div>

<p align=center>图1:使用DNN作为GMM-HMM系统的特征提取器。带阴影的层的输出将作为后续的GMM-HMM系统的特征</p>
  
因为隐层提取的特征之后将用GMM来建模，我们应该仅仅使用激励值(在进行非线性的激活函数之前的输出)，而不是经过激活函数之后的输出值来作为特征。特别是使用sigmoid非线性函数时就更是如此，因为sigmoid函数的输出值域是$[0,1]$,并且主要集中于0和1两个极值处。更需要考虑的是，即使我们使用瓶颈层来提取特征，瓶颈层特征的维度依然很大，而且各个维度之间是相关的。出于这些原因，再将特征运用于GMM-HMM系统之前先使用PCA或者HLDA处理一下会很有帮助，如图2所示

<div align=center>
    <img src="zh-cn/img/ch3/p3.png" /> 
</div>

<p align=center>图2:在GMM-HMM系统中使用Tandem(或者瓶颈层)特征。DNN用于提取Tandem或者瓶颈层特征，然后拼接上原始特征。合并后的特征在使用GMM-HMM进行建模前通过使用PCA或者HLDA进行压缩降维和去相关</p>

注意到因为Tandem或瓶颈层特征与GMM-HMM系统的训练是独立的，所以很难知道哪一层隐层可以提取更好的特征。同样，添加更多的隐层对性能是否有帮助也很难得知。比如文献`[7]`(表1)展示了在声音搜索数据集中一个拥有5个隐层的深度神经网络比拥有3个或者7个隐层的神经网络性能更好。表1同样指出了使用生成性的训练在同一个数据集上对识别效果有帮助。在试验中他们使用了39维的MFCC特征，前后连续5帧共11帧作为DNN的输入，瓶颈层拥有39个神经元，非瓶颈层拥有2048个，在训练时的学习率是$1.5e^{-5}$。在精细调整训练的前6轮学习率是$3e^{-4}$，后6轮是$8e^{-6}$。batch size是256，采用随机梯度下降进行训练。提取到的瓶颈层特征随后直接使用或链接上原始的MFCC特征去训练GMM-HMM。无论是直接使用瓶颈层特征还是链接上原始的MFCC特征，这些特征都通过PCA来去相关，并且降维到39维。

<!-- div css style 内容居中 -->
<style>
.center_me 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>


<div class="center_me">

|   隐层个数   |   3   |    5  |  7    |
| :----: | ---- | ---- | ---- |
|  不使用DBN预训练    |   41.1%   |   34.3%   |  36.1%    |
|    使用DBN预训练  |    34.3%  |    33.4%  |    34.1%  |


</div>
<p align="center">表1：使用不同深度的DNN来提取瓶颈层特征的开发集句错率(SER)比较(来自Yu和Seltzer 文献[7]) </p>

文献`[7]`还揭示出，使用聚类后的状态作为识别目标相比使用单因素或者使用无监督的方法性能更好。如表2所示，无监督的瓶颈层特征提取和使用单音素或者聚类后的状态作为训练目标提取瓶颈特征的性能相差巨大。这清晰的显示出，使用任务相关的信息用于训练特征是非常重要的。

<div class="center_me">

|   瓶颈特征训练的标注   |   开发集句错率   |  测试集句错率    |
| :--: | :----: | :----: |
|   无   |  39.4%    |   42.1%   |
|   单音素状态   |   35.2%   |  37.0%    |
|   从聚类后的状态转换过来的单音素状态   |   34.0%   |   35.7%   |
|    聚类后的状态(senones)  |  33.4%    |   34.8%   |

</div>
<p align="center">表2：在不同的监督标注情况下的句错率(SER)比较，所有的情况都进行了DBN预训练(文献[7]) </p>


### 2.DNN-HMM混合系统和采用深度特征的GMM-HMM系统的比较

!> 关于DNN-HMM的混合系统我们将在下一章介绍 [#4 DNN-HMM Hybrid](zh-cn/04_DNN-HMM-Hybrid)

DNN-HMM混合系统与采用深度特征（即从DNN中提取特征）的GMM-HMM系统最主要的区别是分类器的使用。在Tandem或瓶颈层特征系统中，GMM被用于代替对数线性模型（深度神经网络中的softmax层）。当使用同样的特征时，GMM拥有比对数线性模型更好的建模能力。实际上，在Heigold等的文章`[8]`里指出，在对数线性模型中使用使用1阶和2阶特征的时候有，GMM和对数线性模型是等价的。其结果也说明了GMM可以被一个拥有非常宽的隐层，同时隐层和输出层连接很稀疏的单隐层神经网络建模。从另一个角度说，因为隐层和输出层的对数线性分类器的训练是同时优化的，在DNN-HMM混合系统中的隐层特征与分类器的匹配会比Tandem和瓶颈层特征更好。这两个原因相互抵消，最后的结果是这两个系统的性能几乎是相等的。然而在实际中CD-DNN-HMM系统运用起来更简单。

在使用GMM-HMM系统中深度神经网络提取的特征的主要好处是可以使用现存的已经能很好的训练和自适应GMM-HMM的系统工具。同样，也可以使用训练数据的一个子集去训练提取特征的深度神经网络，然后使用所有的数据提取深度神经网络特征用于训练GMM-HMM系统。

在文献`[9]`中，Yan等系统的通过实验比较了CD-DNN-HMM系统和使用DNN提取特征的GMM-HMM系统。在论文中，他们使用了最后一个隐层的激励作为DNN提取的特征，如图1a所示。然后提取的特征通过PCA压缩并且链接上原始的谱特征。扩展后的特征继续使用HLDA压缩`[10]`,使得最后的维度以适合GMM-HMM。实验的数据是Switchboard(SWB),这里把使用深度特征的GMM-HMM系统称为DNN-GMM-HMM,在试验中，解码和基于词网格的序列级训练时使用的声学模型的缩放系数为0.5（即简单的把声学的对数似然乘以0.5）这个设置能得到最好的识别正确率。试验中使用和其他工作(`[11,12,13]`)相同的PLP特征以及训练和测试配置以进行结果比较。

表3总结于文献`[13,9]`,它比较了CD-DNN-HMM和Tandem。在paper中，他们使用了309个小时的SWB数据进行训练，在SWB Hub5'00测试集合进行性能验证。可以观察到，虽然区域相关的线性变换(region dependent linear transformation,RDLT)`[14,15]`将性能从17.8%改善到16.1%。使用MMI训练的DNN-GMM-HMM依然比同样使用MMI训练的CD-DNN-HMM要差。

<div align=center>
    <img src="zh-cn/img/ch3/p4.png" /> 
</div>

<p align=center>表3:SWB Hub5‘00测试集上的词错率使用309小时训数据。DNN拥有7个隐层，每个隐层约2000个神经元，输出层有约9300个聚类后的状态([9,13])</p>

表4比较了使用2000小时训练数据的时候CD-DNN-HMM和DNN-GMM-HMM的性能。可以观察到使用了RDLT和MMI的DNN-GMM-HMM性能比使用MMI训练的CD-DNN-HMM略好。综合这两个表，我们可以观察到，DNN-GMM-HMM相比其提升的复杂度，性能上的提升并不明显。

<div align=center>
    <img src="zh-cn/img/ch3/p5.png" /> 
</div>

<p align=center>表4:使用2000小时数据训练的模型SWB Hub5’00测试集上的词错误率。DNN拥有7层隐层，每个隐层约2000个神经元，输出层有约18000个聚类后的状态([9])</p>

在前面的讨论中，DNN的衍生特征都来源于隐层，在文献`[16]`中，Sainath等探索了一个不那么直接的方法。在其设置中，DNN拥有6个隐层，每个隐层1024个神经元，输出层是384个HMM的状态，与文献`[9]`中相同的是DNN没有瓶颈层，所以他能比使用瓶颈层的DNN更好的对HMM状态进行分类。不同于文献`[9]`，他们使用了输出层的激励（softmax调用前的输出），而不是最后一个隐层作为特征。384维的激励值随后通过`384-128-40-384`的自编码器被压缩到40维。由于瓶颈层出现在自编码器中，而不是深度神经网络中，这个网络被称为自编码器瓶颈网络(AE-BN)

他们在英语广播新闻任务上（English broadcast news)比较了（表5）使用和不使用AE-BN特征的GMM-HMM系统。这个数据集拥有430小时的训练数据。从表5可以观察到，在使用相同的训练方法的情况下，特征空间说话人自适应（FSA）,特征空间增强型MMI（fBMMI），模型级增强型MMI(BMMI)(`[17]`)以及最大似然回归自适应（MLLR）（`[18]`）等系统中，使用AE-BN特征的GMM-HMM系统和一般的CD-DNN-HMM。通过比较文献`[16]`和`[19]`中的结果，我们可以观察到，在使用同样的训练准则时，CD-DNN-HMM比AE-BN系统性能略好。


<div class="center_me">

|  训练方法    |  基线GMM-HMM    |  采用AE-BN特征的GMM-HMM    |
| :--: | :----: | :----: |
|   FSA   |  20.2%    |  17.6%    |
|   +fBMMI   |  17.7%    |  16.6%    |
|   +BMMI   |   16.5%   |   15.8%   |
|   +MLLR   |   16.0%   |    15.5%  |


</div>
<p align="center">表5：比较AE-BN系统和GMM-HMM系统。使用403小时训练数据，在英文广播新闻数据集上的词错率[16] </p>



### 3.Reference

[1] Hermansky,H.,Ellis,D.P.,Sharma,S. Tandem connectionist feature extraction for conventional HMM systems.ICASSP, vol.3,pp.1635-1638(2000).

[2] Grezl,F., Fousek,P. Optimizing bootle-neck features for LVCSR. ICASSP, Vancouver,Canada(2013). 

[3] Grezl,F.,Karafiat,M., Probabilistic and bottle-neck features for LVCSR of meetings. ICASSP, pp.757-760(2007). 

[4] Fousek,P.,Lamel,L. et al. Transcribing broadcast data using MLP features. INTERSPEECH, pp.1433-1436(2008).

[5] Valente,F.,Doss, M.M. et al. A comparative large scale study of MLP features for mandarin ASR. INTERSPEECH, pp.2630-2633(2010). 

[6] Vergyri,D.,Mandal,A. et al. Development of the SRI/nightingale Arabic ASR system. INTERSPEECH, pp.1437-1440(2008). 

[7] Yu, D.,Seltzer, M.L. et al Improved bottleneck features using pretrained deep neural networks. INTERSPEECH, pp.237-240(2011).

[8] Heigold,G., et al. Equivalence of generative and log-linear models.IEEE 19(5),1138-1148(2011). 

[9] Yan,J., et al. A scalable approach to using DnNN-derived features in GMM-HMM based acoustic modeling fir LVCSR. INTERSPEECH,(2013).

[10] Kumar,N., et al. Heteroscedastic discriminant analysis and reduced rank HMMs for improved speech recognition. Speech Communication 26(4),283-297(1998).

[11] Seide,F., et al. Feature engineering in context-dependent deep neural networks for conversational speech transcription. ASRU,pp.24-29(2011). 

[12] Seide,F., et al. Conversational speech transcription usng context-dependent deep neural networks. INTERSPEECH,pp. 437-440(2011). 

[13] Su,H., et al. Error back propagation for sequence training of context-dependent deep networks for conversational speech transcription. ICASSP, (2013). 

[14] Yan, Z.J. et al. Tied-state based discriminative training of context-expanded region-dependent feature transforms for LVCSR. ICASSP, pp.6940-6944(2013). 

[15] Zhang, B. et al.  Discriminatively trained region dependent feature transform for speech recognition. ICASSP, vol.1, pp.I-I(2006). 

[16] Sainath,T.N. et al. Auto-encoder bottle-neck features using deep belief networks. ICASSP, pp.4153-4156(2012).

[17] Povey,D. et al. Boosted MMI for model and feature-space discriminative training. ICASSP,pp.4057-4060(2008).

[18] Gales,M.J. et al. Mean and variance adaptation within the mllr framework.Computer Speech and Language, 10(4),249-264(1996).

[19] Sainath, T.N. et al. Making deep belief networks effective for large vocabulary continuous speech recognition. IEEE(ASRU), pp.30-35(2011).

!> 我们可以看到这些参考文献大部分都是十几年前的paper.



