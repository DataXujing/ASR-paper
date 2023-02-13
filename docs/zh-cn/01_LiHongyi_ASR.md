## 李宏毅语音识别课程学习笔记
------

!> [b站视频](https://www.bilibili.com/video/BV1we411L7yw?p=7&vd_source=def8c63d9c5f9bf987870bf827bfcb3d)

!> [李宏毅课程资料下载](http://speech.ee.ntu.edu.tw/~tlkagk/courses_DLHLP20.html)

!> [李宏毅个人主页](https://speech.ee.ntu.edu.tw/~hylee/index.php)

<!-- !> [https://www.bilibili.com/read/cv18472465](https://www.bilibili.com/read/cv18472465) -->


<!-- <div align=center>
    <img src="zh-cn/img/word2vec/p8.png" /> 
</div> -->

### 1. Class 1: OverView

1.50年前科学家们对ASR的认知：犹如将水变成汽油，在海洋中提取黄金，治愈癌症，登陆月球，言外之意ASR在那个年代的科学家的认知中是不可完成的事情。

<div align=center>
    <img src="zh-cn/img/ch1/class1/p1.png" /> 
</div>

2.语音识别在解决什么事情？

<div align=center>
    <img src="zh-cn/img/ch1/class1/p2.png" /> 
</div>


输入1段语音，这段语音往往被量化为$[T,d]$维的数据，$T$是长度，$d$是每一个时间点的编码的向量的维度。输出是1段文本，$N$是文本的长度，$V$是Token词表的大小，往往$T>N$.

3.语音识别的Token的定义

<div align=center>
    <img src="zh-cn/img/ch1/class1/p3.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/ch1/class1/p4.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/ch1/class1/p4-1.png" /> 
</div>

+ Phoneme（音素）发音的基本单位，类似于音标比音标更小（1个音标可能有若干音素构成），如果想要换算成文本，需要有个Lexicon(词典表)，例如cat ⟶ K AE T
+ Grapheme（字母）书写文本的最小单元对于中文来说是`字`对于英文来说是`字母`,当然也包含一些特殊的符号比如英文中的空格和其他标点符号等，这种方式的好处是不需要Lexicon
+ Word（词）英文单词或中文中的词语，这种方式的缺点是往往词表比较大，比如英文常用词组数量>100K
+ Morpheme（词元） 介于词和字之间的词元，比如英文中的词根，词缀。这些词元是由语言学家或统计学的方式得到的
+ Bytes(二进制的编码) 常用编码比如UTF-8编码，好处是V是固定大小比如256，并且相同的方式表示符号和不同的语言

4.看一个Token为Word的缺点

<div align=center>
    <img src="zh-cn/img/ch1/class1/p5.png" /> 
</div>

Word不是适合所有的语种的，比如土耳其语中，相同读音意义接近的词在词表中表示的词不同会导致词表数量非常巨大。


5.统计2019年100篇paper以上得到的使用的Token的占比情况

<div align=center>
    <img src="zh-cn/img/ch1/class1/p6.png" /> 
</div>

+ ASR中三大顶会：INTERSPEECH，ICASSP, ASRU的超过100篇paper的统计结果
+ 使用`字`作为Token还是最常见的方式
+ 其次是使用`音素`作为Token
+ 使用`词`作为Token占比最少


6.ASR中的端到端训练任务

<div align=center>
    <img src="zh-cn/img/ch1/class1/p7.png" /> 
</div>

可以联合训练ASR,比如由语音直接到词向量，由语音直接到翻译的结果，由语音直接到意图识别和槽填充这在QA中用的比较多。

7.声学特征

<div align=center>
    <img src="zh-cn/img/ch1/class1/p8.png" /> 
</div>

+ frame是指包含$N$个采样点的小片段，一般控制在$25ms-35ms$ (红色的也叫window)
+ 以16KHz的采样率来说，每毫秒包含16个采样点(每个采样点就是1个值)
+ $25ms$中的时间片段中，包含400个sample point
+ frame移动的step为$10ms$,所以$1s$的语音片段，会有100个frames
+ 也可以将该frame转换为39维的MFCC或80维的fbank

8.常用的声学特征

<div align=center>
    <img src="zh-cn/img/ch1/class1/p9.png" /> 
</div>

+ 原始音频通过离散傅里叶变换（DFT）得到语谱图（横坐标是时间，纵坐标是频率，颜色深浅代表能量）
+ 进过专家提供的滤波器得到fbank（一般滤波器是专家基于人体声学设计的）
+ 进一步的对数变换，离散余弦变换（DCT）得到MFCC

!> 关于音频特征提取在此做相应的补充!

**1.声音特性**

+ 声音（sound)是由物体振动产生的声波。是通过介质传播并能被人或动物听觉器官所感知的波动现象。最初发出振动的物体叫声源。声音以波的形式振动传播。声音是声波通过任何介质传播形成的运动。
+ 频率：是每秒经过一给定点的声波数量，它的测量单位为赫兹，1千赫或1000赫表示每秒经过一给定点的声波有1000个周期，1兆赫就是每秒钟有1,000,000个周期，等等。
+ 音节：就是听觉能够自然察觉到的最小语音单位，音节有声母、韵母、声调三部分组成。一个汉字的读音就是一个音节，一个英文单词可能有一个或多个音节构成，并且按照音节的不同，可以分为不同的种类。
+ 音素：它是从音节中分析出来的最小语音单位，语音分析到音素就不能再分了。比如，“她穿红衣服”是5个音节，而“红”又可进一步分为3个音素–h,o,ng。音素的分析需要一定的语音知识，但是，如果我们读的慢一点是还可以体会到的。
+ 音位：是指能够区分意义的音素，比如bian,pian,bu,pu就是靠b，p两个音素来区分的，所以b，p就是两个音位。 人耳能听到的音频范围：20HZ–20KHZ。人说话的声音频率：300HZ–3.4KHZ。乐器的音频范围：20HZ–20KHZ。


**2.时域图、频谱图、语谱图（时频谱图）**

<!-- https://blog.csdn.net/Robin_Pi/article/details/109204672 -->

2.1 概述

（1）什么是信号的时域和频域？

时域和频域是信号的基本性质，用来分析信号的 不同角度 称为 域 ，一般来说，时域的表示较为形象与直观，频域分析则更为简练，剖析问题更为深刻和方便。目前，信号分析的趋势是从时域向频域发展。然而，它们是互相联系，缺一不可，相辅相成的。

（2）时频域的关系是什么？

时域分析与频域分析是对模拟信号的两个观察面。对信号进行时域分析时，有时一些信号的时域参数相同，但并不能说明信号就完全相同。因为信号不仅随时间变化，还与频率、相位等信息有关，这就需要进一步分析信号的频率结构，并在频率域中对信号进行描述。动态信号从时间域变换到频率域主要通过傅立叶级数和傅立叶变换实现。周期信号的变换采用傅立叶级数，非周期信号的变换采用傅立叶变换。

（3）信号的时域和频域表达方式各有什么特点？

我们描述信号的方式有时域和频域两种方式，时域是描述数学函数或物理信号对时间的关系，而频域是描述信号在频率方面特性时用到的一种坐标系，简单来说，横坐标一个是时间，一个是频率。
一般正弦信号可由幅值、频率、相位三个基本特征值就可以唯一确定。但对于两个形状相似的非正弦波形，从时域角度，很难看出两个信号之间的本质区别，这就需要用到频域表达方式。

+ 时域：自变量是时间，即横轴是时间，纵轴是信号的变化（振幅）。其动态信号$x(t)$是描述信号在不同时刻取值的函数.
+ 频域：自变量是频率，即横轴是频率，纵轴是该频率信号的幅度（振幅），就是指的信号电压大小，也就是通常说的频谱图.

2.2 （时域）波形和频域：用几张对比图来区分

(1) 时域和频域

时域vs频域：

<div align=center>
    <img src="zh-cn/img/ch1/class1/p10.gif" /> 
</div>

时域波形、频域谱线：

<div align=center>
    <img src="zh-cn/img/ch1/class1/p11.png" /> 
</div>

+ 时域图：表现的是一段音频在一段时间内音量的变化，其横轴是时间方向，纵轴是振幅方向。波形实质上是将各个频率的波形叠加在了一起（波形是由各频率不同幅值和相位的简单正弦波复合叠加得到的。）
+ 频谱图：表现的是一段音频在某一时刻各个频率的音量的高低，其横轴是频率方向，纵轴为振幅方向。
+ 将复合波形进行傅里叶变换，拆解还原成每个频率上单一的正弦波构成，相当于把二维的波形图往纸面方向拉伸，变成了三维的立体模型，而拉伸方向上的那根轴叫频率，现在从小到大每个频率点上都对应着一条不同幅值和相位的正弦波。
+ 频谱则是在这个立体模型的频率轴方向上进行切片，丢去时间轴（即在每个时刻都可以拿刀在与时间轴垂直的方向上进行切片），形成以横坐标为频率，纵坐标为幅值的频谱图，表示的是一个静态的时间点上各频率正弦波的幅值大小的分布状况。
再说的直白一点，频谱就是为了找出一个波是由多少波复合而成的！

从下面的频谱图中可以得出这样的结论:

+ 原始波由三个正弦波叠加而成
+ 横轴为这些正弦波分量的频率，纵轴为这些正弦波分量的振幅

!> 关于为什么是正弦波?

<!-- https://blog.csdn.net/Robin_Pi/article/details/103699019 -->
时域分析与频域分析是对模拟信号的两个观察面。时域分析是以时间轴为坐标表示动态信号的关系；频域分析是把信号变为以频率轴为坐标表示出来。

<div align=center>
    <img src="zh-cn/img/ch1/class1/p12.png" /> 
</div>

动态信号从时间域变换到频率域主要通过傅立叶级数和傅立叶变换实现。周期信号靠傅立叶级数，非周期信号靠傅立叶变换。时域越宽，频域越短。

正弦波是频域中唯一存在的波形，这是频域中最重要的规则，即正弦波是对频域的描述，因为频域中的任何波形都可用正弦波合成。 这是正弦波的一个非常重要的性质。正弦波有四个性质使它可以有效地描述其他任一波形：

+ 频域中的任何波形都可以由正弦波的组合完全且惟一地描述。
+ 任何两个频率不同的正弦波都是正交的。如果将两个正弦波相乘并在整个时间轴上求积分，则积分值为零。这说明可以将不同的频率分量相互分离开。
+ 正弦波有精确的数学定义。
+ 正弦波及其微分值处处存在，没有上下边界。

**“任何”周期信号都可以用一系列成谐波关系的正弦曲线来表示。(狄里赫利条件)**

(2) 区分：时频谱图（语谱图）

语谱图：先将语音信号作傅里叶变换，然后以横轴为时间，纵轴为频率，用颜色表示幅值即可绘制出语谱图。在一幅图中表示信号的频率、幅度随时间的变化，故也称“时频图”。

<div align=center>
    <img src="zh-cn/img/ch1/class1/p13.png" /> 
</div>

如下面两张图分别为数字0-10的波形图和语谱图：

+ 数字0-10的波形图:

<div align=center>
    <img src="zh-cn/img/ch1/class1/p14.png" /> 
</div>

+ 数字0-10的语谱图:

<div align=center>
    <img src="zh-cn/img/ch1/class1/p15.png" /> 
</div>

```
附：

频宽、带宽、频带？
频带（frequency band）：对信号而言，频带就是信号包含的最高频率与最低频率这之间的频率范围(当然频率分量必须大于一定的值)。对信道而言，频带就是允许传送的信号的最高频率与允许传送的信号的最低频率这之间的频率范围(当然要考虑衰减必须在一定范围内)
频带宽度（band width）：简称带宽，有时称必要宽度，指为保证某种发射信息的速率和质量所需占用的频带宽度容许值，以赫（Hz）、千赫（KHz）、兆赫（MHz）表示。
注意区分：网络带宽，是指在单位时间能传输的数据量，亦即数据传输率。
宽带和窄带？
“窄”和“宽”是一个相对概念，并无严格数字界限，相对于什么呢？是指信道特性相对于信号特性。第一，什么叫宽带信号，“有待传输的信号”我们称为信源，信源是具备一定的频谱特征的。信源信号通常需要一个载波信号来调制它，才能发送到远方。信源信号带宽远小于载波中心频率的是窄带信号,反之，二者大小可比拟的称为宽带信号。
第二，实际通信中，分配给你的频带资源+真实的传播环境, 我们称之为信道。信道也具备一定的频谱特征。通常情况下，分配到的频带资源越宽，传播环境越稳定，信道能够承载的数据速率就越高。
```


**3.语音的时域特性和频域特性**

<!-- https://blog.csdn.net/weixin_44885180/article/details/122912224?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166556025816800192225858%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166556025816800192225858&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~baidu_landing_v2~default-12-122912224-null-null.nonecase&utm_term=%E8%AF%AD%E9%9F%B3%E8%AF%86%E5%88%AB&spm=1018.2226.3001.4450 -->


3.1 语音时域特性

语音信号有时变特性，是一个非平稳的随机过程。但在一个短时间范围内其特性基本 保持不变，即语音的“短时平稳性”。

在时域，语音信号可以直接用它的时间波形表示出来。其中，清音段类似于白噪声，具有较高的频率，但振幅很小，没有明显的周期性；而浊音都具有明显的周期性，且幅值较大，频率相对较低。语音信号的这些时域特征可以通过短时能量、短时过零率等方法来分析。

**短时能量**

由于语音信号的能量随时间而变化，清音和浊音之间的能量差别相当显著。因此，对短时能量和短时平均幅度进行分析，可以描述语音的这种特征变化情况。

定义$n$时刻某语音信号的短时平均能量为：

$$E_n=\sum^{+ \infty}_{m=- \infty}[x(m)w(n-m)]^2= \sum ^{n} _{m=n-(N-1)}[x(m)w(n-m)]^2$$

式中，$N$为窗长，可见短时能量为一帧样点值的加权平方和，特殊的，当窗函数为矩形窗时，有：

$$E_n=\sum^{n}_{m=n-(N-1)}x^2(m)$$


**短时振幅**

短时能量的一个主要问题是对信号电平值过于敏感。由于需要计算信号样值的平方和，在定点实现时很容易产生溢出。为了克服这个缺点，可以定义一个短时平均幅度函数来衡量语音幅度的变化：

$$E_n=\sum^{+ \infty}_{m=- \infty}|x(m)|w(n-m)= \sum ^{n} _{m=n-(N-1)}|x(m)|w(n-m)$$

上式可以理解为$w(n)$对$|x(n)|$的线性滤波运算。与短时能量比较，短时平均幅度相当于用绝对值之后代替了平方和，简化了运算。

**短时过零率**

短时平均过零率是语音信号时域分析中的一种特征参数。它是指每帧内信号通过零值的次数。

①对有时间横轴的连续语音信号，可以观察到语音的时域波形通过横轴的情况。

②在离散时间语音信号情况下，如果相邻的采样具有不同的代数符号就称为发生了过零，因此可以计算过零的次数。

单位时间内过零的次数就称为过零率。一段长时间内的过零率称为平均过零率。如果是正弦信号，其平均过零率就是信号频率的两倍除以采样频率，而采样频率是固定的。因此过零率在一定程度上可以反映信号的频率信息。短时平均过零率的定义为：

<div align=center>
    <img src="zh-cn/img/ch1/class1/p16.png" width=30% /> 
</div>

<div align=center>
    <img src="zh-cn/img/ch1/class1/p17.png" width=30%/> 
</div>


3.2 语音频域特性

计算信号能量（作用在单位电阻上的电压信号 释放的能量）可以将信号分为：

+ 功率信号：能量无限，不能用能量表示，所以用平均功率表示；
+ 能量信号：能量有限，平均功率为0；

**频谱**

功率信号的频谱（离散）:

<div align=center>
    <img src="zh-cn/img/ch1/class1/p18.png" width=30%/> 
</div>

含义： 周期功率信号幅值（频率为$f0$）经过傅里叶级数展开，被多个离散倍频$nf0$表征，各频点的幅值$C(nf0)$也即该频点的贡献权系数。

**功率谱密度**

功率信号的功率谱密度（连续）:

<div align=center>
    <img src="zh-cn/img/ch1/class1/p19.png" width=30%/> 
</div>

含义：

+ 将信号的功率按照频点贡献铺在频谱之上；
+ 因其能量是无穷的，所以不能把能量铺上去，只能用有限的功率；
+ 对功率谱密度进行积分，能得到局部频段承载的功率；
+ 相比功率信号的频谱突出各频点对功率信号的信号幅值的贡献，功率谱密度突出各频点对功率信号的功率的贡献。

**频谱密度**

能量信号的频谱密度（连续）：

<div align=center>
    <img src="zh-cn/img/ch1/class1/p20.png" width=30%/> 
</div>

含义：

+ 通过傅里叶变换将能量信号转换到连续频域上；
+ 但因能量有限，不能使用离散贡献频点权系数（几乎为0），只能使用频谱密度来表征。

**能量谱密度**

能量信号的能量谱密度（连续）：

<div align=center>
    <img src="zh-cn/img/ch1/class1/p21.png" width=20%/> 
</div>

含义：

+ 将信号能量铺在频谱之上；
+ 对能量谱密度进行局部积分，能得到局部频段承载的能量；
+ 相比能量信号的频谱密度突出连续频点对功率信号的信号幅值的贡献，能量谱密度突出连续频点对能量信号的能量的贡献。


**4.Mel频率倒谱系数（MFCC）**

<!-- http://www.javashuo.com/article/p-ktkuqtwf-ny.html

http://fancyerii.github.io/books/mfcc/

https://zhuanlan.zhihu.com/p/181718235

https://blog.csdn.net/qq_36002089/article/details/120014722

https://zhuanlan.zhihu.com/p/493160516

https://blog.csdn.net/zouxy09/article/details/9156785/

http://pract icalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/ -->


<!-- https://blog.csdn.net/qq_36002089/article/details/120014722

https://blog.csdn.net/u012223913/article/details/77449201 -->

在语音识别（Speech Recognition）和说话者识别（Speaker Recognition）中，最常用到的语音特征就是梅尔倒谱系数（Mel-scale Frequency Cepstral Coefficients，简称MFCC）。根据人耳听觉机理的研究发现，人耳对不同频率的声波有不同的听觉敏感度。从$200Hz$到$5000Hz$的语音信号对语音的清晰度影响对大。两个响度不等的声音作用于人耳时，则响度较高的频率成分的存在会影响到对响度较低的频率成分的感受，使其变得不易察觉，这种现象称为掩蔽效应。由于频率较低的声音在内耳蜗基底膜上行波传递的距离大于频率较高的声音，故一般来说，低音容易掩蔽高音，而高音掩蔽低音较困难。在低频处的声音掩蔽的临界带宽较高频要小。所以，人们从低频到高频这一段频带内按临界带宽的大小由密到疏安排一组带通滤波器，对输入信号进行滤波。将每个带通滤波器输出的信号能量作为信号的基本特征，对此特征经过进一步处理后就可以作为语音的输入特征。由于这种特征不依赖于信号的性质，对输入信号不做任何的假设和限制，又利用了听觉模型的研究成果。因此，这种参数比基于声道模型的LPCC相比具有更好的鲁邦性，更符合人耳的听觉特性，而且当信噪比降低时仍然具有较好的识别性能.

美尔尺度是建立从人类的听觉感知的频率——Pitch到声音实际频率直接的映射。人耳对于低频声音的分辨率要高于高频的声音。通过把频率转换成美尔尺度，我们的特征能够更好的匹配人类的听觉感知效果。从频率到美尔频率的转换公式如下：

$$M(f)=1125\ln(1+f/700)$$

而从美尔频率到频率的转换公式为：

$$M^{−1}(m)=700(e^{m/1125−1})$$

流程图：

<div align=center>
    <img src="zh-cn/img/ch1/class1/p22.png" width=50%/> 
</div>

4.1预加重(Pre-Emphasis)


预加重的目的是提升高频部分，使信号的频谱变得平坦，保持在低频到高频的整个频带中，能用同样的信噪比求频谱。同时，也是为了消除发声过程中声带和嘴唇的效应，来补偿语音信号受到发音系统所抑制的高频部分，也为了突出高频的共振峰。预加重处理其实是将语音信号通过一个高通滤波器：

共振峰：指在声音的频谱中能量相对集中的一些区域。

$$H(z) = 1-\mu z^{-1}$$

时域表示形式：
$$y(n)=x(n)-\alpha x(n-1)$$

一般$\alpha$取值为0.95/0.97.其作用：

+ 加强高频信息，因为一般高频能量比低频小
+ 避免FFT操作中的数值问题
+ 可能增大信噪比（Signal to Noise Ratio）

注意的是，现代的系统可以将这步用`mean normalization`代替

4.2分帧(framing)

由于语音信号的非平稳特性 和 短时平稳特性，将语音信号分分帧。
这里的帧（frame）代表一小段时间t的语音数据。帧由N个采样点组成。
我们要对语音数据做傅里叶变换，将信息从时域转化为频域。但是如果对整段语音做FFT，就会损失时序信息。因此，我们假设在很短的一段时间t内的频率信息不变，对长度为t的帧做傅里叶变换，就能得到对语音数据的频域和时域信息的适当表达。

一般来说，帧的长度取值区间在20ms到40ms之间，相邻帧有50%的重叠（overlapping）(为了避免相邻两帧的变化过大，因此会让两相邻帧之间有一段重叠区域）。常用的参数设置： 帧长25ms，步长（stride）10ms（15ms的重叠）。帧长（T），语音数据采样频率（F ）和帧的采样点（N）之间的关系:

$$T=\frac{N}{F}$$

如N的值为256或512，涵盖的时间约为20~30ms左右。为了避免相邻两帧的变化过大，平缓过度，因此会让两相邻帧之间有一段重叠区域，此重叠区域包含了M个取样点，通常M的值约为N的1/2或1/3。通常语音识别所采用语音信号的采样频率为8KHz或16KHz，以8KHz来说，若帧长度为256个采样点，则对应的时间长度是256/8000×1000=32ms。


4.3加窗（window）

将信号分帧后,我们将每一帧代入窗函数，窗外的值设定为0，其目的是消除各个帧两端可能会造成的信号不连续性（即谱泄露 spectral leakage）。常用的窗函数有矩形窗、汉明窗和汉宁窗，高斯窗等[https://zhuanlan.zhihu.com/p/24318554](https://zhuanlan.zhihu.com/p/24318554)

<div align=center>
    <img src="zh-cn/img/ch1/class1/p23.png" width=50%/> 
</div>

根据窗函数的频域特性，常采用汉明窗（hamming window）。公式如下：

$$w[n]=0.54−0.46cos(2πnN−1)$$

窗口长度为N，$0≤n≤N−1$, 该函数形状如下：

<div align=center>
    <img src="zh-cn/img/ch1/class1/p24.png" width=50%/> 
</div>


将每一帧乘以一个窗函数，如汉明窗，海宁窗。假设分帧后的信号为$S(n), n=0,1…,N-1$, N为帧的大小，那么乘上汉明窗$W(n)$：
$$S^{'}(n)=S(n) \times W(n)$$

4.4快速傅里叶变换（Fast-Fourier-Transform）

由于信号在时域上的变换通常很难看出信号的特性，所以通常将它转换为频域上的能量分布来观察，不同的能量分布，就能代表不同语音的特性。所以在乘上汉明窗后，每帧还必须再经过快速傅里叶变换以得到在频谱上的能量分布。对分帧加窗后的各帧信号进行快速傅里叶变换得到各帧的频谱。

<div align=center>
    <img src="zh-cn/img/ch1/class1/p25.png" width=25%/> 
</div>

式中$x(n)$为输入的语音信号（即为4.3中的$S^{'}(n)$，N表示傅里叶变换的点数。

谱线能量(功率谱)：对FFT后的频谱取模平方得到语音信号的谱线能量

$$P_i(k)=\frac{1}{N}|S_i(k)|^2$$

上式得到的是周期图的功率谱估计。通常我们会进行512点的DFT并且保留前257个系数。

4.5计算通过Mel滤波器的能量

将能量谱通过一组Mel尺度的三角形滤波器组，定义一个有M个滤波器的滤波器组（滤波器的个数和临界带的个数相近），采用的滤波器为三角滤波器，中心频率为$f(m)$ 。M通常取22-26。各$f(m)$之间的间隔随着$m$值的减小而缩小，随着$m$值的增大而增宽，如图所示：

<div align=center>
    <img src="zh-cn/img/ch1/class1/p26.png" width=50%/> 
</div>

三角滤波器的频率响应定义为:

<div align=center>
    <img src="zh-cn/img/ch1/class1/p27.png" width=50%/> 
</div>

三角带通滤波器有三个主要目的：

+ 三角形是低频密、高频疏的，这可以模仿人耳在低频处分辨率高的特性；
+ 对频谱进行平滑化，并消除谐波的作用，突显原先语音的共振峰。（因此一段语音的音调或音高，是不会呈现在 MFCC 参数内，换句话说，以 MFCC 为特征的语音辨识系统，并不会受到输入语音的音调不同而有所影响；在每个三角形内积分，就可以消除精细结构，只保留音色的信息）
+ 还可以减少数据量，降低运算量。

计算每个滤波器组输出的对数能量为：

<div align=center>
    <img src="zh-cn/img/ch1/class1/p28.png" width=30%/> 
</div>

!> Mel三角滤波器原理和设计


P1.为什么会产生出Mel 这种尺度的机制呢？

Mel就是单词Melody（调，旋律）的简写，是根据人耳的听觉感知刻画出来的频率。

+ 人耳朵具有特殊的功能，可以使得人耳朵在嘈杂的环境中，以及各种变异情况下仍能正常的分辨出各种语音
+ 其中，耳蜗有关键作用，耳蜗实质上的作用相当于一个滤波器组，耳蜗的滤波作用是在对数频率尺度上进行的，在1000HZ以下为线性尺度，1K HZ以上为对数尺度，使得人耳对低频信号敏感，高频信号不敏感。
例如，人们可以比较容易地发现500和1000Hz的区别，但很难发现7500和8000Hz的区别。

P2.Mel刻度定义:

<div align=center>
    <img src="zh-cn/img/ch1/class1/p29.png" width=20%/> 
</div>

有时，还会写成：

<div align=center>
    <img src="zh-cn/img/ch1/class1/p30.png" width=18%/> 
</div>

区别也看显然，一个是log以10为底，一个是ln。

Mel频率是Hz的非线性变换log，对于以Mel-Scale为单位的信号，可以做到人们对于相同频率差别的信号的感知能力几乎相同。

<div align=center>
    <img src="zh-cn/img/ch1/class1/p31.png" width=60%/> 
</div>

+ 当频率较小时，Mel随Hz变化较快(人耳对低频音调的感知较灵敏)；
+ 当频率很大时，Mel的上升很缓慢，曲线的斜率很小。

我们一般用三角滤波器来设计Mel滤波器组，当然也可以用其他的滤波器，三角是Mel设计时最常用的。

<div align=center>
    <img src="zh-cn/img/ch1/class1/p32.png" width=60%/> 
</div>

如下图所示，40个三角滤波器组成滤波器组，低频处滤波器密集，门限值大，高频处滤波器稀疏，门限值低。恰好对应了频率越高人耳越迟钝这一客观规律。该滤波器形式叫做等面积梅尔滤波器（Mel-filter bank with same bank area），在人声领域（语音识别，说话人辨认）等领域应用广泛。

<div align=center>
    <img src="zh-cn/img/ch1/class1/p33.png" width=60%/> 
</div>

但是如果用到非人声领域，常用的就是等高梅尔滤波器组（Mel-filter bank with same bank height），如下图：

<div align=center>
    <img src="zh-cn/img/ch1/class1/p34.png" width=60%/> 
</div>


4.5对数能量取log, 计算DCT倒谱

(1)对能量取log

对于滤波器组的能量，我们对它取log。这也是受人类听觉的启发：人类对于声音大小的感受不是线性的。为了使人感知的大小变成2倍，我们需要提高8倍的能量。这意味着如果声音原来足够响亮，那么再增加一些能量对于感知来说并没有明显区别。log这种压缩操作使得我们的特征更接近人类的听觉。为什么是log而不是立方根呢？因为log可以让我们使用倒谱均值减(cepstral mean subtraction)这种信道归一化技术（这可以归一化掉不同信道的差别）。

(2)DCT

经离散余弦变换（DCT）得到MFCC系数 :
<div align=center>
    <img src="zh-cn/img/ch1/class1/p35.png" width=30%/> 
</div>
将上述的对数能量带入离散余弦变换，求出L阶的Mel参数。L阶指MFCC系数阶数，通常取12-16。这里M是三角滤波器个数。

4.6Deltas和Delta-Deltas特征

Deltas和Delta-Deltas通常也叫(一阶)差分系数和二阶差分(加速度)系数。MFCC特征向量描述了一帧语音信号的功率谱的包络信息，但是语音识别也需要帧之间的动态变化信息，比如MFCC随时间的轨迹，实际证明把MFCC的轨迹变化加入后会提高识别的效果。因此我们可以用当前帧前后几帧的信息来计算Delta和Delta-Delta：

<div align=center>
    <img src="zh-cn/img/ch1/class1/p36.png" width=20%/> 
</div>

上式得到的$d_t$是Delta系数，计算第t帧的Delta需要t-N到t+N的系数，N通常是2。如果对Delta系数$d_t$再使用上述公式就可以得到Delta-Delta系数，这样我们就可以得到`3*12=36`维的特征。上面也提到过，我们通常把能量也加到12维的特征里，对能量也可以计算一阶和二阶差分，这样最终可以得到39维的MFCC特征向量。

因此，MFCC的全部组成其实是由： N维MFCC参数（N/3 MFCC系数+ N/3 一阶差分参数+ N/3 二阶差分参数）+帧能量（此项可根据需求替换）。

这里的帧能量是指一帧的音量（即能量），也是语音的重要特征，而且非常容易计算。因此，通常再加上一帧的对数能量（定义：一帧内信号的平方和，再取以10为底的对数值，再乘以10）使得每一帧基本的语音特征就多了一维，包括一个对数能量和剩下的倒频谱参数。另外，解释下最开始说的40维是怎么回事，假设离散余弦变换的阶数取13，那么经过一阶二阶差分后就是39维了再加上帧能量总共就是40维，当然这个可以根据实际需要动态调整。

```python
import numpy as np 
from scipy import signal
from scipy.fftpack import dct
import pylab as plt

def enframe(wave_data, nw, inc, winfunc):
    '''将音频信号转化为帧。
    参数含义：
    wave_data:原始音频型号
    nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
    inc:相邻帧的间隔（同上定义）
    '''
    wlen=len(wave_data) #信号总长度
    if wlen<=nw: #若信号长度小于一个帧的长度，则帧数定义为1
        nf=1
    else: #否则，计算帧的总长度
        nf=int(np.ceil((1.0*wlen-nw+inc)/inc))
    pad_length=int((nf-1)*inc+nw) #所有帧加起来总的铺平后的长度
    zeros=np.zeros((pad_length-wlen,)) #不够的长度使用0填补，类似于FFT中的扩充数组操作
    pad_signal=np.concatenate((wave_data,zeros)) #填补后的信号记为pad_signal
    indices=np.tile(np.arange(0,nw),(nf,1))+np.tile(np.arange(0,nf*inc,inc),(nw,1)).T  #相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
    indices=np.array(indices,dtype=np.int32) #将indices转化为矩阵
    frames=pad_signal[indices] #得到帧信号
    win=np.tile(winfunc,(nf,1))  #window窗函数，这里默认取1
    return frames*win   #返回帧信号矩阵

Df=5
fs=8000
N=fs/Df
t = np.arange(0,(N-1)/fs,1/fs)  	
wave_data=np.sin(2*np.pi*200*t)
#预加重
#b,a = signal.butter(1,1-0.97,'high')
#emphasized_signal = signal.filtfilt(b,a,wave_data)
#归一化倒谱提升窗口
lifts=[]
for n in range(1,13):
    lift =1 + 6 * np.sin(np.pi * n / 12)
    lifts.append(lift)
#print(lifts)	

#分帧、加窗 
winfunc = signal.hamming(256) 
X=enframe(wave_data, 256, 80, winfunc)    #转置的原因是分帧函数enframe的输出矩阵是帧数*帧长
frameNum =X.shape[0] #返回矩阵行数18，获取帧数
#print(frameNum)
for i in range(frameNum):
    y=X[i,:]
    #fft
    yf = np.abs(np.fft.fft(y)) 
    #print(yf.shape)
    #谱线能量
    yf = yf**2
    #梅尔滤波器系数
    nfilt = 24
    low_freq_mel = 0
    NFFT=256
    high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # 把 Hz 变成 Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # 将梅尔刻度等间隔
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # 把 Mel 变成 Hz
    bin = np.floor((NFFT + 1) * hz_points / fs)
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(yf[0:129], fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # 数值稳定性
    filter_banks = 10 * np.log10(filter_banks)  # dB 
    filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
    #print(filter_banks)
    #DCT系数
    num_ceps = 12
    c2 = dct(filter_banks, type=2, axis=-1, norm='ortho')[ 1 : (num_ceps + 1)] # Keep 2-13
    c2 *= lifts
print(c2)
plt.plot(c2)
plt.show()

```

**5.FBank**

<!-- https://www.jianshu.com/p/b25abb28b6f8 -->

Filter bank和MFCC的计算步骤基本一致，只是没有做IDFT而已。

FBank与MFCC对比：

1. 计算量：MFCC是在FBank的基础上进行的，所以MFCC的计算量更大
2. 特征区分度：FBank特征相关性较高（相邻滤波器组有重叠），MFCC具有更好的判别度，这也是在大多数语音识别论文中用的是MFCC，而不是FBank的原因
3. 使用对角协方差矩阵的GMM由于忽略了不同特征维度的相关性，MFCC更适合用来做特征。
4. DNN/CNN可以更好的利用这些相关性，使用FBank特征可以更多地降低WER。

DNN做声学模型时，一般用FilterBank feature，不用MFCC，因为FBank信息更多 (MFCC是由Mel FBank有损变换得到的）。MFCC一般是GMM做声学模型时用的，因为通常GMM假设是diagonal协方差矩阵，而cepstral coefficient更符合这种假设。linear spectrogram里面冗余信息太多了，维度也高，所以一般也不用。

说了这么多终于把语音信号处理和语音识别中相关音频特征提取的方法将清楚了！

9.paper中关于声学特征的使用

<div align=center>
    <img src="zh-cn/img/ch1/class1/p37.png" /> 
</div>

+ 2019年三大语音识别顶会超过100篇paper的统计结果
+ 使用FBank的占75%
+ 其次是MFCC

10.语音识别训练数据量

<div align=center>
    <img src="zh-cn/img/ch1/class1/p38.png" /> 
</div>

------

### 2. Class2: Listen,Attend,and Spell(LAS)

<div align=center>
    <img src="zh-cn/img/ch1/class2/p1.png" /> 
</div>

+ 语音识别模型主要有两种，一种基于seq2seq,一种基于HMM
+ seq2seq的模型主要有LAS，CTC，RNN-T，Neural Transducer，MoChA (Li主讲模型)
+ 2019语音3大顶会的模型类型统计LAS和CTC占比较多

1.Listen(Encoder)

Listen的输入就是一串acoustic features（声学特征），输出另外一串向量。

<div align=center>
    <img src="zh-cn/img/ch1/class2/p2.png" /> 
</div>

Encoder可以将语音里面的杂讯去掉，只抽出跟语音相关的讯息。
Encoder有很多做法，比如RNN（循环神经网络），RNN不只是单向的，还是双向的。https://www.jianshu.com/p/4096bb1ef45b

<div align=center>
    <img src="zh-cn/img/ch1/class2/p3.png" /> 
</div>

再比如`1-D CNN`。它采用一个filter沿着时间的方向扫过这些acoustic features，每一个filter会吃一个范围之内的acoustic features进去，然后得到一个数值。

当我们输出$b^1,b^2$等这些值的时候，我们不仅考虑$x^1$或者$x^2$，而是考虑$x^1$周边的讯号，$x^2$周边的讯号，再进行处理后得到$b^1$或者$b^2$。

如果$b^1$看的是$x^1,x^2$；$b^2$看的是$x^1,x^2,x^3$；$b^3$看的是$x^2,x^3,x^4$。再往上叠加一个filter，这个filter覆盖了$b^1,b^2,b^3$。就说明已经把$x^1,x^2,x^3,x^4$已经全面吃掉。

<div align=center>
    <img src="zh-cn/img/ch1/class2/p4.png" /> 
</div>

还可以替换为self-attention layer!

2.Down Sampling

如果要把一段声音信号表示成为acoustic features的话，会非常长（1s的信号会有100个向量）。因此为了要节省我们在语音识别的计算量，让我们的训练更有效率，我们就采用Down Sampling，这个方法。

+ Pyramid RNN: 在每一层的RNN输出后，都做一个聚合操作。把两个向量加起来，变成一个向量。
+ Pooling Over time: 两个time step的向量，只选其中一个，输入到下一层。
+ Time-delay DNN: 是CNN常用的一个变形。通常CNN是计算一个窗口内每个元素的加权之和，而TDDNN则只计算第一个和最后一个元素。（类似于空洞卷积）
+ Truncated self-attention: 是自注意力的一个变形。通常自注意力会对一个序列中每个元素都去注意，而Truncated的做法是只让当前元素去注意一个窗口内的元素。

<div align=center>
    <img src="zh-cn/img/ch1/class2/p5.png" /> 
</div>

3.Attention

通过Attention层得到$c^0$(context vector)。

<div align=center>
    <img src="zh-cn/img/ch1/class2/p7.png" /> 
</div>

注意上述两种注意力的不同。

4.Spell

这个归一化的分布向量和之前的$z^0$会作为解码器RNN的输入，输出是隐层$z^1$，和一个对词表V中所有可能词预测的概率分布向量。我们取max就可以解码得到最可能的第一个token。

再拿$z^1$与原编码器的隐层向量做注意力，得到一个新的注意力分布$z^2$。它与$c^1$一同输入给RNN，同样的方式就能解码得到第二个token。以此类推，直到解码得到的token是一个终止符，就结束。

<div align=center>
    <img src="zh-cn/img/ch1/class2/p10.png" /> 
</div>


5.Beam Search

<div align=center>
    <img src="zh-cn/img/ch1/class2/p14.png" /> 
</div>

<!-- https://blog.csdn.net/xyz1584172808/article/details/89220906 -->

Beam search 算法在文本生成中用得比较多，用于选择较优的结果（可能并不是最优的）。接下来将以seq2seq机器翻译为例来说明这个Beam search的算法思想。在机器翻译中，beam search算法在测试的时候用的，因为在训练过程中，每一个decoder的输出是有与之对应的正确答案做参照，也就不需要beam search去加大输出的准确率。
有如下从中文到英语的翻译：

中文：
```
我 爱 学习，学习 使 我 快乐
```

英语：
```
I love learning, learning makes me happy
```
在这个测试中，中文的词汇表是`{我，爱，学习，使，快乐}`，长度为5。英语的词汇表是`{I, love, learning, make, me, happy}`（全部转化为小写），长度为6。那么首先使用seq2seq中的编码器对中文序列（记这个中文序列为$X$）进行编码，得到语义向量$C$。

<div align=center>
    <img src="zh-cn/img/ch1/class2/p15.png" /> 
</div>

得到语义向量$C$后，进入解码阶段，依次翻译成目标语言。在正式解码之前，有一个参数需要设置，那就是beam search中的beam size，这个参数就相当于top-k中的k，选择前k个最有可能的结果。在本例中，我们选择beam size=3。

来看解码器的第一个输出$y_1$,在给定语义向量$C$的情况下，首先选择英语词汇表中最有可能k个单词，也就是依次选择条件概率$P(y_1|C)$前3大对应的单词，比如这里概率最大的前三个单词依次是`I, learning, happy`。

接着生成第二个输出$y_2$在这个时候我们得到了哪些东西呢，首先我们得到了编码阶段的语义向量$C$，还有第一个输出$y_1$.此时有个问题，$y_1$有三个，怎么作为这一时刻的输入呢（解码阶段需要将前一时刻的输出作为当前时刻的输入），答案就是都试下，具体做法是：

+ 确定`I`为第一时刻的输出，将其作为第二时刻的输入，得到在已知`(C, I)`的条件下，各个单词作为该时刻输出的条件概率$P(y_2|C,I)$，有6个组合，每个组合的概率为$P(I|C)P(y_2|C, I)$。
+ 确定`learning`为第一时刻的输出，将其作为第二时刻的输入，得到该条件下，词汇表中各个单词作为该时刻输出的条件概率P(y_2|C, learning)$，这里同样有6种组合；
+ 确定`happy`为第一时刻的输出，将其作为第二时刻的输入，得到该条件下各个单词作为输出的条件概率$P(y_2|C, happy)$，得到6种组合，概率的计算方式和前面一样。

这样就得到了18个组合，每一种组合对应一个概率值$P(y_1|C)P(y_2|C, y_1)$，接着在这18个组合中选择概率值top3的那三种组合，假设得到 `I love`，`I happy`，`learning make`。

接下来要做的重复这个过程，逐步生成单词，直到遇到结束标识符停止。最后得到概率最大的那个生成序列。其概率为：

$$P(Y|C)=P(y_1|C)P(y_2|C,y_1),...,P(y_6|C,y_1,y_2,y_3,y_4,y_5)$$

以上就是Beam search算法的思想，当beam size=1时，就变成了贪心算法。

Beam search算法也有许多改进的地方，根据最后的概率公式可知，该算法倾向于选择最短的句子，因为在这个连乘操作中，每个因子都是小于1的数，因子越多，最后的概率就越小。解决这个问题的方式，最后的概率值除以这个生成序列的单词数（记生成序列的单词数为$N$），这样比较的就是每个单词的平均概率大小。

此外，连乘因子较多时，可能会超过浮点数的最小值，可以考虑取对数来缓解这个问题。

6.Training (Teacher Forcing)

<div align=center>
    <img src="zh-cn/img/ch1/class2/p16.png" /> 
</div>


7.LAS中的Attention

<div align=center>
    <img src="zh-cn/img/ch1/class2/p18.png" /> 
</div>

8.LAS的缺点

+ 在小数据集上不怎么work
+ 满足不了流式识别，需要看完整个音频片段才能给出第一个token

------

### 3. class3: CTC,RNA,RNN-T,Neural Transducer,MoCha

1.CTC(Connectionist Temporal Classification)

和LAS相比，CTC能够实现实时识别的功能，CTC模型的基本结构如下图所示：

<div align=center>
    <img src="zh-cn/img/ch1/class3/p1.png" /> 
</div>

首先，模型先通过一个encoder结构将输入的token转化为一个高维隐层嵌入，然后对于每一个token的输出使用一个分类器（全连接网络）进行分类，最终的到每个token对应的预测结果；虽然CTC网络没有Attention机制，但Encoder往往使用LSTM网络，从而每个token也能够得到上下文的信息；CTC会遇到如下两个问题：因为 CTC模型的输入是音位，因此多个相邻的token可能出现重复或者某个token的输出为空的情况：

+ 当某个token没有合适的输出时，我们输出$\Phi$，并在最后将输出结果中的$\Phi$符号删除
+ 当多个相邻 token 对应的输出重复时，我们会在最后将多个重复的输出结果合并

<div align=center>
    <img src="zh-cn/img/ch1/class3/p2.png" /> 
</div>


同样因为 CTC模型的输入是音位，因此我们无法准确的到每个序列对应的标签，以下边的例子为例，同样对于好棒这个语音的音位序列，他的标签可以是下边标签的任意一个，问题是我们要用哪一个做为这个语音序列的标签呢？CTC其实是用到了下边的所有标签，原理这里暂且不做讲解(在下文！)

<div align=center>
    <img src="zh-cn/img/ch1/class3/p3.png" /> 
</div>

!> Does CTC work?

<div align=center>
    <img src="zh-cn/img/ch1/class3/p6.png" /> 
</div>

+ 左图显示了CTC分别在以字为token和以词为token下均可以很好的识别
+ 右图显示可CTC+LM的结果会更好，从模型结构可以看大CTC的Decoder仅仅为一个线性变换，可以将语言模型作为后处理模型单独训练，同时可以减少CTC带来的`结巴`的情形的出现

!> CTC的缺点

<div align=center>
    <img src="zh-cn/img/ch1/class3/p7.png" /> 
</div>

+ 会出现`结巴`的情形，原因是CTC的Decoder仅仅attend了当前的input的vector，并且每个output的独立的，也就是下一个时刻的预测不依赖于前一个时刻，这很容易导致`结巴`的情形出现

应该怎样改变呢？RNA和RNN-T就在逐步解决这些问题！

2.RNA(Recurrent Neural Aligner)

在认识 RNN-T（2012）之前，首先要认识一下RNA（Recurrent Neural Aligner）（2017）网络(RNA在RNN-T之后出现）；前边我们了解了CTC网络，RNA网络就是将CTC中Encoder后的多个分类器（Decoder）换成了一个RNN网络，使网络能够参考序列上下文信息

<div align=center>
    <img src="zh-cn/img/ch1/class3/p8.png" /> 
</div>

+ CTC的Decoder是输入一个vector输出一个token
+ RNA添加了依赖，通过RNN使得当前output依赖于之前的output和output的隐层单元

!> 可否一个input vector映射为多个token?

3.RNN-T

RNN-T 网络在RNA网络的基础上使每个输入vector可以连续输出多个token，当每个token输出符号为$\Phi$时，RNN-T网络再开始接受下一个frame的vector，具体过程如下图所示：

<div align=center>
    <img src="zh-cn/img/ch1/class3/p9.png" /> 
</div>

其实，在RNN-T中，RNN网络的的输出并不是简单的将上一时刻的输出作为当然时刻的一个输入，而是将上一时刻的输出放入一个额外的RNN中，然后将额外RNN的输出作为当前时刻的一个输入；这个额外的RNN可以认为是一个语言模型，可以单独在语料库上进行训练，因为在一般的语料库上并不包含$\Phi$符号，因此这个额外的RNN网络在训练时会忽略符号 $\Phi$

<div align=center>
    <img src="zh-cn/img/ch1/class3/p11.png" /> 
</div>


!> 关于训练数据

<div align=center>
    <img src="zh-cn/img/ch1/class3/p13.png" /> 
</div>

+ 如何生成训练集及如何训练（将在下文解决）

4.Neural Transducer

Neural Transducer 和 RNN-T 网络相比，每次接受多个输入，并对这些输入做Attention，然后得到多个输出的语音识别模型；
和 LAS 对整个输入序列做Attenton不同，Neural Transducer只对窗口内的多个输入做Attention。Neural Transducer 模型结构如下图所示：

<div align=center>
    <img src="zh-cn/img/ch1/class3/p14.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/ch1/class3/p16.png" /> 
</div>

+ 没有Attention,Neural Transducer随着windows size的增大此时用到的信息就是最后一个vector的信息导致训练不work
+ 加了Attention的Neural Transducer在不同的windows size下表现基本一致

5.MoChA(Monotonic Chunkwise Attention)

MoCha 是一个窗口可变的语音识别模型，和 Neural Transducer 最大的区别是MoCha每次得到的窗口大小可以动态变化，每次的窗口大小是模型学习的一个参数；同时因为MoCha的窗口动态可变，因此MoCha的decoder端每次只输出一个token，MoCha模型结构如下图所示：

<div align=center>
    <img src="zh-cn/img/ch1/class3/p17.png" /> 
</div>


6. summery

<div align=center>
    <img src="zh-cn/img/ch1/class3/p21.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/ch1/class3/p22.png" /> 
</div>

------

### 4. class4: HMM(Hidden Markov Model)

<!-- https://blog.csdn.net/YanqiangWang/article/details/113685208 -->

1.GMM-HMM

以前语音识别用的是统计模型，而现在，深度学习的方法有很多思想也还是借鉴的HMM。

<div align=center>
    <img src="zh-cn/img/ch1/class4/p1.png" /> 
</div>

+ $X$是输入语音序列，$Y$是输出文字，我们的目标是穷举所有可能的$Y$，找到一个$Y^{*}$使得$P(Y|X)$最大化。这个过程叫作**解码**。
+ 根据贝叶斯定律，我们可以把它变成:
$$\frac{P(X|Y)P(Y)}{P(X)}$$
+ 由于$P(X)$与我们的解码任务是无关的，因为不会随着$Y$变化而变化，所以我们只需要算$argmax P(X|Y)P(Y)$。
+ 前面这项$P(X|Y)$是Acoustic Model(声学模型)，HMM可以建模，后面那项$P(Y)$是Language Model(语言模型)，有很多种建模方式。

<div align=center>
    <img src="zh-cn/img/ch1/class4/p2.png" /> 
</div>

+ 音素、字或词，这些token，对HMM的隐变量来说，都太大了.所以我们需要为P(X|Y)建模，变成为P(X|S)建模。S为状态，是人定义的.
+ 通常S是比音素Phoneme还要小的单位.
+ Tri-phone:序列中的每一个音素，都会受到前后音素单位发音的影响，把当前的每一个音素，都加上它前后的音素，相当于把原来的音素切得更细。相同的音素在Tri-phone中可能会划分的更细，比如`waht do you think`中的前后`o`的发音，在音素中相同，但在Tri-phone中被划分的更细,这样d后面的uw，和y后面的uw表达出来就会是不同的单位了。（这其实是为了后期更好的估计$P(X|S)$
+ state: ASR中HMM中的状态S比Tri-phone更细致，往往一个Tri-phone被3种或更多种的state描述.

<div align=center>
    <img src="zh-cn/img/ch1/class4/p3.png" /> 
</div>

+ $P(X|S)$: 给定状态S下某特定Acoustic Feature发生的概率被描述
+ 假设我们有一段声音信号$X$，它内容包含了一系列特征向量。我们会先从Start状态转移到第一个状态，发射出一些向量，再到第二个状态，发射出一些向量……以此类推，直至走到END状态。

<div align=center>
    <img src="zh-cn/img/ch1/class4/p4.png" /> 
</div>

+ 上述过程存在两种概率计算，状态转移概率$P(b|a)$即由当前状态a转移到状态b的概率；发射概率，如$P(x|"t-d+uw1")$,即该状态`t-d+uw1`发射出某种特定声学特征向量的概率
+ 可以假设每一种状态，他可以产生出的声学特征向量的分布是一个特定分布(想一想为什么我们要把state划分的那么细致？)，这个分布可以用高斯混合模型GMM来表示这个概率分布
+ 这便是为什么我们要用比Phoneme还要小的单位来表示状态。因为我们要假设每个状态发射出来的分布稳定
+ 为什么我们不用字符单位来当作状态呢？比如，c这个字母它的发音不是固定的，它在很多时候是发"ke"，但它在h后面就发音"ch"。这样就不适合拿来当作HMM的状态
+ 问题来了，state的数量太多，另外很多的state在训练中只出现一两次，根本就估计不准它的高斯混合分布
+ 衍生出来一中Tied-state的技术，它假设某些state的发音就是一样的，所以可以共用同样的高斯混合分布，来减少使用的高斯混合模型的数量。这就好比你有两个命名不一样的指针，都指向了同样的内存。
+ 这个方法在经过很多年很多年的研究之后，就产生了一个终极形态Subspace GMM。所有的State都共用同一个高斯混合模型。
+ 原理好比：它有一个池子，里面有很多高斯分布。每一个state，就好比是一个碗，每一个状态去这个池子中捞几个高斯分布出来，当作自己要发射的高斯混合分布。这样保证每个state既有不同的高斯分布，又有相同的高斯分布。
+ (这篇是来自ICASSP 2010年的论文。有趣的是，Hinton在同年也在该会议上发表了一篇关于深度学习的ASR的论文。但当时大家的注意力都在前一篇论文上，Hinton的研究并没有受到很多重视。原因在，它的表现当时不如SOTA。)

<div align=center>
    <img src="zh-cn/img/ch1/class4/p6.png" /> 
</div>

+ 假设我们已经用给定好的数据算好了发射概率和转移概率，但我们还是算不出$P(X|S)$的概率。这关键技术在Alignment。
我们需要知道哪一个声学特征，是由哪一个状态产生的，才有办法用发射概率和转移概率去计算$P(X|S)$
+ 给定的候选对齐状态不同，算出来产生的声学特征的概率也就会不一样。我们就需要穷举所有可能，找到它产生与观测$X$的声学特征概率最大，最一致的对齐方式。

2.HMM与深度学习的结合

<div align=center>
    <img src="zh-cn/img/ch1/class4/p7.png" /> 
</div>

方法1：Tandem(串联)

+ 之前的声学特征是MFCC格式，深度学习方法输入一个MFCC，预测它属于哪个状态的概率。
+ 这个深度学习模型即为一个状态分类器，输入MFCC输出对应的状态
+ 训练好的这个深度学习模型的最后的output或最后的Hidden Layer或BottleNeck Layer的特征提取出来
+ 接着我们把HMM的输入，原为声学特征（MFCC)，由深度学习的输出特征取代掉。(我们也可以取最后一个隐层或者是瓶颈层(bottleneck)。)

方法2：DNN-HMM Hybrid(混合模型)

+ HMM中有一个高斯混合模型。我们想把它用DNN取代掉
+ 高斯混合模型是给定一个状态，预测声学特征向量的分布，即$P(X|a)$
+ 而DNN是训练一个状态分类器，计算给定一个声学特征下，它是某个状态的概率，即$P(a|X)$
+ 由贝叶斯定律，
$$P(x|a)=\frac{P(a|x)P(x)}{P(a)}$$
这里$P(a)$可以由训练数据统计得到，$P(x)$可以被忽略
+ 有人会觉得这个混合模型的厉害之处在把生成模型HMM中加入了判别模型!
+ 也有人觉得他厉害之处在于用有更多参数的深度神经网络替换了GMM。但这小看了参数量大起来时GMM的表征能力
+ 实际上，这篇论文的贡献在，它让所有的给定观测，计算当前可能状态概率，都共用了一个模型。而不是像GMM那样，有很多不同的模型。类似于一个超级Subspace GMM.
+ DNN可以是任何神经网络，比如CNN或者LSTM

<div align=center>
    <img src="zh-cn/img/ch1/class4/p9.png" /> 
</div>

!> 我们要如何训练一个状态分类器呢？

+ 它的输入是一个声学特征，输出是它是某个状态的概率。
+ 我们训练这个之前，需要知道每个声学特征和状态之间的对应关系。但实际中的标注数据都是没对齐的。
+ 过去的做法是训练一个HMM-GMM，那这个粗糙的模型去做找出一个概率最大的对齐。然后再根据声学特征与状态之间的对齐数据，去训练状态分类器DNN1。
+ 接着，我们再拿这个训练好的状态分类器DNN1，替换掉原来的HMM-GMM，再对数据对齐，来训练出一个更好的状态分类器DNN2。
+ 我们反复重复这个过程。用训练得到的DNN去对数据做对齐，再用对齐的数据去训练一个新的DNN。

<div align=center>
    <img src="zh-cn/img/ch1/class4/p12.png" /> 
</div>

+ 这个方法很强。微软在2016年的时候，让语音识别超过了人类水平。实际生产中，因为要考虑到推断速度，端对端的深度学习模型并不多，除了谷歌的手机助理。大部分都是混合模型。(2020 Lee说)
+ 语音识别的公认错误率指标大概在5%左右，就已经很强了。专业听写人员就在这个水平。
+ 模型能达到5%算是极限了，因为正确答案也是人标注的，也存在5%左右的错误率。再往上提升，学到的只可能是错误信息。
+ 在微软的文献中，他们训练了一个49层的残差神经网络。输出有9000个状态类别，输出是一个向量，用的是Softmax作归一化。
+ 网络层数也变得越来越深
+ 这些技术都是十年前的技术了，现在都是端到端了！

!> 回到端到端ASR中！

这里将统一HMM,CTC,RNN-T的alignments的问题，当然LAS不存在aligments,探讨E2E ASR中如何穷举对齐，如何计算所有对齐的和，如何训练网络和做网络推断！

------

### 5. class5: E2E ASR中的alignments

<!-- https://blog.csdn.net/oldmao_2001/article/details/108740924 -->
<!-- https://zhuanlan.zhihu.com/p/127403727 -->

<div align=center>
    <img src="zh-cn/img/ch1/class5/p1.png" /> 
</div>

+ 上节的时候讲HMM，使用的是state来作为最小单位，End-to-end模型则可以直接使用较大的token进行计算，也就是直接计算给定acoustic feature(声音特征)条件下，某个token sequence出现的最大概率。
+ 也即Decoding:
$$Y^{*}=argmax_{Y} \log P(Y|X)$$
+ 在训练阶段，就是要找到一个参数$\theta$，使得给定$X$出现正确答案$\hat {Y}$的几率越大越好：
$$\theta^{*}=argmax_{ \theta } \log P_{ \theta }(\hat {Y}|X)$$
+ 由于每个分布难以穷举计算，所以一般取每个分布的前几个进行Beam Search
+ 对于LAS,可以直接计算$P(Y|X)$:
$$P(Y|X)=p(a|X)p(b|a,X)...$$
+ LAS的Decoder，input第一个context vector：$c^0$，产生一个概率分布，橙色那个：$p(a)$
并得到这个概率分布产生a的几率。
+ a出来之后，将a和下一个context vector：$c^1$作为Decoder的输入，得到下一个概率分布，获得b
+ 以此类推直到得到`<EOS>`
+ CTC,RNN-T如何计算$P(Y|X)$,CTC，RNN-T和HMM一样需要alignment
+ 我们需要在token sequence插入一些东西（一般是插入NULL或者重复值），使得token sequence与acoustic feature长度一样，例如上面的例子token sequence变成：
  $$h=a\Phi b \Phi$$
把这四个东西对应到下面四个时间步的橙色输出。
+ 然后把所有可能的alignment取出来，然后计算每个acoustic feature产生alignment的几率加起来：
$$P(Y|X)=\sum_{h\in align(Y)}P(h|X)$$

<div align=center>
    <img src="zh-cn/img/ch1/class5/p3.png" /> 
</div>

+ 对于HMM:
$$P_{\theta}(X|S)=\sum_{h\in align(S)}P(X|h)$$
+ 对于CTC,RNN-T:
$$P_{\theta}(Y|X)=\sum_{h\in align(Y)}P(h|X)$$
+ 后面内容主要讲解：
    - 如何穷举所有的alignments
    - 如何累加所有的alignments
    - 如何训练CTC,RNN-T，CTC,RNN-T是NN，需要用反向传播(GD)求解，也就是要求导，即：

     $$\theta^{*}=argmax_{\theta} logP_{\theta}(\hat{Y}|X)$$
    - 如何测试Testing (Inference, decoding): 就是要解：

        $$Y^{*}=argmax_{Y}logP(Y|X)$$

<div align=center>
    <img src="zh-cn/img/ch1/class5/p4.png" /> 
</div>

+ LAS是直接算，没有用alignment，我们这里对比一下HMM，CTC，RNN-T的alignments有什么不一样。
我们假设acoustic feature（6个）和token sequence（3个）.
+ HMM：将state中某些值进行重复，直到长度与acoustic feature长度相同
+ CTC：将token中某些值进行重复或者插入NULL，直到长度与acoustic feature长度相同
+ RNN-T：插入T个NULL

<div align=center>
    <img src="zh-cn/img/ch1/class5/p5.png" /> 
</div>

+ 除了第一步start是斜向下之外，每个列中都有两种选择，一是横着走，一个是右下走
+ 约束是要从起点（橙色）开始，终点（蓝色）结束。下面的黑色箭头是不对的
+ 从开始到结束的所有路径就是所有alignments的集合

<div align=center>
    <img src="zh-cn/img/ch1/class5/p7.png" /> 
</div>

+ CTC：将token中某些值进行重复或者插入NULL(可以在开始或者结束的地方插入NULL，NULL可以有可以没有)，直到长度与acoustic feature长度相同
+ 当在NULL位置，不能走马步（中国象棋术语）否则会跳过token

<div align=center>
    <img src="zh-cn/img/ch1/class5/p12.png" /> 
</div>

+ 上面是集中不同的合法alignment

<div align=center>
    <img src="zh-cn/img/ch1/class5/p10.png" /> 
</div>

+ 所以在起始点，NULL，token三种位置上有不同的走法
+ 注意：当token中有重复值的时候，就不能跳到下一个token (eg: `see`其一种合法的alignment: $s\Phi e \Phi e$,e和e之间必须插入$\Phi$,而不能直接将重复的e直接拼接，根据CTC的alignment的规则，此时`ee`=`e`)

<div align=center>
    <img src="zh-cn/img/ch1/class5/p14.png" /> 
</div>

+ 如下图所示，前面部分插入不插入NULL都可以，但是最后一定至少要有一个NULL，所有的NULL加起来要有T个。
+ 右下角是有一个格子的，符合最后至少有一个NULL的约束

<div align=center>
    <img src="zh-cn/img/ch1/class5/p16.png" /> 
</div>

+ 上面是两个合法的alignment,注意路径不能走出表格


<div align=center>
    <img src="zh-cn/img/ch1/class5/p18.png" /> 
</div>

+ 最后我们可以统一HMM, CTC和RNN-T的alignment的表示！

------

### 6. class6: E2E ASR中的Training & Testing

<!-- https://blog.csdn.net/oldmao_2001/article/details/108820753 -->

1.How to sum over all the alignments

<div align=center>
    <img src="zh-cn/img/ch1/class6/p1.png" /> 
</div>

+ 用的RNN-T做为例子，CTC原理是一样的
+ 我们观察上面的那个路径(alignments)：$h=\Phi c \Phi \Phi a \Phi t \Phi \Phi$
+ 那么上面这个alignment出现的概率计算方法为：(先算$\phi$出现在句首的几率，在计算句首产生$\phi$条件下，c产生的几率，以此类推)
$$P(h|X)=P(\Phi|X) \times P(c|X,\Phi) \times P(\Phi|X,\Phi c)......$$

<div align=center>
    <img src="zh-cn/img/ch1/class6/p2.png" /> 
</div>

+ RNN-T先读取第一个acoustic feature，然后通过encoder，变成一个vector的形式：$h^1$
<div align=center>
    <img src="zh-cn/img/ch1/class6/p3.png" /> 
</div>
+ RNN-T还有一个RNN结构，专门吃encoder产生的token，产生结果对后面的输出产生影响，由于是刚开始，encoder没有产生token，我们设置一个`<BOS>`作为RNN的输入

<div align=center>
    <img src="zh-cn/img/ch1/class6/p4.png" /> 
</div>

得到一个结果：$l^0$ 

+ 然后我们把$h^1$和$l^0$都丢到Decoder中去，得到概率分布：$p_{1,0}$,下标`1`代表吃第1个acoustic feature，下标`0`代表已经产生0个token的概率分布

<div align=center>
    <img src="zh-cn/img/ch1/class6/p5.png" /> 
</div>

+ 那么$\phi$出现在句首的概率是从概率分布：$p_{1,0}$中产生$phi$的几率
+ 有了$\phi$之后，再看生成c的几率,这个$\phi$对于RNN结构而言不会产生影响，RNN结构只处理token。对于Decoder而言，遇到$\phi$之后，它就知道这个acoustic feature的信息已用尽，要读取下一个acoustic feature，于是encoder再读取第二个acoustic feature，并得到对应的vector：$h^2$，由于RNN结构的$l^0$不变，因此Decoder吃$h^2$和$l^0$得到概率分布：$p_{2,0}$,这里的下标代表读取到第二个acoustic feature，已产生0个token的概率分布

<div align=center>
    <img src="zh-cn/img/ch1/class6/p6.png" /> 
</div>

+ 然后在句首产生$\phi$条件下，c产生的概率就是从概率分布：$p_{2,0}$取到c的概率
+ 接下来算出现$\phi c$后，出现$\phi$ 的几率，这个时候由于产生了token，RNN结构就会根据token c产生第二个结果：$l^1$

<div align=center>
    <img src="zh-cn/img/ch1/class6/p7.png" /> 
</div>

+ 在Decoder的部分，由于没有看到$\phi$，因此输入还是$h^2$

<div align=center>
    <img src="zh-cn/img/ch1/class6/p8.png" /> 
</div>

+ 这个时候Decoder产生的概率分布为：$p_{2,1}$,下标表示第二个acoustic feature参数，已经产生第1个token的概率分布。出现$\phi c$后，出现$\phi$的几率就是从概率分布：$p_{2,1}$取到$\phi$的概率
+ 下面就按上面的模式不断重复

<div align=center>
    <img src="zh-cn/img/ch1/class6/p9.png" /> 
</div>

+ 最后的计算就是把所有最上面从概率分布中取出对应的alignments的几率，并且相乘起来

!> 重点理解：

<div align=center>
    <img src="zh-cn/img/ch1/class6/p10.png" /> 
</div>

+ 注意观察上图，上图中的每一个格子其实都是对应一个概率分布，就是前面讲的$p_{i,j}$,i代表第几个acoustic feature，这里是$x^i$，上面讲用的是$h^i$，其实都是一样，后者是前者经过encoder得到的结果。j代表当前已经产生了几个token。
+ **这里的每个格子的概率分布$p_{i,j}$是固定的，与如何走到该格子的路径无关**
+ 所以上图中无论怎么从虚线走到$p_{4,2}$,$ p_{4,2}$ 是固定的，下面看一下这个如何理解：

<div align=center>
    <img src="zh-cn/img/ch1/class6/p11.png" /> 
</div>

+ 上面讲了，RNN-T中的RNN结构是不吃NULL的，只吃token，所以NULL在什么位置并不会影响RNN结构的输出，例如下面三种序列，RNN的输出都是一样的。都会输出$l^2$
+ 而acoustic feature的输入只和NULL的个数有关，有三个NULL就会输入$h^4$，和NULL的位置也没有关系。

理解这个了之后，就可以开始算概率和了。

<div align=center>
    <img src="zh-cn/img/ch1/class6/p13.png" /> 
</div>

+ 记：$\alpha_{i,j}$为the summation of the scores of all the alignments that read i-th acoustic features and output j-th tokens
然后我们就可以把$\alpha_{i,j}$对应到每个格子
+ 这里的$\alpha_{4,2}$意思是将四个acoustic feature产生的alignments和产生两个token的alignments的概率累加起来。具体而言$\alpha_{4,2}$怎么求？
    - $\alpha_{4,2}$可以由$\alpha_{4,1}$ 和$\alpha_{3,2}$两个状态过来：
    - $\alpha_{4,1}$是已经读取了四个acoustic feature，已经产生了一个token，还需要再生成一个token：a，就变成了$\alpha_{4,2}$；
    - $\alpha_{3,2}$是已经读取了三个acoustic feature，已经产生了两个token，还需要再读取一个acoustic feature：$x^4$，就变成了$\alpha_{4,2}$.计算公式为：

        $$\alpha_{4,2}=\alpha_{4,1}P_{4,1}(a)+\alpha_{3,2}P_{3,2}(\phi)$$
+ 那么最右下角的$\alpha_{i,j}$就是对应的alignments的总和，也就是要算每个格子对应的$\alpha$

2.Training

$$\theta^{*}=argmax_{\theta}logP_{\theta}(\hat(Y)|X)$$

后向传播要求导：
$$\frac{\partial P_{\theta}(\hat(Y)|X)}{\partial \theta}$$

经过第二个问题的学习，我们知道了：

$$P(\hat{Y}|X)=\sum_{h}P(h|X)$$

上式中的h是由一连串的概率相乘得来

<div align=center>
    <img src="zh-cn/img/ch1/class6/p15.png" /> 
</div>

所以整个式子就是一连串token产生的概率相乘后相加得来。

这些概率都是受到参数$\theta$的影响，然后概率有去影响我们要计算的总概率

<div align=center>
    <img src="zh-cn/img/ch1/class6/p16.png" /> 
</div>

根据链式法则：

<div align=center>
    <img src="zh-cn/img/ch1/class6/p17.png" /> 
</div>


下面来看$\theta$对上图中每个箭头的偏导怎么求。

+ 第一项

$$\frac{\partial P_{4,1}(a)}{\partial \theta}$$

把$\theta$看做下面这个网络的参数，$p_{4,1}$ 是输出的一个概率分布，从NN的角度来说，就是求反向传播就可以得到偏导结果。这里不展开。

<div align=center>
    <img src="zh-cn/img/ch1/class6/p18.png" /> 
</div>

+ 第二项

下面再来看另外一项偏导如何求，例如：

$$\frac{P_{\theta}(\hat(Y)|X)}{\partial p_{4,1}(a)}$$

<div align=center>
    <img src="zh-cn/img/ch1/class6/p19.png" /> 
</div>

+ 我们看它的分母是一个求和项，可以看做是带$p_{4,1}(a)$的alignments项和不带有$p_{4,1}(a)p$的alignments项的求和。
+ 求偏导的时候，后面那项不用考虑，为0，然后前面那项我们可以写成：
<div align=center>
    <img src="zh-cn/img/ch1/class6/p20.png" /> 
</div>

那么：
<div align=center>
    <img src="zh-cn/img/ch1/class6/p21.png" /> 
</div>

所以最后：
<div align=center>
    <img src="zh-cn/img/ch1/class6/p22.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/ch1/class6/p23.png" /> 
</div>

+ 下面记：$\beta_{i,j}$: the summation of the score of all the alignments staring from i-th acoustic features and j-th tokens.
就是从当前位置到结束位置所有alignments 概率之和。
+ 有了$\alpha,\beta$后，就可以算所有包含某个$(i,j)$位置的alignments路径概率总和。同样以$p_{4,1}$为例

<div align=center>
    <img src="zh-cn/img/ch1/class6/p24.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/ch1/class6/p25.png" /> 
</div>

也就是有了$\alpha,\beta$后，可以把整个第二项算出来，联合第一项就可以完成整个求偏导的过程，从而完成训练。

3.Testing(Inference,Decoding)

$$Y^{*}=argmax_{Y} logP(Y|X)$$

理想上求使得Y最大公式为：
$$argmax_{Y} log\sum_{h\in align(Y)}P(h|X)$$

这个公式要计算所有alignment的可能，并求和，不是不能求，特别麻烦，老师也没给具体方法。

现在有一个近似公式：

$$argmax_{Y} \max_{h \in align(Y)} logP(h|X)$$

就是不穷举所有的alignment序列h，而是找一个使得几率P最大的那个alignment序列$h^*$来替换掉求和。
$$h^{*}=argmax_{Y} logP(h|X)$$

其中$P(h|X)=P(h_1|X)P(h_2|X,h_1)P(h_3|X,h_1,h_2)...$

h是alignment序列：

<div align=center>
    <img src="zh-cn/img/ch1/class6/p26.png" /> 
</div>

然后再根据alignment序列$h^*$求最大的$Y^*$.

实操的时候是用RNN-T中Decoder生成的概率分布中几率最大那个结果拿出来组合为$h^*$

<div align=center>
    <img src="zh-cn/img/ch1/class6/p27.png" /> 
</div>

当然，使用各个概率分布的最大值进行相乘得到的结果不一定最优，可以用Beam Search来做。这里不展开。

!> 小结

<div align=center>
    <img src="zh-cn/img/ch1/class6/p28.png" /> 
</div>


------

### 7. class7: Language Model for ASR

<!-- https://blog.csdn.net/oldmao_2001/article/details/108841873 -->

1.为什么需要LM

<div align=center>
    <img src="zh-cn/img/ch1/class7/p1.png" /> 
</div>

+ 估测Token sequence出现的概率：$P(y_1,y_2,......,y_n)$, 注意这里的单位是token，一般的NLP中是用word为单位。
+ 对于HMM:
  $$Y^{*}=argmax_{Y} P(X|Y)P(Y)$$
  后面那个$P(Y)$就是LM
+ 对于end-to-end的DL模型LAS来说: 
   $$Y^{ *}=argmax_{Y} P(Y|X)$$
  从公式上看貌似不需要LM，实际上写成：
   $$Y^{ *}=argmax_{Y} P(Y|X)P(Y)$$
  也就是加上LM后会使得效果更加好。原因在于：LM is usually helpful when your model outputs text.

<div align=center>
    <img src="zh-cn/img/ch1/class7/p2.png" /> 
</div>

+ $P(Y∣X)$通常需要成对的训练数据，比较难收集；例如：谷歌给出的文献中提到：

`Words in Transcribed Audio：12,500 hours transcribed audio= 12,500 小时x 60分钟 x 130字/分钟≈1亿`

+ $P(Y)$只需要有文字就可以，容易获得。Just Words，例如：BERT (一个巨大的 LM) 用了
30 亿个以上的词
下面来看如何估计$P(y_1,y_2,......,y_n)$

2.N-gram

<div align=center>
    <img src="zh-cn/img/ch1/class7/p3.png" /> 
</div>

+ 我们的词语组合太多，有些句子的组合没有在训练数据中出现过，但是并不带代表这个句子出现的概率为0
+  2-gram:
  
  $$P(y_1,y_2,...,y_n)=P(y_1|BOS)P(y_2|y_1)...P(y_n|y_{n-1})$$

例如：

$$P("wreck \quad a \quad nice \quad beach")=P(wreck|START)P(a|wreck)P(nice|a)P(beach|nice)$$

其中计算$P(beach∣nice)$的方式为：

$$P(beach|nice)=\frac{C(nice \quad beach)}{C(nice)}$$

分子是`nice beach`出现的次数，分母是`nice`出现的次数
+  It is easy to generalize to 3-gram, 4-gram ……

3.Challenge of N-gram

<div align=center>
    <img src="zh-cn/img/ch1/class7/p4.png" /> 
</div>

改进就是：

4.Continuous LM

这个东西借鉴了推荐系统中的思路，例如下图中ABCDE五个用户分别有对4个动漫有打分，那么可以根据其他用户的打分和本身打分的特征推断一些空白位置的分数（可以用Matrix Factorization）：

<div align=center>
    <img src="zh-cn/img/ch1/class7/p5.png" /> 
</div>

把这个思想引入到LM上，创建一张表格列举出每个词（第一行）后面跟另外一个词（第一列）在语料库出现过次数（中间的数字），0代表在语料库没有出现过这个组合，但是并不代表这个组合出现的几率为0

<div align=center>
    <img src="zh-cn/img/ch1/class7/p6.png" /> 
</div>

现在我们分别用向量h和v表示第一行和第一列的词的属性，这两组向量是从上面的表格中估算出来的，步骤如下：

用$n_{ij}$表示第i个词后面接第j个词的次数，我们要学习$v^i,v^j$，并假定：

$$n_{12}=v^1.h^2$$
$$n_{21}=v^2.h^1$$

那么就可以写出损失函数：

$$L=\sum_{(i,j)} (v^i.h^j-n_{ij})^2$$

$v^i,v^j$用梯度下降更新

+ 然后就可以有如下操作：

History “dog” and “cat” can have similar vector $h^{dog}$ 
  and $h^{cat}$.
If $v^{jumped}\cdot h^{cat}$ is large, $v^{jumped}\cdot h^{dog}$
  would be large accordingly.

Smoothing is automatically done.

其实Continuous LM可以看做是只有一个隐藏层的NN的简化版本。
输入是独热编码，下面例子是橙色向量，表示只有狗这个单词，然后中间是要学习的蓝色$h^{dog}$向量，它和绿色的单词向量相乘得到的结果要和ground truth越接近越好。

<div align=center>
    <img src="zh-cn/img/ch1/class7/p7.png" /> 
</div>

因此，在这个上面进行扩展，就得到：

5.NN-based LM

其训练过程就是先收集数据：

<div align=center>
    <img src="zh-cn/img/ch1/class7/p8.png" /> 
</div>

然后就根据数据Learn to predict the next word：

<div align=center>
    <img src="zh-cn/img/ch1/class7/p9.png" /> 
</div>

回到前面那个例子就是：
P(“wreck a nice beach”)=P(wreck|START)P(a|wreck)P(nice|a)P(beach|nice)
P(b|a): the probability of NN predicting the next word.

<div align=center>
    <img src="zh-cn/img/ch1/class7/p10.png" /> 
</div>

当然最早把NN用于LM的还是大佬Bengio，具体可以参考相关论文，李老师提到说这个论文的实验在当时算力很难实现，但是现在已经可以很容易实现。

<div align=center>
    <img src="zh-cn/img/ch1/class7/p11.png" /> 
</div>

6.RNN-based LM

由于输入的句子长度各种不一样，因此使用普通的NN模型无法很好处理这种变长的输入，因此就有了基于RNN的LM

<div align=center>
    <img src="zh-cn/img/ch1/class7/p12.png" /> 
</div>

7.Use LM to improve LAS

可以把LM和DL的模型结合分为三种情况，具体看下面表格：

<div align=center>
    <img src="zh-cn/img/ch1/class7/p13.png" /> 
</div>

+ Shallow Fusion： 有两个模型LM和LAS，都是训练好的，然后把两个模型的输出的两个概率分布取log后按权重（用来调整是哪个模型所占比重更大一些）相加。然后再从结果中取最大（当然也可以用Beam Search）的结果作为token

<div align=center>
    <img src="zh-cn/img/ch1/class7/p14.png" /> 
</div>

+ Deep Fusion： 把两个模型LM和LAS（都是训练好的）的隐藏层拉出来，丢到一个network（灰色那个），再由network输出概率分布。
灰色那个network还需要进行再一次训练

<div align=center>
    <img src="zh-cn/img/ch1/class7/p15.png" /> 
</div>

但是这里有一个地方要注意，就是当我们需要换LM的时候，需要重新训练灰色那个模块：

```
为什么需要换LM？
应为每个语料都有应用范围，例如计算机类的术语和普通术语不一样，老师举例：
程式和城市发音一样，但是一个是计算机用语（台湾的程序的说法），一个是普通术语。
在不同领域进行识别的时候就要切换LM，否则可能识别不准确。
```

如果不想更换LM后重新训练network模块，那么就不要抽取模型的隐藏层，而是抽取softmax前的一层，这一层的大小是和输入向量大小size一样，因为在不同LM中，不同中间的隐藏层每个维度代表的含义可能是不一样的，但是不同LM的最后一层因为要接入softmax的同一个维度含义肯定都一样，而且大小都一样。
当然这样做还是有一个缺点就是词表很大的时候V就很大，接入softmax的维度就很大，计算量比较大。
还有一点这里的LM可以用传统的N-GRAM模型也可以，因为N-GRAM模型也可以生成词表大小的概率分布。

<div align=center>
    <img src="zh-cn/img/ch1/class7/p16.png" /> 
</div>

+ Cold Fusion: 这个模式先训练好LM，LAS是还没有训练，是和灰色模块一起训练的，这个模型可以加快LAS的收敛速度。因为语言顺序部分已经有LM来搞定了，LAS只用关注语音对应词部分即可，所以收敛速度快。
缺点是不能随便换LM，换LM要重新训练LAS。

<div align=center>
    <img src="zh-cn/img/ch1/class7/p17.png" /> 
</div>


到这里，恭喜你Lee的ASR课程就完结了！