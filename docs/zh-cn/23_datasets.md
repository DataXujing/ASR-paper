## 开源语音数据集介绍

<!-- https://wiki.ailemon.net/docs/asrt-doc/asrt-doc-1deoef82nv83e -->

<!-- http://yqli.tech/page/data.html -->

<!-- https://blog.ailemon.net/2018/11/21/free-open-source-chinese-speech-datasets/ -->

工欲善其事必先利其器，做机器学习，我们需要有利器，才能完成工作，数据就是我们最重要的利器之一。做中文语音识别，我们需要有对应的中文语音数据集，以帮助我们完成和不断优化改进项目。我们可能很难拿到成千上万小时的语音数据集，但是这里有一些免费开源的语音数据集，大家一定不要错过。我们也非常感谢相关单位和团体为国内的开源界做出的贡献。我们将首先给出语音识别任务中的常用中文和英文数据集的列表和链接地址,第二部分将对开源的中文语音识别数据集进行介绍。

!> 关于和TTS或音轨分离，语音唤醒等相关的数据集我们将在对应任务的教程中介绍，这里只介绍语音识别相关的数据集。

### 1.语音识别常用开源数据集


+ 中文普通话


<div align=center>

|      | 数据                            | 描述                                     | 链接                                                         |
| ---- | ------------------------------- | ---------------------------------------- | ------------------------------------------------------------ |
| 1    | WenetSpeech                     | 10000小时，强烈推荐                      | [link](https://github.com/wenet-e2e/WenetSpeech)             |
| 2    | Aishell-1                       | 178小时                                  | [link](https://www.aishelltech.com/kysjcp)                   |
| 3    | Aishell-2                       | 1000小时                                 | [link](http://www.aishelltech.com/aishell_2)                 |
| 4    | mozilla common voice            | 提供各种语言的音频,目前14122小时87中语言 | [link](https://commonvoice.mozilla.org/zh-CN/datasets)       |
| 5    | OpenSLR                         | 提供各种语言的合成、识别等语料           | [link](https://www.openslr.org/resources.php)                |
| 6    | open speech corpora             | 各类数据搜集                             | [link](https://github.com/coqui-ai/open-speech-corpora)      |
| 7    | AiShell-4                       | 211场会议，120小时                       | [link](http://www.aishelltech.com/aishell_4)                 |
| 8    | AliMeeting                      | 118.75小时会议数据                       | [link](https://www.openslr.org/119/)                         |
| 9    | Free ST Chinese Mandarin Corpus | 855发音人102600句手机录制                | [link](https://www.openslr.org/38/)                          |
| 10   | aidatatang_200zh                | 200小时600发音人文本准确98%              | [link](https://www.openslr.org/62/)                          |
| 11   | magicData-RAMC                  | 180小时中文spontaneous conversation      | [link](https://www.magicdatatech.com/datasets/mdt2021s003-1647827542) [link](https://openslr.magicdatatech.com/resources/18/) |
| 12   | TAL_CSASR                       | 中英混合587小时                          | [link](https://ai.100tal.com/dataset)                        |
| 13   | TAL_ASR                         | 100小时讲课                              | [link](https://ai.100tal.com/dataset)                        |
| 14   | THCHS30                         | 清华大学THCHS30中文语音数据集            | [link](http://openslr.magicdatatech.com/resources/18/data_thchs30.tgz) |

</div>


+ 英文

<div align=center>

|      | 数据                 | 描述                                     | 链接                                                         |
| ---- | -------------------- | ---------------------------------------- | ------------------------------------------------------------ |
| 1    | GigaSpeech           | 10000小时，强烈推荐                      | [link](https://github.com/SpeechColab/GigaSpeech)            |
| 2    | mozilla common voice | 提供各种语言的音频,目前14122小时87中语言 | [link](https://commonvoice.mozilla.org/zh-CN/datasets)       |
| 3    | OpenSLR              | 提供各种语言的合成、识别等语料           | [link](https://www.openslr.org/resources.php)                |
| 4    | Chime-4              |                                          | [link](http://spandh.dcs.shef.ac.uk/chime_challenge/CHiME4/download.html) |
| 5    | People's speech      | 30000小时英文                            | [link](https://arxiv.org/pdf/2111.09344.pdf)                 |
| 6    | LibriSpeech          | 1000小时audiobooks                       | [link](http://www.openslr.org/12/)                           |
| 7    | earnings21           | 39小时电话会议                           | [link](https://github.com/revdotcom/speech-datasets/tree/main/earnings21) |
| 8    | MLS                  | 50000小时多语言语料                      | [link](http://www.openslr.org/94/)                           |
| 9    | open speech corpora  | 各类数据搜集                             | [link](https://github.com/coqui-ai/open-speech-corpora)      |
| 10   | TED-LIUM 3           | 452小时                                  | [link](https://www.openslr.org/51/)                          |
| 11   | VoxForge             | 讲话转录                                 | [link](http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/) |

</div>

+ 其他语言

<div align=center>

|      | 数据                    | 描述                                                         | 链接                                                    |
| ---- | ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------- |
| 1    | M-AILABS                | 1000小时，German,English,Spanish,Italian,Ukrainian,Russsian,French,Polish | [link](https://www.caito.de/?p=242)                     |
| 2    | mozilla common voice    | 提供各种语言的音频,目前14122小时87中语言                     | [link](https://commonvoice.mozilla.org/zh-CN/datasets)  |
| 3    | OpenSLR                 | 提供各种语言的合成、识别等语料                               | [link](https://www.openslr.org/resources.php)           |
| 4    | CI-AVSR                 | cantonese粤语车内auido-visual数据.8.3小时                    | [link](https://github.com/HLTCHKUST/CI-AVSR)            |
| 5    | open speech corpora     | 各类数据搜集                                                 | [link](https://github.com/coqui-ai/open-speech-corpora) |
| 6    | Hindi                   | 1111小时                                                     | [link](https://www.openslr.org/118/)                    |
| 7    | Samrómur Queries 21.12  | Samrómur Icelandic Speech corpus 20小时                      | [link](https://www.openslr.org/116/)                    |
| 8    | Samrómur Children 21.09 | Icelandic Speech from children                               | [link](https://www.openslr.org/117/)                    |
| 9    | Golos                   | 1240小时Russian                                              | [link](https://www.openslr.org/114/)                    |
| 10   | MediaSpeech             | 10小时French, Arabic, Turkish and Spanish media speech       | [link](https://www.openslr.org/108/)                    |


</div>


### 2.普通话语音识别数据集介绍


+ **THCHS30**

THCHS30是一个很经典的中文语音数据集了，包含了1万余条语音文件，大约40小时的中文语音数据，内容以文章诗句为主，全部为女声。它是由清华大学语音与语言技术中心（CSLT）出版的开放式中文语音数据库。原创录音于2002年由朱晓燕教授在清华大学计算机科学系智能与系统重点实验室监督下进行，原名为“TCMSD”，代表“清华连续”普通话语音数据库’。13年后的出版由王东博士发起，并得到了朱晓燕教授的支持。他们希望为语音识别领域的新入门的研究人员提供玩具级别的数据库，因此，数据库对学术用户完全免费。

license: Apache License v.2.0

+ **ST-CMDS**

ST-CMDS是由一个AI数据公司发布的中文语音数据集，包含10万余条语音文件，大约100余小时的语音数据。数据内容以平时的网上语音聊天和智能语音控制语句为主，855个不同说话者，同时有男声和女声，适合多种场景下使用。

License: Creative Common BY-NC-ND 4.0 (Attribution-NonCommercial-NoDerivatives 4.0 International)

+ **AISHELL-1开源版**

AISHELL-1是由北京希尔公司发布的一个中文语音数据集，其中包含约178小时的开源版数据。该数据集包含400个来自中国不同地区、具有不同的口音的人的声音。录音是在安静的室内环境中使用高保真麦克风进行录音，并采样降至16kHz。通过专业的语音注释和严格的质量检查，手动转录准确率达到95％以上。该数据免费供学术使用。他们希望为语音识别领域的新研究人员提供适量的数据。

License: Apache License v.2.0

+ **Primewords Chinese Corpus Set 1**

Primewords包含了大约100小时的中文语音数据，这个免费的中文普通话语料库由上海普力信息技术有限公司发布。语料库由296名母语为英语的智能手机录制。转录准确度大于98％，置信水平为95％，学术用途免费。抄本和话语之间的映射以JSON格式给出。


+ **aidatatang**

Aidatatang_200zh是由北京数据科技有限公司（数据堂）提供的开放式中文普通话电话语音库。

语料库长达200小时，由Android系统手机（16kHz，16位）和iOS系统手机（16kHz，16位）记录。邀请来自中国不同重点区域的600名演讲者参加录音，录音是在安静的室内环境或环境中进行，其中包含不影响语音识别的背景噪音。参与者的性别和年龄均匀分布。语料库的语言材料是设计为音素均衡的口语句子。每个句子的手动转录准确率大于98％。

+ **MAGICDATA Mandarin Chinese Read Speech Corpus**

Magic Data技术有限公司的语料库，语料库包含755小时的语音数据，其主要是移动终端的录音数据。邀请来自中国不同重点区域的1080名演讲者参与录制。句子转录准确率高于98％。录音在安静的室内环境中进行。数据库分为训练集，验证集和测试集，比例为51：1：2。诸如语音数据编码和说话者信息的细节信息被保存在元数据文件中。录音文本领域多样化，包括互动问答，音乐搜索，SNS信息，家庭指挥和控制等。还提供了分段的成绩单。该语料库旨在支持语音识别，机器翻译，说话人识别和其他语音相关领域的研究人员。因此，语料库完全免费用于学术用途。

+ **AISHELL-2 高校学术免费授权版数据集**

希尔贝壳中文普通话语音数据库AISHELL-2的语音时长为1000小时，其中718小时来自AISHELL-ASR0009-[ZH-CN]，282小时来自AISHELL-ASR0010-[ZH-CN]。录音文本涉及唤醒词、语音控制词、智能家居、无人驾驶、工业生产等12个领域。录制过程在安静室内环境中， 同时使用3种不同设备： 高保真麦克风（44.1kHz，16bit）；Android系统手机（16kHz，16bit）；iOS系统手机（16kHz，16bit）。AISHELL-2采用iOS系统手机录制的语音数据。1991名来自中国不同口音区域的发言人参与录制。经过专业语音校对人员转写标注，并通过严格质量检验，此数据库文本正确率在96%以上。（支持学术研究，未经允许禁止商用。）

[AISHELL-2 中文语音数据库申请链接](https://link.ailemon.net/?target=http://www.aishelltech.com/aishell_2)


+ **数据堂1505小时中文语音数据集（高校学术免费授权版）**

数据有效时长达1505小时，。录音内容超过3万条口语化句子，由6408名来自中国不同地区的录音人参与录制。经过专业语音校对及人员转写标注，通过严格质量检验，句准确率达98%以上，是行业内句准确率的最高标准。

[数据堂1050小时数据集申请获取链接](https://link.ailemon.net/?target=https://datatang.com/opensource)


+ **Speechocean 10小时中文普通话语音识别语料库**

这是一个10.33小时的语料库，它同时通过4个不同的麦克风收集。在安静的办公室中，由20位说话者（10位男性和10位女性）录制​​了该语料库。每个说话人在一个通道中记录了大约120声。包括转录文件。句子的转录精度高于98％。它完全免费用于学术目的。

+ **AISHELL-4**

AISHELL-4是由8通道圆形麦克风阵列采集的大型实录普通话语音数据集，用于会议场景中的语音处理。该数据集由 211 个记录的会议会话组成，每个会话包含 4 到 8 个演讲者，总时长为 120 小时。该数据集旨在三个方面连接多扬声器处理的高级研究和实际应用场景。AISHELL-4通过真实录制的会议，在对话中提供逼真的声学和丰富的自然语音特征，如短暂的停顿、语音重叠、快速切换发言者、噪音等。同时，在AISHELL-4中为每次会议提供准确的转录和发言者语音激活。这使研究人员能够探索会议处理的不同方面，从语音前端处理、语音识别和说话人分类等单个任务，到相关任务的多模态建模和联合优化。“我们”还发布了一个基于 PyTorch 的培训和评估框架作为基线系统，以促进该领域的可重复研究。

+ **hkust**

中文电话录音数据集，主要内容为电话中语音对话录音。包含200h时长，采样率为16khz，采样位宽为16bit。

官方下载链接： http://catalog.ldc.upenn.edu/LDC2005S15

+ **Common Voice**

由Mozilla构建的全球语音数据集开源平台，人人均可贡献，也可免费获取世界各国语言的语音数据集，其中，中文的语音数据集分为“中国大陆”“香港”和“台湾”三部分，AI柠檬博主认为这在一定程度上为针对中国不同地区的口音做技术适配提供了方便。

https://commonvoice.mozilla.org/zh-CN/datasets

+ **TAL_ASR数据集**

好未来AI开放平台中公开了若干语音数据集，其中有100小时的普通话语音数据集，详情：

https://ai.100tal.com/dataset

下载数据集需注册登录，本文不列举该数据集的下载链接。

+ **WeNetSpeech**

Wenet团队开源的10000小时语音数据集，详情：

https://wenet-e2e.github.io/WenetSpeech/

下载数据集需使用其指定脚本，本文不列举该数据集的下载链接。

+ **AISHELL-DMASH 中文普通话麦克风阵列家居场景语音数据库**

AISHELL-DMASH 数据集录制于具有两个不同房间的真实智能家居场景中。该数据集包含 30000 小时的语音数据。录音设备包括一个近距离通话麦克风和位于房间七个不同位置的七组设备。一组录音设备包括一部iPhone、一部Android手机、一部iPad、一个麦克风、一个半径为5cm的圆形麦克风阵列。该数据集包括 511 位演讲者，每位演讲者访问 3 次，间隔 7-15 天。 AISHELL-DMASH数据集由专业的语音标注人员以高QA流程转录，单词准确率达98%，可用于声纹识别、语音识别、唤醒词识别等研究。

该数据集下载方式详见官方网站：http://www.aishelltech.com/DMASH_Dataset


### 3.说话人验证数据集介绍

+ **cn-celeb**

此数据是“在野外”收集的大规模说话人识别数据集。该数据集包含来自1000位中国名人的13万种语音，涵盖了现实世界中的11种不同流派。所有音频文件都编码为单通道，并以16位精度以16kHz采样。数据收集过程由清华大学语音与语言技术中心组织。它也由国家自然科学基金61633013和博士后科学基金2018M640133资助。

+ **HI-MIA**

内容为中文和英文的唤醒词“嗨，米娅”。使用麦克风阵列和Hi-Fi麦克风在实际家庭环境中收集数据。下文描述了基准系统的收集过程和开发。挑战中使用的数据是从1个高保真麦克风和1/3/5米的16通道圆形麦克风阵列中提取的。内容是中文唤醒词。整个集合分为火车（254人），开发（42人）和测试（44人）子集。测试子集提供了成对的目标/非目标答案，以评估验证结果。


### 4.唤醒词数据集

+ **MobvoiHotwords**

MobvoiHotwords是从Mobvoi的商业智能扬声器收集的唤醒单词的语料库。它由关键字和非关键字语音组成。对于关键字数据，将收集包含“ Hi xiaowen”或“ Nihao Wenwen”的关键字语音。对于每个关键字，大约有36k语音。所有关键字数据均收集自788名年龄在3-65岁之间的受试者，这些受试者与智能扬声器的距离（1、3和5米）不同。在采集过程中，具有不同声压级的不同噪声（例如音乐和电视等典型的家庭环境噪声）会在后台播放。

+ **AISHELL-WakeUp-1 中英文唤醒词语音数据库**

AISHELL-WakeUp-1语音数据库共唤醒词语音3936003条，1561.12小时。录音语言，中文和英文；录音地区，中国。录音文本为“你好，米雅” “hi, mia”唤醒词。邀请254名发言人参与录制。录制过程在真实家居环境中，设置7个录音位，使用6个圆形16路PDM麦克风阵列录音板做远讲拾音(16kHz，16bit)、1个高保真麦克风做近讲拾音(44.1kHz，16bit)。此数据库经过专业语音校对人员转写标注，并通过严格质量检验，字正确率100%。可用于声纹识别、语音唤醒识别等研究使用。

该数据集下载方式详见官方网站： http://www.aishelltech.com/wakeup_data



### 5.参考文献

[1].https://wiki.ailemon.net/docs/asrt-doc/asrt-doc-1deoef82nv83e

[2].https://blog.ailemon.net/2018/11/21/free-open-source-chinese-speech-datasets/

[3].http://yqli.tech/page/data.html