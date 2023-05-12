## WeNet

<!-- 1. wennet的环境搭建
2. wenet训练AIsell-1数据集
3. wenet torchlib web api部署
4. wenet 转onnx TensorRT部署
5. wenet  Triton部署 -->

!> https://github.com/wenet-e2e/wenet

### 1.WeNet环境搭建

+ Install prebuilt python package

如果仅仅使用WeMet在python中作为语音识别的一个应用，可以进安装runtime,要求python 3.6+

```shell
pip3 install wenetruntime
```

+ Install for training

如果为了训练，可以做如下安装：

```shell
git clone https://github.com/wenet-e2e/wenet.git
cd wenet
pip3 install -r requirements.txt
```


+ Build for deployment
  
如果使用x86 runtime或语言模型（LM)，可以进行如下操作：

```shell
# runtime build requires cmake 3.14 or above
cd runtime/libtorch
mkdir build && cd build && cmake -DGRAPH_TOOLS=ON .. && cmake --build .
```


### 2.WeNet训练自己的数据集

#### 2.1预训练模型下载

+ Checkpoint Model: 后缀是`.pt`,模型是WeNet训练过程保存的checkpoint,可以用来骨头建 runtime model或进行继续训练
+ Runtime Model: 后缀为`.zip`，可以直接用于x86或android runtime. runtime model 通过Pytorch JIT由checkpoint model 导出。并进行了量化，用来权衡模型大小和精度。

<table>
<thead>
<tr>
<th>Datasets</th>
<th>Languages</th>
<th>Checkpoint Model</th>
<th>Runtime Model</th>
<th>Contributor</th>
</tr>
</thead>
<tbody>
<tr>
<td><a href="../examples/aishell/s0/README.md">aishell</a></td>
<td>CN</td>
<td><a href="https://docs.qq.com/form/page/DZnRkVHlnUk5QaFdC">Conformer</a></td>
<td><a href="https://docs.qq.com/form/page/DZnRkVHlnUk5QaFdC">Conformer</a></td>
<td><a href="https://www.chumenwenwen.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/chumenwenwen.png" width="100px"></a></td>
</tr>
<tr>
<td><a href="../examples/aishell2/s0/README.md">aishell2</a></td>
<td>CN</td>
<td><a href="https://docs.qq.com/form/page/DZnRkVHlnUk5QaFdC">Conformer</a></td>
<td><a href="https://docs.qq.com/form/page/DZnRkVHlnUk5QaFdC">Conformer</a></td>
<td><a href="https://www.chumenwenwen.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/chumenwenwen.png" width="100px"></a></td>
</tr>
<tr>
<td><a href="../examples/gigaspeech/s0/README.md">gigaspeech</a></td>
<td>EN</td>
<td><a href="https://docs.qq.com/form/page/DZnRkVHlnUk5QaFdC">Conformer</a></td>
<td><a href="https://docs.qq.com/form/page/DZnRkVHlnUk5QaFdC">Conformer</a></td>
<td><a href="https://www.chumenwenwen.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/chumenwenwen.png" width="100px"></a></td>
</tr>
<tr>
<td><a href="../examples/librispeech/s0/README.md">librispeech</a></td>
<td>EN</td>
<td><a href="https://docs.qq.com/form/page/DZnRkVHlnUk5QaFdC">Conformer</a></td>
<td><a href="https://docs.qq.com/form/page/DZnRkVHlnUk5QaFdC">Conformer</a></td>
<td><a href="https://www.chumenwenwen.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/chumenwenwen.png" width="100px"></a></td>
</tr>
<tr>
<td><a href="../examples/multi_cn/s0/README.md">multi_cn</a></td>
<td>CN</td>
<td><a href="https://docs.qq.com/form/page/DZnRkVHlnUk5QaFdC">Conformer</a></td>
<td><a href="https://docs.qq.com/form/page/DZnRkVHlnUk5QaFdC">Conformer</a></td>
<td><a href="https://www.jd.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/jd.jpeg" width="100px"></a></td>
</tr>
<tr>
<td><a href="../examples/wenetspeech/s0/README.md">wenetspeech</a></td>
<td>CN</td>
<td><a href="https://docs.qq.com/form/page/DZnRkVHlnUk5QaFdC">Conformer</a></td>
<td><a href="https://docs.qq.com/form/page/DZnRkVHlnUk5QaFdC">Conformer</a></td>
<td><a href="https://horizon.ai" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/hobot.png" width="100px"></a></td>
</tr>
</tbody>
</table>

#### 2.2 以自己构建的数据集为例的WeNet模型训练

<!-- https://blog.csdn.net/ALL_BYA/article/details/124377701 -->

**>>> step 1:数据集的准备生成`wav.scp`和`txt`文件**

数据集结构如下：
```
./AISHELL-1_sample
├── S0150
│   └── S0150_mic
│     ├── BAC009S0150W0001.txt  # 文本标注
│     ├── BAC009S0150W0001.wav  # wav文件
│     ├── BAC009S0150W0002.txt
│     ├── BAC009S0150W0002.wav
│     ├── BAC009S0150W0003.txt
│     ├── BAC009S0150W0003.wav
│     ├── BAC009S0150W0500.txt
│     └── BAC009S0150W0500.wav
└── S0252
    └── S0252_mic
        ├── BAC009S0252W0001.txt
        ├── BAC009S0252W0001.wav
        ├── BAC009S0252W0002.txt
        ├── BAC009S0252W0002.wav
        ├── BAC009S0252W0003.txt
        ├── BAC009S0252W0003.wav
        ├── BAC009S0252W0364.wav
        └── BAC009S0252W0365.txt

```

将其存放在wennet项目下的`sample/mydata/s0`文件夹下，注意自定义数据集的文件结构可以和上面文件结构不同，比如数据集下直接存放train,dev和test的数据，我们只需要修改`01_prepare_data.sh`中的shell脚本即可，其shell脚本如下：


```shell

# -----准备 wav.scp text--------------
#!/bin/bash
. ./path.sh

# 数据集存放的位置
sample_data=/workspace/wenet/examples/mydata/s0/AISHELL-1_sample
# 数据生成的地方
data=/workspace/wenet/examples/mydata/s0/data_
if [ ! -d $data ];then
    mkdir -p $data
fi
# 初始化
rm -rf $data/wav.scp
rm -rf $data/text
# 1.准备 wav.scp text
# 注：如果不是上述数据结构该部分可以自己调整比较简单！
for sub_dir in `ls ${sample_data}`;do
    wav_txt_dir=${sample_data}/${sub_dir}/${sub_dir}_mic
    echo $wav_txt_dir
    for file in `ls $wav_txt_dir`;do
        if [ ${file#*.} != "txt" ];then
            # 准备wav.scp
            echo "${file%.*} $wav_txt_dir/${file%.*}.wav" >> $data/wav.scp
            # echo `wc -l $data/wav.scp`
            # 准备text
            txt=`cat $wav_txt_dir/${file%.*}.txt`
            echo "${file%.*} $txt" >> $data/text
        fi
    done
done
echo "wav.scp and text done!"

```

如下执行，在`data_`文件夹下生成`text`和`wav.scp`文件

```
sudo chmod -R 777 01_prepara_data.sh
./01_prepara_data.sh
```

其中`text`文件内容如下：

```shell
cd data_
head text
```

```
BAC009S0150W0001 设定二十九度
BAC009S0150W0002 设定三十度
BAC009S0150W0003 设定三十一度
BAC009S0150W0004 设定三十二度
BAC009S0150W0005 调高温度
BAC009S0150W0006 哈根达斯
BAC009S0150W0007 九井日本料理
BAC009S0150W0008 老洋房花园饭店
BAC009S0150W0009 上海孙中山故居纪念馆
BAC009S0150W0010 孔家花园
```
`wav.scp`文件内容如下：

```shell
cd data_
head wav.scp
```

```
BAC009S0150W0001 /workspace/wenet/examples/mydata/s0/AISHELL-1_sample/S0150/S0150_mic/BAC009S0150W0001.wav
BAC009S0150W0002 /workspace/wenet/examples/mydata/s0/AISHELL-1_sample/S0150/S0150_mic/BAC009S0150W0002.wav
BAC009S0150W0003 /workspace/wenet/examples/mydata/s0/AISHELL-1_sample/S0150/S0150_mic/BAC009S0150W0003.wav
BAC009S0150W0004 /workspace/wenet/examples/mydata/s0/AISHELL-1_sample/S0150/S0150_mic/BAC009S0150W0004.wav
BAC009S0150W0005 /workspace/wenet/examples/mydata/s0/AISHELL-1_sample/S0150/S0150_mic/BAC009S0150W0005.wav
BAC009S0150W0006 /workspace/wenet/examples/mydata/s0/AISHELL-1_sample/S0150/S0150_mic/BAC009S0150W0006.wav
BAC009S0150W0007 /workspace/wenet/examples/mydata/s0/AISHELL-1_sample/S0150/S0150_mic/BAC009S0150W0007.wav
BAC009S0150W0008 /workspace/wenet/examples/mydata/s0/AISHELL-1_sample/S0150/S0150_mic/BAC009S0150W0008.wav
BAC009S0150W0009 /workspace/wenet/examples/mydata/s0/AISHELL-1_sample/S0150/S0150_mic/BAC009S0150W0009.wav
BAC009S0150W0010 /workspace/wenet/examples/mydata/s0/AISHELL-1_sample/S0150/S0150_mic/BAC009S0150W0010.wav
```

+ `wav.scp` each line records two tab-separated columns : `wav_id` and `wav_path`
+ `text each` line records two tab-separated columns : `wav_id` and `text_label`


**>>> step 2:生成WeNet数据格式`data.list`**

读取step 1生成的`wav.scp`和`text`两个文件，生成`data.list`

```shell

# -------2.准备data.list-------

#!/bin/bash
. ./path.sh

# 数据集存放的位置
sample_data=/workspace/wenet/examples/mydata/s0/AISHELL-1_sample
# 数据生成的地方
data=/workspace/wenet/examples/mydata/s0/data_

exec 3<$data/wav.scp 
exec 4<$data/text
exec 5<$data/text
rm -rf $data/data.list
while read wav <&3 && read txt <&4 && read txt1 <&5
do  
    key=`echo $wav | awk -F ' ' '{ printf $1}'`
    wav=`echo $wav | awk -F ' ' '{ printf $2}'`
    txt=`echo $txt | awk -F ' ' '{ printf $2}'`
    echo "{\"key\":\"${key}\",\"wav\":\"${wav}\",\"txt\":\"${txt}\" }" >> $data/data.list
done
echo "data.list done!"

```

执行上述脚本：
```shell
sudo chmod -R 777 02_get_wenet_data.sh
./02_get_wenet_data.sh
```

执行完毕后在`data_`文件夹下生成`data.list`文件，其内容如下：

```shell
cd data_
head data.list
```

```json

{"key":"BAC009S0150W0001","wav":"/workspace/wenet/examples/mydata/s0/AISHELL-1_sample/S0150/S0150_mic/BAC009S0150W0001.wav","txt":"设定二十九度" }
{"key":"BAC009S0150W0002","wav":"/workspace/wenet/examples/mydata/s0/AISHELL-1_sample/S0150/S0150_mic/BAC009S0150W0002.wav","txt":"设定三十度" }
{"key":"BAC009S0150W0003","wav":"/workspace/wenet/examples/mydata/s0/AISHELL-1_sample/S0150/S0150_mic/BAC009S0150W0003.wav","txt":"设定三十一度" }
{"key":"BAC009S0150W0004","wav":"/workspace/wenet/examples/mydata/s0/AISHELL-1_sample/S0150/S0150_mic/BAC009S0150W0004.wav","txt":"设定三十二度" }
{"key":"BAC009S0150W0005","wav":"/workspace/wenet/examples/mydata/s0/AISHELL-1_sample/S0150/S0150_mic/BAC009S0150W0005.wav","txt":"调高温度" }
{"key":"BAC009S0150W0006","wav":"/workspace/wenet/examples/mydata/s0/AISHELL-1_sample/S0150/S0150_mic/BAC009S0150W0006.wav","txt":"哈根达斯" }
{"key":"BAC009S0150W0007","wav":"/workspace/wenet/examples/mydata/s0/AISHELL-1_sample/S0150/S0150_mic/BAC009S0150W0007.wav","txt":"九井日本料理" }
{"key":"BAC009S0150W0008","wav":"/workspace/wenet/examples/mydata/s0/AISHELL-1_sample/S0150/S0150_mic/BAC009S0150W0008.wav","txt":"老洋房花园饭店" }
{"key":"BAC009S0150W0009","wav":"/workspace/wenet/examples/mydata/s0/AISHELL-1_sample/S0150/S0150_mic/BAC009S0150W0009.wav","txt":"上海孙中山故居纪念馆" }
{"key":"BAC009S0150W0010","wav":"/workspace/wenet/examples/mydata/s0/AISHELL-1_sample/S0150/S0150_mic/BAC009S0150W0010.wav","txt":"孔家花园" }
```

+ key: key of the utterance
+ wav: audio file path of the utterance
+ txt: normalized transcription of the utterance, the transcription will be tokenized to the model units on-the-fly at the training stage

**>>> step 3:生成词典dict**


首先我们将从训练音频的标注文本数据数据中获取词典，这里中文语音识别我们使用的token单位是`字符`,关于token的基本单位可以是其他的自行实现即可。其代码保存在`get_lang_char.py`中如下：

```python

#获取词典
# token的单位是字
import os

text_dir = "./data_/text"
lang_char = set()
with open(text_dir,'r',encoding='utf-8') as rfile:
    lines = rfile.readlines()
    for line in lines:
        text = line.split(" ")[1].strip("\n")
        for char in text:
            lang_char.add(char)

print("<blank> 0")
print("<unk> 1")
id=0
for id,char in enumerate(lang_char):
    print(char,id+2)
print("<sos/eos>",id+3)

```

调用该代码的shell脚本`03_get_dict.sh`如下：

```shell
# -------3.获取词典-------

#!/bin/bash
. ./path.sh

# 数据集存放的位置
sample_data=/workspace/wenet/examples/mydata/s0/AISHELL-1_sample
# 数据生成的地方
data=/workspace/wenet/examples/mydata/s0/data_

python3 get_lang_char.py >  $data/lang_char.txt
```

最后我们执行该脚本，将打印的字典保存在`data_`文件夹下的`lang_char.txt`文件中。

```shell
sudo chmod -R 777 03_get_dict.sh
./03_get_dict.sh

cd data_
head lang_char.txt
```

最后看到其词典样例如下：

```
<blank> 0
<unk> 1
嘉 2
析 3
念 4
阶 5
完 6
缘 7
储 8
块 9
```

**>>> step 4:计算CMVN**

首先这里我们解释一下CMVN(倒谱均值方差归一化)如何计算，假设我们使用的特征是fbank公80维，对于所有的训练集的所有fbank我们每一维度都会计算均值方差，这样根据训练集我们得到80个均值和80个方差，在开发集和测试集中我们使用相同的训练集得到的均值和方差对开发集和测试集上的fbank进行CMVN操作。所以我们需要知道模型配置文件中使用的特征提取的方法以及是否需要计算CMVN。

其自动化的shell脚本如下：

```shell
#--------4. 计算CMVN--------------

#!/bin/bash
. ./path.sh

# 数据集存放的位置
sample_data=/workspace/wenet/examples/mydata/s0/AISHELL-1_sample
# 数据生成的地方
data=/workspace/wenet/examples/mydata/s0/data_

config=conf/train_u2++_conformer.yaml # 这里假设我们使用WeNet2.0的U2++机构，shared encoder使用confirmer
# 打开该配置文件我们可以看到相关的特征提取和语音增强的信息在dataset_conf字段中

# 4.cmvn，计算CMVN的代买在tools下存放

python3  ../../../tools/compute_cmvn_stats.py \
    --num_workers 8 \
    --train_config $config \
    --in_scp data_/wav.scp \
    --out_cmvn data_/global_cmvn

```

我们将上述自动化脚本保存在`04_get_cmvn.sh`中并执行：

```shell

sudo chmod -R 777 04_get_cmvn.sh
./04_get_cmvn.sh
```
生成的CMVN结果被保存在`data_`文件夹下的`global_cmvn`文件，其文件内容如下：

```shell
cd data_
more global_cmvn

```


```

{"mean_stat": [3540739.0, 3572564.5, 3860528.5, 4308749.5, 4716555.0, 4849995.0, 4859065.5, 4714601.0, 4440956.5, 4384438.5, 4387894.5, 4384674.5, 4339638.5, 4207373.0, 4168026.5, 4273772.0, 4373294.0, 437798
5.5, 4358829.5, 4223823.0, 4201052.5, 4408381.5, 4424157.0, 4473661.5, 4417506.5, 4465812.0, 4462348.0, 4517914.5, 4492202.5, 4456262.5, 4428069.0, 4420249.0, 4383139.5, 4382924.0, 4470329.5, 4627179.5, 47612
45.0, 4824135.5, 4800084.5, 4735988.5, 4766333.0, 4677045.0, 4563108.5, 4389866.0, 4479932.5, 4633543.5, 4669204.0, 4652215.5, 4731626.0, 4836636.0, 4875178.0, 4904197.0, 4958564.0, 4986275.5, 4961490.5, 4952
704.0, 4883098.0, 4693577.5, 4570524.5, 4506493.5, 4478836.5, 4458747.0, 4548594.0, 4659745.5, 4700327.0, 4798825.5, 4852884.5, 4878495.0, 4861365.0, 4902382.0, 4952537.0, 4982979.0, 5016769.0, 5082762.5, 519
3796.0, 5136638.5, 5119462.5, 5176910.5, 5050679.0, 4425360.0], 

"var_stat": [34027920.0, 34979080.0, 41920680.0, 52512752.0, 63424448.0, 67689344.0, 68415048.0, 64470032.0, 57174324.0, 55901668.0, 56396988.0,
 56503956.0, 55508632.0, 52615492.0, 51838572.0, 54349976.0, 56694456.0, 56721212.0, 55953916.0, 52594956.0, 52100348.0, 57060344.0, 57524176.0, 58337480.0, 56476444.0, 57168128.0, 56875304.0, 58114540.0, 574
27524.0, 56516852.0, 55979756.0, 55881892.0, 55056728.0, 55164556.0, 57442200.0, 61539588.0, 65079680.0, 66799120.0, 66160880.0, 64423260.0, 65123004.0, 62621528.0, 59337968.0, 54605916.0, 56846980.0, 6093225
6.0, 61794720.0, 61210356.0, 63331828.0, 66292896.0, 67504272.0, 68313096.0, 69739032.0, 70467320.0, 69462568.0, 69268504.0, 67331448.0, 61887828.0, 58406072.0, 56748728.0, 55871204.0, 55406024.0, 57837432.0,
 60852736.0, 61956364.0, 64735244.0, 66217248.0, 67051920.0, 66671304.0, 67954288.0, 69416504.0, 70158080.0, 71027896.0, 72943160.0, 76361560.0, 74397552.0, 73810240.0, 75626344.0, 71969080.0, 55882308.0], 
 "frame_num": 375198}

```

我们可以看到3个字段"mean_stat"共80个，"var_stat"共80个表示每个维度的均值和方差，但这里的值都很大原因是过程中并没有做样本数量的除法，因此有第三个字段“frame_num"用来计算最终的cmvn


**>>> step 5:训练模型**

有了上述准备过程，最终我们可以训练WeNet了，训练代码存放在`wenet/bin/train.py`中。我们可以通过如下训练脚本进行训练:

```shell
#--------5. 模型训练--------------

#!/bin/bash
. ./path.sh

# 数据集存放的位置
sample_data=/workspace/wenet/examples/mydata/s0/AISHELL-1_sample
# 数据生成的地方
data=/workspace/wenet/examples/mydata/s0/data_

config=conf/train_u2++_conformer.yaml # 这里假设我们使用WeNet2.0的U2++机构，shared encoder使用confirmer
# 打开该配置文件我们可以看到相关的特征提取和语音增强的信息在dataset_conf字段中

# 模型训练

python3 ../../../wenet/bin/train.py \
    --config $config \
    --data_type raw \
    --symbol_table data_/lang_char.txt \
    --train_data data_/data.list \
    --model_dir data_/model \
    --cv_data data_/data.list \
    --num_workers 8 \
    --cmvn data_/global_cmvn \
    --gpu 0 \
    --pin_memory

```

该训练脚本我们保存在`05_train_model.sh`中，训练代码的其他参数请参考训练代码，执行如下：

```shell

sudo chmod -R 777 05_train_model.sh
./05_train_model.sh

```

<div align=center>
    <img src="zh-cn/img/ch26/p1.png"   /> 
</div>

我们使用单块V100(32G),执行上述代码可以得到如图所示的训练过程！

查看训练过程：

```shell
tensorboard --logdir tensorboard --port 5000 --bind_all
```

<div align=center>
    <img src="zh-cn/img/ch26/p2.png"   /> 
</div>

忽略精度的变化，这里仅用部分数据训练了若干周期。


**>>> step 6:模型平均**

```shell
#--------6. 模型平均--------------

#!/bin/bash
. ./path.sh

# 数据集存放的位置
sample_data=/workspace/wenet/examples/mydata/s0/AISHELL-1_sample
# 数据生成的地方
data=/workspace/wenet/examples/mydata/s0/data_

config=conf/train_u2++_conformer.yaml # 这里假设我们使用WeNet2.0的U2++机构，shared encoder使用confirmer
# 打开该配置文件我们可以看到相关的特征提取和语音增强的信息在dataset_conf字段中

# 模型训练

python3 ../../../wenet/bin/average_model.py \
      --dst_model data_/average.pt \
      --src_path data_/model \
      --num 30 \
      --val_best
```

执行上述脚本将在`data_`文件夹中生成`average.pt`

**>>> step 7:模型测试**

```shell
#--------7. 模型测试--------------

#!/bin/bash
. ./path.sh

# 数据集存放的位置
sample_data=/workspace/wenet/examples/mydata/s0/AISHELL-1_sample
# 数据生成的地方
data=/workspace/wenet/examples/mydata/s0/data_

python3 ../../../wenet/bin/recognize.py \
    --mode "attention_rescoring" \
    --config data_/model/train.yaml \
    --data_type raw \
    --test_data data_/data.list \
    --checkpoint data_/average.pt \
    --beam_size 10 \
    --batch_size 1 \
    --gpu=1 \
    --penalty 0.0 \
    --dict data_/lang_char.txt \
    --ctc_weight 1.0 \
    --reverse_weight 0 \
    --result_file data_/result.txt

python3 ../../../tools/compute-wer.py --char=1 --v=1 \
    data_/text data_/result.txt > data_/wer.txt

```

在`data_`文件夹下生成`result.txt`和`wer.txt`文件，其结构如下：

```
# result.txt

BAC009S0150W0001 设定二十九度
BAC009S0150W0002 设定三十度
BAC009S0150W0003 设定三十一度
BAC009S0150W0004 设定三十二度
BAC009S0150W0005 调高温度
BAC009S0150W0006 万根达斯
BAC009S0150W0007 就井日本料理
BAC009S0150W0008 老洋房花园饭店
BAC009S0150W0009 上海孙山国居纪年馆
BAC009S0150W0010 孔家花园

```

```
utt: BAC009S0150W0001
WER: 0.00 % N=6 C=6 S=0 D=0 I=0
lab: 设 定 二 十 九 度
rec: 设 定 二 十 九 度


utt: BAC009S0150W0002
WER: 0.00 % N=5 C=5 S=0 D=0 I=0
lab: 设 定 三 十 度
rec: 设 定 三 十 度


utt: BAC009S0150W0003
WER: 0.00 % N=6 C=6 S=0 D=0 I=0
lab: 设 定 三 十 一 度
rec: 设 定 三 十 一 度

```



**>>> step 8:添加训练语言模型**

略

**>>> step 9:模型转换JIT,ONNX**

+ 模型转Pytorch JIT

```shell
export PYTHONPATH=../../../:$PYTHONPATH

python3 ../../../wenet/bin/export_jit.py \
    --config data_/model/train.yaml \
    --checkpoint data_/average.pt \
    --output_file data_/final.zip \

```


<div align=center>
    <img src="zh-cn/img/ch26/p4.png"   /> 
</div>

成功生成JIT模型文件`final.zip`,存放在了`data_`文件夹下。

+ 模型转onnx

```shell
export PYTHONPATH=../../../:$PYTHONPATH

python3 ../../../wenet/bin/export_onnx_gpu.py \
    --config data_/model/train.yaml \
    --checkpoint data_/average.pt \
    --output_onnx_dir data_ \
```

有其他参数请详细看`export_onnx_gpu.py`

<div align=center>
    <img src="zh-cn/img/ch26/p5.png"   /> 
</div>


如上图所示我们成功导出decoder.onnx和encoder.onnx,在2022年的天池-NVIDIA Transformer异构计算模型优化大赛中初赛我们优化了WeNet使其在TensorRT中可以正常推断，我们团队是600余支队伍中第一个优化成功的参赛队伍，也是第一个将LayerNorm Plugin用在WeNet的TensorRT异构计算中并提高了WeNet在TensorRT下的运行速度和精度,并在 初赛中以Top30进入决赛，并最终综合初赛和决赛取得了该场比赛的第一名。我们也将在下一节关于WeNet的Triton Inference Server的部署中详细介绍部署方案。


### 3.WeNet Server x86 ASR Demo

+ 有LM和没有LM的解码方式

<div align=center>
    <img src="zh-cn/img/ch26/p6.png"   /> 
</div>

+ U2的工作方式

<div align=center>
    <img src="zh-cn/img/ch26/p6.gif"   /> 
</div>


**>>> step 1:准备模型**

为了测试效果这部分我们直接使用WeNet提供的中文预训练网络进行测试，我们直接申请下载了JIT转换后的模型，如果你下载的是checkpoint请安装section 2中的方式进行转换。

模型中包含3个文件：

+ final.zip: JIT 模型
+ train.yaml: 模型配置文件
+ units.txt: 词典


**>>> step 2:build服务**

+ 对于CPU

```shell
cd wenet/runtime/libtorch
mkdir build && cd build && cmake .. && cmake --build .
```

+ 对于GPU

```shell
cd wenet/runtime/libtorch
mkdir build && cd build && cmake -DGPU=ON -DGRAPH_TOOLS=ON -DONNX=ON -DTORCH=ON .. && cmake --build .
```


稍等片刻即可编译成功，如下图所示：

<div align=center>
    <img src="zh-cn/img/ch26/p7.png"   /> 
</div>

**>>> step 3: Testing,RTF(real time factor)被显示在console**

```shell
export GLOG_logtostderr=1
export GLOG_v=2
wav_path=/workspace/wenet/my_web/BAC009S0150W0224.wav
model_dir=/workspace/wenet/my_web/20220506_u2pp_conformer_libtorch
./build/bin/decoder_main \
    --chunk_size -1 \
    --wav_path $wav_path \
    --model_path $model_dir/final.zip \
    --unit_path $model_dir/units.txt 2>&1 | tee log.txt

```

运行上述代码输出如下：

<div align=center>
    <img src="zh-cn/img/ch26/p8.png"   /> 
</div>

**>>> step 4: 开启WebSocket服务**

```shell

export GLOG_logtostderr=1
export GLOG_v=2
model_dir=/workspace/wenet/my_web/20220506_u2pp_conformer_libtorch
./build/bin/websocket_server_main \
    --port 5001 \
    --chunk_size 16 \
    --model_path $model_dir/final.zip \
    --unit_path $model_dir/units.txt 2>&1 | tee server.log

```

将上述文件保存在`wenet/runtime/libtorch/websocket_server.sh`中，并在后台运行该服务：

```shell

nohup ./websocket_server.sh &

# 查看进程，方便结束
ps -ef | grep websocket_server.sh
```

修改`wenet/runtime/libtorch/web`中的相关端口信息，然后运行：

```shell
python3 app.py
```

在本地浏览器打开http服务即可测试，如下所示：

<!-- <video src="zh-cn/img/ch26/p9.mp4" width=90%></video> -->

<video id="video" controls="" preload="none" poster="封面" width=90%>
      <source id="mp4" src="zh-cn/img/ch26/p9.mp4" type="video/mp4">
</video>



**>>> step 5: 开启gRPC服务和http服务**

runtime同时也支持gRPC和http服务，只需在build时开启相应的编译选项:

+ grpc

```shell
cd wenet/runtime/libtorch
mkdir build && cd build && cmake -DGRPC=ON .. && cmake --build .
```

server:

```shell
export GLOG_logtostderr=1
export GLOG_v=2
model_dir=your_model_dir
./build/bin/grpc_server_main \
    --port 10086 \
    --workers 4 \
    --chunk_size 16 \
    --model_path $model_dir/final.zip \
    --unit_path $model_dir/units.txt 2>&1 | tee server.log
```

client:

```shell
export GLOG_logtostderr=1
export GLOG_v=2
wav_path=your_test_wav_path
./build/bin/grpc_client_main \
    --hostname 127.0.0.1 --port 10086 \
    --wav_path $wav_path 2>&1 | tee client.log

```

+ http

```shell
mkdir build && cd build && cmake -DHTTP=ON .. && cmake --build 
```

server: simply replace grpc_server_main with http_server_main  in gRPC.

client: simply replace grpc_client_main with http_client_main in gRPC.


### 4.WeNet Triton Inference Server Demo

这一届将介绍如何将离线或在线的WeNet模型部署到Triton Inference Server。

<div align=center>
    <img src="zh-cn/img/ch26/p10.jpg"   /> 
</div>

**>>> step 1: 将模型转换成ONNX**

```shell

pip install onnxruntime-gpu onnxmltools
cd wenet/examples/aishell2 && . ./path.sh
model_dir=<absolute path to>/20211025_conformer_exp
onnx_model_dir=<absolute path>
mkdir $onnx_model_dir

python3 wenet/bin/export_onnx_gpu.py \
        --config=$model_dir/train.yaml \
        --checkpoint=$model_dir/final.pt \
        --cmvn_file=$model_dir/global_cmvn \
        --ctc_weight=0.5 \
        --output_onnx_dir=$onnx_model_dir \
        --fp16

cp $model_dir/words.txt $model_dir/train.yaml $onnx_model_dir/
```

如果需要streaming,则执行：

```shell

python3 wenet/bin/export_onnx_gpu.py 
        --config=$model_dir/train.yaml \
        --checkpoint=$model_dir/final.pt \
        --cmvn_file=$model_dir/global_cmvn  \
        --ctc_weight=0.1 \
        --reverse_weight=0.4 \
        --output_onnx_dir=$onnx_model_dir \
        --fp16 \
        --streaming

```

**>>> step 2:构建server docker**

```shell
docker build . -f Dockerfile/Dockerfile.server -t wenet_server:latest --network host
# offline model
docker run --gpus '"device=0"' -it -v $PWD/model_repo:/ws/model_repo -v $onnx_model_dir:/ws/onnx_model -p 8000:8000 -p 8001:8001 -p 8002:8002 --shm-size=1g --ulimit memlock=-1  wenet_server:latest /workspace/scripts/convert_start_server.sh
# streaming model
docker run --gpus '"device=0"' -it -v $PWD/model_repo_stateful:/ws/model_repo -v $onnx_model_dir:/ws/onnx_model -p 8000:8000 -p 8001:8001 -p 8002:8002 --shm-size=1g --ulimit memlock=-1  wenet_server:latest /workspace/scripts/convert_start_server.sh

```

注意：自己的模型需要修改Triton的配置文件。

**>>> step 3:client**

```shell
docker build . -f Dockerfile/Dockerfile.client -t wenet_client:latest --network host

AUDIO_DATA=<path to your wav data>
docker run -ti --net host --name wenet_client -v $PWD/client:/ws/client -v $AUDIO_DATA:/ws/test_data wenet_client:latest
# In docker
# offline model test
cd /ws/client
# test one wav file
python3 client.py --audio_file=/ws/test_data/mid.wav --url=localhost:8001

# test a list of wav files & cer
python3 client.py --wavscp=/ws/dataset/test/wav.scp --data_dir=/ws/dataset/test/ --trans=/ws/dataset/test/text

```

如果是streaming的，client调用时需要指定：

```shell
python3 client.py --wavscp=/ws/test_data/data_aishell2/test/wav.scp --data_dir=/ws/test_data/data_aishell2/test/ --trans=/ws/test_data/data_aishell2/test/trans.txt --model_name=streaming_wenet --streaming
```



