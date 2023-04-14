## NeMo

!> https://github.com/NVIDIA/NeMo

<!-- https://github.com/NVIDIA/NeMo/tree/main/tutorials/asr -->

NeMo是NVIDIA的一个NLP的工具集可以用来进行ASR,NLP,TTS等任务的训练和推断，本节我们介绍我们之前参加的NVIDIA的一个比赛中用到的ASR在NeMo中的训练过程
以及离线的基于VAD的CTC语音识别的部署和流式语音识别部署过程三部分内容，系统的介绍NeMo的使用。


<div align=center>
    <img src="zh-cn/img/ch25/p1.png"   /> 
</div>

### 1.使用NeMo训练和测试ASR模型

我们以我们参加的比赛为基本例子对NeMo的训练和推断进行介绍！


<!-- markdown 插入pdf -->
<object data="zh-cn/img/ch25/NeMo语音识别模型训练.pdf" type="application/pdf" width=100% height="750px">
    <embed src="http://www.africau.edu/images/default/sample.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="zh-cn/img/ch25/NeMo语音识别模型训练.pdf">Download PDF</a>.</p>
    </embed>
</object>



### 2.NeMo流式语音识别

这一部分我们将介绍使用NeMo的一个预训练网络进行流式的推断。我们将首先介绍为什么需要流式识别以及NeMo提供给我们的一种流式识别的方式。

!> Why Stream?

+ 实时或者接近实时的推断
+ 一个离线的非常长的音频无法一次加载的推断

这里我们以Conformer-CTC模型为例进行介绍，并以一个离线的非常长的音频文件无法加载的场景延时流式语言识别的构建过程。

一种长音频的基于Conformer-CTC的ASR需要将音频分割成连续的chunk(小块)，并在每个chunk进行inference.应该注意在任意边缘都有足够的音频上下文，以便进行准确的ASR。让我们在这里介绍一些术语，以帮助我们浏览本教程的其余部分。

+ Buffer size(缓冲区大小)是推断的音频长度
+ Chunk size(块大小)是添加到缓冲区的新音频的长度

audio buffer(音频缓冲区)由一个音频chunk(块)和来自前一个chunk(块)的一些填充音频组成。为了在buffer(缓冲区)的开始和结束部分有足够的上下文的信息进行最佳预测，我们只处理长度等于每个（chunk)块大小的buffer(缓冲区)中间部分的token。

让我们假设conformer-large model可以转录的音频的最大长度是20s,那么我们可以使用20s作为buffer(缓冲区)大小，使用15s（举例）作为chunk(块)大小.因此一个小时的音频被分成240个chunk（块），每个chunk(块)15秒。让我们来看看可能为此音频创建的一些audio buffer(音频缓冲区)。

```python
# 产生chunk的迭代器
# A simple iterator class to return successive chunks of samples
class AudioChunkIterator():
    def __init__(self, samples, frame_len, sample_rate):
        self._samples = samples
        self._chunk_len = chunk_len_in_secs*sample_rate
        self._start = 0
        self.output=True
   
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.output:
            raise StopIteration
        last = int(self._start + self._chunk_len)
        if last <= len(self._samples):
            chunk = self._samples[self._start: last]
            self._start = last
        else:
            chunk = np.zeros([int(self._chunk_len)], dtype='float32')
            samp_len = len(self._samples) - self._start
            chunk[0:samp_len] = self._samples[self._start:len(self._samples)]
            self.output = False
   
        return chunk
```

```python
# 音频数据转numpy 数组，进行重采样16K
# a helper function for extracting samples as a numpy array from the audio file
import soundfile as sf
def get_samples(audio_file, target_sr=16000):
    with sf.SoundFile(audio_file, 'r') as f:
        sample_rate = f.samplerate
        samples = f.read()
        if sample_rate != target_sr:
            samples = librosa.core.resample(samples, orig_sr=sample_rate, target_sr=target_sr)
        samples = samples.transpose()
        return samples
```

让我们来看看用于解码的每一个语音chunk。

```python
import matplotlib.pyplot as plt
samples = get_samples(concat_audio_path)
sample_rate  = model.preprocessor._cfg['sample_rate'] 
chunk_len_in_secs = 1            
chunk_reader = AudioChunkIterator(samples, chunk_len_in_secs, sample_rate)
count = 0
for chunk in chunk_reader:
    count +=1
    plt.plot(chunk)
    plt.show()
    if count >= 5:
        break
```

进而可以可视化当一个新chunk进入buffer后的buffer。 audio buffer可以被认为是一个固定大小的队列，每个传入的chunk被添加到buffer的末尾，而旧的样本被再开始位置删除：

```python
import numpy as np
context_len_in_secs = 1  # 上下文的秒数，此时延时是1s

buffer_len_in_secs = chunk_len_in_secs + 2* context_len_in_secs  # bufer的长度：context_before+chunk+contex_last

buffer_len = sample_rate*buffer_len_in_secs
sampbuffer = np.zeros([buffer_len], dtype=np.float32)

chunk_reader = AudioChunkIterator(samples, chunk_len_in_secs, sample_rate)
chunk_len = sample_rate*chunk_len_in_secs
count = 0
for chunk in chunk_reader:
    count +=1
    sampbuffer[:-chunk_len] = sampbuffer[chunk_len:]
    sampbuffer[-chunk_len:] = chunk
    plt.plot(sampbuffer)
    plt.show()
    if count >= 5:
        break
```

现在我们完成了将长音频分割成小chunk的方法，我们就可以读每个buffer进行ASR并合并输出获得整个长音频的输出。首选我们写一些辅助方法用来帮助加载buffer数据：

```python
from nemo.core.classes import IterableDataset

def speech_collate_fn(batch):
    """collate batch of audio sig, audio len
    Args:
        batch (FloatTensor, LongTensor):  A tuple of tuples of signal, signal lengths.
        This collate func assumes the signals are 1d torch tensors (i.e. mono audio).
    """

    _, audio_lengths = zip(*batch)

    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()
   
    
    audio_signal= []
    for sig, sig_len in batch:
        if has_audio:
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            audio_signal.append(sig)
        
    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None

    return audio_signal, audio_lengths

# simple data layer to pass audio signal
class AudioBuffersDataLayer(IterableDataset):
    

    def __init__(self):
        super().__init__()

        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._buf_count == len(self.signal) :
            raise StopIteration
        self._buf_count +=1
        return torch.as_tensor(self.signal[self._buf_count-1], dtype=torch.float32), \
               torch.as_tensor(self.signal_shape[0], dtype=torch.int64)
        
    def set_signal(self, signals):
        self.signal = signals
        self.signal_shape = self.signal[0].shape
        self._buf_count = 0

    def __len__(self):
        return 1

```

接下来实现ASR 一个audio buffer并获取chunk的识别结果。对于每个buffer，我们选择中间的chunk,左边的和右边的分别作为历史contex和未来context，用来attend当前的chunk。

比如：一个chunk是1s,buffer大小是3s,我们实际输出是1s-2s的chunk的ASR结果。对于Confermer-CTC模型的stride是4，即时域中每4个特征向量产生1个token。MelSpectrogram特征每10毫秒生成一次，因此每40毫秒的音频会生成一个token.

注意：这里的固有假设是，来自模型的输出token与相应的音频片段很好地对齐。这对于一些CTC Loss训练的模型可能不总是正确的，因此这种streaming inference对于CTC based 模型不总是有效的。

```python

from torch.utils.data import DataLoader
import math
class ChunkBufferDecoder:

    def __init__(self,asr_model, stride, chunk_len_in_secs=1, buffer_len_in_secs=3):
        self.asr_model = asr_model
        self.asr_model.eval()
        self.data_layer = AudioBuffersDataLayer()
        self.data_loader = DataLoader(self.data_layer, batch_size=1, collate_fn=speech_collate_fn)
        self.buffers = []
        self.all_preds = []
        self.chunk_len = chunk_len_in_secs
        self.buffer_len = buffer_len_in_secs
        assert(chunk_len_in_secs<=buffer_len_in_secs)
        
        feature_stride = asr_model._cfg.preprocessor['window_stride']
        self.model_stride_in_secs = feature_stride * stride
        self.n_tokens_per_chunk = math.ceil(self.chunk_len / self.model_stride_in_secs)
        self.blank_id = len(asr_model.decoder.vocabulary)
        self.plot=False
        
    @torch.no_grad()    
    def transcribe_buffers(self, buffers, merge=True, plot=False):
        self.plot = plot
        self.buffers = buffers
        self.data_layer.set_signal(buffers[:])
        self._get_batch_preds()      
        return self.decode_final(merge)
    
    def _get_batch_preds(self):

        device = self.asr_model.device
        for batch in iter(self.data_loader):

            audio_signal, audio_signal_len = batch

            audio_signal, audio_signal_len = audio_signal.to(device), audio_signal_len.to(device)
            log_probs, encoded_len, predictions = self.asr_model(input_signal=audio_signal, input_signal_length=audio_signal_len)
            preds = torch.unbind(predictions)
            for pred in preds:
                self.all_preds.append(pred.cpu().numpy())
    
    def decode_final(self, merge=True, extra=0):
        self.unmerged = []
        self.toks_unmerged = []
        # index for the first token corresponding to a chunk of audio would be len(decoded) - 1 - delay
        delay = math.ceil((self.chunk_len + (self.buffer_len - self.chunk_len) / 2) / self.model_stride_in_secs)

        decoded_frames = []
        all_toks = []
        for pred in self.all_preds:
            ids, toks = self._greedy_decoder(pred, self.asr_model.tokenizer)
            decoded_frames.append(ids)
            all_toks.append(toks)

        for decoded in decoded_frames:
            self.unmerged += decoded[len(decoded) - 1 - delay:len(decoded) - 1 - delay + self.n_tokens_per_chunk]
        if self.plot:
            for i, tok in enumerate(all_toks):
                plt.plot(self.buffers[i])
                plt.show()
                print("\nGreedy labels collected from this buffer")
                print(tok[len(tok) - 1 - delay:len(tok) - 1 - delay + self.n_tokens_per_chunk])                
                self.toks_unmerged += tok[len(tok) - 1 - delay:len(tok) - 1 - delay + self.n_tokens_per_chunk]
            print("\nTokens collected from succesive buffers before CTC merge")
            print(self.toks_unmerged)


        if not merge:
            return self.unmerged
        return self.greedy_merge(self.unmerged)
    
    
    def _greedy_decoder(self, preds, tokenizer):
        s = []
        ids = []
        for i in range(preds.shape[0]):
            if preds[i] == self.blank_id:
                s.append("_")
            else:
                pred = preds[i]
                s.append(tokenizer.ids_to_tokens([pred.item()])[0])
            ids.append(preds[i])
        return ids, s
         
    def greedy_merge(self, preds):
        decoded_prediction = []
        previous = self.blank_id
        for p in preds:
            if (p != previous or previous == self.blank_id) and p != self.blank_id:
                decoded_prediction.append(p.item())
            previous = p
        hypothesis = self.asr_model.tokenizer.ids_to_text(decoded_prediction)
        return hypothesis
```

为了了解这个基于chunk（块）的解码器是如何组合在一起的，让我们用我们从长音频文件中创建的一些buffer(缓冲区)来调用解码器。一些有趣的实验是看看chunk大小和上下文的变化如何影响ASR模型的准确性。

```python
chunk_len_in_secs = 4
context_len_in_secs = 2

buffer_len_in_secs = chunk_len_in_secs + 2* context_len_in_secs

n_buffers = 5

buffer_len = sample_rate*buffer_len_in_secs
sampbuffer = np.zeros([buffer_len], dtype=np.float32)

chunk_reader = AudioChunkIterator(samples, chunk_len_in_secs, sample_rate)
chunk_len = sample_rate*chunk_len_in_secs
count = 0
buffer_list = []
for chunk in chunk_reader:
    count +=1
    sampbuffer[:-chunk_len] = sampbuffer[chunk_len:]
    sampbuffer[-chunk_len:] = chunk
    buffer_list.append(np.array(sampbuffer))
   
    if count >= n_buffers:
        break
```

```python
stride = 4 # 8 for Citrinet
asr_decoder = ChunkBufferDecoder(model, stride=stride, chunk_len_in_secs=chunk_len_in_secs, buffer_len_in_secs=buffer_len_in_secs )
transcription = asr_decoder.transcribe_buffers(buffer_list, plot=True)
```

```python
# Final transcription after CTC merge
print(transcription)
```

下面是对流式推断的结果评估：

```python
# WER calculation
from nemo.collections.asr.metrics.wer import word_error_rate
# Collect all buffers from the audio file
sampbuffer = np.zeros([buffer_len], dtype=np.float32)

chunk_reader = AudioChunkIterator(samples, chunk_len_in_secs, sample_rate)
buffer_list = []
for chunk in chunk_reader:
    sampbuffer[:-chunk_len] = sampbuffer[chunk_len:]
    sampbuffer[-chunk_len:] = chunk
    buffer_list.append(np.array(sampbuffer))

asr_decoder = ChunkBufferDecoder(model, stride=stride, chunk_len_in_secs=chunk_len_in_secs, buffer_len_in_secs=buffer_len_in_secs )
transcription = asr_decoder.transcribe_buffers(buffer_list, plot=False)
wer = word_error_rate(hypotheses=[transcription], references=[ref_transcript])

print(f"WER: {round(wer*100,2)}%")
```


### 3.基于麦克风的流式语音识别

下面我们介绍使用麦克风进行流式语音识别，本教程不建议在真实产品中使用该方式，如果是生产环境中部署建议使用[RIVA](https://developer.nvidia.com/riva)这个框架。

!> 本项目使用了PyAudio库，ubuntu建议如下安装方式：

```
sudo apt install python3-pyaudio
pip install pyaudio
```

```python
import numpy as np
import pyaudio as pa
import os, time

import nemo
import nemo.collections.asr as nemo_asr

# sample rate, Hz
SAMPLE_RATE = 16000

```

1.从NGC加载模型（当然也可以在本地）

```
# 模型加载
asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained('QuartzNet15x5Base-En')

```

2.查看模型配置文件

```python
from omegaconf import OmegaConf
import copy

# Preserve a copy of the full config
cfg = copy.deepcopy(asr_model._cfg)
print(OmegaConf.to_yaml(cfg))
```

3.调整推断过程前处理的参数

```python

# Make config overwrite-able
OmegaConf.set_struct(cfg.preprocessor, False)

# some changes for streaming scenario
cfg.preprocessor.dither = 0.0
cfg.preprocessor.pad_to = 0

# spectrogram normalization constants
normalization = {}
normalization['fixed_mean'] = [
     -14.95827016, -12.71798736, -11.76067913, -10.83311182,
     -10.6746914,  -10.15163465, -10.05378331, -9.53918999,
     -9.41858904,  -9.23382904,  -9.46470918,  -9.56037,
     -9.57434245,  -9.47498732,  -9.7635205,   -10.08113074,
     -10.05454561, -9.81112681,  -9.68673603,  -9.83652977,
     -9.90046248,  -9.85404766,  -9.92560366,  -9.95440354,
     -10.17162966, -9.90102482,  -9.47471025,  -9.54416855,
     -10.07109475, -9.98249912,  -9.74359465,  -9.55632283,
     -9.23399915,  -9.36487649,  -9.81791084,  -9.56799225,
     -9.70630899,  -9.85148006,  -9.8594418,   -10.01378735,
     -9.98505315,  -9.62016094,  -10.342285,   -10.41070709,
     -10.10687659, -10.14536695, -10.30828702, -10.23542833,
     -10.88546868, -11.31723646, -11.46087382, -11.54877829,
     -11.62400934, -11.92190509, -12.14063815, -11.65130117,
     -11.58308531, -12.22214663, -12.42927197, -12.58039805,
     -13.10098969, -13.14345864, -13.31835645, -14.47345634]
normalization['fixed_std'] = [
     3.81402054, 4.12647781, 4.05007065, 3.87790987,
     3.74721178, 3.68377423, 3.69344,    3.54001005,
     3.59530412, 3.63752368, 3.62826417, 3.56488469,
     3.53740577, 3.68313898, 3.67138151, 3.55707266,
     3.54919572, 3.55721289, 3.56723346, 3.46029304,
     3.44119672, 3.49030548, 3.39328435, 3.28244406,
     3.28001423, 3.26744937, 3.46692348, 3.35378948,
     2.96330901, 2.97663111, 3.04575148, 2.89717604,
     2.95659301, 2.90181116, 2.7111687,  2.93041291,
     2.86647897, 2.73473181, 2.71495654, 2.75543763,
     2.79174615, 2.96076456, 2.57376336, 2.68789782,
     2.90930817, 2.90412004, 2.76187531, 2.89905006,
     2.65896173, 2.81032176, 2.87769857, 2.84665271,
     2.80863137, 2.80707634, 2.83752184, 3.01914511,
     2.92046439, 2.78461139, 2.90034605, 2.94599508,
     2.99099718, 3.0167554,  3.04649716, 2.94116777]

cfg.preprocessor.normalize = normalization

# Disable config overwriting
OmegaConf.set_struct(cfg.preprocessor, True)
```

```python
asr_model.preprocessor = asr_model.from_config_dict(cfg.preprocessor)
# Set model to inference mode
asr_model.eval();

asr_model = asr_model.to(asr_model.device)
```

4.设置数据进行流式识别

```python
from nemo.core.classes import IterableDataset
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
import torch
from torch.utils.data import DataLoader
```

```python
# simple data layer to pass audio signal
class AudioDataLayer(IterableDataset):
    @property
    def output_types(self):
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(self, sample_rate):
        super().__init__()
        self._sample_rate = sample_rate
        self.output = True
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.output:
            raise StopIteration
        self.output = False
        return torch.as_tensor(self.signal, dtype=torch.float32), \
               torch.as_tensor(self.signal_shape, dtype=torch.int64)
        
    def set_signal(self, signal):
        self.signal = signal.astype(np.float32)/32768.
        self.signal_shape = self.signal.size
        self.output = True

    def __len__(self):
        return 1
```

```python
data_layer = AudioDataLayer(sample_rate=cfg.preprocessor.sample_rate)
data_loader = DataLoader(data_layer, batch_size=1, collate_fn=data_layer.collate_fn)
```

```python
# inference method for audio signal (single instance)
def infer_signal(model, signal):
    data_layer.set_signal(signal)
    batch = next(iter(data_loader))
    audio_signal, audio_signal_len = batch
    audio_signal, audio_signal_len = audio_signal.to(asr_model.device), audio_signal_len.to(asr_model.device)
    log_probs, encoded_len, predictions = model.forward(
        input_signal=audio_signal, input_signal_length=audio_signal_len
    )
    return log_probs

```
```python
# class for streaming frame-based ASR
# 1) use reset() method to reset FrameASR's state
# 2) call transcribe(frame) to do ASR on
#    contiguous signal's frames
class FrameASR:
    
    def __init__(self, model_definition,
                 frame_len=2, frame_overlap=2.5, 
                 offset=10):
        '''
        Args:
          frame_len: frame's duration, seconds
          frame_overlap: duration of overlaps before and after current frame, seconds
          offset: number of symbols to drop for smooth streaming
        '''
        self.vocab = list(model_definition['labels'])
        self.vocab.append('_')
        
        self.sr = model_definition['sample_rate']
        self.frame_len = frame_len
        self.n_frame_len = int(frame_len * self.sr)
        self.frame_overlap = frame_overlap
        self.n_frame_overlap = int(frame_overlap * self.sr)
        timestep_duration = model_definition['AudioToMelSpectrogramPreprocessor']['window_stride']
        for block in model_definition['JasperEncoder']['jasper']:
            timestep_duration *= block['stride'][0] ** block['repeat']
        self.n_timesteps_overlap = int(frame_overlap / timestep_duration) - 2
        self.buffer = np.zeros(shape=2*self.n_frame_overlap + self.n_frame_len,
                               dtype=np.float32)
        self.offset = offset
        self.reset()
        
    def _decode(self, frame, offset=0):
        assert len(frame)==self.n_frame_len
        self.buffer[:-self.n_frame_len] = self.buffer[self.n_frame_len:]
        self.buffer[-self.n_frame_len:] = frame
        logits = infer_signal(asr_model, self.buffer).cpu().numpy()[0]
        # print(logits.shape)
        decoded = self._greedy_decoder(
            logits[self.n_timesteps_overlap:-self.n_timesteps_overlap], 
            self.vocab
        )
        return decoded[:len(decoded)-offset]
    
    @torch.no_grad()
    def transcribe(self, frame=None, merge=True):
        if frame is None:
            frame = np.zeros(shape=self.n_frame_len, dtype=np.float32)
        if len(frame) < self.n_frame_len:
            frame = np.pad(frame, [0, self.n_frame_len - len(frame)], 'constant')
        unmerged = self._decode(frame, self.offset)
        if not merge:
            return unmerged
        return self.greedy_merge(unmerged)
    
    def reset(self):
        '''
        Reset frame_history and decoder's state
        '''
        self.buffer=np.zeros(shape=self.buffer.shape, dtype=np.float32)
        self.prev_char = ''

    @staticmethod
    def _greedy_decoder(logits, vocab):
        s = ''
        for i in range(logits.shape[0]):
            s += vocab[np.argmax(logits[i])]
        return s

    def greedy_merge(self, s):
        s_merged = ''
        
        for i in range(len(s)):
            if s[i] != self.prev_char:
                self.prev_char = s[i]
                if self.prev_char != '_':
                    s_merged += self.prev_char
        return s_merged
```

5.流式推断

流式识别的精度依赖于frame的长度和buffer size.可以通过实验得到这些参数

```python
# duration of signal frame, seconds
FRAME_LEN = 1.0
# number of audio channels (expect mono signal)
CHANNELS = 1

CHUNK_SIZE = int(FRAME_LEN*SAMPLE_RATE)
asr = FrameASR(model_definition = {
                   'sample_rate': SAMPLE_RATE,
                   'AudioToMelSpectrogramPreprocessor': cfg.preprocessor,
                   'JasperEncoder': cfg.encoder,
                   'labels': cfg.decoder.vocabulary
               },
               frame_len=FRAME_LEN, frame_overlap=2, 
               offset=4)
```

```python
asr.reset()

p = pa.PyAudio()
print('Available audio input devices:')
input_devices = []
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev.get('maxInputChannels'):
        input_devices.append(i)
        print(i, dev.get('name'))

if len(input_devices):
    dev_idx = -2
    while dev_idx not in input_devices:
        print('Please type input device ID:')
        dev_idx = int(input())

    empty_counter = 0

    def callback(in_data, frame_count, time_info, status):
        global empty_counter
        signal = np.frombuffer(in_data, dtype=np.int16)
        text = asr.transcribe(signal)
        if len(text):
            print(text,end='')
            empty_counter = asr.offset
        elif empty_counter > 0:
            empty_counter -= 1
            if empty_counter == 0:
                print(' ',end='')
        return (in_data, pa.paContinue)

    stream = p.open(format=pa.paInt16,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    input_device_index=dev_idx,
                    stream_callback=callback,
                    frames_per_buffer=CHUNK_SIZE)

    print('Listening...')

    stream.start_stream()
    
    # Interrupt kernel and then speak for a few more words to exit the pyaudio loop !
    try:
        while stream.is_active():
            time.sleep(0.1)
    finally:        
        stream.stop_stream()
        stream.close()
        p.terminate()

        print()
        print("PyAudio stopped")
    
else:
    print('ERROR: No audio input device found.')
```
