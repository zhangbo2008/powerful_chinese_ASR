epoch=10
# 2021-08-04,17点25   这套代码已经完美了. 这套跑完. 会发现输入的语句经过finetune可以完美的输出结果了!!!!!!!!!!!!


import librosa
import torch
import torchaudio
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
# 配置好cache,不然就跑c盘去了.c盘太宝贵了555555555
processor = Wav2Vec2Processor.from_pretrained("ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt")
tokenizer = processor
model = Wav2Vec2ForCTC.from_pretrained("ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt",cache_dir='E:/saving_model')  # 这种写法就把模型存到这里面了.然后下次继续这么写,会在这里面读取.

resampler = torchaudio.transforms.Resample(48_000, 16_000)
print('训练之前测试一下')
if 1:


    # Preprocessing the datasets.
    # We need to read the aduio files as arrays
    # def speech_file_to_array_fn(batch):
    #     speech_array, sampling_rate = torchaudio.load(batch["path"])
    #     batch["speech"] = resampler(speech_array).squeeze().numpy()
    #     return batch
    #========run demo
    # test_dataset = test_dataset.map(speech_file_to_array_fn)
    import soundfile as sf
    name='chinese.wav'
    src_sig, sr = sf.read(name)  # name是要 输入的wav 返回 src_sig:音频数据  sr:原采样频率
    dst_sig = librosa.resample(src_sig, sr, 16000)
    inputs = processor(dst_sig, sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)

    print("Prediction:", processor.batch_decode(predicted_ids))
    print("答案",'宋朝末年年间定居粉岭围')
    # print("Reference:", test_dataset[:2]["sentence"])



#=========run valid


#
# test_dataset = load_dataset("common_voice", "zh-CN", split="test",cache_dir='E:/dataForVoice')
# savepath=test_dataset['path'][0]
# savesentence=test_dataset['sentence'][0]
# print(1)
#
#
# resampler = torchaudio.transforms.Resample(48_000, 16_000)
#
# # Preprocessing the datasets.
# # We need to read the aduio files as arrays
# def speech_file_to_array_fn(batch):
#     speech_array, sampling_rate = torchaudio.load(batch["path"])
#     batch["speech"] = resampler(speech_array).squeeze().numpy()
#     return batch
#
#
# # test_dataset = test_dataset.map(speech_file_to_array_fn)
#
# aaa=      resampler(  torchaudio.load(savepath)[0]  ).squeeze().numpy()
# # test_dataset = test_dataset.map(speech_file_to_array_fn)
# inputs = processor(aaa, sampling_rate=16_000, return_tensors="pt", padding=True)
#
# with torch.no_grad():
#     logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
#
# predicted_ids = torch.argmax(logits, dim=-1)
#
# print("Prediction:", processor.batch_decode(predicted_ids))
# print("Reference:", savesentence)
# #======================train   下面是重要的训练代码. 进行finetune.








# 2021-06-05,16点06 思路是从英文直接超过来试试.
import soundfile as sf
name='chinese.wav'
src_sig, sr = sf.read(name)  # name是要 输入的wav 返回 src_sig:音频数据  sr:原采样频率
dst_sig = librosa.resample(src_sig, sr, 16000)
input_values = processor(dst_sig, sampling_rate=16_000, return_tensors="pt", padding=True).input_values





 # compute loss
target_transcription = "宋朝末年年间定居粉岭围"
processor.tokenizer.save_vocabulary("saving_dict") # 把字典文件存下来, 可以看看里面都是什么内容.
 # wrap processor as target processor to encode labels
with processor.as_target_processor():
  labels = processor(target_transcription, return_tensors="pt").input_ids       # 把答案也进行编码,跟语音编码是一样的.上面用的特征提取器, 下面用的nlp编码.
# 编码之后也是38这个长度.




from transformers import AutoTokenizer, AutoModelWithLMHead, AdamW

class A():
    pass
args=A()
args.learning_rate=3e-5
args.adam_epsilon=1e-8
args.weight_decay=0
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
    },
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]

optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
# 开启finetune模式 ,,,,,,,C:\Users\Administrator\.PyCharm2019.3\system\remote_sources\-456540730\-337502517\transformers\data\processors\squad.py 从这个里面进行抄代码即可.
model.zero_grad()
model.train()

print('start_train')
'''

Using cls_token, but it is not set yet.
Using mask_token, but it is not set yet.
Using sep_token, but it is not set yet.

这个不用管,其实在json里面都有.
'''
for _ in range(epoch):




    loss = model(input_values, labels=labels).loss
    loss.backward()
    optimizer.step()

    model.zero_grad()
    print(loss)
print("train_over!!!!!!!!!!")





print('训练之后下面进行测是')



if 1:


    # Preprocessing the datasets.
    # We need to read the aduio files as arrays
    # def speech_file_to_array_fn(batch):
    #     speech_array, sampling_rate = torchaudio.load(batch["path"])
    #     batch["speech"] = resampler(speech_array).squeeze().numpy()
    #     return batch
    #========run demo
    # test_dataset = test_dataset.map(speech_file_to_array_fn)
    import soundfile as sf
    name='chinese.wav'
    src_sig, sr = sf.read(name)  # name是要 输入的wav 返回 src_sig:音频数据  sr:原采样频率
    dst_sig = librosa.resample(src_sig, sr, 16000)
    inputs = processor(dst_sig, sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)

    print("Prediction:", processor.batch_decode(predicted_ids))
    print("答案",'宋朝末年年间定居粉岭围')




