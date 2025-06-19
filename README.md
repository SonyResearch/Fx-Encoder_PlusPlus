# Fx-Encoder++

Convert audio effects from your music into encoded representations suitable for audio effects processing and analysis tasks.

## About Fx-Encoder++ 
We adopt the codebase of [CLAP](https://github.com/LAION-AI/CLAP) for this project.

An audio effects representation learning based on SimCLR. 
### Architecture 
![fxencoder_plusplus](assets/model_arc.png)

## Usage
### Installation 
```
pip install fxencoder_plusplus
```

### Usage 
**Notice: The input to Fx-Encoder++ should be stereo**

Initialize Models 
```
from fxencoder_plusplus import load_model 

# Load default base model (auto-downloads if needed)
DEVICE = 'cuda'
model = load_model(
    'default',
    device=DEVICE,
)
```

Extract audio effects representations from mixture tracks or stem tracks, where a single representation encodes the overall audio effects style of the entire input.
```
import torch 
import librosa 
audio_path = librosa.example('trumpet')
wav, sr = librosa.load(audio_path, sr=44100, mono=False)
wav = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).repeat(1, 2, 1).to(DEVICE) # [1, 2, seq_len]

fx_emb = model.get_fx_embedding(wav)
print(fx_emb.shape) # [1, embed_dim], [1, 128]
```

Extract instrument-specific audio effects representations from mixture tracks. For example, extract the audio effects representation of just the vocals within a full mix. 

1. Audio Reference:
```
import torchaudio 
import julius 
mixture_path = "/path/to/mixture.wav"
mixture, sr = torchaudio.load(mixture_path, num_frames=441000)
mixture = mixture.unsqueeze(0).to(DEVICE) # [1, channel, seq_len]

query_path = "/path/to/inst.wav"
query, sr = torchaudio.load(query_path, frame_offset=441000, num_frames=441000)
query = query.unsqueeze(0).to(DEVICE) # [1, channel, seq_len]
query = julius.resample_frac(query, int(44100), int(48000))

_, fx_emb = model.get_fx_embedding_by_audio_query(mixture, query)
print(fx_emb.shape) # [1, embed_dim], [1, 128]
``` 
2. Text Reference: 
```
import torchaudio 
mixture_path = "/path/to/mixture.wav"
mixture, sr = torchaudio.load(mixture_path, num_frames=441000)
mixture = mixture.unsqueeze(0).to(DEVICE) # [1, channel, seq_len]

query = "the sound of vocals"

_, fx_emb = model.get_fx_embedding_by_text_query(mixture, query)
print(fx_emb.shape) # [1, embed_dim], [1, 128]
```


## Training 

###  Env 

1. Create environment with conda 
```
conda create --name fxenc python=3.10.14
```
2. Install 
```
pip install -r requirements.txt 
```

### Prepare Fx-Normalized Dataset 
Because the dataset has copyright restriction, unfortunatly we cannot directly share preprocessed datasets. 

0. Download [MUSDB](https://sigsep.github.io/datasets/musdb.html), [MoisesDB](https://github.com/moises-ai/moises-db)
1. Please check [FxNorm-automix](https://github.com/sony/fxnorm-automix) for preparing audio effects normalized dataset

### Run 
```
bash scripts/train_proposed.sh 
```

## Evaluation 
We develop a retrieval-based evaluation pipeline (Using MUSDB dataset as the example)

0. Check [FxNorm-automix](https://github.com/sony/fxnorm-automix) for preparing audio effects normalized dataset
1. Synthesize evaluation dataset: check [build_musdb.py](./build_eval_data/build_musdb.py) 
2. Run retrieval-based evaluation: check [eval_retrieval.py](./evaluation/eval_retrieval.py)

## LICENSE 
This library is released under the CC BY-NC 4.0 license. Please refer to the LICENSE file for more details.
