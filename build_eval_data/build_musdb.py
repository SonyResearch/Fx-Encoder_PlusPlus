import sys 
sys.path.append('.')
sys.path.append('..')
import os
import torch
import json
import random 
import numpy as np 
import soundfile as sf
from pathlib import Path

from fx_aug import Random_FX_Chain
from fx_chain.constants import ALL_PROCESSORS, ALL_PROCESSORS_WITHOUT_GAIN
from train_utils import Meter, loudness_normalize


import warnings 
warnings.filterwarnings("ignore", message="Grid size 1 will likely result in GPU under-utilization")
warnings.filterwarnings("ignore", message="Grid size 4 will likely result in GPU under-utilization")


# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def traverse_dir(
        root_dir,
        extension='.wav',
        amount=None,
        str_include=None,
        str_exclude=None,
        is_pure=False,
        is_sort=False,
        is_ext=True):

    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                # path
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path

                # amount
                if (amount is not None) and (cnt == amount):
                    if is_sort:
                        file_list.sort()
                    return file_list
                
                # check string
                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue
                
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                file_list.append(pure_path)
                cnt += 1
    if is_sort:
        file_list.sort()
    return file_list

def convert_tensor_to_numpy(tensor, is_squeeze=True):
    if is_squeeze:
        tensor = tensor.squeeze()
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()

# > ========= Variables ========= < 
NUMBER_OF_STYLES = 5 # number of possible styles, all with different effects
NUMBER_OF_CONTENTS = NUMBER_OF_STYLES * 2
DURATION_OF_SEGMENT = 10 
DEVICE = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu' 
SR = 44100
METER = Meter(SR)
FX_AUG_CHAIN = Random_FX_Chain(SR, DEVICE)
NUMBER_OF_EFFECTS = 2
ALL_CONTENTS = {}


SOURCE_PATH = '/home/yytung/projects/ntu_nas/musdb_fx_norm_inst_wise'
DIR_PATH = '/home/yytung/projects/fxencoder_plusplus/eval_data/musdb'
source_path = Path(SOURCE_PATH)
audio_files = traverse_dir(source_path, is_pure=False, is_sort=True)
print('> len(audio_files): ', len(audio_files))

# Prepare contents for retrieval dataset
for i in range(NUMBER_OF_CONTENTS):
    inst_ok = {
        'drums': False,
        'bass': False,
        'other': False,
        'vocals': False
    }
    instrument_2_wav = {
        'drums': None,
        'bass': None,
        'other': None,
        'vocals': None
    }
    for fn in audio_files:
        fn = Path(fn)
        instrument = fn.stem
        if instrument == 'mixture':
            continue
        if inst_ok[instrument]:
            continue
        start_point = torch.randint(low=0, high=sf.SoundFile(fn).frames-SR*DURATION_OF_SEGMENT-1, size=(1,))[0]
        wav, sr = sf.read(fn, start=start_point, frames=SR*DURATION_OF_SEGMENT)
        wav = wav.transpose(1, 0)
        instrument_2_wav[instrument] = wav
    ALL_CONTENTS[f'content_{i}'] = instrument_2_wav


# > ========= Main Loop ========= < 
for num_fx in range(1, NUMBER_OF_EFFECTS+1): # choose number of fx processors 
    content_id = 0 
    for style_id in range(NUMBER_OF_STYLES): # 
        print('> Processing style: ', style_id)
        print('> num_fx: ', num_fx)
        if num_fx == 1:
            processors_order = random.sample(ALL_PROCESSORS_WITHOUT_GAIN, k=num_fx) 
        else:
            processors_order = random.sample(ALL_PROCESSORS, k=num_fx)
        print('> Processors_order: ', processors_order)
        random.shuffle(processors_order)

        nn_param = None 
        activate = None
        random_loudness_values = None
        must_use = None
        style_mixtures = {}
        
        for _ in range(2):
            print('Processing content: ', content_id)
            drum = ALL_CONTENTS[f'content_{content_id}']['drums']
            drum_wav = torch.from_numpy(drum).unsqueeze(0).to(DEVICE).float()
            bass = ALL_CONTENTS[f'content_{content_id}']['bass']
            bass_wav = torch.from_numpy(bass).unsqueeze(0).to(DEVICE).float()
            vocal = ALL_CONTENTS[f'content_{content_id}']['vocals']
            vocal_wav = torch.from_numpy(vocal).unsqueeze(0).to(DEVICE).float()
            other = ALL_CONTENTS[f'content_{content_id}']['other']
            other_wav = torch.from_numpy(other).unsqueeze(0).to(DEVICE).float()
            batch_wise_stems = torch.cat([drum_wav, bass_wav, vocal_wav, other_wav], dim=0)

            # initialize parameters
            if nn_param is None and activate is None and random_loudness_values is None and must_use is None:
                must_use = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
                activate = torch.bernoulli(must_use.expand(batch_wise_stems.shape[0], -1))
                random_loudness_values = -18 + 4 * torch.rand(batch_wise_stems.shape[0]) 

            processed_stems, nn_param, activate = FX_AUG_CHAIN(batch_wise_stems, nn_param, activate, processors_order)
            # per-track loudness normalized 
            processed_stems = loudness_normalize(processed_stems, random_loudness_values)

            # already loudness normalized
            drum_stem = processed_stems[0, :, :] 
            bass_stem = processed_stems[1, :, :]
            vocal_stem = processed_stems[2, :, :]
            other_stem = processed_stems[3, :, :]
            
            mixture = torch.sum(processed_stems, dim=0)
            # final loudness normalized 
            mixture = loudness_normalize(mixture.unsqueeze(0), None).squeeze(0)

            mixture = convert_tensor_to_numpy(mixture).transpose(1, 0)
            drums = convert_tensor_to_numpy(drum_stem).transpose(1, 0)
            bass = convert_tensor_to_numpy(bass_stem ).transpose(1, 0)
            vocal = convert_tensor_to_numpy(vocal_stem).transpose(1, 0)
            other = convert_tensor_to_numpy(other_stem).transpose(1, 0)

            saved_dir = f'{DIR_PATH}/{num_fx}_fx_processors/style_id_{style_id}'
            os.makedirs(saved_dir, exist_ok=True)
            sf.write(os.path.join(saved_dir, f'mixture_{content_id}_{style_id}.wav'), mixture, SR)
            sf.write(os.path.join(saved_dir, f'drums_{content_id}_{style_id}.wav'), drums, SR)
            sf.write(os.path.join(saved_dir, f'bass_{content_id}_{style_id}.wav'), bass, SR)
            sf.write(os.path.join(saved_dir, f'vocals_{content_id}_{style_id}.wav'), vocal, SR)
            sf.write(os.path.join(saved_dir, f'other_{content_id}_{style_id}.wav'), other, SR)
            content_id += 1 
