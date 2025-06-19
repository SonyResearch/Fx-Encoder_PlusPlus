import os
import glob
import torch
import random
import numpy as np
import time
from tqdm import tqdm
from typing import List
import torch
import soundfile as sf 

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

# > ======== FX Normalized MoisesDB ========
class MoisesDB_Norm_Dataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        root_path,
        sample_rate,
        win_len,
        batch_size,
        hop_len=None
    ):
        self.sample_rate = sample_rate
        self.win_len = win_len
        self.hop_len = hop_len if hop_len is not None else win_len
        self.batch_size = batch_size // 2
        self.instrument_folders = sorted(glob.glob(os.path.join(root_path, "*")))
        self.instrument_segments = {}
        self._prepare_data()
        min_segments = min(len(segments) for segments in self.instrument_segments.values())
        print(f"Minimum segments across instruments: {min_segments}")
        
        self.total_batches = min_segments // self.batch_size
        self.all_indices = self._generate_indices()
    
    def _prepare_data(self):
        for folder in tqdm(self.instrument_folders, desc="Indexing audio files"):
            instrument_name = os.path.basename(folder)
            
            if instrument_name == 'mixture' or os.path.splitext(instrument_name)[1] == '.wav':
                continue
            self.instrument_segments[instrument_name] = []
            
            for audio_file in glob.glob(os.path.join(folder, "**/*.wav"), recursive=True):
                try:
                    info = sf.info(audio_file)
                    if info.samplerate != self.sample_rate:
                        print(f"Warning: File {audio_file} has different sample rate: {info.samplerate}")
                        continue
                    
                    total_length = int(info.frames)
                    for offset in range(0, total_length - self.win_len + 1, self.hop_len):
                        self.instrument_segments[instrument_name].append({
                            'file': audio_file,
                            'offset': offset
                        })
                        
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
                    continue
            
            print(f"Found {len(self.instrument_segments[instrument_name])} segments for {instrument_name}")
    
    def _generate_indices(self):
        min_segments = min(len(segments) for segments in self.instrument_segments.values())
        total_batches = min_segments // self.batch_size
        
        all_batch_indices = []
        for _ in range(total_batches):
            batch_indices = []
            for _ in range(self.batch_size):
                segment_indices = {}
                for instrument in self.instrument_segments:
                    segment_indices[instrument] = random.randint(0, len(self.instrument_segments[instrument]) - 1)
                batch_indices.append(segment_indices)
            all_batch_indices.append(batch_indices)
        
        return all_batch_indices
    
    def _load_audio_chunk(self, file_path, offset, duration):
        audio, sr = sf.read(
            file_path,
            start=offset,
            frames=duration,
            dtype='float32'
        )
        
        if len(audio.shape) == 1:
            audio = np.expand_dims(audio, axis=0)
        else:
            audio = audio.T
            
        if audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)
            
        return audio
    
    def __getitem__(self, index):
        if index >= len(self.all_indices):
            raise IndexError(f"Index {index} out of bounds")
        
        batch_indices = self.all_indices[index]
        batch_audio = []
        
        for sample_indices in batch_indices:
            sample_tracks = []
            
            for instrument in sorted(self.instrument_segments.keys()):
                segment_info = self.instrument_segments[instrument][sample_indices[instrument]]
                
                try:
                    audio = self._load_audio_chunk(
                        segment_info['file'],
                        segment_info['offset'],
                        self.win_len
                    )
                    
                    sample_tracks.append(audio)
                    
                except Exception as e:
                    print(f"Error loading {segment_info['file']}: {e}")
                    sample_tracks.append(np.zeros((2, self.win_len), dtype=np.float32))
            
            batch_audio.append(np.stack(sample_tracks))
            batch_audio.append(np.stack(sample_tracks))
        
        batch_audio = np.stack(batch_audio)  # [batch_size, n_instruments(11), channels(2), samples]
        
        return {
            'audio': torch.from_numpy(batch_audio).float(),
        }
    
    def __len__(self):
        return len(self.all_indices)
    
    def reset_indices(self):
        # Set a different random seed for each epoch
        seed = int(time.time() * 1000) % 1000000
        random.seed(seed)
        self.all_indices = self._generate_indices()
        # Reset the random seed to None to use system time
        random.seed(None)

