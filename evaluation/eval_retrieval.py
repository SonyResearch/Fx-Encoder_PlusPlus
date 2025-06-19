import sys 
sys.path.append('.')
sys.path.append('..')
import os
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import torch.nn.functional as F
from fxencoder_plusplus import load_model

# > Configuration (take musdb as example)
NUM_OF_FX = 8
INSTRUMENT_CANDIDATES = ['mixture', 'drums', 'bass', 'vocals', 'other']
DATA_PATH = f'/home/yytung/projects/fxencoder_plusplus/eval_data/musdb'
FINAL_REPORT_JSON_PATH = '/home/yytung/projects/fxencoder_plusplus/eval_data/musdb'
DEVICE = 'cpu'
MODEL = load_model(device=DEVICE)
class AudioDataset(Dataset):
    def __init__(
        self, 
        root_dir: str = './dataset',
        instrument: str = 'drums',  
        sample_rate: int = 44100
    ):
        self.root_dir = root_dir
        self.instrument = instrument
        self.sample_rate = sample_rate
        
        self.style_groups = {} 
        
        for style_folder in sorted(os.listdir(root_dir)):
            if not style_folder.startswith('style_id_'): 
                continue
                
            style_path = os.path.join(root_dir, style_folder)
            try:
                style_id = int(style_folder.split('_')[2])  
            except ValueError:
                print(f"Skipping folder with invalid format: {style_folder}")
                continue

            audio_files = []
            for audio_file in os.listdir(style_path):
                if not audio_file.startswith(instrument):
                    continue
                    
                try:
                    content_id = int(audio_file.split('_')[1])
                    file_path = os.path.join(style_path, audio_file)
                    audio_files.append((file_path, content_id))
                except (ValueError, IndexError):
                    print(f"Skipping file with invalid format: {audio_file}")
                    continue
            
            if len(audio_files) >= 2:  
                self.style_groups[style_id] = audio_files
        
        self.pairs = []
        for style_id, files in self.style_groups.items():
            for i in range(len(files)):
                for j in range(i + 1, len(files)):
                    if files[i][1] != files[j][1]: 
                        self.pairs.append({
                            'style_id': style_id,
                            'content_a': files[i][1],
                            'content_b': files[j][1],
                            'file_a': files[i][0],
                            'file_b': files[j][0]
                        })
        
        print(f"Found {len(self.pairs)} valid pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        waveform_a, sr = torchaudio.load(pair['file_a'])
        waveform_b, sr = torchaudio.load(pair['file_b'])
        
        return {
            'waveform_a': waveform_a,
            'waveform_b': waveform_b,
            'style_id': pair['style_id'],
            'content_a': pair['content_a'],
            'content_b': pair['content_b'],
            'file_a': pair['file_a'],
            'file_b': pair['file_b']
        }

def extract_features(model, dataset, device):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    features_a = []
    features_b = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            waveforms_a = batch['waveform_a'].to(device)
            waveforms_b = batch['waveform_b'].to(device)

            batch_a_features = model.get_fx_embedding(waveforms_a)
            batch_b_features = model.get_fx_embedding(waveforms_b)
            
            features_a.append(batch_a_features.cpu().numpy())
            features_b.append(batch_b_features.cpu().numpy())

    features_a = np.concatenate(features_a, axis=0)
    features_b = np.concatenate(features_b, axis=0)
    return features_a, features_b 

def main_retrieval(model, data_path, target_instrument, device):
    dataset = AudioDataset(
        root_dir=data_path,
        instrument=target_instrument,
    )

    features_a, features_b = extract_features(model, dataset, device)
    
    features_a = torch.tensor(features_a) 
    features_b = torch.tensor(features_b)  
    groundtruth = torch.arange(features_a.shape[0]).view(-1, 1)
    
    # calculate similarity matrix
    similarity = features_a @ features_b.t() 
    ranking = torch.argsort(similarity, descending=True) 
    preds = torch.where(ranking == groundtruth)[1]
    preds = preds.cpu().numpy()

    metrics = {}
    metrics["mean_rank"] = preds.mean() + 1
    metrics["median_rank"] = np.floor(np.median(preds)) + 1
    for k in [1, 5, 10]:
        metrics[f"R@{k}"] = np.mean(preds < k)
        metrics["mAP@10"] = np.mean(np.where(preds < 10, 1 / (preds + 1), 0.0))
    
    print(f'fxencoder_plusplus_{target_instrument}: ', metrics)
    return metrics

# # Main evaluation loop
final_report = {}
for instrument in INSTRUMENT_CANDIDATES:
    final_report[instrument] = {}
    for num_fx in range(1, NUM_OF_FX+1):
        final_report[instrument][f'{num_fx}'] = {}

for num_fx in range(1, NUM_OF_FX+1):
    data_path = os.path.join(DATA_PATH, f'{num_fx}_fx_processors')
    for target_instrument in INSTRUMENT_CANDIDATES:
        print('> DEVICE: ', DEVICE)
        metrics = main_retrieval(MODEL, data_path, target_instrument, DEVICE)
        final_report[target_instrument][f'{num_fx}'] = metrics

# Save results
os.makedirs(FINAL_REPORT_JSON_PATH, exist_ok=True)
with open(os.path.join(FINAL_REPORT_JSON_PATH, 'final.json'), 'w') as f:
    json.dump(final_report, f)
print(' > ========== FINISHED ========== <')
