import torch
import random
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataclasses import dataclass
from dataloader.moisesdb import MoisesDB_Norm_Dataset
from dataloader.musdb import MusDB_Norm_Dataset

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler

def get_fxNorm_dataset(
    args, 
    sample_rate, 
    win_len, 
    hop_len, 
    is_train = True,
):

    eval_mode = not is_train
    if eval_mode:
        dataset = MusDB_Norm_Dataset(
            root_path=args.val_data,
            sample_rate=sample_rate,
            win_len=win_len,
            batch_size=args.batch_size * 2,
            hop_len=win_len,)
    else:
        dataset = MusDB_Norm_Dataset(
            root_path=args.val_data,
            sample_rate=sample_rate,
            win_len=win_len,
            batch_size=args.batch_size,
            hop_len=hop_len,)
        # dataset = MoisesDB_Norm_Dataset(
        #     root_path=args.train_data,
        #     sample_rate=sample_rate,
        #     win_len=win_len,
        #     batch_size=args.batch_size,
        #     hop_len=hop_len,)

    num_samples = len(dataset)
    sampler = (
        DistributedSampler(dataset, shuffle=False)
        if args.distributed and is_train
        else None
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        sampler=sampler,
        drop_last=is_train
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    
    return DataInfo(dataloader, sampler)
    
    

def get_data(args, sample_rate, win_len, hop_len):
    data = {}
    
    if args.train_data:
        data["train"] = get_fxNorm_dataset(args, sample_rate, win_len, hop_len, is_train=True)
        
    if args.val_data:
        data["val"] = get_fxNorm_dataset(args, sample_rate, win_len, hop_len, is_train=False)
        
    return data


