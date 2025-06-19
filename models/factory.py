import sys 
sys.path.append('.')
sys.path.append('..')
import json
import logging
import os
import pathlib
import re
from copy import deepcopy
from pathlib import Path
from packaging import version
import torch

# > ============================================= Factory model =========================================== <
from typing import Tuple, Any
from dataclasses import dataclass
from .fxenc_plusplus import FxEncoderPlusPlus

def load_state_dict(checkpoint_path: str, map_location="cpu", skip_params=True):
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    return state_dict

_MODEL_CONFIGS = {
    'PANN': {
        'embed_dim': 2048, 
        'mixture_cfg': 
            {
                'audio_length': 2048, 
                'clip_samples': 44100*20, 
                'mel_bins': 64, 
                'sample_rate': 44100, 
                'window_size': 2048, 
                'hop_size': 512, 
                'fmin': 50, 
                'fmax': 18000, 
                'class_num': 527, 
                'model_type': 'PANN', 
                'model_name': 'Cnn14'
            }
    }
}

@dataclass
class EncoderConfig:
    model_name: str
    pretrained: bool
    enable_fusion: bool
    fusion_type: str

def create_fx_encoder(
    model_name: str,
    pretrained: str = "",
    enable_fusion: bool = False,
    fusion_type: str = 'None'):
    pretrained_orig = pretrained
    pretrained = pretrained.lower()
    if model_name in _MODEL_CONFIGS:
        logging.info(f"Loading {model_name} model config.")
        model_cfg = deepcopy(_MODEL_CONFIGS[model_name])
    
    
    model_cfg["enable_fusion"] = enable_fusion
    model_cfg["fusion_type"] = fusion_type

    model_cfg["audio_clap_module"] = True
    model_cfg["text_clap_module"] = False
    model_cfg["extractor_module"] = True
    model = FxEncoderPlusPlus(**model_cfg)
    
    return model, model_cfg

def create_model(args) -> Tuple[Any, Any]:
    config = EncoderConfig(
        model_name=args.model,
        pretrained="",
        enable_fusion=False,
        fusion_type='None',
    )
    return create_fx_encoder(
        config.model_name,
        config.pretrained,
        enable_fusion=config.enable_fusion,
        fusion_type=config.fusion_type
    )
    


