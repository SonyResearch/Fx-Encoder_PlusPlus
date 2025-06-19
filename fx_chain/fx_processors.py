import torch 
from dasp_pytorch.modules import Processor

from .ddsp_cores import *
from .constants import *

class Distortion(Processor):
    def __init__(
        self,
        sample_rate: int,
        min_gain_db: float = 0.0,
        max_gain_db: float = 24.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.process_fn = distortion
        self.param_ranges = {
            "drive_db": (min_gain_db, max_gain_db),
        }
        self.num_params = len(self.param_ranges)

class Limiter(Processor):
    def __init__(
        self,
        sample_rate: int,
        min_threshold_db: float = -60.0,
        max_threshold_db: float = 0.0-EPS,
        min_attack_ms: float = 5.0,
        max_attack_ms: float = 100.0,
        min_release_ms: float = 5.0,
        max_release_ms: float = 100.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.process_fn = limiter
        self.param_ranges = {
            "threshold": (min_threshold_db, max_threshold_db),
            "at": (min_attack_ms, max_attack_ms),
            "rt": (min_release_ms, max_release_ms),
        }
        self.num_params = len(self.param_ranges)

class Multiband_Compressor(Processor):
    def __init__(
        self,
        sample_rate: int,
        min_threshold_db_comp: float = -60.0,
        max_threshold_db_comp: float = 0.0-EPS,
        min_ratio_comp: float = 1.0+EPS,
        max_ratio_comp: float = 20.0,
        min_attack_ms_comp: float = 5.0,
        max_attack_ms_comp: float = 100.0,
        min_release_ms_comp: float = 5.0,
        max_release_ms_comp: float = 100.0,
        min_threshold_db_exp: float = -60.0,
        max_threshold_db_exp: float = 0.0-EPS,
        min_ratio_exp: float = 0.0+EPS,
        max_ratio_exp: float = 1.0-EPS,
        min_attack_ms_exp: float = 5.0,
        max_attack_ms_exp: float = 100.0,
        min_release_ms_exp: float = 5.0,
        max_release_ms_exp: float = 100.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.process_fn = multiband_compressor
        self.param_ranges = {
            "low_cutoff": (20, 300),
            "high_cutoff": (2000, 12000), 
            "parallel_weight_factor": (0.2, 0.7),

            "low_shelf_comp_thresh": (min_threshold_db_comp, max_threshold_db_comp),
            "low_shelf_comp_ratio": (min_ratio_comp, max_ratio_comp),
            "low_shelf_exp_thresh": (min_threshold_db_exp, max_threshold_db_exp),
            "low_shelf_exp_ratio": (min_ratio_exp, max_ratio_exp),
            "low_shelf_at": (min_attack_ms_exp, max_attack_ms_exp),
            "low_shelf_rt": (min_release_ms_exp, max_release_ms_exp),
            
            "mid_band_comp_thresh": (min_threshold_db_comp, max_threshold_db_comp),
            "mid_band_comp_ratio": (min_ratio_comp, max_ratio_comp),
            "mid_band_exp_thresh": (min_threshold_db_exp, max_threshold_db_exp),
            "mid_band_exp_ratio": (min_ratio_exp, max_ratio_exp),
            "mid_band_at": (min_attack_ms_exp, max_attack_ms_exp),
            "mid_band_rt": (min_release_ms_exp, max_release_ms_exp),
            
            "high_shelf_comp_thresh": (min_threshold_db_comp, max_threshold_db_comp),
            "high_shelf_comp_ratio": (min_ratio_comp, max_ratio_comp),
            "high_shelf_exp_thresh": (min_threshold_db_exp, max_threshold_db_exp),
            "high_shelf_exp_ratio": (min_ratio_exp, max_ratio_exp),
            "high_shelf_at": (min_attack_ms_exp, max_attack_ms_exp),
            "high_shelf_rt": (min_release_ms_exp, max_release_ms_exp),
        }
        self.num_params = len(self.param_ranges)

class Delay(Processor):
    def __init__(
        self,
        sample_rate: int,
        min_delay_ms: float = 0.0,
        max_delay_ms: float = 24.0,
        min_wet: float = 0.1, 
        max_wet: float = 0.7
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.process_fn = delay
        self.param_ranges = {
            "delay_samples": (int(self.sample_rate * min_delay_ms), int(self.sample_rate * max_delay_ms)),
            "wet": (min_wet, max_wet)
        }
        self.num_params = len(self.param_ranges)

class Imager(Processor):
    def __init__(
        self,
        sample_rate: int,
        min_width: float = 0.0,
        max_width: float = 1.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.process_fn = stereo_widener
        self.param_ranges = {
            "width": (min_width, max_width),
        }
        self.num_params = len(self.param_ranges)

class Reverb(Processor):
    def __init__(
        self,
        sample_rate: int,
        device: str = 'cuda',
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.device = device
        self.process_fn = noise_shaped_reverberation
        self.param_ranges = {
            "band0_gain": (0.0, 1.0),
            "band1_gain": (0.0, 1.0),
            "band2_gain": (0.0, 1.0),
            "band3_gain": (0.0, 1.0),
            "band4_gain": (0.0, 1.0),
            "band5_gain": (0.0, 1.0),
            "band6_gain": (0.0, 1.0),
            "band7_gain": (0.0, 1.0),
            "band8_gain": (0.0, 1.0),
            "band9_gain": (0.0, 1.0),
            "band10_gain": (0.0, 1.0),
            "band11_gain": (0.0, 1.0),
            "band0_decay": (0.0, 1.0),
            "band1_decay": (0.0, 1.0),
            "band2_decay": (0.0, 1.0),
            "band3_decay": (0.0, 1.0),
            "band4_decay": (0.0, 1.0),
            "band5_decay": (0.0, 1.0),
            "band6_decay": (0.0, 1.0),
            "band7_decay": (0.0, 1.0),
            "band8_decay": (0.0, 1.0),
            "band9_decay": (0.0, 1.0),
            "band10_decay": (0.0, 1.0),
            "band11_decay": (0.0, 1.0),
            "mix": (0.0, 1.0),
        }
        self.num_params = len(self.param_ranges)

    def process_normalized(self, x: torch.Tensor, param_tensor: torch.Tensor):
        
        # extract parameters from tensor
        param_dict = self.extract_param_dict(param_tensor)
        
        # denormalize parameters to full range
        denorm_param_dict = self.denormalize_param_dict(param_dict)

        # now process audio with denormalized parameters
        y = self.process_fn(
            x,
            self.sample_rate,
            **denorm_param_dict,
        )

        return y
    
