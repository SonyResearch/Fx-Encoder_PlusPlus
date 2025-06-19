import torch 
import torch.nn as nn 
import time

from fx_chain.constants import *
from fx_chain.fx_processors import * 
from fx_chain.constants import * 

# > ====== Setup processor based on config ====== <
def select_processors(processor_name, sample_rate):
    if processor_name == 'eq':
        cur_fx_module = dasp_pytorch.ParametricEQ(
            sample_rate=sample_rate, 
            min_gain_db = -12.0, 
            max_gain_db = 12.0, 
            min_q_factor = 0.5, 
            max_q_factor = 5.0
        )
    elif processor_name == 'distortion':
        cur_fx_module = Distortion(
            sample_rate=sample_rate, 
            min_gain_db = 0.0, 
            max_gain_db = 8.0
        )
    elif processor_name == 'multiband_comp':
        cur_fx_module = Multiband_Compressor(
            sample_rate=sample_rate,
            min_threshold_db_comp = -30.0,
            max_threshold_db_comp = -5.0,
            min_ratio_comp = 1.5,
            max_ratio_comp = 6.0,
            min_attack_ms_comp = 1.0,
            max_attack_ms_comp = 20.0,
            min_release_ms_comp = 20.0,
            max_release_ms_comp = 500.0,
            min_threshold_db_exp = -30.0,
            max_threshold_db_exp = -5.0,
            min_ratio_exp = 0.0+EPS,
            max_ratio_exp = 1.0-EPS,
            min_attack_ms_exp = 1.0,
            max_attack_ms_exp = 20.0,
            min_release_ms_exp = 20.0,
            max_release_ms_exp = 500.0,
        )
    elif processor_name == 'gain':
        cur_fx_module = dasp_pytorch.Gain(
            sample_rate=sample_rate, 
            min_gain_db = 6.0, 
            max_gain_db = 12.0
        )
    elif processor_name == 'limiter':
        cur_fx_module = Limiter(
            sample_rate=sample_rate,
            min_threshold_db = -20.0,
            max_threshold_db = 0.0-EPS,
            min_attack_ms = 0.1,
            max_attack_ms = 5.0,
            min_release_ms = 20.0,
            max_release_ms = 1000.0,
        )
    elif processor_name == 'imager':
        cur_fx_module = Imager(
            sample_rate=sample_rate,
            min_width=0.0,
            max_width=1.0
        )
    elif processor_name == 'delay':
        cur_fx_module = Delay(
            sample_rate=sample_rate,
            min_delay_ms = 0.0,
            max_delay_ms = 300.0,
            min_wet = 0.1, 
            max_wet = 0.7
        )
    elif processor_name == 'reverb':
        cur_fx_module = Reverb(
            sample_rate=sample_rate,
        )
    return cur_fx_module

def generate_all_processors(sample_rate):
    FX_PROCESSORS = {}
    TOTAL_NUM_PARAMS = 0
    for cur_fx in ALL_PROCESSORS:
        FX_PROCESSORS[cur_fx] = select_processors(cur_fx, sample_rate)
    TOTAL_NUM_PARAMS = sum([FX_PROCESSORS[cur_fx].num_params for cur_fx in FX_PROCESSORS])
    return FX_PROCESSORS, TOTAL_NUM_PARAMS

class Random_FX_Chain(nn.Module):
    def __init__(
        self, 
        sample_rate, 
        device, 
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.device = device
        # Initialize all possible processors
        self.fx_processors, self.total_num_param = generate_all_processors(sample_rate)

        # Fx Information (probability, indices)
        self.fx_indices = {fx_name: indx for indx, fx_name in enumerate(ALL_PROCESSORS)}
        self.fx_prob = [0] * len(self.fx_indices)
        self.param_range = [(0, 0)] * len(self.fx_indices)

        start = 0
        for fx_name in self.fx_indices:
            end = start + self.fx_processors[fx_name].num_params
            self.param_range[self.fx_indices[fx_name]] = (start, end)
            start = end
            self.fx_prob[self.fx_indices[fx_name]] = DEFAULT_FX_PROB[fx_name]
        self.fx_prob = torch.tensor(self.fx_prob).to(self.device)
        
    def update_fx_prob(self, fx_prob): # if our model is too focus on specific effect, do fx probability scheduling
        self.fx_prob = [0] * len(self.fx_indices)
        for fx_name in self.fx_indices:
            self.fx_prob[self.fx_indices[fx_name]] = fx_prob[fx_name]
        self.fx_prob = torch.tensor(self.fx_prob).to(self.device)

    def sample_fx(self): # randomly sample one effect from the probability distribution
        fx_index = torch.multinomial(self.fx_prob, 1).item()
        selected_fx = list(self.fx_indices.keys())[list(self.fx_indices.values()).index(fx_index)]
        return selected_fx
    
    # network forward operation 
    def forward(
        self, 
        x, 
        nn_param = None, 
        activate = None,
        processors_order = None
    ):
        # nn_param: parameters for each effect 
        # activate: binary mask that controls which effects are actually applied
        flag = True
        detect_nan = True
        while flag:
            batch_size = x.shape[0]
            # if not provided, randomly generate one possible fx parameters 
            if nn_param is None:
                nn_param = torch.rand((batch_size, self.total_num_param)).to(self.device)
            else: # if provided, examine the shape 
                assert nn_param.shape[0] == batch_size and nn_param.shape[1] == self.total_num_param
            
            if activate is None:
                activate = torch.bernoulli(self.fx_prob.unsqueeze(0).expand(batch_size, -1))#.to(self.device)
            activate = activate.to(self.device)
            
            for cur_fx in processors_order:
                param_start, param_end = self.param_range[self.fx_indices[cur_fx]]
                cur_input_param = nn_param[:, param_start:param_end]
                x_processed = self.fx_processors[cur_fx].process_normalized(x, cur_input_param)
                if torch.isnan(x_processed).any():
                    break
                cur_mask = activate[:, self.fx_indices[cur_fx]].unsqueeze(-1).unsqueeze(-1).int() ##
                x = x_processed * cur_mask + x * ~cur_mask
            
            detect_nan = False
            if detect_nan == False:
                flag = False
        return x, nn_param, activate

# different processors vs different processors
class Random_Single_FX_Chain(nn.Module):
    def __init__(
        self, 
        sample_rate, 
        device, 
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.device = device
        # Initialize all possible processors
        self.fx_processors, self.total_num_param = generate_all_processors(sample_rate, device)

        self.fx_indices = {element: indx for indx, element in enumerate(ALL_PROCESSORS)}
        self.idx_to_fx = {indx: element for element, indx in self.fx_indices.items()}
        
        self.param_range = [(0, 0)] * len(self.fx_indices)
        start = 0
        for k in self.fx_indices:
            end = start + self.fx_processors[k].num_params
            self.param_range[self.fx_indices[k]] = (start, end)
            start = end

    def get_fx_mapping(self):
        return self.fx_indices.copy()

    def get_num_fx(self):
        return len(self.fx_indices)

    def idx_to_name(self, indices):
        if torch.is_tensor(indices):
            return [self.idx_to_fx[idx.item()] for idx in indices]
        else:
            return self.idx_to_fx[indices]

    def name_to_idx(self, names):
        if isinstance(names, list):
            return [self.fx_indices[name] for name in names]
        else:
            return self.fx_indices[names]

    def forward(
        self, 
        x, 
        nn_param = None, 
        activate = None,
        labels = None
    ):
        flag = True
        detect_nan = False
        while flag:
            batch_size = x.shape[0]
            if labels is None:
                labels = torch.randint(0, len(self.fx_indices), (batch_size,), device=self.device) # [batch_size]

            if nn_param is None:
                nn_param = torch.rand((batch_size, self.total_num_param)).to(self.device)
            else:
                assert nn_param.shape[0] == batch_size and nn_param.shape[1] == self.total_num_param
            
            if activate is None:
                activate = torch.zeros((batch_size, len(self.fx_indices)), device=self.device)
                activate[torch.arange(batch_size), labels] = 1

            x_processed = x.clone()
            
            for fx_name, fx_idx in self.fx_indices.items():
                mask = (labels == fx_idx)
                if not mask.any():
                    continue
                    
                param_start, param_end = self.param_range[fx_idx]
                current_batch = x[mask]
                current_params = nn_param[mask, param_start:param_end]
                
                processed = self.fx_processors[fx_name].process_normalized(current_batch, current_params)
                
                if torch.isnan(processed).any():
                    detect_nan = True
                    break
                    
                x_processed[mask] = processed

            if detect_nan == False:
                flag = False
                x = x_processed

        return x, nn_param, activate, labels




