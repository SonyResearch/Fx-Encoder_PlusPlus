import torch
import laion_clap
import logging
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)


    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x

class CLAP_AUDIO_ENCODER(torch.nn.Module):
    def __init__(self, pretrained: bool = True, frozen: bool = False) -> None:
        super().__init__()
        self.pretrained = pretrained
        self.frozen = frozen

        # load the model
        self.encoder = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-tiny', tmodel='roberta')
        if self.pretrained:
            self.encoder.load_ckpt()  # download the default pretrained checkpoint.

        self.embed_dim = 512

    def forward(self, x: torch.Tensor):
        if self.frozen:
            with torch.no_grad():
                embed = self.encoder.get_audio_embedding_from_data(
                    x=x, use_tensor=True
                )
        else:
            embed = self.encoder.get_audio_embedding_from_data(
                    x=x, use_tensor=True
                )

        return embed
            
class CLAP_TEXT_ENCODER(torch.nn.Module):
    def __init__(self, pretrained: bool = True, frozen: bool = False) -> None:
        super().__init__()
        self.pretrained = pretrained
        self.frozen = frozen

        # load the model
        self.encoder = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-tiny', tmodel='roberta')
        if self.pretrained:
            self.encoder.load_ckpt()  # download the default pretrained checkpoint.

        self.embed_dim = 512

    def forward(self, x):
        if self.frozen:
            with torch.no_grad():
                embed = self.encoder.get_text_embedding(
                    x=x, use_tensor=True
                )
        else:
            embed = self.encoder.get_text_embedding(
                    x=x, use_tensor=True
                )

        return embed 
    
# > ================ Proposed =================== <
class MixtureFxEncoder(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, enable_fusion=False, fusion_type='None'):
        super().__init__()
        self.enable_fusion = enable_fusion
        self.fusion_type = fusion_type

        window = "hann"
        center = True
        pad_mode = "reflect"
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.input_norm = "minmax"
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True,
        )

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=2, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc_1 = nn.Linear(2048, 2048, bias=True)

        self.init_weight()
        
    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc_1)

    def forward(self, x):
        """
        Input: (batch_size, 2, data_length)
        """
        batch_size, chs, seq_len = x.size()

        # move to batch dim
        x = x.view(batch_size * chs, seq_len)
        
        # extract logmel features
        x = self.spectrogram_extractor(x)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        if self.input_norm == "batchnorm":
            # this normalizes over mel bins which is problematic for equalization
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)
        elif self.input_norm == "minmax":
            x = x.clamp(-80, 40.0)  # clamp the logmels between -80 and 40
            x = (x + 80) / 120  # normalize the logmels between 0 and 1
            x = (x * 2) - 1  # normalize the logmels between -1 and 1
        elif self.input_norm == "none":
            pass
        else:
            raise ValueError(f"Invalid input_norm: {self.input_norm}")

        x = x.view(batch_size, chs, x.size(-2), x.size(-1))
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.relu_(self.fc_1(x))
        embedding = x

        output_dict = {
            'embedding': embedding, 
        }

        return output_dict

def create_MixtureFxEncoder():
    model = MixtureFxEncoder(
        sample_rate = 44100, #audio_cfg.sample_rate,
        window_size = 2048, #audio_cfg.window_size,
        hop_size = 512, #audio_cfg.hop_size,
        mel_bins = 64, #audio_cfg.mel_bins,
        fmin = 50, #audio_cfg.fmin,
        fmax = 18000, #audio_cfg.fmax,
    )
    return model

class MLPLayers(nn.Module):
    def __init__(self, units=[512, 512, 512], nonlin=nn.ReLU(), dropout=0.1):
        super(MLPLayers, self).__init__()
        self.nonlin = nonlin
        self.dropout = dropout

        sequence = []
        for u0, u1 in zip(units[:-1], units[1:]):
            sequence.append(nn.Linear(u0, u1))
            sequence.append(self.nonlin)
            sequence.append(nn.Dropout(self.dropout))
        sequence = sequence[:-2]

        self.sequential = nn.Sequential(*sequence)

    def forward(self, X):
        X = self.sequential(X)
        return X

class BernoulliDynamicDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.p_min = 0.75
        self.p_max = 0.95
    
    def get_random_dropout_rate(self):
        return torch.empty(1).uniform_(self.p_min, self.p_max).item()
    
    def forward(self, x):
        if self.training:
            p = self.get_random_dropout_rate()
            mask = torch.bernoulli(torch.full_like(x, 1-p))
            return x * mask / (1 - p)
        return x

class AudioExtracter(nn.Module):
    def __init__(self, fx_embedding_dim=128, clap_embedding_dim=512):
        super().__init__()
        
        # Simple fusion network
        self.fusion = nn.Sequential(
            nn.Linear(fx_embedding_dim+clap_embedding_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128),
        )
    
    def forward(self, mixture_emb, query_emb):
        # Concatenate and project
        x = torch.cat([mixture_emb, query_emb], dim=-1)  # [B, 2D]
        stem_emb = self.fusion(x)  # [B, D]
        
        return stem_emb

@dataclass
class AudioCfg:
    model_type: str = "PANN"
    model_name: str = "Cnn14"
    sample_rate: int = 44100
    # Param
    audio_length: int = 1024
    window_size: int = 1024
    hop_size: int = 1024
    fmin: int = 50
    fmax: int = 14000
    mel_bins: int = 64
    clip_samples: int = 441000
    class_num: int = 527
    condition_dim: int = 512

class FxEncoderPlusPlus(nn.Module): 
    def __init__(
        self,
        embed_dim: int = 2048,
        mixture_cfg: AudioCfg = None,
        enable_fusion: bool = False,
        fusion_type: str = 'None',
        joint_embed_shape: int = 128,
        mlp_act: str = 'relu',
        audio_clap_module: bool = True,
        text_clap_module: bool = False,
        extractor_module: bool = True,
    ):
        super().__init__()
        
        self.mixture_cfg = mixture_cfg
        self.enable_fusion = enable_fusion
        self.fusion_type = fusion_type
        self.joint_embed_shape = joint_embed_shape
        self.mlp_act = mlp_act

        if mlp_act == 'relu':
            mlp_act_layer = nn.ReLU()
        elif mlp_act == 'gelu':
            mlp_act_layer = nn.GELU()
        else:
            raise NotImplementedError

        # > ========================= FX Encoder ========================= <
        self.fx_encoder = create_MixtureFxEncoder()
        self.fx_encoder_transform = MLPLayers(units=[self.joint_embed_shape, self.joint_embed_shape, self.joint_embed_shape], dropout=0.1)
        
        self.fx_encoder_projection = nn.Sequential(
            nn.Linear(embed_dim, self.joint_embed_shape),
            mlp_act_layer,
            nn.Linear(self.joint_embed_shape, self.joint_embed_shape)
        )
        
        self.logit_scale_m = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_t = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        if audio_clap_module:
            # Freeze all layers
            # print("Loading CLAP Audio Model")
            self.audio_clap_model = CLAP_AUDIO_ENCODER(pretrained=True, frozen=True)
            for param in self.audio_clap_model.parameters():
                param.requires_grad = False
            self.clap_dropout = BernoulliDynamicDropout()
        if text_clap_module:
            # Freeze all layers
            print("Loading CLAP Text Model")
            self.text_clap_model = CLAP_TEXT_ENCODER(pretrained=True, frozen=True)
            for param in self.text_clap_model.parameters():
                param.requires_grad = False
            
        if extractor_module:
            # extractor 
            self.extractor = AudioExtracter()

        self.use_audio_clap_module = audio_clap_module
        self.use_text_clap_module = text_clap_module 
        self.use_extractor_module = extractor_module

    def get_fx_embedding(self, x):
        fx_emb = self.fx_encoder(x) 
        fx_emb = self.fx_encoder_projection(fx_emb["embedding"])
        fx_emb = F.normalize(fx_emb, dim=-1)
        return fx_emb
    
    def get_fx_embedding_pre_proj(self, x, normalized = True):
        fx_emb = self.fx_encoder(x)["embedding"]
        if normalized:
            fx_emb = F.normalize(fx_emb, dim=-1)            
        return fx_emb
    
    def get_fx_embedding_by_audio_query(self, x, audio_query):
        # mixture fx embedding 
        fx_mixture_emb = self.fx_encoder(x) 
        fx_mixture_emb = self.fx_encoder_projection(fx_mixture_emb["embedding"])
        fx_mixture_emb = F.normalize(fx_mixture_emb, dim=-1)
        
        # stem fx embedding 
        query_content_embeded = self.audio_clap_model(torch.mean(audio_query, dim=1))
        fx_stem_emb = self.extractor(fx_mixture_emb, query_content_embeded)
        fx_stem_emb = F.normalize(fx_stem_emb, dim=-1)
        return fx_mixture_emb, fx_stem_emb
    
    def get_fx_embedding_by_text_query(self, x, text_query):
        # mixture fx embedding 
        fx_mixture_emb = self.fx_encoder(x) 
        fx_mixture_emb = self.fx_encoder_projection(fx_mixture_emb["embedding"])
        fx_mixture_emb = F.normalize(fx_mixture_emb, dim=-1)

        # stem fx embedding 
        query_embeded = self.text_clap_model(text_query)
        fx_stem_emb = self.extractor(fx_mixture_emb, query_embeded)
        fx_stem_emb = F.normalize(fx_stem_emb, dim=-1)
        return fx_mixture_emb, fx_stem_emb
    

    def forward(
        self, 
        mixture_a, 
        mixture_b, 
        stem_a, 
        query_stem, 
        device = None
    ):

        if device is None:
            if mixture_a is not None:
                device = mixture_a.device
            elif mixture_b is not None:
                device = mixture_b.device
        if mixture_a is None and mixture_b is None:
            # a hack to get the logit scale
            return self.logit_scale_m.exp(), self.logit_scale_t.exp()
        
        # ======== Global ========
        mixture_a_features = self.fx_encoder_projection(
            self.fx_encoder(mixture_a)["embedding"]
        )
        mixture_a_features = F.normalize(mixture_a_features, dim=-1)
        
        mixture_b_features = self.fx_encoder_projection(
            self.fx_encoder(mixture_b)["embedding"]
        )
        mixture_b_features = F.normalize(mixture_b_features, dim=-1)

        mixture_a_features_mlp = self.fx_encoder_transform(mixture_a_features)
        mixture_b_features_mlp = self.fx_encoder_transform(mixture_b_features)
        
        # ======= Local ========
        stem_a_features = self.fx_encoder_projection(
            self.fx_encoder(stem_a)["embedding"]
        )
        stem_a_features = F.normalize(stem_a_features, dim=-1)
        
        if self.use_audio_clap_module and self.use_extractor_module:
            query_stem_content_embeded = self.clap_dropout(
                self.audio_clap_model(
                    torch.mean(query_stem, dim=1)
                )
            )
            extracted_stem_a_features = self.extractor(mixture_a_features, query_stem_content_embeded)
            extracted_stem_a_features = F.normalize(extracted_stem_a_features, dim=-1)
        elif self.use_text_clap_module and self.use_extractor_module:
            query_stem_content_embeded = self.text_clap_model(query_stem)
            extracted_stem_a_features = self.extractor(mixture_a_features, query_stem_content_embeded)
            extracted_stem_a_features = F.normalize(extracted_stem_a_features, dim=-1)
        
        return (
            mixture_a_features, # global
            mixture_b_features, # global 
            stem_a_features, # local 
            extracted_stem_a_features, # local 
            mixture_a_features_mlp,
            mixture_b_features_mlp,
            self.logit_scale_m.exp(),
            self.logit_scale_t.exp(),
        )

    def get_logit_scale(self):
        return self.logit_scale_m.exp(), self.logit_scale_t.exp()

def load_model(model_path, device):
    
    model = FxEncoderPlusPlus(
        embed_dim = 2048, 
        audio_clap_module = True, 
        extractor_module = True
    )

    # load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "epoch" in checkpoint:
        # resuming a train checkpoint w/ epoch and optimizer state
        start_epoch = checkpoint["epoch"]
        sd = checkpoint["state_dict"]
        if next(iter(sd.items()))[0].startswith(
                "module"
        ):
            sd = {k[len("module."):]: v for k, v in sd.items()}
        model.load_state_dict(sd)
        logging.info(
            f"=> resuming checkpoint '{model_path}' (epoch {start_epoch})"
        )
    else:
        # loading a bare (model only) checkpoint for fine-tune or evaluation
        model.load_state_dict(checkpoint)
        start_epoch = 0
    
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

