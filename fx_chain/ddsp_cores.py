import torch 
import dasp_pytorch
from .dsp_signal import * 
import torch.nn.functional as F
from numba import config
config.CUDA_ENABLE_PYNVJITLINK = 1
config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
import torchcomp
from functools import partial

# > ======= Gain ======= <
def gain(x, sample_rate, gain_db):
    bs, chs, seq_len = x.size()

    gain_db = gain_db.view(bs, 1, 1)
    # convert gain from db to linear
    gain_lin = 10 ** (gain_db.repeat(1, chs, 1) / 20.0)
    return x * gain_lin

# > ======= Distortion ======= <
def distortion(x, sample_rate, drive_db):
    bs, chs, seq_len = x.size()
    drive_db = drive_db.view(bs, 1, 1)
    return torch.tanh(x * (10 ** (drive_db.repeat(1, chs, 1) / 20.0)))

# > ======= EQ ======= <
def parametric_eq(
    x,
    sample_rate,
    low_shelf_gain_db,
    low_shelf_cutoff_freq,
    low_shelf_q_factor,
    band0_gain_db,
    band0_cutoff_freq,
    band0_q_factor ,
    band1_gain_db ,
    band1_cutoff_freq ,
    band1_q_factor ,
    band2_gain_db ,
    band2_cutoff_freq ,
    band2_q_factor ,
    band3_gain_db ,
    band3_cutoff_freq ,
    band3_q_factor ,
    high_shelf_gain_db ,
    high_shelf_cutoff_freq ,
    high_shelf_q_factor,):

    bs, chs, seq_len = x.size()

    # reshape to move everything to batch dim
    low_shelf_gain_db = low_shelf_gain_db.view(-1, 1, 1)
    low_shelf_cutoff_freq = low_shelf_cutoff_freq.view(-1, 1, 1)
    low_shelf_q_factor = low_shelf_q_factor.view(-1, 1, 1)
    band0_gain_db = band0_gain_db.view(-1, 1, 1)
    band0_cutoff_freq = band0_cutoff_freq.view(-1, 1, 1)
    band0_q_factor = band0_q_factor.view(-1, 1, 1)
    band1_gain_db = band1_gain_db.view(-1, 1, 1)
    band1_cutoff_freq = band1_cutoff_freq.view(-1, 1, 1)
    band1_q_factor = band1_q_factor.view(-1, 1, 1)
    band2_gain_db = band2_gain_db.view(-1, 1, 1)
    band2_cutoff_freq = band2_cutoff_freq.view(-1, 1, 1)
    band2_q_factor = band2_q_factor.view(-1, 1, 1)
    band3_gain_db = band3_gain_db.view(-1, 1, 1)
    band3_cutoff_freq = band3_cutoff_freq.view(-1, 1, 1)
    band3_q_factor = band3_q_factor.view(-1, 1, 1)
    high_shelf_gain_db = high_shelf_gain_db.view(-1, 1, 1)
    high_shelf_cutoff_freq = high_shelf_cutoff_freq.view(-1, 1, 1)
    high_shelf_q_factor = high_shelf_q_factor.view(-1, 1, 1)

    eff_bs = x.size(0)

    # six second order sections
    sos = torch.zeros(eff_bs, 6, 6).type_as(low_shelf_gain_db)
    # ------------ low shelf ------------
    b, a = biquad(
        low_shelf_gain_db,
        low_shelf_cutoff_freq,
        low_shelf_q_factor,
        sample_rate,
        "low_shelf",
    )
    sos[:, 0, :] = torch.cat((b, a), dim=-1)
    # ------------ band0 ------------
    b, a = biquad(
        band0_gain_db,
        band0_cutoff_freq,
        band0_q_factor,
        sample_rate,
        "peaking",
    )
    sos[:, 1, :] = torch.cat((b, a), dim=-1)
    # ------------ band1 ------------
    b, a = biquad(
        band1_gain_db,
        band1_cutoff_freq,
        band1_q_factor,
        sample_rate,
        "peaking",
    )
    sos[:, 2, :] = torch.cat((b, a), dim=-1)
    # ------------ band2 ------------
    b, a = biquad(
        band2_gain_db,
        band2_cutoff_freq,
        band2_q_factor,
        sample_rate,
        "peaking",
    )
    sos[:, 3, :] = torch.cat((b, a), dim=-1)
    # ------------ band3 ------------
    b, a = biquad(
        band3_gain_db,
        band3_cutoff_freq,
        band3_q_factor,
        sample_rate,
        "peaking",
    )
    sos[:, 4, :] = torch.cat((b, a), dim=-1)
    # ------------ high shelf ------------
    b, a = biquad(
        high_shelf_gain_db,
        high_shelf_cutoff_freq,
        high_shelf_q_factor,
        sample_rate,
        "high_shelf",
    )
    sos[:, 5, :] = torch.cat((b, a), dim=-1)
    x_out = sosfilt_via_fsm(sos, x)
    # move channels back
    x_out = x_out.view(bs, chs, seq_len)

    return x_out

# > ======= Compressor ======= <
def compressor(
    x,
    sample_rate,
    threshold_db,
    ratio  ,
    attack_ms  ,
    release_ms  ,
    knee_db  ,
    makeup_gain_db  ,
    eps = 1e-8,
    lookahead_samples= 0,):
    bs, chs, seq_len = x.size()  # check shape

    # if multiple channels are present create sum side-chain
    x_side = x.sum(dim=1, keepdim=True)
    x_side = x_side.view(-1, 1, seq_len)
    threshold_db = threshold_db.view(-1, 1, 1)
    ratio = ratio.view(-1, 1, 1)
    attack_ms = attack_ms.view(-1, 1, 1)
    release_ms = release_ms.view(-1, 1, 1)
    knee_db = knee_db.view(-1, 1, 1)
    makeup_gain_db = makeup_gain_db.view(-1, 1, 1)
    eff_bs = x_side.size(0)

    # compute time constants
    normalized_attack_time = sample_rate * (attack_ms / 1e3)
    # normalized_release_time = sample_rate * (release_ms / 1e3)
    constant = torch.tensor([9.0]).type_as(attack_ms)
    alpha_A = torch.exp(-torch.log(constant) / normalized_attack_time)
    # alpha_R = torch.exp(-torch.log(constant) / normalized_release_time)
    # note that release time constant is not used in the smoothing filter

    # compute energy in db
    x_db = 20 * torch.log10(torch.abs(x_side).clamp(eps))

    # static characteristic with soft knee
    x_sc = x_db.clone()#.float()
    #x_sc = x_sc.float()
    # when signal is less than (T - W/2) leave as x_db

    # when signal is at the threshold engage knee
    idx1 = x_db >= (threshold_db - (knee_db / 2))
    idx2 = x_db <= (threshold_db + (knee_db / 2))
    idx = torch.logical_and(idx1, idx2)
    x_sc_below = x_db + ((1 / ratio) - 1) * (
        (x_db - threshold_db + (knee_db / 2)) ** 2
    ) / (2 * knee_db)
    x_sc = x_sc.double()
    x_sc_below = x_sc_below.double()
    x_sc[idx] = x_sc_below[idx]

    # when signal is above threshold linear response
    idx = x_db > (threshold_db + (knee_db / 2))
    x_sc_above = threshold_db + ((x_db - threshold_db) / ratio)
    x_sc[idx] = x_sc_above[idx]

    # output of gain computer
    g_c = x_sc - x_db

    # design attack/release smoothing filter
    b = torch.cat(
        [(1 - alpha_A), torch.zeros(eff_bs, 1, 1).type_as(alpha_A)],
        dim=-1,
    ).squeeze(1)
    a = torch.cat(
        [torch.ones(eff_bs, 1, 1).type_as(alpha_A), -alpha_A],
        dim=-1,
    ).squeeze(1)
    g_c_attack = lfilter_via_fsm(g_c, b, a)

    # look-ahead by delaying the input signal in relation to gain reduction
    if lookahead_samples > 0:
        x = torch.roll(x, lookahead_samples, dims=-1)
        x[:, :, :lookahead_samples] = 0

    # add makeup gain in db
    g_s = g_c_attack + makeup_gain_db

    # convert db gains back to linear
    g_lin = 10 ** (g_s / 20.0)

    # apply time-varying gain and makeup gain
    y = x * g_lin

    # move channels back to the channel dimension
    y = y.view(bs, chs, seq_len)
    
    return y

# > ======= Reverb ======= <
# reverb from christian
def noise_shaped_reverberation(
    x  ,
    sample_rate,
    band0_gain,
    band1_gain,
    band2_gain,
    band3_gain,
    band4_gain,
    band5_gain,
    band6_gain,
    band7_gain,
    band8_gain,
    band9_gain,
    band10_gain,
    band11_gain,
    band0_decay,
    band1_decay,
    band2_decay,
    band3_decay,
    band4_decay,
    band5_decay,
    band6_decay,
    band7_decay,
    band8_decay,
    band9_decay,
    band10_decay,
    band11_decay,
    mix,
    num_samples= 65536,
    num_bandpass_taps= 1023,):
    assert num_bandpass_taps % 2 == 1, "num_bandpass_taps must be odd"

    bs, chs, seq_len = x.size()
    assert chs <= 2, "only mono/stereo signals are supported"

    # if mono copy to stereo
    if chs == 1:
        x = x.repeat(1, 2, 1)
        chs = 2

    # stack gains and decays into a single tensor
    band_gains = torch.stack(
        [
            band0_gain,
            band1_gain,
            band2_gain,
            band3_gain,
            band4_gain,
            band5_gain,
            band6_gain,
            band7_gain,
            band8_gain,
            band9_gain,
            band10_gain,
            band11_gain,
        ],
        dim=1,
    )
    band_gains = band_gains.unsqueeze(-1)

    band_decays = torch.stack(
        [
            band0_decay,
            band1_decay,
            band2_decay,
            band3_decay,
            band4_decay,
            band5_decay,
            band6_decay,
            band7_decay,
            band8_decay,
            band9_decay,
            band10_decay,
            band11_decay,
        ],
        dim=1,
    )
    band_decays = band_decays.unsqueeze(-1)

    # create the octave band filterbank filters
    filters = octave_band_filterbank(num_bandpass_taps, sample_rate)
    filters = filters.type_as(x)
    num_bands = filters.shape[0]

    # reshape gain, decay, and mix parameters
    band_gains = band_gains.view(bs, 1, num_bands, 1)
    band_decays = band_decays.view(bs, 1, num_bands, 1)
    mix = mix.view(bs, 1, 1)

    # generate white noise for IR generation
    pad_size = num_bandpass_taps - 1
    wn = torch.randn(bs * 2, num_bands, num_samples + pad_size).type_as(x)

    # filter white noise signals with each bandpass filter
    wn_filt = torch.nn.functional.conv1d(
        wn,
        filters,
        groups=num_bands,
        # padding=self.num_taps -1,
    )
    # shape: (bs * 2, num_bands, num_samples)
    wn_filt = wn_filt.view(bs, 2, num_bands, num_samples)

    # apply bandwise decay parameters (envelope)
    t = torch.linspace(0, 1, steps=num_samples).type_as(x)  # timesteps
    band_decays = (band_decays * 10.0) + 1.0
    env = torch.exp(-band_decays * t.view(1, 1, 1, -1))
    wn_filt *= env * band_gains

    # sum signals to create impulse shape: bs, 2, 1, num_samp
    w_filt_sum = wn_filt.mean(2, keepdim=True)

    # apply impulse response for each batch item (vectorized)
    x_pad = torch.nn.functional.pad(x, (num_samples - 1, 0))
    vconv1d = torch.vmap(partial(torch.nn.functional.conv1d, groups=2), in_dims=0)
    y = vconv1d(x_pad, torch.flip(w_filt_sum, dims=[-1]))

    # create a wet/dry mix
    y = (1 - mix) * x + mix * y

    return y

# > ======= Stereo imager ======= <
def stereo_widener(x, sample_rate, width):
    bs, chs, seq_len = x.size()
    assert chs == 2, "Input tensor must have shape (bs, 2, seq_len)"

    sqrt2 = np.sqrt(2)
    mid = (x[..., 0, :] + x[..., 1, :]) / sqrt2
    side = (x[..., 0, :] - x[..., 1, :]) / sqrt2

    _width = (1 - width).unsqueeze(-1)
    _side_width = width.unsqueeze(-1)
    mid *= 2 * _width
    side *= 2 * _side_width
    # covert back to stereo
    left = (mid + side) / sqrt2
    right = (mid - side) / sqrt2

    return torch.stack((left, right), dim=-2)

# > ======= Panning ======= <
def panning(x, sample_rate, gain_db_l, gain_db_r):
    bs, chs, seq_len = x.size()
    gain_db_l = gain_db_l.view(bs, 1, 1)
    gain_db_r = gain_db_r.view(bs, 1, 1)
    # convert gain from db to linear
    gain_lin_l = 10 ** (gain_db_l.repeat(1, 1, 1) / 20.0)
    gain_lin_r = 10 ** (gain_db_r.repeat(1, 1, 1) / 20.0)
    gain = torch.cat((gain_lin_l, gain_lin_r), dim=1)
    y = x * gain 
    return y

# > ======= Multi-band Compressor ======= <
def linkwitz_riley_4th_order(
    x: torch.Tensor, 
    cutoff_freq: torch.Tensor,
    sample_rate: float, 
    filter_type: str):
    q_factor = torch.ones(cutoff_freq.shape) / torch.sqrt(torch.tensor([2.0]))
    gain_db = torch.zeros(cutoff_freq.shape)
    q_factor = q_factor.to(x.device)
    gain_db = gain_db.to(x.device)

    b, a = dasp_pytorch.signal.biquad(
        gain_db,
        cutoff_freq,
        q_factor,
        sample_rate,
        filter_type
    )

    del gain_db
    del q_factor
    
    eff_bs = x.size(0)
    # six second order sections
    sos = torch.cat((b, a), dim=-1).unsqueeze(1)

    # apply filter twice to phase difference amounts of 360°
    x = dasp_pytorch.signal.sosfilt_via_fsm(sos, x)
    x_out = dasp_pytorch.signal.sosfilt_via_fsm(sos, x)

    return x_out

def multiband_compressor(
    x: torch.Tensor,
    sample_rate: float,

    low_cutoff: torch.Tensor,
    high_cutoff: torch.Tensor, 
    parallel_weight_factor: torch.Tensor,

    low_shelf_comp_thresh: torch.Tensor,
    low_shelf_comp_ratio: torch.Tensor,
    low_shelf_exp_thresh: torch.Tensor,
    low_shelf_exp_ratio: torch.Tensor,
    low_shelf_at: torch.Tensor,
    low_shelf_rt: torch.Tensor,
    
    mid_band_comp_thresh: torch.Tensor,
    mid_band_comp_ratio: torch.Tensor,
    mid_band_exp_thresh: torch.Tensor,
    mid_band_exp_ratio: torch.Tensor,
    mid_band_at: torch.Tensor,
    mid_band_rt: torch.Tensor,
    
    high_shelf_comp_thresh: torch.Tensor,
    high_shelf_comp_ratio: torch.Tensor,
    high_shelf_exp_thresh: torch.Tensor,
    high_shelf_exp_ratio: torch.Tensor,
    high_shelf_at: torch.Tensor,
    high_shelf_rt: torch.Tensor,):
    """Multiband (Three-band) Compressor.

    Low-shelf -> Mid-band -> High-shelf

    Args:
        x (torch.Tensor): Time domain tensor with shape (bs, chs, seq_len)
        sample_rate (float): Audio sample rate.
        low_cutoff (torch.Tensor): Low-shelf filter cutoff frequency in Hz.
        high_cutoff (torch.Tensor): High-shelf filter cutoff frequency in Hz.
        low_shelf_comp_thresh (torch.Tensor): 
        low_shelf_comp_ratio (torch.Tensor): 
        low_shelf_exp_thresh (torch.Tensor): 
        low_shelf_exp_ratio (torch.Tensor): 
        low_shelf_at (torch.Tensor): 
        low_shelf_rt (torch.Tensor): 
        mid_band_comp_thresh (torch.Tensor): 
        mid_band_comp_ratio (torch.Tensor): 
        mid_band_exp_thresh (torch.Tensor): 
        mid_band_exp_ratio (torch.Tensor): 
        mid_band_at (torch.Tensor): 
        mid_band_rt (torch.Tensor): 
        high_shelf_comp_thresh (torch.Tensor): 
        high_shelf_comp_ratio (torch.Tensor): 
        high_shelf_exp_thresh (torch.Tensor): 
        high_shelf_exp_ratio (torch.Tensor): 
        high_shelf_at (torch.Tensor): 
        high_shelf_rt (torch.Tensor): 

    Returns:
        y (torch.Tensor): Filtered signal.
    """
    bs, chs, seq_len = x.size()

    low_cutoff = low_cutoff.view(-1, 1, 1)
    high_cutoff = high_cutoff.view(-1, 1, 1) 
    parallel_weight_factor = parallel_weight_factor.view(-1, 1, 1)

    eff_bs = x.size(0)

    ''' cross over filter '''
    # Low-shelf band (low frequencies)
    low_band = linkwitz_riley_4th_order(x, low_cutoff, sample_rate, filter_type="low_pass")
    # High-shelf band (high frequencies)
    high_band = linkwitz_riley_4th_order(x, high_cutoff, sample_rate, filter_type="high_pass")
    # Mid-band (band-pass)
    mid_band = x - low_band - high_band  # Subtract low and high bands from original signal

    ''' compressor '''
    try:
        x_out_low = low_band * torchcomp.compexp_gain(low_band.sum(axis=1).abs(),
                                            comp_thresh=low_shelf_comp_thresh, \
                                            comp_ratio=low_shelf_comp_ratio, \
                                            exp_thresh=low_shelf_exp_thresh, \
                                            exp_ratio=low_shelf_exp_ratio, \
                                            at=torchcomp.ms2coef(low_shelf_at, sample_rate), \
                                            rt=torchcomp.ms2coef(low_shelf_rt, sample_rate)).unsqueeze(1)
        # dasp_pytorch.functional.compressor(
        #                                 low_band,
        #                                 sample_rate: float,
        #                                 threshold_db: torch.Tensor,
        #                                 ratio: torch.Tensor,
        #                                 attack_ms: torch.Tensor,
        #                                 release_ms: torch.Tensor,
        #                                 knee_db: torch.Tensor,
        #                                 makeup_gain_db: torch.Tensor,
        #                                 eps: float = 1e-8,
        #                                 lookahead_samples: int = 0,
        #                             )
    except:
        x_out_low = low_band
        #print('\t!!!failed computing low-band compression!!!')
    try:
        x_out_high = high_band * torchcomp.compexp_gain(high_band.sum(axis=1).abs(),
                                            comp_thresh=high_shelf_comp_thresh, \
                                            comp_ratio=high_shelf_comp_ratio, \
                                            exp_thresh=high_shelf_exp_thresh, \
                                            exp_ratio=high_shelf_exp_ratio, \
                                            at=torchcomp.ms2coef(high_shelf_at, sample_rate), \
                                            rt=torchcomp.ms2coef(high_shelf_rt, sample_rate)).unsqueeze(1)
    except:
        x_out_high = high_band
        #print('\t!!!failed computing high-band compression!!!')
    try:
        x_out_mid = mid_band * torchcomp.compexp_gain(mid_band.sum(axis=1).abs(),
                                            comp_thresh=mid_band_comp_thresh, \
                                            comp_ratio=mid_band_comp_ratio, \
                                            exp_thresh=mid_band_exp_thresh, \
                                            exp_ratio=mid_band_exp_ratio, \
                                            at=torchcomp.ms2coef(mid_band_at, sample_rate), \
                                            rt=torchcomp.ms2coef(mid_band_rt, sample_rate)).unsqueeze(1)
    except:
        x_out_mid = mid_band
        #print('\t!!!failed computing mid-band compression!!!')
    x_out = x_out_low + x_out_high + x_out_mid

    # parallel computation
    x_out = parallel_weight_factor * x_out + (1-parallel_weight_factor) * x

    # move channels back
    x_out = x_out.view(bs, chs, seq_len)

    return x_out

# > ======= Limiter ======= <
def limiter(
    x: torch.Tensor,
    sample_rate: float,
    threshold: float,
    at: float,
    rt: float,):
    """Limiter.

    from Chin-yun's paper

    Args:
        x (torch.Tensor): Time domain tensor with shape (bs, chs, seq_len)
        sample_rate (float): Audio sample rate.
        threshold (torch.Tensor): Limiter threshold in dB.
        at (torch.Tensor): Attack time.
        rt (torch.Tensor): Release time.
        
    Returns:
        y (torch.Tensor): Limited signal.
    """
    bs, chs, seq_len = x.size()

    x_out = x * torchcomp.limiter_gain(x.sum(axis=1).abs(), 
                                        threshold=threshold,
                                        at=torchcomp.ms2coef(at, sample_rate), 
                                        rt=torchcomp.ms2coef(rt, sample_rate)).unsqueeze(1)

    # move channels back
    x_out = x_out.view(bs, chs, seq_len)

    return x_out

# > ======= Delay ======= <
def unwrap_phase(phase, dim=-1):
    """Unwrap phase to ensure continuous delay response."""
    diff = torch.diff(phase, dim=dim)
    # Wrap differences to [-π, π]
    diff_wrapped = torch.remainder(diff + np.pi, 2 * np.pi) - np.pi
    # Reconstruct unwrapped phase
    phase_unwrapped = torch.cat([
        phase.narrow(dim, 0, 1),
        phase.narrow(dim, 0, 1) + torch.cumsum(diff_wrapped, dim=dim)
    ], dim=dim)
    return phase_unwrapped

def delay(x, sample_rate, delay_samples, wet):
    batch_size, num_channels, num_samples = x.shape
    
    # Clamp parameters to safe ranges
    max_delay_samples = int(sample_rate * 0.3)
    delay_samples = torch.clamp(delay_samples, 1, max_delay_samples).float()
    wet = torch.clamp(wet, 0.0, 1.0)
    dry = 1 - wet
    
    # Convert delay from samples to seconds
    delay_seconds = delay_samples / sample_rate
    
    # Calculate padding needed for delay
    max_delay_samples_int = int(torch.max(delay_samples).item())
    
    # Calculate FFT size (next power of 2 for efficiency)
    fft_size = 2 ** int(np.ceil(np.log2(num_samples + max_delay_samples_int)))
    
    # Pad input signal
    pad_right = fft_size - (num_samples + max_delay_samples_int)
    x_padded = F.pad(x, (max_delay_samples_int, pad_right))
    
    # Convert to frequency domain
    X = torch.fft.rfft(x_padded, n=fft_size)
    
    # Calculate frequency vector
    freqs = torch.fft.rfftfreq(fft_size, 1/sample_rate).to(x.device)
    
    # Calculate phase shift for delay
    phase = -2 * np.pi * freqs * delay_seconds.view(-1, 1, 1)
    phase = unwrap_phase(phase, dim=-1)
    
    # Apply phase shift
    X_delayed = X * torch.exp(1j * phase).to(X.dtype)
    
    # Convert back to time domain
    x_delayed_padded = torch.fft.irfft(X_delayed, n=fft_size)
    
    # Trim to original length
    x_delayed = x_delayed_padded[..., max_delay_samples_int:max_delay_samples_int + num_samples]
    
    # Mix dry and wet signals
    wet = wet.view(batch_size, 1, 1)
    dry = dry.view(batch_size, 1, 1)
    output = dry * x + wet * x_delayed
    
    return output



