ALL_PROCESSORS = [
    'eq', 
    'distortion', 
    'multiband_comp', 
    'gain', 
    'imager',
    'limiter', 
    'delay', 
    'reverb'
]

ALL_PROCESSORS_WITHOUT_GAIN = [
    'eq', 
    'distortion', 
    'multiband_comp', 
    'imager', 
    'limiter', 
    'delay', 
    'reverb'
]

FX_TO_LABEL = {
    'eq': 0,
    'distortion': 1,
    'multiband_comp': 2, 
    'gain': 3,
    'imager': 4,
    'limiter': 5,
    'delay': 6, 
    'reverb': 7
}

DEFAULT_FX_PROB = {
    'eq': 0.6, 
    'distortion': 0.3, 
    'multiband_comp': 0.8, 
    'gain': 0.6,
    'imager': 0.6, 
    'limiter': 0.6,
    'delay': 0.6, 
    'reverb': 0.6,
}


EPS = 1e-6