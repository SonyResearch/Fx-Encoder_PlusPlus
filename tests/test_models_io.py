import torch 
from models.fxenc_plusplus import load_model

if __name__ == "__main__":
    model = load_model(
        '/home/yytung/projects/fxencoder_plusplus/ckpts/fxenc_plusplus.pt',
        'cuda'
    )
    print(model)