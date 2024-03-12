import json
import torch
import torch.nn as nn
from tab_transformer_pytorch import TabTransformer, FTTransformer

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Since features and labels are already tensors, no conversion is needed
        return self.features[idx], self.labels[idx]

def load_model(model_type, model_path, dim, depth, heads, attn_dropout, ff_dropout, categories, num_continuous):

    if model_type == 'TabTransformer':
        model = TabTransformer(
            categories=categories,  # tuple containing the number of unique values within each category
            num_continuous=num_continuous,                              # number of continuous values
            dim=dim,                                               # dimension, paper set at 32
            dim_out=1,                                                  # binary prediction, but could be anything
            depth=depth,                                           # depth, paper recommended 6
            heads=heads,                                           # heads, paper recommends 8
            attn_dropout=attn_dropout,                             # post-attention dropout
            ff_dropout=ff_dropout ,                                # feed forward dropout
            mlp_hidden_mults=(4, 2),                                    # relative multiples of each hidden dimension of the last mlp to logits
            mlp_act=nn.ReLU()                                           # activation for final mlp, defaults to relu, but could be anything else (selu etc)
        )
    elif model_type == 'FTTransformer':
            model = FTTransformer(
            categories = categories,      # tuple containing the number of unique values within each category
            num_continuous = num_continuous,  # number of continuous values
            dim = dim,                     # dimension, paper set at 192
            dim_out = 1,                        # binary prediction, but could be anything
            depth = depth,                          # depth, paper recommended 3
            heads = heads,                          # heads, paper recommends 8
            attn_dropout = attn_dropout,   # post-attention dropout, paper recommends 0.2
            ff_dropout = ff_dropout                   # feed forward dropout, paper recommends 0.1
        )
    elif args.model_type == 'Transformer':
    model = torch.nn.Transformer(
        d_model=dim,                                           # embedding dimension
        nhead=heads,                                           # number of attention heads
        num_encoder_layers=6,                                       # number of encoder layers
        num_decoder_layers=6,                                       # number of decoder layers
        dim_feedforward=2048,                                       # dimension of the feedforward network
        dropout=0.1,                                                # dropout rate
        activation=torch.nn.ReLU(),                                 # activation function
        custom_encoder=None,                                        # custom encoder
        custom_decoder=None,                                        # custom decoder
        layer_norm_eps=1e-05,                                       # layer normalization epsilon
        batch_first=False,                                          # if True, input and output tensors are provided as (batch, seq, feature)
        norm_first=False,                                           # if True, layer normalization is done before self-attention
        device=device,                                              # device to run the model on
        dtype=None                                                  # data type
    )
    else:
        raise ValueError("Invalid model type. Choose 'Transformer', 'TabTransformer' or 'FTTransformer'.")

    # Load the state dict with fix_state_dict applied
    state_dict = torch.load(model_path)
    model.load_state_dict(fix_state_dict(state_dict))
    model.eval()
    return model