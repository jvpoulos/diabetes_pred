import torch
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch.nn.utils.prune as prune
import seaborn as sns
import pandas as pd
from tab_transformer_pytorch import TabTransformer, FTTransformer
import os
import argparse
import json
from tqdm import tqdm
import numpy as np
from model_utils import load_model
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.cuda.amp as amp
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast
import bitsandbytes as bnb
scaler = amp.GradScaler()

class CustomDataset(Dataset):
    def __init__(self, x_categ, x_cont, labels):
        self.x_categ = x_categ
        self.x_cont = x_cont
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.x_categ[idx], self.x_cont[idx], self.labels[idx]

def extract_embeddings(model, loader, device, args):
    model.eval()
    embeddings_list = []
    with torch.no_grad():
        for batch in loader:
            x_categ, x_cont, _ = batch
            x_categ = x_categ.to(device)
            x_cont = x_cont.to(device)

            with amp.autocast():  # Enable mixed precision
                embeddings = model.module.get_embeddings(x_categ, x_cont, batch_size=args.batch_size)

            embeddings_list.append(embeddings)
            del x_categ, x_cont, embeddings
            torch.cuda.empty_cache()

    embeddings = torch.cat(embeddings_list, dim=0)  # Concatenate all embeddings
    
    # Gather embeddings from all processes using all_gather
    world_size = dist.get_world_size()
    embeddings_gather = [torch.zeros_like(embeddings) for _ in range(world_size)]
    dist.all_gather(embeddings_gather, embeddings)
    
    embeddings = torch.cat(embeddings_gather, dim=0)
    
    return embeddings
    
def plot_embeddings(tsne_df, model_type, model_path):
    plt.figure(figsize=(16, 10))
    sns.scatterplot(x='TSNE1', y='TSNE2', palette=sns.color_palette("hsv", 10),
                    data=tsne_df, legend="full", alpha=0.3)
    plt.title(f'Learned embeddings - {model_type}')
    filename = f"tSNE_embeddings_plot_{model_type}_{os.path.basename(model_path).replace('.pth', '')}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Embeddings plot saved to {filename}")

def main(gpu, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

    print("Loading dataset...")
    if args.dataset_type == 'train':
        features = torch.load('train_features.pt')
        labels = torch.load('train_labels.pt')
    elif args.dataset_type == 'validation':
        features = torch.load('validation_features.pt')
        labels = torch.load('validation_labels.pt')

    with open('column_names.json', 'r') as file:
        column_names = json.load(file)

    with open('encoded_feature_names.json', 'r') as file:
        encoded_feature_names = json.load(file)

    with open('columns_to_normalize.json', 'r') as file:
        columns_to_normalize = json.load(file)

    excluded_columns = ["A1cGreaterThan7"]
    additional_binary_vars = ["Female", "Married", "GovIns", "English", "Veteran"]

    column_names_filtered = [col for col in column_names if col not in excluded_columns]
    encoded_feature_names_filtered = [name for name in encoded_feature_names if name in column_names_filtered]

    binary_features_combined = list(set(encoded_feature_names_filtered + additional_binary_vars))

    binary_feature_indices = [column_names_filtered.index(col) for col in binary_features_combined if col in column_names_filtered]

    numerical_feature_indices = [column_names.index(col) for col in columns_to_normalize if col not in excluded_columns]

    categories = [2] * len(binary_feature_indices)
    print("Categories:", len(categories))

    num_continuous = len(numerical_feature_indices)
    print("Continuous:", num_continuous)

    x_categ = features[:, binary_feature_indices]
    x_cont = features[:, numerical_feature_indices]

    dataset = CustomDataset(x_categ, x_cont, labels)

    data_loader = DataLoader(dataset, batch_size=args.batch_size // 2, shuffle=False, pin_memory=True, num_workers=4)

   # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.pruning and os.path.exists(args.model_path + '_pruned'):
        print("Loading pruned model...")
        model = load_model(args.model_type, args.model_path + '_pruned', args.dim, args.depth, args.heads, args.attn_dropout, args.ff_dropout, categories, num_continuous, device)
    elif args.quantization and os.path.exists(args.model_path + '_quantized.pth'):
        print("Loading quantized model...")
        model = load_model(args.model_type, args.model_path + '_quantized.pth', args.dim, args.depth, args.heads, args.attn_dropout, args.ff_dropout, categories, num_continuous, device, quantized=True)
    else:
        print("Loading model...")
        model = load_model(args.model_type, args.model_path, args.dim, args.depth, args.heads, args.attn_dropout, args.ff_dropout, categories, num_continuous, device)

        if args.pruning:
            print("Model pruning.")
            # Specify the pruning percentage
            pruning_percentage = 0.4

            # Perform global magnitude-based pruning
            parameters_to_prune = [(module, 'weight') for module in model.modules() if isinstance(module, (nn.Linear, nn.Embedding))]
            prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=pruning_percentage)

            # Remove the pruning reparameterization
            for module, _ in parameters_to_prune:
                prune.remove(module, 'weight')
            
            # Save the pruned model
            if args.model_path is not None:
                pruned_model_path =  args.model_path + '_pruned'
                torch.save(model.state_dict(), pruned_model_path)

        if args.quantization:
            print(f"Quantizing model to 8 bits.")
            # Create a bitsandbytes quantization configuration
            qconfig = bnb.QuantizationConfig(bits=8)
            model = bnb.quantize(model, qconfig)

        if args.pruning or args.quantization:
            # Save the pruned and/or quantized model
            if args.model_path is not None:
                model_path_base, model_path_ext = os.path.splitext(args.model_path)
                model_path_ext_pruned = '_pruned' if args.pruning else ''
                model_path_ext_quantized = '_quantized' if args.quantization else ''
                model_path = f"{model_path_base}{model_path_ext_pruned}{model_path_ext_quantized}{model_path_ext}"
                torch.save(model.state_dict(), model_path)

    model = model.to(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=False)

    with autocast():
        embeddings = extract_embeddings(model, data_loader, gpu, args)

    if dist.get_rank() == 0:
        # Move embeddings to CPU and reshape to 2D
        embeddings = embeddings.cpu().numpy().reshape(-1, embeddings.shape[-1])

        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(embeddings)

        tsne_df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])

        plot_embeddings(tsne_df, args.model_type, args.model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract embeddings.')
    parser.add_argument('--dataset_type', type=str, choices=['train','validation'], required=True, help='Specify dataset type for evaluation')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['Transformer','TabTransformer', 'FTTransformer'],
                        help='Type of the model to train: TabTransformer or FTTransformer')
    parser.add_argument('--dim', type=int, default=None, help='Dimension of the model')
    parser.add_argument('--depth', type=int, help='Depth of the model.')
    parser.add_argument('--heads', type=int, help='Number of attention heads.')
    parser.add_argument('--ff_dropout', type=float, help='Feed forward dropout rate.')
    parser.add_argument('--num_encoder_layers', type=float, default=6, help='Number of sub-encoder-layers in the encoder')
    parser.add_argument('--num_decoder_layers', type=float, default=6, help=' Number of sub-decoder-layers in the decoder')
    parser.add_argument('--dim_feedforward', type=float, default=2048, help='Dimension of the feedforward network model ')
    parser.add_argument('--dropout', type=float, default=0.1, help='Attention dropout rate')
    parser.add_argument('--attn_dropout', type=float, default=None, help='Attention dropout rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and validation')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Oath to the saved model file to load.')
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N', help='Number of data loading workers (default: 1)')
    parser.add_argument('-g', '--gpus', default=2, type=int, help='Number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='Ranking within the nodes')
    parser.add_argument('--pruning', action='store_true', help='Enable model pruning')
    parser.add_argument('--quantization', action='store_true', help='Quantization with bit width 8')

    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes

    if args.model_type == 'TabTransformer':
        if args.dim is None:
            args.dim = 32
        if args.attn_dropout is None:
            args.attn_dropout = 0.1
        args.depth = args.depth if args.depth is not None else 6
        args.heads = args.heads if args.heads is not None else 8
        args.ff_dropout = args.ff_dropout if args.ff_dropout is not None else 0.1
    elif args.model_type == 'FTTransformer':
        if args.dim is None:
            args.dim = 192
        if args.attn_dropout is None:
            args.attn_dropout = 0.2
        args.depth = args.depth if args.depth is not None else 3
        args.heads = args.heads if args.heads is not None else 8
        args.ff_dropout = args.ff_dropout if args.ff_dropout is not None else 0.1
    elif args.model_type == 'Transformer':
        if args.heads is None:
            raise ValueError("The 'heads' argument must be provided when using the 'Transformer' model type.")
        if args.dtype is None:
            args.dtype = torch.float32
        if args.dim is None:
            args.dim = 512
        args.heads = args.heads if args.heads is not None else 8

    mp.set_start_method('spawn')  # Set the start method to 'spawn'

    processes = []
    for i in range(args.gpus):
        p = mp.Process(target=main, args=(i, args))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()