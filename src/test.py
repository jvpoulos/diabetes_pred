import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tab_transformer_pytorch import TabTransformer, FTTransformer
import json
import numpy as np
import random
import bitsandbytes as bnb
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from model_utils import load_model, CustomDataset, evaluate_model, worker_init_fn

def main(args):

    # Set random seed for reproducibility
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    # Set deterministic and benchmark flags for PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Loading dataset...")
    if args.dataset_type == 'test':
        # Load the test dataset
        test_dataset = torch.load('test_dataset.pt')
    elif args.dataset_type == 'validation':
        validation_dataset = torch.load('validation_dataset.pt')

    with open('column_names.json', 'r') as file:
        column_names = json.load(file)

    with open('encoded_feature_names.json', 'r') as file:
        encoded_feature_names = json.load(file)

    with open('columns_to_normalize.json', 'r') as file:
        columns_to_normalize = json.load(file)

    with open('binary_feature_indices.json', 'r') as file:
        binary_feature_indices = json.load(file)

    with open('numerical_feature_indices.json', 'r') as file:
        numerical_feature_indices = json.load(file)

    categories = [2] * len(binary_feature_indices)
    print("Categories:", len(categories))

    num_continuous = len(numerical_feature_indices)
    print("Continuous:", num_continuous)

    # Assuming dataset is a TensorDataset containing a single tensor with both features and labels
    dataset_tensor = test_dataset.tensors[0] if args.dataset_type == 'test' else validation_dataset.tensors[0]

    print(f"Original dataset tensor shape: {dataset_tensor.shape}")

    # Extracting indices for features and label
    excluded_columns_indices = [column_names.index('A1cGreaterThan7')]
    feature_indices = [i for i in range(dataset_tensor.size(1)) if i not in excluded_columns_indices]

    label_index = column_names.index('A1cGreaterThan7')

    # Selecting features and labels
    features = dataset_tensor[:, feature_indices]
    labels = dataset_tensor[:, label_index]

    x_categ = features[:, binary_feature_indices]
    x_cont = features[:, numerical_feature_indices]

    features = torch.cat((x_categ, x_cont), dim=1)
    dataset = CustomDataset(features, labels)

    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, worker_init_fn=worker_init_fn)

    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.pruning and os.path.exists(args.model_path + '_pruned'):
        print("Loading pruned model...")
        if args.model_type=='ResNet':
            model = load_model(args.model_type, args.model_path + '_pruned', args.dim, args.depth, args.heads, args.attn_dropout, args.ff_dropout, categories, num_continuous, device, binary_feature_indices=binary_feature_indices, numerical_feature_indices=numerical_feature_indices)
        else:
            model = load_model(args.model_type, args.model_path + '_pruned', args.dim, args.depth, args.heads, args.attn_dropout, args.ff_dropout, categories, num_continuous, device)
    elif args.quantization and os.path.exists(args.model_path + '_quantized.pth'):
        print("Loading quantized model...")
        if args.model_type=='ResNet':
            model = load_model(args.model_type, args.model_path + '_pruned', args.dim, args.depth, args.heads, args.attn_dropout, args.ff_dropout, categories, num_continuous, device, binary_feature_indices=binary_feature_indices, numerical_feature_indices=numerical_feature_indices)
        else:
            model = load_model(args.model_type, args.model_path + '_quantized.pth', args.dim, args.depth, args.heads, args.attn_dropout, args.ff_dropout, categories, num_continuous, device, quantized=True)
    else:
        print("Loading model...")
        if args.model_type == 'ResNet':
            model = load_model(args.model_type, args.model_path, args.dim, args.depth, args.heads, args.attn_dropout, args.ff_dropout, categories, num_continuous, device, binary_feature_indices=binary_feature_indices, numerical_feature_indices=numerical_feature_indices)
        else:
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

    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
    #     model = nn.DataParallel(model)

    print("Model loaded.")

    print("Evaluating model...")
    criterion = nn.BCEWithLogitsLoss()
    evaluate_model(model, test_loader, criterion, device, args.model_type, binary_feature_indices, numerical_feature_indices)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate an attention network on the test set.')
    parser.add_argument('--dataset_type', type=str, choices=['test','validation'], required=True, help='Specify dataset type for evaluation')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['TabTransformer', 'FTTransformer','ResNet'],
                        help='Type of model: TabTransformer or FTTransformer')
    parser.add_argument('--dim', type=int, default=None, help='Dimension of the model')
    parser.add_argument('--depth', type=int, help='Depth of the model.')
    parser.add_argument('--heads', type=int, help='Number of attention heads.')
    parser.add_argument('--ff_dropout', type=float, help='Feed forward dropout rate.')
    parser.add_argument('--num_encoder_layers', type=float, default=6, help='Number of sub-encoder-layers in the encoder')
    parser.add_argument('--num_decoder_layers', type=float, default=6, help=' Number of sub-decoder-layers in the decoder')
    parser.add_argument('--dim_feedforward', type=float, default=2048, help='Dimension of the feedforward network model ')
    parser.add_argument('--dropout', type=float, default=0.1, help='Attention dropout rate')
    parser.add_argument('--attn_dropout', type=float, default=None, help='Attention dropout rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Oath to the saved model file to load.')
    parser.add_argument('--pruning', action='store_true', help='Enable model pruning')
    parser.add_argument('--quantization', action='store_true', help='Quantization with bit width 8')
    args = parser.parse_args()

    # Conditional defaults based on args.model_type
    if args.model_type == 'TabTransformer':
        if args.dim is None:
            args.dim = 32  # Default for TabTransformer
        if args.attn_dropout is None:
            args.attn_dropout = 0.1  # Default for TabTransformer
        args.depth = args.depth if args.depth is not None else 6
        args.heads = args.heads if args.heads is not None else 8
        args.ff_dropout = args.ff_dropout if args.ff_dropout is not None else 0.1
    elif args.model_type == 'FTTransformer':
        if args.dim is None:
            args.dim = 192  # Default for FTTransformer
        if args.attn_dropout is None:
            args.attn_dropout = 0.2  # Default for FTTransformer
        args.depth = args.depth if args.depth is not None else 3
        args.heads = args.heads if args.heads is not None else 8
        args.ff_dropout = args.ff_dropout if args.ff_dropout is not None else 0.1
    elif args.model_type == 'Transformer':
        if args.heads is None:
            raise ValueError("The 'heads' argument must be provided when using the 'Transformer' model type.")
        if args.dtype is None:
            args.dtype = torch.float32  # Set a default value for dtype
        if args.dim is None:
            args.dim = 512
        args.heads = args.heads if args.heads is not None else 8  # Set a default value of 8 if args.heads is None

    main(args)