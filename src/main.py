import argparse
import tiktoken
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from gpt_dataset import GPTDataset, create_dataloader
from gpt_model import GPTModel


def parse_arguments():
    parser = argparse.ArgumentParser(description="GPT Model Trainer/Inference CLI")
    parser.add_argument('--mode', type=str, choices=['train', 'infer'], required=True, help='Mode: train or infer. IMPORTANT: Infer args must match train args.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--stride', type=int, default=128, help='Stride for data loading')
    parser.add_argument('--output_dim', type=int, default=1, help='Embedding dimension for output')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for data loading')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of heads for multi-head attention')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of layers for transformer')
    parser.add_argument('--grad_accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps')
    parser.add_argument('--train_data_path', type=str, default='data/genesis.txt', help='Path to training data')
    parser.add_argument('--save_path_prefix', type=str, default='models/gpt_model.pth', help='Prefix path to save the model. Each epoch will be saved as <save_path_prefix>_<epoch>.pth')
    parser.add_argument('--load_path', type=str, default='models/gpt_model.pth', help='Path to load the model for inference')
    parser.add_argument('--text_to_infer', type=str, default='', help='Text to generate inference from')
    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.mode == 'train':
        # Training mode
        with open(args.train_data_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        # dataset = GPTDataset(raw_text, tokenizer=tiktoken.get_encoding("gpt2"), max_length=args.max_length, stride=args.stride)
        dataloader = create_dataloader(raw_text, max_length=args.max_length, stride=args.stride, batch_size=args.batch_size, tokenizer=tiktoken.get_encoding("gpt2"), num_workers=args.num_workers)
        # Faster GPU matmul with vocab_size multiple of 64 (is normally 50257 for tiktoken gpt-2 tokenizer)
        model = GPTModel(vocab_size=50304, output_dim=args.output_dim, block_size=args.max_length, num_heads=args.num_heads, num_layers=args.num_layers)
        model.train_model(dataloader, num_epochs=args.num_epochs, learning_rate=args.learning_rate, grad_accumulation_steps=args.grad_accumulation_steps, save_path_prefix=args.save_path_prefix)

    elif args.mode == 'infer':
        # Inference mode
        tokenizer = tiktoken.get_encoding("gpt2")
        # Faster GPU matmul with vocab_size multiple of 64 (is normally 50257 for tiktoken gpt-2 tokenizer)
        model = GPTModel(vocab_size=50304, output_dim=args.output_dim, block_size=args.max_length, num_heads=args.num_heads, num_layers=args.num_layers)
        model.load_state_dict(torch.load(args.load_path))
        model.eval()

        # Generate text using the trained model
        text = args.text_to_infer
        input_ids = tokenizer.encode(text)
        if len(input_ids) > args.max_length:
            input_ids_trunc_pad = input_ids[:args.max_length]
        # 50256 is the empty padding token with tiktoken gpt-2 encoding
        input_ids_trunc_pad = input_ids + [50256] * (args.max_length - len(input_ids))
        input_tensor = torch.tensor(input_ids_trunc_pad).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            predicted_token_ids = torch.topk(output, 5, dim=-1)
            print(predicted_token_ids.values.tolist())
            print(predicted_token_ids.indices.tolist())
            for token in predicted_token_ids.indices.tolist():
                print(tokenizer.decode(token))
        
if __name__ == "__main__":
    main()
