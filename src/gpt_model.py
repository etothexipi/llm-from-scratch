import torch
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, ProfilerActivity


class GPTModel(nn.Module):
    """
    A PyTorch module for a GPT-like transformer model.

    This model includes token and position embedding layers, followed by a transformer and a linear output layer.
    """

    def __init__(self, vocab_size, output_dim, block_size, num_heads, num_layers):
        """
        Initialize the GPT Model.

        :param vocab_size: int, the size of the vocabulary.
        :param output_dim: int, the dimensionality of the output space.
        :param block_size: int, the size of each block of text (sequence length).
        :param num_heads: int, the number of heads in the multiheadattention models.
        :param num_layers: int, the number of sub-encoder-layers in the transformer.
        """
        super().__init__()
        self.token_embedding_layer = nn.Embedding(vocab_size, output_dim)
        self.pos_embedding_layer = nn.Embedding(block_size, output_dim)
        self.transformer = nn.Transformer(d_model=output_dim, nhead=num_heads, num_encoder_layers=0, num_decoder_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(output_dim, vocab_size)

    def forward(self, inputs):
        """
        Forward pass of the model.

        :param inputs: torch.Tensor, input tensor.
        :return: torch.Tensor, logits output by the model.
        """
        token_embeddings = self.token_embedding_layer(inputs)
        batch_size, seq_length, _ = token_embeddings.shape
        pos_embeddings = self.pos_embedding_layer(torch.arange(seq_length, device=inputs.device))
        input_embeddings = token_embeddings + pos_embeddings
        # Creating a dummy input for the encoder part
        dummy_src = torch.zeros_like(input_embeddings, device=inputs.device)
        # Creating a mask that will effectively ignore the encoder input
        src_mask = torch.ones((batch_size, seq_length), device=inputs.device).bool()

        transformer_out = self.transformer(src=dummy_src, tgt=input_embeddings, src_mask=src_mask)
        logits = self.linear(transformer_out)
        # print(logits)
        return logits

    def train_model(self, train_dataloader, num_epochs, learning_rate, save_path_prefix, grad_accumulation_steps):
        """
        Train the GPT Model.

        :param train_dataloader: DataLoader, the DataLoader for training data.
        :param num_epochs: int, number of training epochs.
        :param learning_rate: float, learning rate for the optimizer.
        :param save_path_prefix: str, path to save the trained model.
        :param grad_accumulation_steps: int, number of gradient accumulation steps.
        """
        # Check if CUDA is available and set
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        self.to(device)
        self.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scaler = GradScaler()

        for epoch in range(num_epochs):
            total_loss = 0
            optimizer.zero_grad()
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False, with_stack=False) as prof:
            cnt = 0
            for step, (inputs, targets) in enumerate(train_dataloader):
                # pin arrays which allows us to move them to GPU asynchronously (non_blocking=True)
                inputs, targets = inputs.pin_memory().to(device, non_blocking=True), targets.pin_memory().to(device, non_blocking=True)

                # Mixed precision training
                # with autocast():
                logits = self(inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (step + 1) % grad_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    # set_to_none=True for memory efficiency
                    optimizer.zero_grad(set_to_none=True)
                
                total_loss += loss.item()

                if cnt % 100 == 0:
                    print(f"Batches remaining: {len(train_dataloader) - cnt}")
                cnt += 1

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")
            # print(prof.key_averages(group_by_stack_n=10).table(sort_by="self_cuda_time_total", row_limit=10))
            # print(prof.key_averages(group_by_stack_n=10).table(sort_by="self_cpu_time_total", row_limit=10))

            save_path = f"{save_path_prefix}_{epoch+1}.pth"
            torch.save(self.state_dict(), save_path)
            print(f"Model checkpoint saved to {save_path}")
