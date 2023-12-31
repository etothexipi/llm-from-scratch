from datetime import datetime
import math

import torch
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.profiler import profile, ProfilerActivity


# Class taken from: https://github.com/pytorch/examples/blob/30b310a977a82dbfc3d8e4a820f3b14d876d3bd2/word_language_model/model.py#L65C1-L105C31
class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        # Need to adjust the tensor shape to match other modules (batch_size, seq_len, embed_dim)
        x = x.transpose(0, 1)
        # print(f"x: {x[1]}")
        x = x + self.pe[:x.size(0), :]
        # print(f"x_pos: {x[1]}")
        return self.dropout(x.transpose(0, 1))
        # return self.dropout(x)
    

class GPTModel(nn.Transformer):
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
        super().__init__(d_model=output_dim, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers, batch_first=True)
        self.output_dim = output_dim
        # self.num_heads = num_heads
        # self.num_layers = num_layers
        self.token_embedding_layer = nn.Embedding(vocab_size, output_dim)
        # self.pos_embedding_layer = nn.Embedding(block_size, output_dim)
        # self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=output_dim, nhead=num_heads)
        # self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(d_model=output_dim, dropout=0.1, max_len=block_size)
        self.linear_layer = nn.Linear(output_dim, vocab_size)

    def forward(self, inputs):
        """
        Forward pass of the model.

        :param inputs: torch.Tensor, input tensor.
        :return: torch.Tensor, logits output by the model.
        """
        _batch_size, seq_length = inputs.shape
        # pos_embeddings = self.pos_embedding_layer(torch.arange(seq_length, device=inputs.device))
        # input_embeddings = token_embeddings + pos_embeddings
        # print(f"token_embedding_inputs: {self.token_embedding_layer(inputs)}")
        # print(f"sqrt: {math.sqrt(self.output_dim)}")
        token_embeddings = self.token_embedding_layer(inputs) # * math.sqrt(self.output_dim)
        # print(f"token_embeddings: {token_embeddings}")
        src = self.pos_encoder(token_embeddings)
        # print(f"src: {src}")
        mask = torch.log(torch.tril(torch.ones(seq_length,seq_length))).to(inputs.device)
        # print(f"mask: {mask}")
        encoder_output = self.encoder(src)
        # print(f"encoder_output: {encoder_output}")
        decoder_output = self.decoder(src, encoder_output, tgt_mask=mask)
        # print(f"decoder_output: {decoder_output}")
        last_token_output = decoder_output[:, -1, :]
        # print(f"last_token_output: {last_token_output}")
        logits = self.linear_layer(last_token_output)
        # print(f"logits: {logits}")
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
            step_start = datetime.now()
            for step, (inputs, targets) in enumerate(train_dataloader):
                # pin arrays which allows us to move them to GPU asynchronously (non_blocking=True)
                inputs, targets = inputs.pin_memory().to(device, non_blocking=True), targets.pin_memory().to(device, non_blocking=True)
                next_token_targets = targets[:, -1]
                # Mixed precision training
                with autocast():
                    logits = self(inputs)
                    loss = criterion(logits.view(-1, logits.size(-1)), next_token_targets.view(-1))
                    scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (step + 1) % grad_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    # set_to_none=True for memory efficiency
                    optimizer.zero_grad(set_to_none=True)
                
                total_loss += loss.item()

                # Estimate time remaining for epoch
                if step % 1000 == 0 and step != 0:
                    step_end = datetime.now()
                    time_elapsed = step_end - step_start
                    time_remaining = time_elapsed * (len(train_dataloader) - step) / 100
                    print(f"Estimated time remaining for epoch: {time_remaining}")
                    step_start = datetime.now()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")
            # print(prof.key_averages(group_by_stack_n=10).table(sort_by="self_cuda_time_total", row_limit=10))
            # print(prof.key_averages(group_by_stack_n=10).table(sort_by="self_cpu_time_total", row_limit=10))

            if epoch % 10 == 0:
                save_path = f"{save_path_prefix}_{epoch+1}.pth"
                torch.save(self.state_dict(), save_path)
                print(f"Model checkpoint saved to {save_path}")