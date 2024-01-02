import copy
from datetime import datetime
import math

import torch
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.profiler import profile, ProfilerActivity


# Class modified from PyTorch 2.1 TransformerDecoderLayer class
class CustomTransformerDecoderLayer(nn.Module):
    """Modified from original PyTorch class. The difference is we are removing the two
    attention steps and the `memory` tensor input since this is a generative model
    and we do not have any `memory` or encoder output to pass in, only the input sequence.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise it's done after.
            Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.
    """
    __constants__ = ['norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=True, bias=True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            bias=bias, **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation


    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)


    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, tgt_is_causal=False):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            tgt_mask: the mask for the tgt sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``False``.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm2(x + self._ff_block(x))

        return x


    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           is_causal=is_causal,
                           need_weights=False)[0]
        return self.dropout1(x)


    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


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
        x = x + self.pe[:x.size(0), :].requires_grad_(False)
        # print(f"x_pos: {x[1]}")
        return self.dropout(x.transpose(0, 1))
        # return self.dropout(x)
    

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
        self.output_dim = output_dim
        # self.num_heads = num_heads
        # self.num_layers = num_layers
        self.token_embedding_layer = nn.Embedding(vocab_size, output_dim)
        # self.pos_embedding_layer = nn.Embedding(block_size, output_dim)
        self.transformer_decoder_layer = CustomTransformerDecoderLayer(d_model=output_dim, nhead=num_heads, dim_feedforward=1024, dropout=0.0, batch_first=True)
        self.transformer_decoder = nn.ModuleList([copy.deepcopy(self.transformer_decoder_layer) for i in range(num_layers)])
        self.pos_encoder = PositionalEncoding(d_model=output_dim, dropout=0.0, max_len=block_size)
        self.linear_layer = nn.Linear(output_dim, vocab_size)

    def forward(self, inputs):
        """
        Forward pass of the model.

        :param inputs: torch.Tensor, input tensor.
        :return: torch.Tensor, logits output by the model.
        """
        _batch_size, seq_length = inputs.shape
        token_embeddings = self.token_embedding_layer(inputs)
        pos_encoder_embeddings = self.pos_encoder(token_embeddings)
        # Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
        # Unmasked positions are filled with float(0.0).
        mask = torch.triu(torch.full((seq_length, seq_length), float('-inf')), diagonal=1)
        decoder_output = self.transformer_decoder(tgt=pos_encoder_embeddings, tgt_mask=mask, tgt_is_causal=True)
        last_token_output = decoder_output[:, -1, :]
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
                avg_loss = total_loss / (step + 1)

                # Estimate time remaining for epoch
                # if step % 1000 == 0 and step != 0:
                #     step_end = datetime.now()
                #     time_elapsed = step_end - step_start
                #     time_remaining = time_elapsed * (len(train_dataloader) - step) / 100
                #     print(f"Estimated time remaining for epoch: {time_remaining}")
                #     step_start = datetime.now()

            print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")
            # print(prof.key_averages(group_by_stack_n=10).table(sort_by="self_cuda_time_total", row_limit=10))
            # print(prof.key_averages(group_by_stack_n=10).table(sort_by="self_cpu_time_total", row_limit=10))

            if epoch % 10 == 0:
                save_path = f"{save_path_prefix}_{epoch+1}.pth"
                torch.save(self.state_dict(), save_path)
                print(f"Model checkpoint saved to {save_path}")