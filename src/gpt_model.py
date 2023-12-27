import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity


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
        self.transformer = nn.Transformer(output_dim, num_heads, num_layers, batch_first=True)
        self.linear = nn.Linear(output_dim, vocab_size)

    def forward(self, inputs):
        """
        Forward pass of the model.

        :param inputs: torch.Tensor, input tensor.
        :return: torch.Tensor, logits output by the model.
        """
        token_embeddings = self.token_embedding_layer(inputs)
        _, seq_length, _ = token_embeddings.shape
        pos_embeddings = self.pos_embedding_layer(torch.arange(seq_length, device=inputs.device))
        input_embeddings = token_embeddings + pos_embeddings
        transformer_out = self.transformer(input_embeddings, input_embeddings)
        logits = self.linear(transformer_out)
        return logits

    def train_model(self, train_dataloader, num_epochs, learning_rate, save_path):
        """
        Train the GPT Model.

        :param train_dataloader: DataLoader, the DataLoader for training data.
        :param num_epochs: int, number of training epochs.
        :param learning_rate: float, learning rate for the optimizer.
        :param save_path: str, path to save the trained model.
        """
        # Check if CUDA is available and set
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        self.to(device)  # Move model to the device specified above
        self.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            total_loss = 0
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False, profile_memory=False) as prof:
                cnt = 0
                for inputs, targets in train_dataloader:
                    with record_function("model_training"):
                        inputs, targets = inputs.to(device), targets.to(device)
                        optimizer.zero_grad()
                        logits = self(inputs)
                        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                        if cnt % 100 == 0:
                            print(f"Batches remaining: {len(train_dataloader) - cnt}")
                        cnt += 1

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

            torch.save(self.state_dict(), save_path)
            print(f"Model checkpoint saved to {save_path}")
