import torch
from torch.utils.data import Dataset

class GPTDataset(Dataset):
    """
    A PyTorch Dataset for GPT-style training.

    This class takes a text input and tokenizes it into input-target pairs using a sliding window approach.
    """

    def __init__(self, text, tokenizer, max_length, stride):
        """
        Initialize the dataset.

        :param text: str, the raw text to be tokenized.
        :param tokenizer: a tokenizer instance compatible with the GPT model.
        :param max_length: int, the maximum length of the input-target pairs.
        :param stride: int, the number of tokens to slide over the text for the next sequence.
        """
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)
        self._create_sequences(token_ids, max_length, stride)

    def _create_sequences(self, token_ids, max_length, stride):
        """
        Create input-target sequences from the tokenized text.

        :param token_ids: list of int, tokenized representation of the text.
        :param max_length: int, the maximum length of the input-target pairs.
        :param stride: int, the number of tokens to slide over the text for the next sequence.
        """
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        """
        Return the total number of items in the dataset.
        """
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        Retrieve the input-target pair at the specified index.

        :param idx: int, the index of the item to retrieve.
        :return: tuple of torch.Tensor, (input_ids, target_ids).
        """
        return self.input_ids[idx], self.target_ids[idx]
