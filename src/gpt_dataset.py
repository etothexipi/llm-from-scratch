import torch
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    """
    A PyTorch Dataset for GPT-style training.

    This class tokenizes text input into input-target pairs using a sliding window approach.
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
        self._tokenize_and_create_sequences(text, max_length, stride)

    def _tokenize_and_create_sequences(self, text, max_length, stride):
        """
        Tokenize the text and create input-target sequences.

        :param text: str, the text to tokenize.
        :param max_length: int, the maximum length of the input-target pairs.
        :param stride: int, the number of tokens to slide over the text for the next sequence.
        """
        token_ids = self.tokenizer.encode(text)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))
            # if i < 2:
            #     print(text[:100])
            #     # print(token_ids[:17])
            #     print(input_chunk)
            #     print(target_chunk)
            #     print(self.input_ids)
            #     print(self.target_ids)

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


def create_dataloader(text, tokenizer, batch_size, max_length, stride, num_workers):
    """
    Create a DataLoader for the GPT Dataset.

    :param text: str, the raw text to be tokenized and loaded.
    :param tokenizer: a tokenizer instance compatible with the GPT model.
    :param batch_size: int, size of each batch of data.
    :param max_length: int, maximum sequence length for each data sample.
    :param stride: int, stride for creating overlapping sequences.
    :param num_workers: int, number of subprocesses to use for data loading.
    :return: DataLoader, the DataLoader instance for the dataset.
    """
    dataset = GPTDataset(text, tokenizer, max_length, stride)
    return DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)