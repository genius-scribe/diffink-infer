import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class ValDataset(Dataset):
    """Validation dataset loaded from an HDF5 file produced by the DiffInk pipeline."""

    text_cache = None

    def __init__(self, hdf5_file, text_file, writer_file=None, transform=None):
        self.hdf5_file = hdf5_file
        self.transform = transform

        print(f"Loading dataset: {hdf5_file}")
        self.hf = h5py.File(hdf5_file, "r")

        self.keys = []
        self.lengths = []
        for key in self.hf.keys():
            n = len(self.hf[key]["point_seq"])
            if n >= 200:
                self.keys.append(key)
                self.lengths.append(n)

        self.sorted_indices = np.argsort(self.lengths)[::-1]
        self.len = len(self.keys)

        if ValDataset.text_cache is None and text_file is not None:
            ValDataset.text_cache = self._read_chars(text_file)

    def _read_chars(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: i for i, k in enumerate(data.keys())}

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sorted_idx = self.sorted_indices[idx]
        key = self.keys[sorted_idx]

        writer_id = self.hf[key]["writer_id"][()].decode("utf-8")
        point_seq = np.array(self.hf[key]["point_seq"][:], dtype=np.float32)
        char_points_idx = self.hf[key]["char_points_idx"][:]
        line_text = self.hf[key]["line_text"][()].decode("utf-8")

        if self.transform:
            point_seq = self.transform(point_seq)

        return writer_id, torch.tensor(point_seq, dtype=torch.float32), line_text, char_points_idx

    @staticmethod
    def get_text_index(line_text):
        font_data = ValDataset.text_cache
        char_idx = []
        for text in line_text:
            converted = [font_data[c] for c in text if c in font_data]
            converted.append(font_data.get("、", 0))  # trailing marker
            char_idx.append(converted)
        text_tensor = [torch.tensor(lst, dtype=torch.long) for lst in char_idx]
        return pad_sequence(text_tensor, batch_first=True, padding_value=-1)

    @staticmethod
    def collate_fn(batch):
        writer_id, sequences, line_text, char_points_idx = zip(*batch)

        text_index = ValDataset.get_text_index(line_text)

        max_length = max(s.shape[0] for s in sequences)
        while max_length % 8 != 0:
            max_length += 1

        pad_val = torch.tensor([0, 0, 0, 0, 1], dtype=torch.float)
        padded = []
        for seq in sequences:
            if seq.shape[0] < max_length:
                extra = pad_val.unsqueeze(0).expand(max_length - seq.shape[0], -1)
                seq = torch.cat([seq, extra], dim=0)
            padded.append(seq)

        batch_tensor = torch.stack(padded)
        is_padding = (batch_tensor == pad_val).all(dim=-1)
        mask = ~is_padding

        return batch_tensor, mask, text_index, char_points_idx, writer_id
