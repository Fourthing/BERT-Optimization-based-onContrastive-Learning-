import csv
import torch
from torch.utils.data import Dataset

class SentencePairDataset(Dataset):
    def __init__(self, csv_file):
        self.data = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append((row['sentence1'], row['sentence2'], int(row['label'])))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence1, sentence2, label = self.data[idx]
        return sentence1, sentence2, label

# 实际使用，训练里调用用这两句
dataset = SentencePairDataset('BERT-ContrastiveLearning/data/sentence_pairs.csv')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
