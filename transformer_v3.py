import math
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


max_seq_length = 512
batch_size = 1
eval_batch_size = 1
bptt = 35
nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
lr = 5.0 # learning rate
epochs = 3


def log(msg):
	print(str(datetime.now()) + ": " + str(msg))


class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):
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
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


dataset = load_dataset('wmt/wmt19', "ru-en").with_format("torch")
train_loader = DataLoader(dataset["train"].with_format("torch"), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset["validation"].with_format("torch"), batch_size=batch_size)


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", model_max_length=max_seq_length, cache_dir="./sources/tokenizers")
# tokenizer = mistral_tokenizer("./sources/models/Mistral-7B-v0.1/tokenizer.model", max_seq_len=max_seq_length)
transformer = TransformerModel(tokenizer.vocab_size, max_seq_length, nhead, nhid, nlayers, dropout).to(device)

print(torch.cuda.memory_allocated() / 1024**3)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(transformer.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


for j in range(epochs):
    log(f"epoch n{j}, {datetime.now()}")
    transformer.train()
    log("train mode")
    train_loop = tqdm(train_loader, leave=True)
    time.sleep(0.5)

    for i in train_loop:
        src, tgt = (i["translation"]["ru"][0]), (i["translation"]["en"][0])
        # print(src, tgt)
        src, tgt = tokenizer(src), tokenizer(tgt)
        optimizer.zero_grad()
        output = transformer(src)
        print(output.size(), tgt.size())
        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()
        train_loop.set_description(f"Epoch {j}/{epochs}, train_loss={loss.item()}")


    transformer.eval()
    val_loop = tqdm(val_loader, leave=True)
    with torch.no_grad():
        for i in train_loop:
            src, tgt = tokenizer(i["translation"]["ru"]), tokenizer(i["translation"]["en"])
            optimizer.zero_grad()
            output = transformer(src)
            loss = criterion(output, tgt)
            train_loop.set_description(f"Epoch {j}/{epochs}, train_loss={loss.item()}")
