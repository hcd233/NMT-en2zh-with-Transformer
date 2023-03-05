import os
from datetime import datetime

import torch.nn.functional as F
import torch.utils.data as Data
from torch.optim import AdamW

from model import Transformer
from utils import *

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
VOCAB_SIZE = 36000
EPOCH = 3
MAX_LEN = 50
BATCH_SIZE = 256

LR = 3e-5
EMBEDDING_SIZE = 1024  # 768
NUM_HEADS = 12  # 8
PAD_IDX = 0
UNK_IDX = 1

DATASET_PATH = "./dataset"
VOCAB_DIR_PATH = "./vocabs"
TOKENIZED_PATH = "./tokenized_dataset"
PRETRAINED_MODEL_PATH = "./model/model.params"
TMP_PATH = "./tmp_model"

if not os.path.exists(TMP_PATH):
    os.mkdir(TMP_PATH)

def Validation(dev_loader, model):
    dev_loss = []
    model.eval()
    with torch.no_grad():
        for _, (x, y) in enumerate(dev_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            output = model(x, y)
            output = output.reshape(-1, output.shape[2])
            y = y.reshape(-1)
            loss = F.cross_entropy(output, y, ignore_index=PAD_IDX)
            dev_loss.append(loss)
        dev_loss = torch.mean(torch.tensor(dev_loss))
        print("Epoch {} Validation loss: {:.6f}\n".format(epoch, dev_loss))
        return dev_loss


def Train(train_loader, model, optimizer, model_path=None):
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        output = model(x, y[:, :-1])  # 输入前面第 n-1 的词 供它预测第 n 的词
        output = output.reshape(-1, output.shape[2])
        y = y[:, 1:].reshape(-1)
        loss = F.cross_entropy(output, y, ignore_index=PAD_IDX)
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
        optimizer.step()
        if batch_idx % 100 == 0:
            print('[{}] Batch idx: {} Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                batch_idx, epoch, batch_idx * len(x),
                len(train_loader.dataset),
                                  100. * batch_idx / len(train_loader),
                loss.item()))
        if batch_idx % 6000 == 0:
            torch.save(model.state_dict(),
                       TMP_PATH + "/model_epoch{}_batch_idx{}_loss{:.2f}.params".format(epoch, batch_idx, loss.item()))


class TensorsDataset(Data.Dataset):
    def __init__(self, EN_Dataset, ZH_Dataset=None, mode='train'):
        super(TensorsDataset, self).__init__()
        self.mode = mode
        self.en_dataset = EN_Dataset
        self.zh_dataset = ZH_Dataset

    def __getitem__(self, idx):
        if self.mode == 'train' or self.mode == 'dev':
            return self.en_dataset[idx], self.zh_dataset[idx]
        else:
            return self.en_dataset[idx]

    def __len__(self):
        return self.en_dataset.shape[0]


if __name__ == '__main__':
    train_zh = load_tokenized_file("./tokenized_dataset/train_zh.txt")
    train_en = load_tokenized_file("./tokenized_dataset/train_en.txt")
    val_zh = load_tokenized_file("./tokenized_dataset/dev_zh.txt")
    val_en = load_tokenized_file("./tokenized_dataset/dev_en.txt")
    print(f"train dataset　len:{len(train_zh)}\nvalidation dataset len:{len(val_zh)}")

    zh2idx = load_vocab_json(VOCAB_DIR_PATH + "/zh2idx.json")
    en2idx = load_vocab_json(VOCAB_DIR_PATH + "/en2idx.json")
    idx2zh = load_vocab_json(VOCAB_DIR_PATH + "/idx2zh.json", key_eq_idx=True)
    idx2en = load_vocab_json(VOCAB_DIR_PATH + "/idx2en.json", key_eq_idx=True)

    print(f"vocab size:{len(zh2idx)}")

    tensors_train_zh = sentence2tensor(train_zh, zh2idx, max_len=MAX_LEN)
    tensors_train_en = sentence2tensor(train_en, en2idx, max_len=MAX_LEN)
    tensors_val_zh = sentence2tensor(val_zh, zh2idx, max_len=MAX_LEN)
    tensors_val_en = sentence2tensor(val_en, en2idx, max_len=MAX_LEN)

    print("Sentence2Tensor over")

    train = TensorsDataset(tensors_train_en, tensors_train_zh, mode='train')
    dev = TensorsDataset(tensors_val_en, tensors_val_zh, mode='dev')

    print("Load dataset over")

    train_loader = Data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    dev_loader = Data.DataLoader(dev, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    text = ["I am a cat.", "how dare are you?", "So will you stay?", "Take him out first.",
            "I have a question about this scene. Yes? ",
            "Guests enjoy a complimentary Health Club & Fitness Center that includes: gym, sauna, steam room and hot tub.",
            "To get back. ",
            "Been playing for bigger stakes than I normally would.",
            "I mean, I didn't deal with him directly. ",
            "Are there any families in your neighborhood that you don't get along with? ",
            "Than he wants us to have it. So they start shooting at each other. ",
            "Kenwood chef balloon whisk for circlip type hub.",
            "Those associated with obsessive-compulsive disorder, clomipramine may be used.",
            "You don't want to run the risk of losing 100 dollars.",
            "for china to reach its target of quadrupling 2000s national product by 2020 it will need bold reforms in a wide range of areas ",
            "just left but the words still", "well you stopped it before", "im so pleased to be invited here today "]
    model = Transformer(src_vocab_size=VOCAB_SIZE,
                        trg_vocab_size=VOCAB_SIZE,
                        src_pad_idx=0,
                        trg_pad_idx=0,
                        embed_size=EMBEDDING_SIZE,
                        dropout=0.1,
                        num_layers=NUM_HEADS,
                        device=DEVICE).to(DEVICE)
    # model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))
    optimizer = AdamW(model.parameters(), lr=LR)
    best_loss = 100
    print("Start Training")
    for epoch in range(EPOCH):
        Train(train_loader, model, optimizer)
        print("Start Validation")
        dev_loss = Validation(dev_loader, model)
        print("\nSave model at ./model/model_{}_{:.2f}.params".format(epoch, dev_loss))
        torch.save(model.state_dict(), "./model/model_epoch{}_loss{:.2f}.params".format(epoch, dev_loss))
        print("Infer Examples.")
        for i in text:
            pre_result = Infer(i, model, zh2idx, idx2zh, en2idx)
            print(i, "\t", pre_result)
