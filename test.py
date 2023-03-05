from model import Transformer
from train import VOCAB_SIZE, VOCAB_DIR_PATH
from infer import Infer
from utils import *
from random import shuffle

NUM_SENTENCES = 50

PRETRAINED_MODEL_PATH = "./model/model.params"
EMBEDDING_SIZE = 768  # 768
NUM_HEADS = 8  # 8
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

zh2idx = load_vocab_json(VOCAB_DIR_PATH + "/zh2idx.json")
en2idx = load_vocab_json(VOCAB_DIR_PATH + "/en2idx.json")
idx2zh = load_vocab_json(VOCAB_DIR_PATH + "/idx2zh.json", key_eq_idx=True)
# idx2en = load_vocab_json(VOCAB_DIR_PATH + "/idx2en.json", key_eq_idx=True)

if __name__ == '__main__':
    test_en = load_file("./dataset/test.en.txt")
    test_zh = load_file("./dataset/test.ch.txt")
    test_dataset = list(zip(test_en, test_zh))
    shuffle(test_dataset)
    model = Transformer(src_vocab_size=VOCAB_SIZE,
                        trg_vocab_size=VOCAB_SIZE,
                        src_pad_idx=0,
                        trg_pad_idx=0,
                        embed_size=EMBEDDING_SIZE,
                        num_layers=NUM_HEADS,
                        device=DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))
    for en, zh in test_dataset[:NUM_SENTENCES]:
        translate = Infer(en, model, zh2idx, idx2zh, en2idx)
        print("EN: " + en + "\n" +
              "MT: " + translate + "\n" +
              "ZH: " + zh + "\n")
