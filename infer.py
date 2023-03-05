from model import Transformer
from train import MAX_LEN, VOCAB_SIZE, VOCAB_DIR_PATH
from utils import *
import argparse

PRETRAINED_MODEL_PATH = "./model/model.params"
EMBEDDING_SIZE = 768  # 768
NUM_HEADS = 8  # 8
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Quick Inference')
parser.add_argument("-s", "--sentence", metavar=None, type=str, default="This is a simple instance.")
args = parser.parse_args()


zh2idx = load_vocab_json(VOCAB_DIR_PATH + "/zh2idx.json")
en2idx = load_vocab_json(VOCAB_DIR_PATH + "/en2idx.json")
idx2zh = load_vocab_json(VOCAB_DIR_PATH + "/idx2zh.json", key_eq_idx=True)
# idx2en = load_vocab_json(VOCAB_DIR_PATH + "/idx2en.json", key_eq_idx=True)

def Infer(en_text, model, zh2idx, idx2zh, en2idx):
    if not isinstance(en_text, list):
        en_text = [en_text]
    model.eval()
    with torch.no_grad():
        en_text = delete_invalid_char(en_text, isEN=True)
        en_text = tokenize(en_text, isEN=True)
        en_tensor = sentence2tensor(en_text, vocab=en2idx, max_len=MAX_LEN - 1)
        en_tensor = en_tensor.to(DEVICE)
        zh_sentences = [['<sos>']]
        for i in range(MAX_LEN):
            zh_tensor = sentence2tensor(zh_sentences, vocab=zh2idx, max_len=MAX_LEN).to(DEVICE)
            pre = model(en_tensor, zh_tensor)
            word_idx = torch.argmax(pre[0, i, :])
            word = idx2zh[int(word_idx)]
            zh_sentences[0].append(word)
            if word == '<eos>':
                break

    return ''.join(zh_sentences[0][1:-1])

if __name__ == '__main__':
    test_en = load_file("./dataset/test.en.txt")
    test_zh = load_file("./dataset/test.ch.txt")
    test_dataset = list(zip(test_en, test_zh))
    model = Transformer(src_vocab_size=VOCAB_SIZE,
                        trg_vocab_size=VOCAB_SIZE,
                        src_pad_idx=0,
                        trg_pad_idx=0,
                        embed_size=EMBEDDING_SIZE,
                        num_layers=NUM_HEADS,
                        device=DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))
    translate = Infer(args.sentence, model, zh2idx, idx2zh, en2idx)
    print("EN: " + args.sentence + "\n" +
          "MT: " + translate + "\n")
