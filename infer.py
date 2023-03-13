from model import Transformer
from train import MAX_LEN, VOCAB_SIZE, VOCAB_DIR_PATH
from utils import *
import argparse

EMBEDDING_SIZE = 1024  # 768
NUM_HEADS = 12  # 8
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
PRETRAINED_MODEL_PATH = "./model/model_2_1024_12.params"
parser = argparse.ArgumentParser(description='Quick Inference')
parser.add_argument("-s", "--sentence", metavar=None, type=str, default="Input your English sentence, such as "
                                                                        "this instance.")
parser.add_argument("-i", "--infer_mode", metavar=None, type=bool, default=False)
args = parser.parse_args()

zh2idx = load_vocab_json(VOCAB_DIR_PATH + "/zh2idx.json")
en2idx = load_vocab_json(VOCAB_DIR_PATH + "/en2idx.json")
idx2zh = load_vocab_json(VOCAB_DIR_PATH + "/idx2zh.json", key_eq_idx=True)


# idx2en = load_vocab_json(VOCAB_DIR_PATH + "/idx2en.json", key_eq_idx=True)
def InferMode(model, zh2idx, idx2zh, en2idx):
    print("持续推理模式，输入Q可以退出.")
    model.eval()
    with torch.no_grad():
        while True:
            en_text = input("EN: ")
            if not isinstance(en_text, list):
                en_text = [en_text]
            if en_text == ["Q"]:
                break
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
            print("MT: "+"".join(zh_sentences[0][1:-1]))


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
    model = Transformer(src_vocab_size=VOCAB_SIZE,
                        trg_vocab_size=VOCAB_SIZE,
                        src_pad_idx=0,
                        trg_pad_idx=0,
                        embed_size=EMBEDDING_SIZE,
                        num_layers=NUM_HEADS,
                        device=DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location='cuda:0'))
    if args.infer_mode:
        InferMode(model, zh2idx, idx2zh, en2idx)
    else:
        translate = Infer(args.sentence, model, zh2idx, idx2zh, en2idx)
        print("EN: " + args.sentence + "\n" +
              "MT: " + translate + "\n")
