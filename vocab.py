import os.path
from train import VOCAB_DIR_PATH, TOKENIZED_PATH, VOCAB_SIZE
from utils import *

if not os.path.exists(VOCAB_DIR_PATH):
    os.mkdir(VOCAB_DIR_PATH)


def cmp_load_save_vocab(load_vocab, save_vocab):
    if len(load_vocab) != len(save_vocab):
        print("load_len:" + str(len(load_vocab)), "save_len" + str(len(save_vocab)))
        return False
    for i in load_vocab.keys():
        if load_vocab[i] != save_vocab[i]:
            print(i, load_vocab[i], save_vocab[i])
            return False
    return True


if __name__ == '__main__':
    train_zh = load_tokenized_file(TOKENIZED_PATH + "/train_zh.txt")
    train_en = load_tokenized_file(TOKENIZED_PATH + "/train_en.txt")

    print("zh dataset len:", len(train_zh))
    print("en dataset len:", len(train_en))

    zh_freq_dict = build_word_freq_dict(train_zh)
    zh_freq_dict = sorted(zh_freq_dict.items(), key=lambda x: x[1], reverse=True)

    en_freq_dict = build_word_freq_dict(train_en)
    en_freq_dict = sorted(en_freq_dict.items(), key=lambda x: x[1], reverse=True)
    print(len(en_freq_dict), len(zh_freq_dict))
    zh2idx = build_vocab(zh_freq_dict, vocab_size=VOCAB_SIZE)
    en2idx = build_vocab(en_freq_dict, vocab_size=VOCAB_SIZE)
    idx2zh = dict(zip(zh2idx.values(), zh2idx.keys()))
    idx2en = dict(zip(en2idx.values(), en2idx.keys()))
    print("Build vocab over!")

    save_vocab_json(zh2idx, VOCAB_DIR_PATH + "/zh2idx.json")
    save_vocab_json(en2idx, VOCAB_DIR_PATH + "/en2idx.json")
    save_vocab_json(idx2zh, VOCAB_DIR_PATH + "/idx2zh.json")
    save_vocab_json(idx2en, VOCAB_DIR_PATH + "/idx2en.json")
    print("Save vocab over!")

    load_zh2idx = load_vocab_json(VOCAB_DIR_PATH + "/zh2idx.json")
    load_en2idx = load_vocab_json(VOCAB_DIR_PATH + "/en2idx.json")
    load_idx2zh = load_vocab_json(VOCAB_DIR_PATH + "/idx2zh.json", key_eq_idx=True)
    load_idx2en = load_vocab_json(VOCAB_DIR_PATH + "/idx2en.json", key_eq_idx=True)
    print("Load vocab over!")

    print(cmp_load_save_vocab(load_idx2zh, idx2zh),
          cmp_load_save_vocab(load_idx2en, idx2en),
          cmp_load_save_vocab(load_zh2idx, zh2idx),
          cmp_load_save_vocab(load_en2idx, en2idx))
