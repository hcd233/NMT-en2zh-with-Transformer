import os

import matplotlib.pyplot as plt

from utils import *

IMG_DIR_PATH = "./analysis_img"
if not os.path.exists(IMG_DIR_PATH):
    os.mkdir(IMG_DIR_PATH)

train_zh = load_tokenized_file("./tokenized_dataset/train_zh.txt")
train_en = load_tokenized_file("./tokenized_dataset/train_en.txt")
count_zh = Counter([len(s) for s in train_zh])
count_en = Counter([len(s) for s in train_en])

plt.bar(x=count_zh.keys(), height=count_zh.values())
plt.xlabel("word per sentence")
plt.ylabel("number")
plt.ylim(0, 5000)
plt.savefig(IMG_DIR_PATH + "/zh_sentences_len.png", dpi=300)
plt.show()

plt.bar(x=count_en.keys(), height=count_en.values())
plt.xlabel("word per sentence")
plt.ylabel("number")
plt.ylim(0, 5000)
plt.savefig(IMG_DIR_PATH + "/en_sentences_len.png", dpi=300)
plt.show()

zh_freq_dict = build_word_freq_dict(train_zh)
zh_freq_dict = sorted(zh_freq_dict.items(), key=lambda x: x[1], reverse=True)
en_freq_dict = build_word_freq_dict(train_en)
en_freq_dict = sorted(en_freq_dict.items(), key=lambda x: x[1], reverse=True)

print(zh_freq_dict[:10])
print(en_freq_dict[:10])
print(en_freq_dict[50000])
print(zh_freq_dict[50000])
print("zh_vocab_size:", len(zh_freq_dict))
print("en_vocab_size:", len(en_freq_dict))

x = range(2, 50000)
plt.bar(x=x, height=[zh_freq_dict[i][1] for i in x], width=100)
plt.xlabel("zh word frequency rank")
plt.xlim(-100, 50000)
plt.ylim(0, 5000)
plt.ylabel("number")
plt.savefig(IMG_DIR_PATH + "/zh_word_freq.png", dpi=300)
plt.show()

x = range(2, 50000)
plt.bar(x=x, height=[en_freq_dict[i][1] for i in x], width=100)
plt.xlabel("en word frequency rank")
plt.xlim(-100, 50000)
plt.ylim(0, 5000)
plt.ylabel("number")
plt.savefig(IMG_DIR_PATH + "/en_word_freq.png", dpi=300)
plt.show()
