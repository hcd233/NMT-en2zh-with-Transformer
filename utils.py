import string
import torch
import jieba
import json
from collections import Counter

PAD_IDX = 0  # padding的索引
UNK_IDX = 1  # 未知词的索引
cn_char = "；·、 "  # 中文常用符号
en_char = "#$%&()*+-/;<=>@[\]^_`{|}~"
space_char = ",.?"
invalid_char_set = string.punctuation + cn_char  # 无效的符号集合，包括标点符号和中文常用符号


# 读取文件并返回以行为元素的列表
def load_file(src) -> list:
    res = []
    with open(src, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        res.append(line[:-2])
    return res


# 删除无效的字符并返回处理后的句子列表
def delete_invalid_char(orig_dataset, isEN=False) -> list:
    proc_dataset = []
    for sentence in orig_dataset:
        proc_sente = ""
        if isEN:
            sentence = sentence.lower()
            for char in sentence:
                if char in space_char:
                    proc_sente += " "+char
                elif char not in en_char:  # 英文中只删除标点符号
                    proc_sente += char
        else:
            for char in sentence:
                if char not in invalid_char_set:  # 中文中删除标点符号和常用符号
                    proc_sente += char
        proc_dataset.append(proc_sente)
    return proc_dataset


# 对句子进行分词并返回处理后的句子列表
def tokenize(orig_dataset, isEN=False) -> list:
    proc_dataset = []
    for sentence in orig_dataset:
        if not isEN:
            tkz_sente = jieba.lcut(sentence)  # 中文使用jieba进行分词
            proc_dataset.append(['<sos>'] + tkz_sente + ['<eos>'])  # 在句子开头和结尾加上<sos>和<eos>
        else:
            tkz_sente = sentence.split(" ")  # 英文使用空格进行分词
            proc_dataset.append(tkz_sente)
    return proc_dataset


# 加载分词后的文件并返回处理后的句子列表
def load_tokenized_file(file_path):
    dataset = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        total_words = file.read()
    total_words = total_words.split(" \n")[:-1]  # 按空格和换行符分割句子
    for sentence in total_words:
        dataset.append(sentence.split(" "))  # 再按空格分割单词
    return dataset


# 将分词后的数据保存到文件中
def save_tokenized_file(tokenized_dataset, file_path, mode='w'):
    with open(file_path, mode=mode, encoding='utf-8') as file:
        for sentence in tokenized_dataset:
            text = "".join(i + " " for i in sentence) + "\n"  # 将单词拼接成字符串，每个句子以换行符结尾
            file.writelines(text)


# 统计词频并返回一个Counter对象
def build_word_freq_dict(dataset):
    total_chars = [c for s in dataset for c in s]
    return Counter(total_chars)


def build_vocab(freq_dataset, vocab_size=30000):
    vocab = {
        '<pad>': PAD_IDX,  # 用于填充句子长度的保留词汇
        '<unk>': UNK_IDX  # 未出现在词汇表中的单词的索引
    }
    for i in range(2, vocab_size):
        vocab[freq_dataset[i - 2][0]] = i
    return vocab


def save_vocab_json(vocab_dict, json_path):
    """
    将词汇表保存为 JSON 格式的文件。

    参数：
    vocab_dict: 词汇表，每个单词作为键，对应的索引作为值。
    json_path: 保存词汇表的文件路径。

    """
    with open(json_path, encoding='utf-8', mode='w') as file:
        json.dump(vocab_dict, file, indent=4, ensure_ascii=False)


def load_vocab_json(vocab_json, key_eq_idx=False) -> dict:
    """
    从 JSON 格式的文件中加载词汇表。

    参数：
    vocab_json: 保存词汇表的 JSON 文件路径。
    key_eq_idx: 是否将键转换为整数类型的索引。如果为 True，将使用 `eval()` 函数将键转换为整数类型。

    返回：
    vocab: 词汇表，每个单词作为键，对应的索引作为值。

    """
    with open(vocab_json, encoding='utf-8', mode='r') as file:
        vocab = json.load(file)
    assert isinstance(vocab, dict)
    if key_eq_idx:
        vocab = {eval(k): v for k, v in vocab.items()}
    return vocab


# 将句子转换为张量
def sentence2tensor(raw, vocab, max_len=20):
    proc_dataset = []
    for sentence in raw:
        sts_len = len(sentence)  # 计算句子长度
        tsr = []
        for word in sentence:  # 将句子中的每个词转换为对应的索引，如果该词不在词表中，则使用 UNK_IDX
            idx = vocab.get(word)
            if idx is None:
                tsr.append(UNK_IDX)
            else:
                tsr.append(idx)
        if sts_len < max_len:  # 如果句子长度小于最大长度，则在句子末尾填充 PAD_IDX，使其长度达到最大长度
            tsr += [1 for _ in range(max_len - sts_len)]
        else:  # 如果句子长度大于最大长度，则截断句子使其长度为最大长度
            tsr = tsr[:max_len]
        proc_dataset.append(tsr)
    return torch.tensor(proc_dataset)  # 如果句子长度大于最大长度，则截断句子使其长度为最大长度
