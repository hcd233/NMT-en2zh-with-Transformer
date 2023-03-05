# 导入必要的模块
import os.path

# 导入自定义的模块和函数
from train import DATASET_PATH, TOKENIZED_PATH
from utils import *

# 如果数据集路径不存在，创建该路径
if not os.path.exists(DATASET_PATH):
    os.mkdir(DATASET_PATH)

# 如果经过分词的数据集路径不存在，创建该路径
if not os.path.exists(TOKENIZED_PATH):
    os.mkdir(TOKENIZED_PATH)


# 比较并加载、保存文本数据集
def cmp_load_save_text(load_dataset, save_dataset):
    # 如果加载数据集和保存数据集的长度不一致，返回 False
    if len(load_dataset) != len(save_dataset):
        print(len(load_dataset), len(save_dataset))
        return False
    # 按顺序遍历加载数据集和保存数据集的每个数据
    for i in range(len(load_dataset)):
        # 如果加载数据集和保存数据集的某个数据长度不一致，返回 False
        if len(load_dataset[i]) != len(save_dataset[i]):
            print(len(load_dataset[i]), len(save_dataset[i]), load_dataset[i], save_dataset[i])
            return False
        # 按顺序遍历加载数据集和保存数据集的每个子词
        for j in range(len(save_dataset[i])):
            # 如果加载数据集和保存数据集的某个子词不一致，返回 False
            if load_dataset[i][j] != save_dataset[i][j]:
                print(load_dataset[i], save_dataset[i])
                return False
    return True


if __name__ == '__main__':

    train_zh = load_file(DATASET_PATH + "/train.ch.txt")
    train_en = load_file(DATASET_PATH + "/train.en.txt")
    val_zh = load_file(DATASET_PATH + "/dev.ch.txt")
    val_en = load_file(DATASET_PATH + "/dev.en.txt")
    print("Load file over!")  # 加载文件完成提示

    # 删除无效字符
    train_zh = delete_invalid_char(train_zh, isEN=False)
    train_en = delete_invalid_char(train_en, isEN=True)
    val_zh = delete_invalid_char(val_zh, isEN=False)
    val_en = delete_invalid_char(val_en, isEN=True)
    print("Delete invalid char over!")  # 删除无效字符完成提示

    # 分词
    train_zh = tokenize(train_zh, isEN=False)
    train_en = tokenize(train_en, isEN=True)
    val_zh = tokenize(val_zh, isEN=False)
    val_en = tokenize(val_en, isEN=True)
    print("Tokenize over!")  # 分词完成提示

    # 保存
    save_tokenized_file(train_zh, TOKENIZED_PATH + "/train_zh.txt")
    save_tokenized_file(train_en, TOKENIZED_PATH + "/train_en.txt")
    save_tokenized_file(val_zh, TOKENIZED_PATH + "/dev_zh.txt")
    save_tokenized_file(val_en, TOKENIZED_PATH + "/dev_en.txt")
    print("Save over!")  # 保存完成提示

    # 读取数据，用来接下来的测试
    load_train_zh = load_tokenized_file(TOKENIZED_PATH + "/train_zh.txt")
    load_train_en = load_tokenized_file(TOKENIZED_PATH + "/train_en.txt")
    load_val_zh = load_tokenized_file(TOKENIZED_PATH + "/dev_zh.txt")
    load_val_en = load_tokenized_file(TOKENIZED_PATH + "/dev_en.txt")
    print("Start compare!")

    # 进行比较，确保文件IO没有bug
    print(cmp_load_save_text(load_train_zh, train_zh),
          cmp_load_save_text(load_train_en, train_en),
          cmp_load_save_text(load_val_zh, val_zh),
          cmp_load_save_text(load_val_en, val_en))
