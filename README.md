# 使用transformer实现神经机器翻译
## 简介
    任务：实现英文 -> 中文 的神经机器翻译(NMT)

    深度学习框架：Pytorch

    模型：Transformer

    数据集：AI Challenger 2017中的英中机器文本翻译

    训练：GPU Nvidia Tesla A100 40GB * 1
         Epoch = 5
         Train Loss = 1.2
         Valdation Loss = 2.4


    模型性能：BLEU分数：未进行评测

    推理效果：
        EN: I am a cat.   
        ZH: 我是个猫

        EN: how dare you?
        ZH: 你怎么会

        EN: So will you stay?
        CN: 你要留下来吗

        EN: Take him out first.   
        CN: 先把他带走

        EN: I have a question about this scene. Yes?    
        CN: 我有问题关于这个地方的问题是的

        EN: Guests enjoy a complimentary Health Club & Fitness Center that includes: gym, sauna, steam room and hot tub.   
        CN: 客人享受免费的免费的健康设施包括健身中心的健身中心包括健身中心包括游泳池和热水浴缸

        EN: To get back.    
        CN: 回到后面

        EN: Been playing for bigger stakes than I normally would.   
        CN: 我通常会玩更多的<unk>

        EN: I mean, I didn't deal with him directly.    
        CN: 我是说我和他无关

        EN: Are there any families in your neighborhood that you don't get along with?    
        CN: 你家里有家人吗你不和我一起去

        EN: Than he wants us to have it. So they start shooting at each other.    
        CN: 他想让我们让我们这么做所以他们就会打到对方

        EN: You don't want to run the risk of losing 100 dollars.   
        CN: 你不想再<unk>100美元的风险   

## 安装依赖(无版本要求)

```text
torch : https://pytorch.org/
jieba : pip install jieba
```
## 项目代码结构(按执行顺序排序)

    1. preprocess.py ——预处理文件
    2. vocab.py ——构建词表文件
    3. analysis.py ——进行数据分析，确定词表大小还有词长
    4. model.py ——Transformer模型文件
    5. train.py ——训练文件
    6. test.py ——用测试集对模型进行测试
    7. infer.py ——模型推理
    8. utils.py ——编写工具函数的地方

## 项目文件结构

    1. analysis_img ——matplotlib数据分析绘图文件
    2. dataset ——存放数据集
    3. tokenized_dataset ——分词后的数据集
    4. vocabs ——构建的词表
    5. model ——存放训练的模型

## 使用
### 推理
命令如下，需要在infer文件中修改输入参数(还没弄使用argparse的版本)
~~~shell
python test.py
~~~
### 训练
命令如下，还是需要该参数(默认单卡，多卡训练请修改代码，还没弄使用argparse的版本)
~~~shell
python train.py
~~~
### 预处理
命令如下，做的是分词还有构建词表的工作
~~~shell
python preprocess.py
python vocab.py
~~~
### 推理
~~~shell
python infer.py -s "English sentences which you want to translate"
~~~
## 未来项目维护的计划
1. 预处理使用更加高效的方式，更加有效保存词表 BPE
2. 更加方便地使用shell进行训练 利用argparse库
3. 重构代码，这三天写这个项目确实写得很乱，会把不合规范的函数变量重写一次，并且稍作优化
4. 尝试更换模型，Transformer实在是太古老了，但是效果直到现在还是惊人的，会学习并尝试预训练模型
5. 学习些小trick，看看能否增加翻译质量，使用BLEU来评估翻译质量