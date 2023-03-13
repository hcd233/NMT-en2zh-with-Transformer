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
        EN: Do you think the things
        MT: 你觉得事情会怎样吗
        ZH: 你觉得费奇特说的那些

        EN: If we don't get him to a doctor, he will die. Do you understand?
        MT: 如果我们不把他送到医生身上，他会死的。明白吗？
        ZH: 如果不让他看医生他会死的。明白吗？

        EN: I've never been here before. It's okay, Whitney.
        MT: 我以前从没来过过。好吧，惠特尼。
        ZH: 我却没来过这里。没关系，惠特尼。

        EN: I don't know, Jack the Ripper. Oh
        MT: 我不知道，杰克杰克。哦
        ZH: 我不知道... 波兰版开膛手杰克。哦

        EN: Some people force their information on you
        MT: 一些人在你身上<unk>信息
        ZH: 有些人在强迫你接受他们的消息

        EN: I'm not the type of person that can keep things bottled up inside
        MT: 我不是那种能把事情<unk>的人
        ZH: 我不是那种能藏得住话的人

        EN: Each suite features a fully equipped kitchenette, a sitting area with a comfortable sofabed, and two bedrooms- one with a regular bed and one king bed, all         with a Serta mattress.
        MT: 每间套房均配有一个设备齐全的小厨房，一个舒适的休息区，配有一张舒适的沙发，一间卧室配有一张普通的床和一张特大号床，所有配备了一个小厨房。
        ZH: 每一间套房都提供了设施齐全的小厨房、带有舒适沙发床的休息区、两个卧室- 一个带有普通床，另一个带有特大床，全部配有塞塔床垫。

        EN: The three-dragon glazed screen wall is built in Ming Dynasty.
        MT: <unk><unk><unk>建于明代。
        ZH: 寺前有三龙琉璃照壁，是大同市唯一的一座双面照壁，为明代遗物。

        EN: I know what you're thinking. - so?
        MT: 我知道你在想什么。那又怎样？
        ZH: 我知道你在想什么。-所以？

        EN: Come on, boomer!
        MT: 快点，<unk>！
        ZH: 布玛，快走！

        EN: And get other star power to help me.
        MT: 去找另一个明星来帮我。
        ZH: 我可以请其他巨星来帮我。

        EN: I just reckon I'll get the money from a bank
        MT: 我想我可以从银行里拿钱
        ZH: 我觉得还是从银行借钱比较好

        EN: And, well, I'm only being honest with myself when I say that...
        MT: 我是说，我唯一诚实的时候我也是诚实的
        ZH: 嗯，我只是要对我自己说实话……

        EN: I've managed to find our friend.
        MT: 我已经找到了我们的朋友。
        ZH: 我终于找到我们的朋友了。
        
        EN: For the people who lacking basic digital capabilities, they may consider online public services and business transactions as a burden and let alone enjoy           the benefits that are offered by them.
        MT: 对于缺少基本数字能力的人来说，他们可能会考虑在线服务和商业业务，并让人们更愿意享受他们的利益。
        ZH: 对于缺乏基本数字能力的人来说，他们可能会认为在线公共服务和商业交易是一种负担，更不用说享受他们提供的好处了。


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
### 预处理
命令如下，做的是分词还有构建词表的工作
~~~shell
python preprocess.py
python vocab.py
~~~
### 训练
命令如下，还是需要该参数(默认单卡，多卡训练请修改代码，还没弄使用argparse的版本)
~~~shell
python train.py
~~~
### 测试
命令如下，输出测试集中随机几对句子的测试结果
~~~shell
python test.py
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
