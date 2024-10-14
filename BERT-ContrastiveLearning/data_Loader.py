import os
import pandas as pd
from datasets import load_dataset


def load_corpus(filepath):
    # 加载MRPC数据集
    dataset = load_dataset("glue", "mrpc")

    # 选择训练集
    train_dataset = dataset['train']

    # 创建存储路径（如果不存在）
    output_dir = './data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 定义要存储的CSV文件路径
    output_file = os.path.join(output_dir, 'sentence_pairs.csv')

    # 创建一个列表，存储每个样本的句子对和标签
    data = []

    # 遍历训练集中的样本，将句子对和标签存储到列表中
    for i in range(len(train_dataset)):
        sentence1 = train_dataset[i]['sentence1']
        sentence2 = train_dataset[i]['sentence2']
        label = train_dataset[i]['label']  # 标签
        data.append([sentence1, sentence2, label])

    # 使用 pandas 将数据保存到 CSV 文件中
    df = pd.DataFrame(data, columns=["sentence1", "sentence2", "label"])
    df.to_csv(output_file, index=False)

    print(f"数据已成功保存到 {output_file}")

    """加载句子对的数据，返回一个列表，每个元素是一个句子对 (正例/负例)"""
    data = pd.read_csv(filepath)
    # 假设数据是两列，分别是句子1和句子2
    sentences = list(zip(data['sentence1'], data['sentence2']))
    return sentences
