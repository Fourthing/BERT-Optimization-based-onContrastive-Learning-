import pandas as pd

def load_corpus(filepath):
    """加载句子对的数据，返回一个列表，每个元素是一个句子对 (正例/负例)"""
    data = pd.read_csv(filepath)
    # 假设数据是两列，分别是句子1和句子2
    sentences = list(zip(data['sentence1'], data['sentence2']))
    return sentences
