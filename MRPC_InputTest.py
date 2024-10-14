from datasets import load_dataset

# 加载MRPC数据集
dataset = load_dataset("glue", "mrpc")

# 选择训练集
train_dataset = dataset['train']

# 打印前5个样本的 sentence1, sentence2 和 label
for i in range(5):
    sentence1 = train_dataset[i]['sentence1']
    sentence2 = train_dataset[i]['sentence2']
    label = train_dataset[i]['label']
    print(f"Sample {i+1}:")
    print(f"Sentence 1: {sentence1}")
    print(f"Sentence 2: {sentence2}")
    print(f"Label: {label}")
    print("="*50)
