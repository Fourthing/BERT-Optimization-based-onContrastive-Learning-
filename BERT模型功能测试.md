# BERT模型功能测试

**预训练任务**：

- **遮蔽语言模型（Masked Language Model, MLM）**：BERT在预训练过程中，会随机遮蔽输入中的部分单词（如15%），并通过深度双向Transformer网络来预测这些被遮蔽的单词。例如，输入句子“the man went to the [MASK]”中，BERT需要预测出[MASK]为“store”。
- **下一句预测（Next Sentence Prediction, NSP）**：BERT会接收两句话A和B，判断B是否为A的下一句话，还是随机的一句话。通过这种方式来学习句子之间的关系。例如，句子A是“the man went to the store”，句子B是“he bought a gallon of milk”，BERT需要判断B是否是A的下一句。

**下游任务**（Fine-tuning）：

- **句子分类任务**：如情感分析、句子对比任务（例如MultiNLI），通过微调预训练的BERT模型来进行分类。
- **命名实体识别（NER）**：识别文本中的实体（如人名、地点、组织等）。
- **问答任务（如SQuAD）**：BERT被用于回答问题，给定一个问题和一段文本，模型需要从文本中找到问题的答案。

### 1.遮蔽语言模型（MLM）

要实现BERT的**遮蔽语言模型**（MLM），我们可以使用 `transformers` 库中的 `BertForMaskedLM` 模型。这个库提供了对BERT的预训练和微调支持。下面是一个简单的示例代码，展示了如何使用BERT进行MLM预测。

#### 安装依赖
首先确保安装了 `transformers` 和 `torch`：
```bash
pip install transformers torch
```

#### 示例代码

```python
import torch
from transformers import BertTokenizer, BertForMaskedLM

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# 需要预测的句子，使用[MASK]标记需要预测的单词
text = "The man went to the [MASK]."

# 将文本编码为模型输入所需的格式
input_ids = tokenizer.encode(text, return_tensors='pt')

# 找到MASK的位置
mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

# 得到预测结果（logits）
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits

# 找到MASK标记处的预测结果
mask_token_logits = logits[0, mask_token_index, :]

# 获取可能性最高的前5个单词
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

# 打印预测的单词
for token in top_5_tokens:
    predicted_token = tokenizer.decode([token])
    print(f"Predicted token: {predicted_token}")
```

#### 代码解析
- **模型和分词器的加载**：我们使用了 `bert-base-uncased` 这个BERT模型。`BertForMaskedLM` 是专门用于处理MLM任务的模型。
- **输入句子**：我们输入了一个句子 `The man went to the [MASK].`，其中 `[MASK]` 是我们希望模型预测的部分。
- **编码和预测**：使用 `tokenizer.encode` 将句子转换为BERT所需的输入格式，然后通过模型进行预测。
- **提取预测结果**：我们从输出中找到 `[MASK]` 的位置，提取其对应的 logits，最后得到可能性最高的5个单词。

#### 输出示例
```
Predicted token: store
Predicted token: bathroom
Predicted token: hospital
Predicted token: office
Predicted token: house
```

在这个例子中，模型预测了句子中 `[MASK]` 位置可能的单词，如 "store"、"bathroom" 等。

可以根据需要调整输入文本，并观察模型给出的不同

### 2.下一句预测任务（NSP）

接下来是 **Next Sentence Prediction (NSP)**，即下一句预测任务。NSP 的目的是判断两个句子是否是连续的，即第二个句子是否是第一个句子的“下一句”。BERT 的预训练任务之一就是 NSP，因此我们可以直接调用 BERT 模型来完成这个任务。

下面是实现下一句预测（NSP）的示例代码：

```python
import torch
from transformers import BertTokenizer, BertForNextSentencePrediction

# 加载BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

# 输入两个句子
sentence_a = "The man went to the store."
sentence_b = "He bought a gallon of milk."

# 对两个句子进行编码
encoding = tokenizer(sentence_a, sentence_b, return_tensors='pt')

# 使用模型预测
outputs = model(**encoding)
logits = outputs.logits

# 将logits转换为概率
softmax = torch.nn.Softmax(dim=1)
probs = softmax(logits)

# 判断是否为下一句
next_sentence_label = torch.argmax(probs).item()

# 输出结果
if next_sentence_label == 0:
    print(f"'{sentence_b}' 是 '{sentence_a}' 的下一句。")
else:
    print(f"'{sentence_b}' 不是 '{sentence_a}' 的下一句。")
```

#### 代码解释：
1. **加载模型和分词器**：我们使用 `BertTokenizer` 和 `BertForNextSentencePrediction` 类来加载 BERT 模型和分词器。
2. **输入两个句子**：`sentence_a` 是第一句，`sentence_b` 是我们要预测是否为其“下一句”的句子。
3. **编码输入**：`tokenizer` 会对两个句子进行编码，将它们转换为模型能够理解的张量。
4. **模型预测**：我们通过模型的 `logits` 输出，获取两个句子的相关性得分。BERT 使用两个标签来表示结果：`0` 表示句子是相连的，`1` 表示句子不是相连的。
5. **计算结果**：通过 Softmax 将得分转换为概率，`argmax` 得到最终的判断结果。

#### 输出解释：
- **如果 `next_sentence_label == 0`**，说明第二句是第一句的下一句。
- **如果 `next_sentence_label == 1`**，说明第二句不是第一句的下一句。

#### 改进建议：
1. **多语言支持**：可以更换为其他语言的 BERT 模型，如 `bert-base-chinese`，以支持中文或其他语言的句子预测。
2. **批量预测**：可以一次输入多个句子对，进行批量处理，并根据输出的 logits 结果判断哪些句子对是相连的。

---

### 3.句子分类任务（Sentence Classification）

接下来是 **句子分类任务 (Sentence Classification)**。在这个任务中，模型的目标是对输入的句子进行分类。比如，它可以被用来分类情感（积极/消极）、主题分类（新闻类别）、或者其他的文本分类任务。

以下是基于 BERT 模型实现句子分类的示例代码。我们将 BERT 模型用于简单的二分类任务，例如判断句子是否表达“积极”或“消极”的情感。

#### 示例代码：句子分类（情感分类）
```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 加载BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # 二分类任务

# 输入句子
sentence = "I love this movie. It was fantastic!"

# 对输入句子进行编码
inputs = tokenizer(sentence, return_tensors="pt")

# 使用模型进行预测
outputs = model(**inputs)
logits = outputs.logits

# 将logits转换为概率
softmax = torch.nn.Softmax(dim=1)
probs = softmax(logits)

# 得到分类标签
predicted_class = torch.argmax(probs).item()

# 输出结果
if predicted_class == 1:
    print(f"'{sentence}' 表达的是积极的情感。")
else:
    print(f"'{sentence}' 表达的是消极的情感。")
```

#### 代码解释：
1. **加载模型和分词器**：这里我们加载了 `BertForSequenceClassification` 类，它可以用于句子分类任务。在这个例子中，我们将 `num_labels` 参数设置为 2，表示二分类任务（积极/消极）。
2. **输入句子**：可以替换 `sentence` 的值为任意你想分类的句子。
3. **编码输入**：`tokenizer` 会对句子进行编码，将其转换为模型能理解的格式。
4. **模型预测**：通过模型的 `logits` 输出，获取预测的分类标签。`logits` 表示类别的未标准化得分。
5. **计算概率和预测结果**：我们使用 Softmax 函数将 `logits` 转换为概率，然后通过 `argmax` 获取预测的类别标签。`0` 代表消极，`1` 代表积极。

#### 输出解释：
- **如果 `predicted_class == 1`**，模型判断句子表达的是积极情感。
- **如果 `predicted_class == 0`**，模型判断句子表达的是消极情感。

#### 改进建议：
1. **自定义分类任务**：可以调整 `num_labels` 参数，以适应你的特定分类任务，例如多分类（例如新闻分类或话题分类），并加载经过特定任务微调过的模型。
2. **情感分类数据集**：如果你有一个情感分类数据集（例如 IMDB 或其他数据），你可以对模型进行微调，使其更好地适应你的应用场景。

---

### 4.序列标注任务（Token Classification）

接下来我们介绍 **序列标注任务（Token Classification）**，该任务常用于 **命名实体识别 (NER)**、**词性标注 (POS tagging)** 等场景。在这个任务中，我们对输入文本的每一个词（token）进行分类，而不是对整个句子进行分类。

#### 示例代码：命名实体识别 (NER) 任务
在这个示例中，我们将使用 `BertForTokenClassification` 模型，它被设计用来处理序列标注任务。

##### 示例代码：
```python
import torch
from transformers import BertTokenizer, BertForTokenClassification

# 加载BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=9)  # 假设我们有9个NER标签

# NER 标签（BIO标记）
label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

# 输入句子
sentence = "Hawking was a theoretical physicist at Cambridge University."

# 对输入句子进行编码
inputs = tokenizer(sentence, return_tensors="pt", is_split_into_words=False)

# 模型推理
outputs = model(**inputs)
logits = outputs.logits

# 获取每个token的预测标签
predictions = torch.argmax(logits, dim=2)

# 将预测的标签转换为NER标签
predicted_labels = [label_list[pred.item()] for pred in predictions[0]]

# 对应输入的token
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# 输出每个token及其对应的NER标签
for token, label in zip(tokens, predicted_labels):
    print(f"{token}: {label}")
```

#### 代码解释：
1. **加载模型和分词器**：我们加载 `BertForTokenClassification` 模型和 `BertTokenizer`。这里的 `num_labels=9` 假设我们有 9 个不同的命名实体标签（例如人名、组织、地点等）。
2. **输入句子**：示例句子为 "Hawking was a theoretical physicist at Cambridge University."，可以根据需要替换为任意句子。
3. **编码输入**：`tokenizer` 将句子转化为模型可接受的格式，注意 `is_split_into_words=False`，表示整个句子未被分割成单词。
4. **模型推理**：通过模型的 `logits` 输出获取每个词的预测标签。
5. **获取预测标签**：使用 `torch.argmax` 获取每个 token 的预测标签索引，并将其映射到实际的NER标签（如 "B-PER", "I-ORG" 等）。
6. **输出结果**：最后输出每个 token 及其对应的标签。

#### 输出解释：
假设句子中的每个词分别对应如下的NER标签：
```plaintext
Hawking: B-PER
was: O
a: O
theoretical: O
physicist: O
at: O
Cambridge: B-ORG
University: I-ORG
```
- **"B-PER"** 表示这是一个人名实体的开头 ("B" 表示 beginning)。
- **"I-ORG"** 表示这是一个组织实体的后续部分 ("I" 表示 inside)。
- **"O"** 表示不是命名实体。

#### 改进建议：
1. **标签集**：根据不同的数据集和任务，NER标签集可能会有所不同。例如，常见的BIO格式标记可以扩展到更多类别（如产品、日期等）。
2. **模型微调**：如果你有特定的序列标注数据集，可以对BERT模型进行微调，使其更适应你的任务需求。

---

这就是基于 BERT 进行 **序列标注（命名实体识别）** 的基础实现。如果你还需要更多任务的示例代码，或者对其他模型任务有疑问，请继续告诉我！

### 5.问答任务（Q&A）

**问答任务（Question Answering）** 是 BERT 的一个常见应用，例如在 SQuAD（Stanford Question Answering Dataset）数据集中，模型会给定一个问题和一段上下文文本，然后需要从文本中找到并标记出问题的答案。

在这个任务中，BERT 会生成两个输出：
1. **起始位置（start position）**：答案在上下文文本中的起始索引。
2. **结束位置（end position）**：答案在上下文文本中的结束索引。

#### 示例代码：基于 BERT 的问答任务
下面的代码展示了如何使用 BERT 模型回答问题。

##### 示例代码：
```python
import torch
from transformers import BertTokenizer, BertForQuestionAnswering

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

# 输入的上下文段落和问题
context = """Albert Einstein was a German-born theoretical physicist who developed the theory of relativity, 
             one of the two pillars of modern physics (alongside quantum mechanics). His work is also known for 
             its influence on the philosophy of science. He is best known to the general public for his mass–energy 
             equivalence formula E = mc², which has been dubbed "the world's most famous equation"."""
question = "What is Albert Einstein known for?"

# 对问题和上下文进行编码，将二者拼接为模型的输入
inputs = tokenizer(question, context, return_tensors='pt')

# 模型推理，获取 start 和 end logits
outputs = model(**inputs)

# 获取预测的答案起始和结束位置
start_scores = outputs.start_logits
end_scores = outputs.end_logits

# 获取 start 和 end logits 中分数最高的位置
answer_start = torch.argmax(start_scores)
answer_end = torch.argmax(end_scores) + 1  # +1 因为 Python 切片是左闭右开

# 使用tokenizer将模型的输出转换为实际文本答案
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

# 输出结果
print(f"问题: {question}")
print(f"答案: {answer}")
```

#### 代码解析：
1. **加载模型和分词器**：使用 `BertForQuestionAnswering` 模型，它特别用于处理问答任务。`BertTokenizer` 用来将问题和上下文转化为 BERT 可接受的输入格式。
2. **输入上下文和问题**：在代码中，问题是 "What is Albert Einstein known for?"，上下文提供了爱因斯坦的相关描述。
3. **编码输入**：使用 `tokenizer` 将问题和上下文拼接在一起，返回 pytorch tensor 格式。
4. **模型推理**：模型会输出两个 logits：`start_logits` 和 `end_logits`，分别表示答案在上下文中的起始和结束位置。
5. **获取答案位置**：通过 `torch.argmax` 获取答案在上下文中的起始和结束位置（index）。
6. **解码答案**：使用 `convert_ids_to_tokens` 和 `convert_tokens_to_string` 方法将答案索引转换为实际文本。

#### 示例输出：
```plaintext
问题: What is Albert Einstein known for?
答案: his mass–energy equivalence formula E = mc²
```

#### 改进建议：
1. **处理长文本**：BERT 的输入长度有限（一般为 512 个 token），如果上下文太长，可以通过滑动窗口等方法将长段落分片处理。
2. **微调模型**：可以通过 SQuAD 等数据集对 BERT 进行微调，使其适应特定领域的问答任务，效果会更好。
3. **多候选答案**：有时答案的起始和结束位置不一定是连续的，处理多候选答案或非连续答案的任务可以结合 `n-best` 策略，给出多个可能答案。

---

![屏幕截图 2024-10-13 110417](https://gitee.com/De1ores/csdn-picture-bed/raw/master/202410131537361.png)从图片来看，出现了以下问题：

1. **模型权重未完全初始化**：提示部分 `BertForQuestionAnswering` 模型的权重（如 `qa_outputs.bias` 和 `qa_outputs.weight`）没有从预训练的模型中加载出来，而是新初始化的。这通常意味着该模型可能尚未针对问答任务进行微调，因此在问答任务上表现不佳，导致输出为空。

2. **模型未微调**：虽然使用了 `bert-base-uncased` 模型，但这个模型没有经过微调用于问答任务（例如，SQuAD 数据集上的微调）。因此，它缺乏在问答任务上提取有效答案的能力。

#### 解决方法：
1. **使用微调的BERT模型**：可以尝试使用已经在问答任务（如 SQuAD 数据集）上微调过的模型，如 `bert-large-uncased-whole-word-masking-finetuned-squad`。这个模型专门针对问答任务进行了优化。

   示例代码：
   ```python
   from transformers import BertTokenizer, BertForQuestionAnswering

   # 使用在SQuAD上微调的BERT模型
   tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
   model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

   # 输入的上下文段落和问题
   context = """Albert Einstein was a German-born theoretical physicist who developed the theory of relativity, 
                one of the two pillars of modern physics (alongside quantum mechanics). His work is also known for 
                its influence on the philosophy of science. He is best known to the general public for his mass–energy 
                equivalence formula E = mc², which has been dubbed "the world's most famous equation"."""
   question = "What is Albert Einstein known for?"

   # 对问题和上下文进行编码
   inputs = tokenizer(question, context, return_tensors='pt')

   # 推理，获取 start 和 end logits
   outputs = model(**inputs)

   # 获取答案的起始和结束位置
   answer_start = torch.argmax(outputs.start_logits)
   answer_end = torch.argmax(outputs.end_logits) + 1

   # 将 token 转换为字符串
   answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

   print(f"问题: {question}")
   print(f"答案: {answer}")
   ```

2. **检查输入数据**：确保问题和上下文内容编码时没有问题。如果输入过长，也可能会导致模型无法正常推理。你可以缩短上下文文本或者使用滑动窗口策略来处理较长的文本。

