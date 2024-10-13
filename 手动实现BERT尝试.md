# 手动实现BERT模型的主要计算流程

这段代码展示了如何通 过手动实现 BERT 模型的主要计算流程，包括嵌入层、Transformer 层中的自注意力机制和前馈神经网络等。代码使用了预训练的 BERT 模型的权重，并通过 NumPy 进行矩阵运算，复现了与 Hugging Face `transformers` 库中 BertModel 类似的行为。

```python
import torch
import math
import numpy as np
from transformers import BertModel

'''

通过手动矩阵运算实现Bert结构
模型文件下载 https://huggingface.co/models

'''

# bert = BertModel.from_pretrained("本地路径", return_dict=False)
bert = BertModel.from_pretrained("bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()
bert.eval()
x = np.array([2450, 15486, 15167, 2110]) #通过vocab对应输入：深度学习
torch_x = torch.LongTensor([x])  

#pytorch形式输入
# seqence_output, pooler_output = bert(torch_x)
# print(seqence_output.shape, pooler_output.shape)
# print(seqence_output, pooler_output)

# print(bert.state_dict().keys())  #查看所有的权值矩阵名称


#softmax归一化
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=-1, keepdims=True)

#gelu激活函数
def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.power(x, 3))))

class DiyBert:
    #将预训练好的整个权重字典输入进来
    def __init__(self, state_dict):
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.num_layers = 1
        self.load_weights(state_dict)

    def load_weights(self, state_dict):
        #embedding部分
        self.word_embeddings = state_dict["embeddings.word_embeddings.weight"].numpy()
        self.position_embeddings = state_dict["embeddings.position_embeddings.weight"].numpy()
        self.token_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"].numpy()
        self.embeddings_layer_norm_weight = state_dict["embeddings.LayerNorm.weight"].numpy()
        self.embeddings_layer_norm_bias = state_dict["embeddings.LayerNorm.bias"].numpy()
        self.transformer_weights = []
        #transformer部分，有多层
        for i in range(self.num_layers):
            q_w = state_dict["encoder.layer.%d.attention.self.query.weight" % i].numpy()
            q_b = state_dict["encoder.layer.%d.attention.self.query.bias" % i].numpy()
            k_w = state_dict["encoder.layer.%d.attention.self.key.weight" % i].numpy()
            k_b = state_dict["encoder.layer.%d.attention.self.key.bias" % i].numpy()
            v_w = state_dict["encoder.layer.%d.attention.self.value.weight" % i].numpy()
            v_b = state_dict["encoder.layer.%d.attention.self.value.bias" % i].numpy()
            attention_output_weight = state_dict["encoder.layer.%d.attention.output.dense.weight" % i].numpy()
            attention_output_bias = state_dict["encoder.layer.%d.attention.output.dense.bias" % i].numpy()
            attention_layer_norm_w = state_dict["encoder.layer.%d.attention.output.LayerNorm.weight" % i].numpy()
            attention_layer_norm_b = state_dict["encoder.layer.%d.attention.output.LayerNorm.bias" % i].numpy()
            intermediate_weight = state_dict["encoder.layer.%d.intermediate.dense.weight" % i].numpy()
            intermediate_bias = state_dict["encoder.layer.%d.intermediate.dense.bias" % i].numpy()
            output_weight = state_dict["encoder.layer.%d.output.dense.weight" % i].numpy()
            output_bias = state_dict["encoder.layer.%d.output.dense.bias" % i].numpy()
            ff_layer_norm_w = state_dict["encoder.layer.%d.output.LayerNorm.weight" % i].numpy()
            ff_layer_norm_b = state_dict["encoder.layer.%d.output.LayerNorm.bias" % i].numpy()
            self.transformer_weights.append([q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias,
                                             attention_layer_norm_w, attention_layer_norm_b, intermediate_weight, intermediate_bias,
                                             output_weight, output_bias, ff_layer_norm_w, ff_layer_norm_b])
        #pooler层
        self.pooler_dense_weight = state_dict["pooler.dense.weight"].numpy()
        self.pooler_dense_bias = state_dict["pooler.dense.bias"].numpy()


    #bert embedding，使用3层叠加，在经过一个embedding层
    def embedding_forward(self, x):
        # x.shape = [max_len]
        we = self.get_embedding(self.word_embeddings, x)  # shpae: [max_len, hidden_size]
        # position embeding的输入 [0, 1, 2, 3]
        pe = self.get_embedding(self.position_embeddings, np.array(list(range(len(x)))))  # shpae: [max_len, hidden_size]
        # token type embedding,单输入的情况下为[0, 0, 0, 0]
        te = self.get_embedding(self.token_type_embeddings, np.array([0] * len(x)))  # shpae: [max_len, hidden_size]
        embedding = we + pe + te
        # 加和后有一个归一化层
        embedding = self.layer_norm(embedding, self.embeddings_layer_norm_weight, self.embeddings_layer_norm_bias)  # shpae: [max_len, hidden_size]
        return embedding

    #embedding层实际上相当于按index索引，或理解为onehot输入乘以embedding矩阵
    def get_embedding(self, embedding_matrix, x):
        return np.array([embedding_matrix[index] for index in x])

    #执行全部的transformer层计算
    def all_transformer_layer_forward(self, x):
        for i in range(self.num_layers):
            x = self.single_transformer_layer_forward(x, i)
        return x

    #执行单层transformer层计算
    def single_transformer_layer_forward(self, x, layer_index):
        weights = self.transformer_weights[layer_index]
        #取出该层的参数，在实际中，这些参数都是随机初始化，之后进行预训练
        q_w, q_b, \
        k_w, k_b, \
        v_w, v_b, \
        attention_output_weight, attention_output_bias, \
        attention_layer_norm_w, attention_layer_norm_b, \
        intermediate_weight, intermediate_bias, \
        output_weight, output_bias, \
        ff_layer_norm_w, ff_layer_norm_b = weights
        #self attention层
        attention_output = self.self_attention(x,
                                q_w, q_b,
                                k_w, k_b,
                                v_w, v_b,
                                attention_output_weight, attention_output_bias,
                                self.num_attention_heads,
                                self.hidden_size)
        #bn层，并使用了残差机制
        x = self.layer_norm(x + attention_output, attention_layer_norm_w, attention_layer_norm_b)
        #feed forward层
        feed_forward_x = self.feed_forward(x,
                              intermediate_weight, intermediate_bias,
                              output_weight, output_bias)
        #bn层，并使用了残差机制
        x = self.layer_norm(x + feed_forward_x, ff_layer_norm_w, ff_layer_norm_b)
        return x

    # self attention的计算
    def self_attention(self,
                       x,
                       q_w,
                       q_b,
                       k_w,
                       k_b,
                       v_w,
                       v_b,
                       attention_output_weight,
                       attention_output_bias,
                       num_attention_heads,
                       hidden_size):
        # x.shape = max_len * hidden_size
        # q_w, k_w, v_w  shape = hidden_size * hidden_size
        # q_b, k_b, v_b  shape = hidden_size
        q = np.dot(x, q_w.T) + q_b  # shape: [max_len, hidden_size]      W * X + B lINER
        k = np.dot(x, k_w.T) + k_b  # shpae: [max_len, hidden_size]
        v = np.dot(x, v_w.T) + v_b  # shpae: [max_len, hidden_size]
        attention_head_size = int(hidden_size / num_attention_heads)
        # q.shape = num_attention_heads, max_len, attention_head_size
        q = self.transpose_for_scores(q, attention_head_size, num_attention_heads)
        # k.shape = num_attention_heads, max_len, attention_head_size
        k = self.transpose_for_scores(k, attention_head_size, num_attention_heads)
        # v.shape = num_attention_heads, max_len, attention_head_size
        v = self.transpose_for_scores(v, attention_head_size, num_attention_heads)
        # qk.shape = num_attention_heads, max_len, max_len
        qk = np.matmul(q, k.swapaxes(1, 2))
        qk /= np.sqrt(attention_head_size)
        qk = softmax(qk)
        # qkv.shape = num_attention_heads, max_len, attention_head_size
        qkv = np.matmul(qk, v)
        # qkv.shape = max_len, hidden_size
        qkv = qkv.swapaxes(0, 1).reshape(-1, hidden_size)
        # attention.shape = max_len, hidden_size
        attention = np.dot(qkv, attention_output_weight.T) + attention_output_bias
        return attention

    #多头机制
    def transpose_for_scores(self, x, attention_head_size, num_attention_heads):
        # hidden_size = 768  num_attent_heads = 12 attention_head_size = 64
        max_len, hidden_size = x.shape
        x = x.reshape(max_len, num_attention_heads, attention_head_size)
        x = x.swapaxes(1, 0)  # output shape = [num_attention_heads, max_len, attention_head_size]
        return x

    #前馈网络的计算
    def feed_forward(self,
                     x,
                     intermediate_weight,  # intermediate_size, hidden_size
                     intermediate_bias,  # intermediate_size
                     output_weight,  # hidden_size, intermediate_size
                     output_bias,  # hidden_size
                     ):
        # output shpae: [max_len, intermediate_size]
        x = np.dot(x, intermediate_weight.T) + intermediate_bias
        x = gelu(x)
        # output shpae: [max_len, hidden_size]
        x = np.dot(x, output_weight.T) + output_bias
        return x

    #归一化层
    def layer_norm(self, x, w, b):
        x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)
        x = x * w + b
        return x

    #链接[cls] token的输出层
    def pooler_output_layer(self, x):
        x = np.dot(x, self.pooler_dense_weight.T) + self.pooler_dense_bias
        x = np.tanh(x)
        return x

    #最终输出
    def forward(self, x):
        x = self.embedding_forward(x)
        sequence_output = self.all_transformer_layer_forward(x)
        pooler_output = self.pooler_output_layer(sequence_output[0])
        return sequence_output, pooler_output


#自制
db = DiyBert(state_dict)
diy_sequence_output, diy_pooler_output = db.forward(x)
#torch
torch_sequence_output, torch_pooler_output = bert(torch_x)

print(diy_sequence_output)
print(torch_sequence_output)

# print(diy_pooler_output)
# print(torch_pooler_output)
```

![屏幕截图 2024-10-13 083934](C:/Users/Star/Pictures/Screenshots/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-13%20083934.png)

![屏幕截图 2024-10-13 083607](https://gitee.com/De1ores/csdn-picture-bed/raw/master/202410130920767.png)

图片展示的是 Hugging Face 网站上的 BERT 模型页面。在这个页面上，并没有直接的“下载”按钮。不过，你可以通过代码来下载模型，或者使用 Hugging Face 的 `transformers` 库来加载模型。以下是你需要做的步骤：

1. **安装依赖库**：
   首先，你需要安装 Hugging Face 的 `transformers` 库和 PyTorch（或 TensorFlow）：

   ```bash
   pip install transformers torch  # 如果你选择使用 PyTorch
   ```

2. **加载模型**：
   可以通过以下代码下载并加载这个模型。

   - 使用 `pipeline` 高级封装：

     ```python
     from transformers import pipeline

     # 创建填空任务的 pipeline，加载谷歌的中文BERT模型
     pipe = pipeline("fill-mask", model="google-bert/bert-base-chinese")

     # 测试输入示例
     result = pipe("巴黎是[MASK]国的首都。")
     print(result)
     ```

   - 直接加载模型和分词器：

     ```python
     from transformers import AutoTokenizer, AutoModelForMaskedLM
     
     # 加载分词器和模型
     tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")
     model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-chinese")
     
     # 编码输入
     inputs = tokenizer("巴黎是[MASK]国的首都。", return_tensors="pt")
     outputs = model(**inputs)
     
     print(outputs)
     ```

3. **如何下载模型**：
   上述代码在运行时会自动下载模型到本地缓存。模型下载后会保存在 Hugging Face 的本地缓存目录（通常位于 `~/.cache/huggingface` 下）。如果想要手动管理或备份这些模型，也可以从缓存中找到它们。



***接下来分步讲解各步骤：***

### 1. **载入预训练模型权重**

```python
#通过给定的路径加载一个预训练的 BERT 模型。
bert = BertModel.from_pretrained(r"D:\badou\pretrain_model\chinese_bert_likes\bert-base-chinese", return_dict=False)
#获取模型的状态字典，包含了模型的所有参数（权重和偏置）。
state_dict = bert.state_dict()
#将模型设置为评估模式。
bert.eval()
```

这一部分使用 `transformers` 库中的 `BertModel` 载入一个已经训练好的 BERT 模型，并提取其权重字典 `state_dict`，权重包含模型中的各类参数如嵌入层和 Transformer 层的权重。

### 2. **输入处理**

```python
x = np.array([2450, 15486, 15167, 2110])  # 通过vocab对应输入：深度学习
torch_x = torch.LongTensor([x])  # pytorch形式输入
```

`x` 是输入的句子，使用词汇表中的索引表示。这是句子"深度学习"的 token id。然后，将其转换为 PyTorch 的张量形式。

### 3. **自定义 BERT 类 (DiyBert)**

定义了 `DiyBert` 类，该类模仿了 BERT 的前向传播过程。`DiyBert` 类包含以下几个核心部分：

- **`__init__`** 和 **`load_weights`**：初始化类并加载 BERT 的预训练权重到类的各个属性中，包括嵌入层和 Transformer 层的参数。
- **`embedding_forward`**：计算输入的词嵌入（word embeddings）、位置嵌入（position embeddings）和 token 类型嵌入（token type embeddings），并通过归一化层对它们进行归一化处理。
- **`all_transformer_layer_forward` 和 `single_transformer_layer_forward`**：逐层执行 Transformer 层的计算。每一层中有两个主要部分：
  - 自注意力机制（self-attention）
  - 前馈神经网络（feed-forward network）
- **`self_attention`**：计算自注意力机制。首先计算查询（query）、键（key）和值（value）向量，通过这些向量计算注意力权重并进行加权求和，得到注意力输出。
- **`feed_forward`**：实现前馈网络，将自注意力层的输出进行非线性变换和线性变换。
- **`layer_norm`**：层归一化，用于对每一层的输出进行标准化，保持训练的稳定性。
- **`pooler_output_layer`**：在 BERT 模型中，`[CLS]` token 的输出通常用于分类任务。该部分实现了输出层对 `sequence_output` 的池化。

### 4. **前向传播 (forward)**

```python
def forward(self, x):
    x = self.embedding_forward(x)  # 嵌入层前向传播
    sequence_output = self.all_transformer_layer_forward(x)  # Transformer 层前向传播
    pooler_output = self.pooler_output_layer(sequence_output[0])  # 池化层
    return sequence_output, pooler_output
```

`forward` 函数首先通过嵌入层生成输入的嵌入表示，接着通过所有 Transformer 层进行前向传播，最后使用池化层对 `sequence_output` 的 `[CLS]` token 的输出进行池化，生成 `pooler_output`。

### 5. **对比结果**

```python
diy_sequence_output, diy_pooler_output = db.forward(x)  # 自定义 BERT
torch_sequence_output, torch_pooler_output = bert(torch_x)  # PyTorch BERT
```

在这部分，`DiyBert` 的结果 `diy_sequence_output` 和 `diy_pooler_output` 与 `transformers` 库中 PyTorch 的 BERT 模型的输出 `torch_sequence_output` 和 `torch_pooler_output` 进行对比。通过比较自定义实现和 PyTorch 模型的输出，可以验证自定义实现的准确性。



