import torch
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
