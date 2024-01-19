import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# トークナイザーとモデルのロード
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=2)

# モデルの読み込み
model_load_path = "distilbert_finetuned.bin"
model.load_state_dict(torch.load(model_load_path))
model.eval()
