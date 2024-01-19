from sklearn.metrics import accuracy_score
import pandas as pd
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from tqdm import tqdm


def label_salary(salary_str):
    salary_str = salary_str.replace("$", "")
    salary_str = salary_str.replace(",", "")
    salary_str = salary_str.replace(" a year", "")

    salary_list = salary_str.split(" - ")

    return int(((int(salary_list[0]) + int(salary_list[1])) / 2) >= 50_000)


def preprocess(df):
    # Remove rows with no salary
    df = df.dropna(subset=["salary"])

    # Remove rows with non-annual salary
    df = df[df["salary"].str.contains(
        "\\$\\d{1,3}(,\\d{1,3})* - \\$\\d{1,3}(,\\d{1,3})* a year"
    )]

    # 平均値に変換
    df["salary_label"] = df["salary"].apply(label_salary)

    return df


def evaluate(model, dataloader):
    model.eval()  # モデルを評価モードに設定
    total_eval_accuracy = 0
    total_eval_loss = 0
    loss_fn = torch.nn.CrossEntropyLoss()

    for batch in dataloader:
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)

        logits = outputs.logits
        loss = loss_fn(logits, b_labels)
        total_eval_loss += loss.item()

        # 正確さの計算
        preds = torch.argmax(logits, dim=1).flatten()
        total_eval_accuracy += accuracy_score(
            b_labels.cpu().numpy(), preds.cpu().numpy())

    avg_val_accuracy = total_eval_accuracy / len(dataloader)
    avg_val_loss = total_eval_loss / len(dataloader)

    return avg_val_loss, avg_val_accuracy


def __main__():
    # データの読み込みと前処理
    df = pd.read_csv("offers.csv")
    df = preprocess(df)

    # モデルの読み込み
    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased')

    # データのトークン化
    inputs = tokenizer(df["description"].tolist(), padding=True,
                       truncation=True, return_tensors="pt", max_length=512)
    labels = torch.tensor(df["salary_label"].tolist())

    # データセットの作成
    dataset = TensorDataset(inputs["input_ids"],
                            inputs["attention_mask"], labels)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # データローダーの作成
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset),
                                  batch_size=16)
    validation_dataloader = DataLoader(
        val_dataset, sampler=SequentialSampler(val_dataset), batch_size=16)

    # オプティマイザーの設定
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # トレーニングの設定
    epochs = 3

    # トレーニングループ
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('Training...')

        model.train()
        total_train_loss = 0

        for batch in tqdm(train_dataloader):
            b_input_ids, b_input_mask, b_labels = batch

            model.zero_grad()
            outputs = model(
                b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f'Average training loss: {avg_train_loss}')

        print('Validation...')

        model.eval()
        total_eval_loss = 0

        for batch in tqdm(validation_dataloader):
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                outputs = model(
                    b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
                total_eval_loss += loss.item()

        avg_val_loss = total_eval_loss / len(validation_dataloader)
        print(f'Average validation loss: {avg_val_loss}')

    val_loss, val_accuracy = evaluate(model, validation_dataloader)
    print(f'Validation Loss: {val_loss}')
    print(f'Validation Accuracy: {val_accuracy}')

    model.save_pretrained("ditilbert_finetuned.bin")


if __name__ == "__main__":
    __main__()
