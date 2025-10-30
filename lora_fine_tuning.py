#!/usr/bin/env python
# coding: utf-8

# 사용 모델: bert-base-uncased
# 
# 항목             |    일반 파인튜닝 |      loRA 파인튜닝
# 
# 
# Validation Accuracy | 0.92 | 0.496
# 
# 
# Training Time    |    227.99  | 163.930
# 
# 
# GPU 메모리 사용량  |  3108MiB |  3108MiB
# 
# 
# 모델 저장 용량   |    417.73  | 418.900
# 

# In[9]:


from google.colab import drive
drive.mount('/content/drive')

get_ipython().run_line_magic('cd', '/content/drive/MyDrive/git/3-fine-tuning')


# In[18]:


get_ipython().system('ls /content/drive/MyDrive/git/3-fine-tuning')


# In[3]:


# 필수 라이브러리 설치
get_ipython().system('pip install -q torch transformers datasets accelerate peft')

import time, os, torch
os.environ["WANDB_DISABLED"] = "true"

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

# GPU 정보 확인
get_ipython().system('nvidia-smi')

MODEL_NAME = "bert-base-uncased"
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)
dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

train_dataset = dataset["train"].shuffle(seed=42).select(range(2000))
eval_dataset = dataset["test"].select(range(500))


# In[4]:


from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# === 일반 Fine-tuning ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

model_base = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

start = time.time()
training_args = TrainingArguments(
    output_dir="./results_base",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=1,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model_base,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
metrics_base = trainer.evaluate()
end = time.time()

# 결과 저장
torch.save(model_base.state_dict(), "base_model.pt")

results_base = {
    "Validation Accuracy": round(metrics_base["eval_accuracy"], 4) if "eval_accuracy" in metrics_base else round(metrics_base["eval_accuracy"] if "accuracy" in metrics_base else metrics_base["accuracy"], 4),
    "Training Time (s)": round(end - start, 2),
    "Model Size (MB)": round(os.path.getsize("base_model.pt") / (1024 * 1024), 2),
}

print("Fine-tuning 결과")
for k, v in results_base.items():
    print(f"{k}: {v}")


# In[5]:


get_ipython().system('nvidia-smi')


# In[6]:


from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType

# === LoRA Fine-tuning ===

# 평가 함수
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# 모델 불러오기
model_lora = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "value"]  # attention 계층에 LoRA 적용
)

# LoRA 모델 생성
model_lora = get_peft_model(model_lora, lora_config)

model_lora.print_trainable_parameters()

# GPU 메모리 초기화
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# 학습 시작
start = time.time()
training_args = TrainingArguments(
    output_dir="./results_lora",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=1,
    logging_dir="./logs",
    disable_tqdm=True
)

trainer = Trainer(
    model=model_lora,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
metrics_lora = trainer.evaluate()
end = time.time()

# GPU 최대 메모리 사용량 측정
gpu_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB 단위

# 모델 저장
torch.save(model_lora.state_dict(), "lora_model.pt")

# 정확도 가져오기
val_acc = metrics_lora.get("eval_accuracy", metrics_lora.get("accuracy", 0.0))

# 결과 정리
results_lora = {
    "Validation Accuracy": round(val_acc, 4),
    "Training Time (s)": round(end - start, 2),
    "GPU Memory (MB)": round(gpu_peak, 2),
    "Model Size (MB)": round(os.path.getsize("lora_model.pt") / (1024 * 1024), 2),
}

# 결과 출력
print(" LoRA Fine-tuning 결과")
for k, v in results_lora.items():
    print(f"{k}: {v}")


# In[7]:


get_ipython().system('nvidia-smi')


# In[8]:


import pandas as pd

comparison = pd.DataFrame({
    "항목": ["Validation Accuracy", "Training Time (s)", "Model Size (MB)"],
    "일반 파인튜닝": [results_base["Validation Accuracy"], results_base["Training Time (s)"], results_base["Model Size (MB)"]],
    "LoRA 파인튜닝": [results_lora["Validation Accuracy"], results_lora["Training Time (s)"], results_lora["Model Size (MB)"]],
})

print("일반 Fine-tuning vs LoRA Fine-tuning 비교 결과")
display(comparison)

