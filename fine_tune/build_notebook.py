"""Build a single Colab notebook with embedded data."""
import gzip, base64, json, os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_PATH = os.path.join(os.path.dirname(__file__), "train_colab.ipynb")

# Compress and encode data files
data_blobs = {}
for name in ["train_t5.jsonl", "eval_t5.jsonl"]:
    path = os.path.join(DATA_DIR, name)
    with open(path, "rb") as f:
        raw = f.read()
    compressed = gzip.compress(raw, compresslevel=9)
    data_blobs[name] = base64.b64encode(compressed).decode("ascii")
    print(f"{name}: {len(raw)} -> {len(data_blobs[name])} base64 chars")

# Build cells
cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [source]})

def code(source):
    cells.append({"cell_type": "code", "metadata": {}, "source": [source], "execution_count": None, "outputs": []})

md(
    "# ByT5 Fine-tuning for E-Commerce Typo Correction\n\n"
    "Bu notebook, ByT5 modelini e-ticaret typo duzeltme icin fine-tune eder.\n\n"
    "**Kullanim:** Runtime > Change runtime type > **H100 GPU** secin, sonra hucreleri sirayla calistirin.\n\n"
    "Veri dosyalari notebook icinde gomulu olarak gelmektedir."
)

md("## 1. Bagimliliklari Kur")

code(
    "!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\n"
    "!pip install -q transformers accelerate sentencepiece tensorboard"
)

md("## 2. GPU Kontrol")

code(
    'import torch\n'
    '\n'
    'if torch.cuda.is_available():\n'
    '    gpu_name = torch.cuda.get_device_name(0)\n'
    '    cap = torch.cuda.get_device_capability()\n'
    '    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3\n'
    '    print(f"GPU: {gpu_name}")\n'
    '    print(f"Compute capability: {cap[0]}.{cap[1]}")\n'
    '    print(f"VRAM: {vram:.1f} GB")\n'
    'else:\n'
    '    print("GPU bulunamadi! Runtime > Change runtime type > GPU secin.")'
)

md(
    "## 3. Gomulu Veriyi Diske Yaz\n\n"
    "Veri dosyalari notebook icinde gzip+base64 olarak gomulu. Bu hucre onlari diske cikarir."
)

data_cell = (
    'import gzip, base64\n'
    '\n'
    'TRAIN_B64 = """' + data_blobs["train_t5.jsonl"] + '"""\n'
    '\n'
    'EVAL_B64 = """' + data_blobs["eval_t5.jsonl"] + '"""\n'
    '\n'
    'TRAIN_FILE = "/content/train_t5.jsonl"\n'
    'EVAL_FILE = "/content/eval_t5.jsonl"\n'
    '\n'
    'for path, b64data in [(TRAIN_FILE, TRAIN_B64), (EVAL_FILE, EVAL_B64)]:\n'
    '    raw = gzip.decompress(base64.b64decode(b64data))\n'
    '    with open(path, "wb") as f:\n'
    '        f.write(raw)\n'
    '    lines = sum(1 for _ in open(path))\n'
    '    print(f"{path}: {lines} satir yazildi")\n'
    '\n'
    'del TRAIN_B64, EVAL_B64  # Bellek temizle'
)
code(data_cell)

md("## 4. Konfigurasyon")

code(
    '# ===================== AYARLAR =====================\n'
    'MODEL_NAME = "google/byt5-base"       # google/byt5-small, google/byt5-base, google/byt5-large\n'
    'EPOCHS = 15\n'
    'BATCH_SIZE = 12                        # T4 icin: small=16, base=12, large=4\n'
    'LEARNING_RATE = 5e-5                   # small=3e-4, base=5e-5, large=3e-5\n'
    'GRADIENT_ACCUMULATION = 2              # small=1, base=2, large=4\n'
    'GRADIENT_CHECKPOINTING = False         # large icin True yapin\n'
    'LABEL_SMOOTHING = 0.1\n'
    'MAX_LENGTH = 128\n'
    'EVAL_STEPS = 500\n'
    'EVAL_SUBSET = 500                      # Hizli eval icin\n'
    'LOGGING_STEPS = 50\n'
    'PATIENCE = 5                           # Early stopping\n'
    'OUTPUT_DIR = "/content/byt5-typo"\n'
    '# =================================================='
)

md("## 5. Dataset Sinifi")

code(
    'import json\n'
    'from torch.utils.data import Dataset\n'
    '\n'
    '\n'
    'class TypoCorrectionDataset(Dataset):\n'
    '    """JSONL formatinda {"input_text": ..., "target_text": ...} yukler."""\n'
    '\n'
    '    def __init__(self, filepath, tokenizer, max_source_len=128, max_target_len=128):\n'
    '        self.tokenizer = tokenizer\n'
    '        self.max_source_len = max_source_len\n'
    '        self.max_target_len = max_target_len\n'
    '        self.examples = []\n'
    '\n'
    '        with open(filepath, "r", encoding="utf-8") as f:\n'
    '            for line in f:\n'
    '                line = line.strip()\n'
    '                if line:\n'
    '                    self.examples.append(json.loads(line))\n'
    '\n'
    '        print(f"  Loaded {len(self.examples)} examples from {filepath}")\n'
    '\n'
    '    def __len__(self):\n'
    '        return len(self.examples)\n'
    '\n'
    '    def __getitem__(self, idx):\n'
    '        ex = self.examples[idx]\n'
    '        source = ex["input_text"]\n'
    '        target = ex["target_text"]\n'
    '\n'
    '        source_enc = self.tokenizer(\n'
    '            source,\n'
    '            max_length=self.max_source_len,\n'
    '            truncation=True,\n'
    '        )\n'
    '        target_enc = self.tokenizer(\n'
    '            target,\n'
    '            max_length=self.max_target_len,\n'
    '            truncation=True,\n'
    '        )\n'
    '\n'
    '        return {\n'
    '            "input_ids": source_enc["input_ids"],\n'
    '            "attention_mask": source_enc["attention_mask"],\n'
    '            "labels": target_enc["input_ids"],\n'
    '        }'
)

md("## 6. Metrik Fonksiyonlari")

code(
    'import numpy as np\n'
    '\n'
    '\n'
    'def build_compute_metrics(tokenizer):\n'
    '    """Seq2SeqTrainer icin compute_metrics fonksiyonu."""\n'
    '\n'
    '    def _cer(pred, ref):\n'
    '        """Character Error Rate (edit distance)."""\n'
    '        n = len(ref) or 1\n'
    '        if pred == ref:\n'
    '            return 0.0\n'
    '        d = list(range(len(ref) + 1))\n'
    '        for i, pc in enumerate(pred):\n'
    '            nd = [i + 1] + [0] * len(ref)\n'
    '            for j, rc in enumerate(ref):\n'
    '                cost = 0 if pc == rc else 1\n'
    '                nd[j + 1] = min(nd[j] + 1, d[j + 1] + 1, d[j] + cost)\n'
    '            d = nd\n'
    '        return d[-1] / n\n'
    '\n'
    '    def compute_metrics(eval_preds):\n'
    '        predictions, label_ids = eval_preds\n'
    '\n'
    '        # Logits (3D) ise argmax ile token ID ye cevir\n'
    '        if predictions.ndim == 3:\n'
    '            predictions = np.argmax(predictions, axis=-1)\n'
    '\n'
    '        # -100 ve gecersiz ID leri pad token ile degistir\n'
    '        label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)\n'
    '        predictions = np.where(\n'
    '            (predictions >= 0) & (predictions < tokenizer.vocab_size),\n'
    '            predictions,\n'
    '            tokenizer.pad_token_id,\n'
    '        )\n'
    '\n'
    '        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)\n'
    '        decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)\n'
    '\n'
    '        decoded_preds = [p.strip() for p in decoded_preds]\n'
    '        decoded_labels = [l.strip() for l in decoded_labels]\n'
    '\n'
    '        # Sentence-level exact-match accuracy\n'
    '        correct = sum(1 for p, t in zip(decoded_preds, decoded_labels) if p == t)\n'
    '        accuracy = correct / len(decoded_preds) if decoded_preds else 0.0\n'
    '\n'
    '        # Token-level accuracy\n'
    '        total_tokens = 0\n'
    '        correct_tokens = 0\n'
    '        for p, t in zip(decoded_preds, decoded_labels):\n'
    '            p_tok = p.split()\n'
    '            t_tok = t.split()\n'
    '            min_len = min(len(p_tok), len(t_tok))\n'
    '            for i in range(min_len):\n'
    '                if p_tok[i] == t_tok[i]:\n'
    '                    correct_tokens += 1\n'
    '            total_tokens += max(len(p_tok), len(t_tok))\n'
    '        token_acc = correct_tokens / total_tokens if total_tokens else 0.0\n'
    '\n'
    '        # Character Error Rate\n'
    '        cer_scores = [_cer(p, t) for p, t in zip(decoded_preds, decoded_labels)]\n'
    '        avg_cer = sum(cer_scores) / len(cer_scores) if cer_scores else 0.0\n'
    '\n'
    '        return {\n'
    '            "sentence_accuracy": round(accuracy, 4),\n'
    '            "token_accuracy": round(token_acc, 4),\n'
    '            "cer": round(avg_cer, 4),\n'
    '        }\n'
    '\n'
    '    return compute_metrics'
)

md("## 7. Model ve Tokenizer Yukle")

code(
    'from transformers import AutoTokenizer, T5ForConditionalGeneration\n'
    '\n'
    'print(f"Model yukleniyor: {MODEL_NAME} ...")\n'
    'tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n'
    'model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)\n'
    '\n'
    'if GRADIENT_CHECKPOINTING:\n'
    '    model.gradient_checkpointing_enable()\n'
    '    print("Gradient checkpointing aktif")\n'
    '\n'
    'param_count = sum(p.numel() for p in model.parameters())\n'
    'trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)\n'
    'print(f"Parametreler: {param_count:,} toplam, {trainable:,} trainable")'
)

md("## 8. Datasetleri Yukle")

code(
    'print("Datasetler yukleniyor...")\n'
    'train_ds = TypoCorrectionDataset(TRAIN_FILE, tokenizer, MAX_LENGTH, MAX_LENGTH)\n'
    'eval_ds_full = TypoCorrectionDataset(EVAL_FILE, tokenizer, MAX_LENGTH, MAX_LENGTH)\n'
    '\n'
    '# Hizli eval icin subset\n'
    'eval_size = min(EVAL_SUBSET, len(eval_ds_full))\n'
    'if eval_size < len(eval_ds_full):\n'
    '    eval_ds = torch.utils.data.Subset(eval_ds_full, list(range(eval_size)))\n'
    '    print(f"  Eval subset: {eval_size}/{len(eval_ds_full)}")\n'
    'else:\n'
    '    eval_ds = eval_ds_full'
)

md("## 9. Egitim")

code(
    'import os\n'
    'from transformers import (\n'
    '    Seq2SeqTrainer,\n'
    '    Seq2SeqTrainingArguments,\n'
    '    EarlyStoppingCallback,\n'
    '    DataCollatorForSeq2Seq,\n'
    ')\n'
    '\n'
    '# Precision ayari\n'
    'use_fp16 = False\n'
    'use_bf16 = False\n'
    'if torch.cuda.is_available():\n'
    '    cap = torch.cuda.get_device_capability()\n'
    '    if cap[0] >= 8:  # Ampere+ -> BF16\n'
    '        use_bf16 = True\n'
    '        print(f"BF16 kullaniliyor (compute capability {cap[0]}.{cap[1]})")\n'
    '    else:\n'
    '        use_fp16 = True\n'
    '        print("FP16 kullaniliyor")\n'
    '\n'
    '# Data collator\n'
    'data_collator = DataCollatorForSeq2Seq(\n'
    '    tokenizer=tokenizer,\n'
    '    model=model,\n'
    '    padding=True,\n'
    '    label_pad_token_id=-100,\n'
    ')\n'
    '\n'
    '# Warmup hesapla\n'
    'steps_per_epoch = len(train_ds) // (BATCH_SIZE * GRADIENT_ACCUMULATION)\n'
    'total_steps = steps_per_epoch * EPOCHS\n'
    'warmup_steps = int(total_steps * 0.1)\n'
    '\n'
    'os.makedirs(OUTPUT_DIR, exist_ok=True)\n'
    '\n'
    'training_args = Seq2SeqTrainingArguments(\n'
    '    output_dir=OUTPUT_DIR,\n'
    '    num_train_epochs=EPOCHS,\n'
    '    per_device_train_batch_size=BATCH_SIZE,\n'
    '    per_device_eval_batch_size=BATCH_SIZE * 2,\n'
    '    learning_rate=LEARNING_RATE,\n'
    '    lr_scheduler_type="cosine",\n'
    '    warmup_steps=warmup_steps,\n'
    '    weight_decay=0.01,\n'
    '    label_smoothing_factor=LABEL_SMOOTHING,\n'
    '    max_grad_norm=1.0,\n'
    '    fp16=use_fp16,\n'
    '    bf16=use_bf16,\n'
    '    gradient_accumulation_steps=GRADIENT_ACCUMULATION,\n'
    '    eval_strategy="steps",\n'
    '    eval_steps=EVAL_STEPS,\n'
    '    save_strategy="steps",\n'
    '    save_steps=EVAL_STEPS,\n'
    '    save_total_limit=3,\n'
    '    load_best_model_at_end=True,\n'
    '    metric_for_best_model="sentence_accuracy",\n'
    '    greater_is_better=True,\n'
    '    logging_steps=LOGGING_STEPS,\n'
    '    predict_with_generate=True,\n'
    '    generation_max_length=64,\n'
    '    generation_num_beams=1,\n'
    '    report_to=["tensorboard"],\n'
    '    dataloader_num_workers=2,\n'
    '    dataloader_pin_memory=True,\n'
    '    remove_unused_columns=False,\n'
    ')\n'
    '\n'
    'trainer = Seq2SeqTrainer(\n'
    '    model=model,\n'
    '    args=training_args,\n'
    '    train_dataset=train_ds,\n'
    '    eval_dataset=eval_ds,\n'
    '    data_collator=data_collator,\n'
    '    processing_class=tokenizer,\n'
    '    compute_metrics=build_compute_metrics(tokenizer),\n'
    '    callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],\n'
    ')\n'
    '\n'
    'print(f"\\nToplam step: {total_steps}, Warmup: {warmup_steps}")\n'
    'print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")\n'
    'print("\\nEgitim basliyor...\\n")\n'
    'trainer.train()'
)

md("## 10. TensorBoard (Opsiyonel)")

code('%load_ext tensorboard\n%tensorboard --logdir /content/byt5-typo')

md("## 11. Final Evaluation")

code(
    'print("Final evaluation...")\n'
    'metrics = trainer.evaluate()\n'
    'print(json.dumps(metrics, indent=2))'
)

md("## 12. Modeli Kaydet")

code(
    'best_dir = os.path.join(OUTPUT_DIR, "best")\n'
    'os.makedirs(best_dir, exist_ok=True)\n'
    'trainer.save_model(best_dir)\n'
    'tokenizer.save_pretrained(best_dir)\n'
    'print(f"Model kaydedildi: {best_dir}")\n'
    '\n'
    '# Metrikleri kaydet\n'
    'with open(os.path.join(OUTPUT_DIR, "final_metrics.json"), "w") as f:\n'
    '    json.dump(metrics, f, indent=2)\n'
    'print("Metrikler kaydedildi.")'
)

md("## 13. Modeli Test Et")

code(
    '# Hizli test\n'
    'test_inputs = [\n'
    '    "correct: samsng galxy",\n'
    '    "correct: iphne 15 pro maks",\n'
    '    "correct: logitech mause",\n'
    '    "correct: nvidai rtx 4090",\n'
    '    "correct: macbok pro m3",\n'
    '    "correct: kulaklık bluettoh",\n'
    ']\n'
    '\n'
    'model.eval()\n'
    'for text in test_inputs:\n'
    '    inputs = tokenizer(text, return_tensors="pt").to(model.device)\n'
    '    with torch.no_grad():\n'
    '        outputs = model.generate(**inputs, max_length=64)\n'
    '    result = tokenizer.decode(outputs[0], skip_special_tokens=True)\n'
    '    print(f"  {text:40s} -> {result}")'
)

md("## 14. Modeli Indir\n\nEgitim tamamlandiktan sonra modeli ziplayip indirin.")

code(
    'import shutil\n'
    '\n'
    '# Zip olustur\n'
    'zip_path = shutil.make_archive("/content/byt5-typo-best", "zip", best_dir)\n'
    'print(f"Zip olusturuldu: {zip_path}")\n'
    '\n'
    '# Colab dan indir\n'
    'from google.colab import files\n'
    'files.download(zip_path)'
)

md("## 15. Google Drive a Kaydet (Opsiyonel)")

code(
    '# Drive a kaydetmek isterseniz:\n'
    '# from google.colab import drive\n'
    '# drive.mount("/content/drive")\n'
    '# !cp -r /content/byt5-typo/best /content/drive/MyDrive/byt5-typo-best\n'
    '# print("Model Drive a kaydedildi.")'
)

# Build notebook
notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

size_mb = os.path.getsize(OUT_PATH) / 1024 / 1024
print(f"\nNotebook yazildi: {OUT_PATH}")
print(f"Boyut: {size_mb:.1f} MB")
