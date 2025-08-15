#!/usr/bin/env python3
import os, argparse, numpy as np
from datasets import load_dataset
from transformers import (
    WhisperForConditionalGeneration, WhisperProcessor,
    Seq2SeqTrainingArguments, Seq2SeqTrainer
)
import evaluate
import torch
import soundfile as sf

# Quieter / safer defaults
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import torchaudio
    HAS_TORCHAUDIO = True
except Exception:
    HAS_TORCHAUDIO = False

# ---- Speech collator (fallback, no extra deps) ----
class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor):
        self.processor = processor
        self.pad_id = processor.tokenizer.pad_token_id
    def __call__(self, features):
        input_feats = [torch.tensor(f["input_features"], dtype=torch.float32) for f in features]
        batch = {"input_features": torch.stack(input_feats, dim=0)}
        labels = [f["labels"] for f in features]
        padded = self.processor.tokenizer.pad({"input_ids": labels}, padding=True, return_tensors="pt")
        label_ids = padded["input_ids"]
        if self.pad_id is not None:
            label_ids = label_ids.masked_fill(label_ids == self.pad_id, -100)
        batch["labels"] = label_ids
        return batch

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_manifest", default="training/train/manifest.jsonl")
    ap.add_argument("--dev_manifest",   default="training/dev/manifest.jsonl")
    ap.add_argument("--model_name",     default="openai/whisper-small")
    ap.add_argument("--language",       default="hi")  # set "" for code-switch
    ap.add_argument("--task",           default="transcribe", choices=["transcribe","translate"])
    ap.add_argument("--epochs",         type=int, default=4)
    ap.add_argument("--lr",             type=float, default=1e-5)
    ap.add_argument("--train_bs",       type=int, default=4)
    ap.add_argument("--eval_bs",        type=int, default=2)
    ap.add_argument("--grad_accum",     type=int, default=8)
    ap.add_argument("--eval_steps",     type=int, default=500)
    ap.add_argument("--logging_steps",  type=int, default=50)
    ap.add_argument("--output_dir",     default="ckpts/whisper")
    ap.add_argument("--fp16",           action="store_true")
    ap.add_argument("--grad_ckpt",      action="store_true")
    ap.add_argument("--seed",           type=int, default=42)
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Data ----
    data_files = {"train": args.train_manifest, "validation": args.dev_manifest}
    ds = load_dataset("json", data_files=data_files)

    # Filter long utterances (>30s) to prevent OOM
    def keep_short(ex): 
        # our manifest has "duration" — default to 0 if missing
        return float(ex.get("duration", 0.0)) <= 30.0
    ds["train"] = ds["train"].filter(keep_short)
    ds["validation"] = ds["validation"].filter(keep_short)

    # ---- Processor / Model ----
    lang = args.language if args.language != "" else None
    processor = WhisperProcessor.from_pretrained(args.model_name, language=lang, task=args.task)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=lang, task=args.task) if lang else None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    if args.grad_ckpt:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()

    # ---- Preprocess ----
    def preprocess(batch):
        path = batch["audio"]
        wav, sr = sf.read(path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            if HAS_TORCHAUDIO:
                wav = torchaudio.functional.resample(torch.from_numpy(wav).float(), sr, 16000).numpy()
            else:
                from scipy.signal import resample_poly
                wav = resample_poly(wav, 16000, sr)
        feats = processor(wav, sampling_rate=16000).input_features[0]
        labels = processor(text=batch["text"]).input_ids
        return {"input_features": feats, "labels": labels}

    ds = ds.map(preprocess, remove_columns=ds["train"].column_names)

    # ---- Metrics ----
    wer_metric = evaluate.load("wer")
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if preds is not None and preds.dtype != np.float32 and preds.dtype != np.float64:
            pred_ids = preds
        else:
            pred_ids = np.argmax(preds, axis=-1)
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
        label_str = processor.batch_decode(labels, skip_special_tokens=True)
        return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

    # ---- Trainer ----
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    args_hf = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=500,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_strategy="steps",
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,   # keep for live WER
        dataloader_num_workers=4,
        report_to=["tensorboard"],
        seed=args.seed,
        remove_unused_columns=False,
        generation_max_length=256,
        group_by_length=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args_hf,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
        tokenizer=processor,  # processing_class
        compute_metrics=compute_metrics,
    )

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    trainer.train()
    print("Final eval:", trainer.evaluate())
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
