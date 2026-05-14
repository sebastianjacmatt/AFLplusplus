"""Standalone SFT warmup script — produces a saved LoRA adapter checkpoint.

Usage:
    python scripts/sft_warmup.py \
        --config configs/rlm.json \
        --corpus /path/to/jscorpus \
        --steps  3000 \
        --out    ckpts/sft_v1/

The adapter in --out is then loaded by BaseTrainer at AFL++ startup when
TrainingConfig.sft_adapter_path points to that directory.  LoRA config
must match the RL config exactly (same lora_r, lora_alpha, lora_target_modules)
or the adapter weights won't fit.
"""

import argparse
import logging
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Allow running from the rlm_mutator directory without installing as a package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model

from config       import load_config
from masking      import CodeT5SpanMasker
from base_trainer import _SFTDataset, _SFTCollator, _build_corpus_records


def main():
    p = argparse.ArgumentParser(description="SFT warmup — saves a LoRA adapter checkpoint.")
    p.add_argument("--config",     required=True,          help="Path to rlm JSON config.")
    p.add_argument("--corpus",     required=True,          help="Directory of valid program files.")
    p.add_argument("--steps",      type=int, default=None, help="Optimizer steps (default: sft_warmup_steps from config).")
    p.add_argument("--out",        required=True,          help="Output directory for the adapter checkpoint.")
    p.add_argument("--batch-size", type=int, default=8,    help="Per-device train batch size (default 8).")
    p.add_argument("--grad-accum", type=int, default=4,    help="Gradient accumulation steps (default 4, effective batch = batch-size × grad-accum).")
    args = p.parse_args()

    model_cfg, _, train_cfg = load_config(args.config)

    steps = args.steps if args.steps is not None else train_cfg.sft_warmup_steps
    if steps <= 0:
        raise SystemExit(
            "No steps specified: pass --steps N or set sft_warmup_steps in the config."
        )

    if train_cfg.lora_r <= 0:
        raise SystemExit(
            "lora_r must be > 0 — the SFT adapter requires LoRA to be enabled."
        )

    log.info("loading tokenizer and model from %s", model_cfg.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name_or_path)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_cfg.model_name_or_path)
    log.info("wrapping with LoRA r=%d alpha=%d target=%s", train_cfg.lora_r, train_cfg.lora_alpha, train_cfg.lora_target_modules)
    model     = get_peft_model(model, LoraConfig(
        r              = train_cfg.lora_r,
        lora_alpha     = train_cfg.lora_alpha,
        lora_dropout   = train_cfg.lora_dropout,
        target_modules = train_cfg.lora_target_modules_list,
        task_type      = TaskType.SEQ_2_SEQ_LM,
    ))
    model.enable_input_require_grads()  # required for gradient checkpointing with frozen base weights

    masker = CodeT5SpanMasker(
        tokenizer,
        corruption_rate    = model_cfg.mask_probability,
        mean_span_length   = model_cfg.mean_span_length,
        min_span_length    = model_cfg.min_span_length,
        max_span_length    = model_cfg.max_span_length,
        whole_word_masking = model_cfg.whole_word_masking,
    )

    log.info("loading corpus from %s", args.corpus)
    records = _build_corpus_records(tokenizer, masker, args.corpus)
    if not records:
        raise SystemExit(f"No records loaded from {args.corpus}")
    log.info("%d records loaded", len(records))

    batch_size = min(args.batch_size, len(records))
    log.info("effective batch = %d × %d = %d", batch_size, args.grad_accum, batch_size * args.grad_accum)
    sft_args = TrainingArguments(
        output_dir                  = args.out,
        max_steps                   = steps,
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = args.grad_accum,
        gradient_checkpointing      = True,
        learning_rate               = train_cfg.learning_rate,
        warmup_ratio                = 0.1,
        bf16                        = train_cfg.bf16,
        save_strategy               = "no",
        logging_steps               = 10,
        report_to                   = "tensorboard",
    )
    Trainer(
        model         = model,
        args          = sft_args,
        train_dataset = _SFTDataset(records),
        data_collator = _SFTCollator(tokenizer),
    ).train()

    log.info("saving adapter to %s", args.out)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    log.info("done")


if __name__ == "__main__":
    main()
