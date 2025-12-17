"""Model fine-tuning and evaluation"""

import time
import os
import logging

import pandas as pd
import torch
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from sentence_transformers import SentenceTransformer, util
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback

logger = logging.getLogger(__name__)


class FinetuneModel:
    def __init__(self, model_config, sft_config, system_prompt):
        self.model_config = model_config
        self.sft_config = sft_config
        self.system_prompt = system_prompt

    def load_model(self):
        try:
            logger.info(f"Loading model: {self.model_config['model_name']}")
            model, tokenizer = FastModel.from_pretrained(
                model_name=self.model_config['model_name'],
                max_seq_length=self.model_config['max_seq_len'],
                load_in_4bit=True,
            )
            model = FastModel.get_peft_model(
                model,
                r=self.model_config['rank'],
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_alpha=self.model_config['alpha'],
                lora_dropout=self.model_config['dropout'],
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=357841,
                use_rslora=False,
                loftq_config=None,
            )
            tokenizer = get_chat_template(tokenizer, chat_template=self.model_config['chat_template'])
            logger.info("Model loaded successfully")
            return model, tokenizer
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            raise

    def train_model(self, model, tokenizer, train_data, test_data):
        logger.info("Starting training...")
        try:
            supports_bf16 = torch.cuda.is_bf16_supported()

            def process_batch(examples):
                texts = []
                for question, answer in zip(examples["question"], examples["answer"]):
                    question = str(question) if not isinstance(question, str) else question
                    answer = str(answer) if not isinstance(answer, str) else answer
                    conversation = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                    ]
                    text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
                    if "gemma" in self.model_config['chat_template'].lower():
                        text = text.removeprefix('<bos>')
                    texts.append(text)
                return {"text": texts}

            num_proc = os.cpu_count()
            train_dataset = train_data.map(process_batch, batched=True, num_proc=num_proc, remove_columns=train_data.column_names, desc="Formatting Train Data")
            test_dataset = test_data.map(process_batch, batched=True, num_proc=num_proc, remove_columns=test_data.column_names, desc="Formatting Test Data")

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                args=SFTConfig(
                    dataset_text_field="text",
                    per_device_train_batch_size=self.sft_config['batch_size'],
                    gradient_accumulation_steps=1,
                    warmup_steps=int(0.03 * len(train_dataset)),
                    num_train_epochs=self.sft_config['epochs'],
                    max_steps=self.sft_config['epochs'] * int(len(train_dataset) / self.sft_config['batch_size']),
                    learning_rate=self.sft_config['learning_rate'],
                    logging_steps=self.sft_config['logging_steps'],
                    optim="adamw_8bit",
                    weight_decay=0.01,
                    lr_scheduler_type="linear",
                    seed=357841,
                    output_dir="output",
                    report_to="none",
                    eval_accumulation_steps=self.sft_config['eval_accumulation_steps'],
                    save_strategy="steps",
                    save_steps=self.sft_config['save_steps'],
                    save_total_limit=10,
                    eval_strategy="steps",
                    eval_steps=self.sft_config['eval_steps'],
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_loss",
                    greater_is_better=False,
                    fp16=not supports_bf16,
                    bf16=supports_bf16,
                ),
            )

            if self.sft_config['early_stopping_criteria']:
                early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.0)
                trainer.add_callback(early_stopping_callback)

            if "gemma" in self.model_config['chat_template'].lower():
                trainer = train_on_responses_only(trainer, instruction_part="<start_of_turn>user\n", response_part="<start_of_turn>model\n")
            elif "qwen" in self.model_config['chat_template'].lower():
                trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")

            trainer.train()
            logs = trainer.state.log_history
            logs = self.format_logs(logs)
            logger.info("Training completed successfully")
            return model, tokenizer, logs
        except Exception as e:
            logger.error(f"Error during training: {str(e)}", exc_info=True)
            raise

    def evaluate_model(self, model, tokenizer, test_data, template):
        logger.info("Evaluating model...")
        try:
            results = []
            device = "cuda" if torch.cuda.is_available() else "cpu"
            for idx, row in enumerate(test_data):
                start_time = time.time()
                messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": row["question"]}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                if "gemma" in template.lower():
                    text = text.removeprefix("<bos>")
                inputs = tokenizer(text, return_tensors="pt").to(device)
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=125, temperature=1, top_p=0.95, top_k=64)
                prediction = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                correct, similarity = self.semantic_accuracy(prediction, row["answer"])
                precision, recall, f1 = self.token_f1(prediction, row["answer"])
                results.append({"precision": precision, "recall": recall, "f1": f1, "latency": time.time() - start_time})
                if (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx+1}/{len(test_data)}")
            df = pd.DataFrame(results)
            logger.info("Evaluation completed successfully")
            return df[["precision", "recall", "f1", "latency"]].mean().to_dict()
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
            return {}

    def semantic_accuracy(self, prediction, reference, threshold=0.7):
        try:
            embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            emb_pred = embed_model.encode(prediction, convert_to_tensor=True)
            emb_ref = embed_model.encode(reference, convert_to_tensor=True)
            sim = util.cos_sim(emb_pred, emb_ref).item()
            return sim >= threshold, sim
        except Exception:
            return False, 0.0

    def token_f1(self, prediction: str, reference: str):
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()
        common = set(pred_tokens) & set(ref_tokens)
        if len(common) == 0:
            return 0.0, 0.0, 0.0
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return precision, recall, f1

    def format_logs(self, logs):
        train_logs = [log for log in logs if 'loss' in log]
        eval_logs = [log for log in logs if 'eval_loss' in log]
        df_train = pd.DataFrame(train_logs)[['step', 'loss']].rename(columns={'loss': 'train_loss'}) if train_logs else pd.DataFrame(columns=['step', 'train_loss'])
        df_eval = pd.DataFrame(eval_logs)[['step', 'eval_loss']] if eval_logs else pd.DataFrame(columns=['step', 'eval_loss'])
        if not df_train.empty and not df_eval.empty:
            return pd.merge(df_train, df_eval, on='step', how='outer').sort_values(by='step').reset_index(drop=True)
        return df_train if not df_train.empty else pd.DataFrame()