# 修正適用 fastapi_llm_server.py
import os
import torch
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import Optional
from unsloth import FastLanguageModel, FastModel
from trl import SFTTrainer, SFTConfig # SFTTrainer をインポート
from peft import LoraConfig
from datasets import Dataset
import json
from transformers import pipeline, TrainingArguments, Trainer
import shutil
import uvicorn
from evaluate import load
import numpy as np
import logging
import threading # threading をインポート
import sys
import math
import copy

app = FastAPI()

# -------------------------------------------------------------
# ロガーの設定 (ファイルの先頭に近い場所で一度だけ行う)
# print() の代わりに logger を使用
# 例: print(f"Loading model: {MODEL_NAME}") の代わりに  
# logger.info(f"Loading model: {MODEL_NAME}")
# -------------------------------------------------------------
# logger = logging.getLogger(__name__) # アプリケーション固有のロガーに戻す
logger = logging.getLogger("fastapi_llm_server") # LOGGING_CONFIGで設定した名前に合わせる


# MODEL_NAME = "unsloth/gemma-3-4B-it"
MODEL_NAME = "unsloth/gemma-3-1b-it"
# MODEL_NAME = "google/gemma-3-4b-it"
# MODEL_NAME = "google/gemma-2b-it"
ACTIVE_MODEL = None
ACTIVE_MODEL_TYPE = None # "base" or "trained"
TOKENIZER = None # Tokenizerは共通で一度ロード
model_lock = threading.Lock()
TRAINED_PATH = "./gemma3_unsloth_ckc_fine_clm"
DATA_PATH = "attack_ent_with_subtec_ckc_mapping_jp.json"
MAX_SEQ_LENGTH = 2048   # 8192 -> 2048 -> 8192 -> 4096 -> 2048


INITIAL_TRAINING_STATUS  = {
    "status": "Not started",
    "loss": None,
    "val_loss": None,
    "progress": 0,
    "step": 0,
    "epoch": 0,
    "accuracy": 0.0,
    "perplexity": 0.0,
    "metrics": {},
    "history": {
        "train_steps": [],
        "train_loss": [],
        "eval_steps": [],
        "val_loss": [],
    }
}

training_status = INITIAL_TRAINING_STATUS.copy()


class InferenceRequest(BaseModel):
    prompt: str
    use_trained_model: bool = False
    max_new_tokens: int = 2045  # 生成するトークンの最大長。タスクに応じて調整してください。
    do_sample: bool = False     # 決定論的な出力を得るために False に設定
    temperature: float = 0.8    # do_sample=False の場合、この値は無視されます
    top_p: float = 0.9          # do_sample=False の場合、この値は無視されます
    top_k: int = 50             # do_sample=False の場合、この値は無視されます
    repetition_penalty: float = 1.2 # 応答内の不要な繰り返しを抑制します。1.0でペナルティなし。


class TrainConfig(BaseModel):
    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 5e-5
    logging_steps: int = 1
    eval_steps: int = 100
    save_steps: int = 100


# -------------------------------------------------------------
# メモリ使用量を監視するユーティリティ
# -------------------------------------------------------------
def print_gpu_memory():
    if torch.cuda.is_available():
        logger.info(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB / "
                    f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB")


# -------------------------------------------------------------
# メモリをクリアする
# -------------------------------------------------------------
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()


def _create_peft_configured_base_model_and_tokenizer():
    """
    ベースモデルとトークナイザーをロードし、PEFT設定を適用する。
    LoRAを使用するため full_finetuning は False に設定。
    """
    logger.info(f"Creating PEFT-configured base model: {MODEL_NAME} with max_seq_length: {MAX_SEQ_LENGTH}")

    # -------------------------------------------------------------
    # モデル初期化
    # -------------------------------------------------------------
    # UnslothのFastLanguageModelを使用
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        # dtype = torch.float16,
        dtype = torch.bfloat16,
        load_in_4bit = True,  # 4 bit quantization to reduce memory
        load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
        full_finetuning = False, # [NEW!] We have full finetuning now! : LoRAを使用するためFalseに設定
    )

    # -------------------------------------------------------------
    # LoRA設定
    # -------------------------------------------------------------
    logger.info("Model loaded. Applying PEFT (LoRA) settings...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 8, # LoRAランク 16 -> 8
        lola_rank = 8,
        lora_alpha = 16,    # 32 -> 16
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_dropout = 0, # LoRAドロップアウト率  Supports any, but = 0 is optimized 0.05 -> 0
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,    # 0 -> 3407
        max_seq_length = MAX_SEQ_LENGTH,
        use_rslora = True,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    logger.info("PEFT (LoRA) settings applied to base model.")
    return model, tokenizer

def _load_adapter_to_model(model, adapter_path):
    logger.info(f"Loading adapter from {adapter_path} into the provided model.")
    try:
        model.load_adapter(adapter_path, adapter_name="default")
        # ロードしたアダプターを推論で有効にするためにアクティブ化します
        model.set_adapter("default")
        logger.info("Adapter loaded and set to active successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load and set adapter from {adapter_path}: {e}", exc_info=True)
        raise

def switch_active_model(target_model_type: str):
    global ACTIVE_MODEL, ACTIVE_MODEL_TYPE, TOKENIZER
    with model_lock:
        if ACTIVE_MODEL_TYPE == target_model_type and ACTIVE_MODEL is not None:
            logger.info(f"Model type {target_model_type} is already active and loaded. No switch needed.")
            return True

        logger.info(f"Attempting to switch/load active model to: {target_model_type}")
        
        if ACTIVE_MODEL is not None:
            logger.info(f"Clearing existing model ({ACTIVE_MODEL_TYPE}) from memory.")
            del ACTIVE_MODEL
            ACTIVE_MODEL = None
            clear_gpu_memory()
        
        ACTIVE_MODEL_TYPE = None # 切り替え中は不定状態

        try:
            prepared_base_model, local_tokenizer = _create_peft_configured_base_model_and_tokenizer()
            
            if TOKENIZER is None:
                TOKENIZER = local_tokenizer
            
            if target_model_type == "base":
                ACTIVE_MODEL = prepared_base_model
                ACTIVE_MODEL_TYPE = "base"
                logger.info("Successfully switched to BASE_MODEL (PEFT configured).")

            elif target_model_type == "trained":
                if not os.path.exists(TRAINED_PATH):
                    logger.error(f"Trained model path {TRAINED_PATH} does not exist. Cannot load trained model.")
                    logger.warn("Falling back to loading BASE_MODEL as trained model is not found.")
                    ACTIVE_MODEL = prepared_base_model # Fallback to base model
                    ACTIVE_MODEL_TYPE = "base"
                    print_gpu_memory()
                    return False # Indicate that the intended trained model was not loaded

                logger.info(f"Loading adapter from {TRAINED_PATH} into the base model.")
                ACTIVE_MODEL = _load_adapter_to_model(prepared_base_model, TRAINED_PATH)
                ACTIVE_MODEL_TYPE = "trained"
                logger.info("Successfully switched to TRAINED_MODEL (base + adapter).")
            
            else:
                logger.error(f"Unknown target model type: {target_model_type}")
                if prepared_base_model: del prepared_base_model
                clear_gpu_memory()
                return False

            print_gpu_memory()
            return True

        except Exception as e:
            logger.error(f"Error switching/loading model to {target_model_type}: {e}")
            import traceback
            traceback.print_exc()
            if ACTIVE_MODEL: del ACTIVE_MODEL
            ACTIVE_MODEL = None
            ACTIVE_MODEL_TYPE = None
            clear_gpu_memory()
            return False

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup event triggered.")
    if os.path.exists(TRAINED_PATH):
        logger.info(f"Trained model found at {TRAINED_PATH}. Loading as initial model.")
        if not switch_active_model("trained"):
            logger.error("Failed to load trained model on startup. Trying base model...")
            if not switch_active_model("base"):
                raise RuntimeError("Failed to load any model on startup. Server cannot start.")
    else:
        logger.info("No trained model found. Loading base model as initial model.")
        if not switch_active_model("base"):
            raise RuntimeError("Failed to load base model on startup. Server cannot start.")
    
    if ACTIVE_MODEL is None or TOKENIZER is None:
        raise RuntimeError("Model or Tokenizer not initialized correctly on startup.")


@app.post("/infer")
def infer(request: InferenceRequest):
    global ACTIVE_MODEL, ACTIVE_MODEL_TYPE, TOKENIZER, training_status
    training_status.update({"status": "Infer", "progress": 0, "loss": None, "step": 0, "accuracy": 0.0, "perplexity": 0.0, "metrics": {}})

    target_type = "trained" if request.use_trained_model else "base"
    
    if ACTIVE_MODEL_TYPE != target_type or ACTIVE_MODEL is None:
        logger.info(f"Infer request for {target_type}, but current is {ACTIVE_MODEL_TYPE} or model not loaded. Switching model.")
        if not switch_active_model(target_type):
            error_msg = f"Failed to switch/load model to {target_type} for inference."
            logger.error(error_msg)
            # If trained model was requested but not found, switch_active_model might load base.
            # Check if the current active model is what was requested.
            if request.use_trained_model and ACTIVE_MODEL_TYPE != "trained":
                 return {"error": f"Trained model requested but could not be loaded. Path: {TRAINED_PATH}"}
            elif ACTIVE_MODEL is None: # General failure
                 return {"error": error_msg}

    if ACTIVE_MODEL is None or TOKENIZER is None:
        critical_error_msg = "Critical error: ACTIVE_MODEL or TOKENIZER is None before pipeline creation."
        logger.error(critical_error_msg)
        return {"error": critical_error_msg}

    model_source_name = "TRAINED_MODEL" if ACTIVE_MODEL_TYPE == "trained" else "BASE_MODEL"
    logger.info(f"Using {model_source_name} for inference with prompt: '{request.prompt[:50]}...'")

    try:
        pipe = pipeline("text-generation", model=ACTIVE_MODEL, tokenizer=TOKENIZER)
        
        generation_params = {
            "max_new_tokens": request.max_new_tokens,
            "do_sample": request.do_sample,
            "repetition_penalty": request.repetition_penalty,
            "pad_token_id": TOKENIZER.eos_token_id, # Suppress potential warnings
        }
        if request.do_sample:
            generation_params["temperature"] = request.temperature
            generation_params["top_p"] = request.top_p
            generation_params["top_k"] = request.top_k

        # Gemma-ITモデルが期待するチャット形式のプロンプトを作成します。
        prompt_template = [{"role": "user", "content": request.prompt}]
        
        # テンプレートを適用し、パイプラインに渡すためのフォーマット済み「文字列」プロンプトを取得します。
        formatted_prompt = TOKENIZER.apply_chat_template(
            prompt_template,
            tokenize=False,
            add_generation_prompt=True, # モデルに応答を促すためのプロンプトを追加
        )

        logger.info(f"Generating text with params: {generation_params}")
        # パイプラインには、生のプロンプト文字列ではなく、フォーマット済みのプロンプト文字列を渡します。
        outputs = pipe(formatted_prompt, **generation_params)
        full_text = outputs[0]['generated_text']
        
        # 出力にはプロンプト部分も含まれるため、モデルが生成した応答部分のみを抽出します。
        # Gemmaのテンプレートでは、モデルの応答は "<start_of_turn>model\n" の後に続きます。
        generated_text = full_text.split("<start_of_turn>model\n")[-1].strip()
        
        logger.info(f"Generated text (extracted): {generated_text[:200]}...")
        return {"result": generated_text}

    except Exception as e:
        logger.info(f"[ERROR] Error during text generation pipeline with {model_source_name}: {e}")
        import traceback
        traceback.print_exc()

        return {"error": f"Error during inference: {str(e)}"}


@app.post("/train")
def train_model(config: TrainConfig):
    def run_training():
        # global ACTIVE_MODEL, ACTIVE_MODEL_TYPE, TOKENIZER, training_status # Ensure access to global model variables
        global ACTIVE_MODEL, ACTIVE_MODEL_TYPE, TOKENIZER, training_status
        training_status = copy.deepcopy(INITIAL_TRAINING_STATUS)

        training_status.update({"status": "Loading", "progress": 0, "loss": None, "step": 0, "accuracy": 0.0, "perplexity": 0.0, "metrics": {}})

        logger.info("Ensuring base model (PEFT-configured) is active for training.")
        if not switch_active_model("base"):
            log_msg = "Failed to load/switch to base model for training. Aborting."
            training_status.update({"status": "Error", "logmsg": [log_msg]})
            logger.error(log_msg)
            return
        
        current_model_for_training = ACTIVE_MODEL # This is now the PEFT-configured base model
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        training_status.update({"status": "Formatting dataset"})
        logger.info("Formatting dataset...")

        # dataset.map(batched=True)で使うために、バッチを処理するフォーマット関数を定義します。
        EOS_TOKEN = TOKENIZER.eos_token # モデルのEOSトークンを取得
        def formatting_prompts_func(examples):
            # examplesの'id'キーの値がリストかどうかでバッチかどうかを判定し、単一サンプルならバッチ形式に変換
            # .map()は通常バッチで渡すが、念のため単一サンプルにも対応
            if not examples or 'id' not in examples:
                return []

            is_batch = isinstance(examples['id'], list)
            if not is_batch:
                examples = {k: [v] for k, v in examples.items()}

            processed_texts = []

            # `examples` は {'column_name': [value1, value2, ...]} の形式なので、
            # 各行（レコード）を再構築して処理する
            for i in range(len(examples['id'])):
                # 各レコードを辞書形式で取得
                record = {key: examples[key][i] for key in examples}

                # トップレベルの必須文字列キーのチェック
                required_string_keys = ['id', 'name_eng', 'description_jp']
                is_record_valid = True
                for key in required_string_keys:
                    if not (key in record and isinstance(record[key], str) and record[key].strip()):
                        logger.warning(f"Skipping invalid record (missing/empty string key '{key}'): {record}")
                        is_record_valid = False
                        break
                if not is_record_valid:
                    continue # このレコードはスキップ

                # プロンプトの基本部分を構築
                prompt_parts = [
                    f"### PROMPT: {record['id']}::{record['name_eng']}"
                ]

                # CKC Phases の追加
                # ckc_map_info はリストであると期待される
                ckc_map_info = record.get('ckc_map_info')
                if ckc_map_info and isinstance(ckc_map_info, list):
                    prompt_parts.append("\n[CKC PHASES]")
                    for phase in ckc_map_info:
                        # 各フェーズ内の必須キーのチェック
                        phase_keys = ['ckc_phase_number', 'ckc_phase_name_en', 'ckc_description_jp']
                        # ckc_phase_number は数値である可能性があるので、str()で文字列に変換してstrip()
                        if all(k in phase and phase[k] is not None and str(phase[k]).strip() != '' for k in phase_keys):
                            prompt_parts.append(
                                f"{phase['ckc_phase_number']}. {phase['ckc_phase_name_en']}: {phase['ckc_description_jp']}"
                            )
                        else:
                            logger.warning(f"Skipping invalid CKC phase record: {phase}")

                elif ckc_map_info is not None: # 存在はするがリストではない場合
                    logger.warning(f"Skipping invalid ckc_map_info (not a list): {ckc_map_info}")

                # Techniques と Subtechniques の追加
                # techniques はリストであると期待される
                techniques = record.get('techniques')
                if techniques and isinstance(techniques, list):
                    prompt_parts.append("\n[SUBTECHNIQUES]")
                    for tech in techniques:
                        # 各テクニック内の必須キーのチェック
                        tech_keys = ['id', 'name_eng', 'description_jp']
                        if all(k in tech and tech[k] is not None and str(tech[k]).strip() != '' for k in tech_keys):
                            prompt_parts.append(
                                f"{tech['id']}::{tech['name_eng']}: {tech['description_jp']}"
                            )

                            # サブテクニックの追加
                            subtechniques = tech.get('subtechniques')
                            if subtechniques and isinstance(subtechniques, list):
                                prompt_parts.append("  [SUBTECHNIQUES]") # インデント
                                for subtech in subtechniques:
                                    # 各サブテクニック内の必須キーのチェック
                                    subtech_keys = ['id', 'name_eng', 'description_jp']
                                    if all(k in subtech and subtech[k] is not None and str(subtech[k]).strip() != '' for k in subtech_keys):
                                        prompt_parts.append(
                                            f"  {subtech['id']}::{subtech['name_eng']}: {subtech['description_jp']}"
                                        )
                                    else:
                                        logger.warning(f"Skipping invalid subtechnique record: {subtech}")
                            elif subtechniques is not None: # 存在はするがリストではない場合
                                logger.warning(f"Skipping invalid subtechniques (not a list) for technique: {tech.get('id')}")
                        else:
                            logger.warning(f"Skipping invalid technique record: {tech}")
                elif techniques is not None: # 存在はするがリストではない場合
                    logger.warning(f"Skipping invalid techniques (not a list): {techniques}")

                # プロンプトとレスポンスを結合
                full_prompt = "\n".join(prompt_parts)
                text = f"{full_prompt}\n### RESPONSE: {record['description_jp']}{EOS_TOKEN}"
                processed_texts.append(text)

            # .map()はキーを持つ辞書を返す必要がある
            return {"text": processed_texts}

        # SFTTrainerにformatting_funcを直接渡すため、ここではデータセットをロードするだけにします。
        # フォーマットの適用はTrainer内部で効率的に行われます。
        full_dataset = Dataset.from_list(json_data)

        # .map()を使用してデータセット全体を事前にフォーマットします。
        # これにより、SFTTrainerに渡す前にデータが正しい形式になっていることを保証できます。
        formatted_dataset = full_dataset.map(
            formatting_prompts_func,
            batched=True,
            remove_columns=list(json_data[0].keys()) # 元の列は不要なので削除
        )

        # データセットを訓練用と検証用に分割 (90% train, 10% validation)
        split_dataset = formatted_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        logger.info(f"Dataset split into {len(train_dataset)} training samples and {len(eval_dataset)} validation samples.")

        # フォーマット後にデータセットが空になっていないかチェック
        if len(train_dataset) == 0 or len(eval_dataset) == 0:
            log_msg = "Training or evaluation dataset is empty after formatting. Aborting training."
            training_status.update({"status": "Error", "logmsg": [log_msg]})
            logger.error(log_msg)
            return

        def compute_metrics(eval_pred):
            """
            Trainerはeval_lossを自動計算するため、この関数ではaccuracyなど追加のメトリクスのみを計算します。
            """
            logits, labels = eval_pred
            # ラベルが-100のトークンは損失計算から除外されているため、精度計算でも同様に除外します。
            predictions = np.argmax(logits, axis=-1)

            valid_labels_mask = labels != -100
            accuracy = np.sum(predictions[valid_labels_mask] == labels[valid_labels_mask]) / np.sum(valid_labels_mask)
            
            return {"accuracy": float(accuracy)}

        # 物理的なバッチサイズはメモリ制約のため1に固定します。
        per_device_batch_size = 1
        # 実質的なバッチサイズがリクエストされた値になるように勾配累積ステップを計算します。
        # これにより、GPUメモリ使用量を抑えつつ、より大きなバッチサイズと同等の学習効果を得られます。
        gradient_accumulation_steps = config.batch_size // per_device_batch_size

        if config.batch_size % per_device_batch_size != 0:
            logger.warning(f"Requested batch_size ({config.batch_size}) is not a multiple of per_device_batch_size ({per_device_batch_size}). "
                           f"Effective batch size will be rounded down to {per_device_batch_size * gradient_accumulation_steps}.")

        logger.info(f"Training with effective batch size: {config.batch_size} "
                    f"(per_device_train_batch_size = {per_device_batch_size}, "
                    f"gradient_accumulation_steps = {gradient_accumulation_steps})")        

        # -------------------------------------------------------------
        # Hyperパラメータ設定（Streamlit側からのパラメータ指定を考慮
        # -------------------------------------------------------------
        # training_args = SFTConfig(
        #     output_dir=TRAINED_PATH,
        #     per_device_train_batch_size=per_device_batch_size,
        #     gradient_accumulation_steps=gradient_accumulation_steps,
        #     max_steps = 200,
        #     warmup_ratio = 0.1,
        #     num_train_epochs=config.epochs,
        #     learning_rate=config.learning_rate,
        #     logging_steps=config.logging_steps,
        #     eval_steps=config.eval_steps,
        #     save_steps=config.save_steps if config.save_steps > 0 else None,
        #     save_strategy="steps" if config.save_steps > 0 else "no",
        #     logging_strategy="steps",
        #     report_to="none",
        #     fp16 = False,
        #     bf16 = True,
        #     optim = "paged_adamw_8bit",
        #     weight_decay = 0.01,
        #     lr_scheduler_type = "cosine",
        #     seed = 42,
        # )

        # SFTConfigを使用して学習引数を設定します。
        # TypeErrorを回避するため、evaluation_strategyは初期化後に設定します。
        training_args = SFTConfig(
            output_dir=TRAINED_PATH,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            # max_steps = 200, # num_train_epochsが優先されるため、UIからのエポック数指定を有効にするためにコメントアウト
            warmup_ratio = 0.1,
            num_train_epochs=config.epochs, # UIからのエポック数を反映
            learning_rate=config.learning_rate,
            logging_steps=config.logging_steps,
            eval_steps=config.eval_steps,
            save_steps=config.save_steps if config.save_steps > 0 else None,
            save_strategy="steps" if config.save_steps > 0 else "no",
            logging_strategy="steps",
            report_to="none",
            fp16 = False,
            bf16 = True,
            optim = "paged_adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = 42,
            # packing=True をSFTTrainerに渡すため、ここでは設定しない
        )

        # ★★★ 評価を確実に有効化するための重要なステップ ★★★
        # 初期化後に evaluation_strategy を設定することで、TypeError を回避します。
        training_args.evaluation_strategy = "steps"

        class CustomSFTTrainer(SFTTrainer): # Inherit from SFTTrainer
            def log(self, logs: dict, *args, **kwargs):
                super().log(logs)
                step = self.state.global_step
                # 訓練ロスのログ
                if "loss" in logs:
                    training_status.update({
                        "loss": float(logs.get("loss")),
                        "epoch": self.state.epoch,
                        "step": step,
                        "total_steps": self.state.max_steps,
                        "progress": min(1.0, step / self.state.max_steps) if self.state.max_steps > 0 else 0,
                        "metrics": logs,
                        "status": "Training",
                    })

                    # 訓練ロス履歴の記録は、訓練ログがある場合のみ実行
                    training_status["history"]["train_steps"].append(step)
                    training_status["history"]["train_loss"].append(float(logs.get("loss")))

                # 検証ロスのログ
                if "eval_loss" in logs:
                    eval_loss = float(logs["eval_loss"])
                    training_status["val_loss"] = eval_loss
                    training_status["history"]["eval_steps"].append(step)
                    training_status["history"]["val_loss"].append(eval_loss)

                    # Perplexityをeval_lossから計算してステータスに追加
                    perplexity = math.exp(eval_loss)
                    training_status["perplexity"] = perplexity

                    # compute_metricsから返されるaccuracyを取得
                    if "eval_accuracy" in logs:
                        training_status["accuracy"] = float(logs["eval_accuracy"])

        current_model_for_training.config.use_cache = False # Important for training

        # -------------------------------------------------------------
        # トレーナー設定
        # -------------------------------------------------------------
        trainer = CustomSFTTrainer( # Use the custom SFTTrainer
            model=current_model_for_training,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=TOKENIZER, # Pass tokenizer to SFTTrainer
            dataset_text_field="text", # フォーマット済みデータセットのテキストフィールド名を指定
            packing=True, # データセットが非常に小さい場合、packingによりサンプル数が0になる可能性があるため無効化
            # packing=False, # packingを無効化し、各サンプルを個別に処理
            max_seq_length = MAX_SEQ_LENGTH, # パッキング後、この長さにチャンク化される
            compute_metrics=compute_metrics,
        )


        # --- 学習実行前の最終チェック ---
        # Trainerの内部状態から、計算された総学習ステップ数を取得します。
        total_training_steps = trainer.state.max_steps
        first_eval_step = training_args.eval_steps
        logger.info(f"Trainer initialized. Total training steps: {total_training_steps}, Evaluation will occur every {first_eval_step} steps.")

        # 総学習ステップ数が最初の評価ステップに満たない場合の警告
        if total_training_steps < first_eval_step:
            logger.warning(
                f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                f"WARNING: Total training steps ({total_training_steps}) is less than the evaluation interval ({first_eval_step}).\n"
                f"As a result, NO EVALUATION will be performed and 'eval_loss' will not be available.\n"
                f"To fix this, please either INCREASE the number of epochs or DECREASE the 'Evaluation Interval' in the UI.\n"
                f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )


        trainer.train()

        training_status.update({"status": "Saving", "progress": 1.0})
        logger.info(f"Saving trained adapter from the model instance to {TRAINED_PATH}.")
        current_model_for_training.save_pretrained(TRAINED_PATH) # Save adapter from the model that was trained
        TOKENIZER.save_pretrained(TRAINED_PATH)
        
        logger.info("Training complete. Refreshing active model to newly trained version.")
        if not switch_active_model("trained"):
            logger.error("Failed to switch to newly trained model after training.")
            training_status.update({"status": "Training complete, but failed to load new model", "logmsg": ["Failed to load newly trained model."]})

            logger.info("Training status has been reset to initial state after failure to load new model.")
        else:
            logger.info("Successfully switched to newly trained model after training.")
            training_status.update({"status": "Training complete"})

            logger.info("Training status has been reset to initial state after successful training.")


    thread = threading.Thread(target=run_training, daemon=True) # daemon=True を追加してメインスレッド終了時に自動終了
    thread.start()
    return {"status": "Training started"}


@app.get("/status")
def get_status():
    global training_status
    return training_status


@app.post("/uploadfile/")
# def upload_file(file: UploadFile = File(...)):
#     dest = os.path.join("./uploaded_files", file.filename)
#     os.makedirs("./uploaded_files", exist_ok=True)
#     with open(dest, "wb") as f:
#         shutil.copyfileobj(file.file, f)
#     return {"filename": file.filename, "status": "uploaded"}
def upload_file(file: UploadFile = File(...)):
    """
    学習用のJSONファイルをアップロードし、既存のDATA_PATHのファイルを上書きします。
    """
    try:
        logger.info(f"Received request to upload new training data: {file.filename}. This will overwrite {DATA_PATH}.")
        with open(DATA_PATH, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"filename": file.filename, "status": f"Uploaded and set as current training data ({DATA_PATH})"}
    except Exception as e:
        logger.error(f"Failed to upload and save training data: {e}", exc_info=True)
        return {"error": f"Failed to process uploaded file: {str(e)}"}


@app.get("/")
async def read_root():
    logger.info("Minimal / (root) endpoint called")
    return {"Hello": "World"}

@app.get("/testlog")
def test_log_endpoint():
    logger.info("Minimal /testlog endpoint called (INFO)")
    logger.debug("Minimal /testlog endpoint called (DEBUG)") # DEBUGログもテスト
    return {"message": "Minimal test log endpoint called"}


if __name__ == "__main__":
    # logger.info("Starting FastAPI server with Uvicorn directly from script.") # このログは表示されるはず
    # uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")

    # この時点でのロガー名 __name__ は "__main__" になります。
    # Uvicornがモジュールとしてロードする際のロガー名は "fastapi_llm_server" です。
    # LOGGING_CONFIG で両方、または "fastapi_llm_server" を設定します。
    current_module_logger = logging.getLogger(__name__) # これは "__main__" ロガー
    app_module_logger_name = "fastapi_llm_server" # FastAPIファイル内のgetLogger(__name__)がこの名前になる

    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
                "use_colors": True, # ターミナルが色表示に対応していない場合はFalseに
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(asctime)s - %(client_addr)s - "%(request_line)s" %(status_code)s',
                "datefmt": "%Y-%m-%d %H:%M:%S",
                "use_colors": True,
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr", # 標準エラー出力へ (sys.stdoutも試す価値あり)
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout", # 標準出力へ
            },
        },
        "loggers": {
            # ... (他のロガー設定) ...
            "": { # ルートロガー
                "handlers": ["default"],
                "level": "DEBUG", # アプリケーション全体のログレベルをDEBUGに設定
                "level": "INFO", # アプリケーション全体のログレベルをDEBUGに設定
                "propagate": False # ルートロガーなので伝播は不要
            },
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            # app_module_logger_name の propagate も False に変更することを検討
            # app_module_logger_name: {"handlers": ["default"], "level": "DEBUG", "propagate": True}, # アプリケーションロガー
            app_module_logger_name: {"handlers": ["default"], "level": "DEBUG", "propagate": False}, # アプリケーションロガー
            # "__main__": {"handlers": ["default"], "level": "DEBUG", "propagate": True}, # スクリプト直接実行時のロガー
        },
    }

    # current_module_logger.info("*** Attempting to start FastAPI server with Uvicorn (custom log config)...") # このログは表示されるか？
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=LOGGING_CONFIG)
    # uvicorn.run(app, host="0.0.0.0", port=8000, log_config="debug")