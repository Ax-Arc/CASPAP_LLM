# rag_agent_server_streaming.py
# ストリーミング応答に対応したRAG & Agentサーバー

import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, AsyncGenerator
from unsloth import FastLanguageModel
from transformers import pipeline, TextIteratorStreamer
import shutil
import uvicorn
import logging
import threading
import re
import json
import pandas as pd
import requests
import asyncio

# -------------------------------------------------------------
# 基本設定
# -------------------------------------------------------------
app = FastAPI(
    title="Streaming RAG & Agent LLM Server",
    description="Fine-tuned Gemma model with RAG, Agent, and Streaming capabilities.",
    version="1.1.0",
)

# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- モデルとデータパスの設定 ---
# 提供されたスクリプトから設定を継承
MODEL_NAME = "unsloth/gemma-3-1b-it"
TRAINED_PATH = "./gemma3_unsloth_ckc_fine_sft"  # ファインチューニング用データパス
MAX_SEQ_LENGTH = 8192   # 4096 -> 8192
# --- RAG用データパス ---
ASSET_DATA_PATH = "asset_management.csv"

# --- エージェント用設定 ---
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★ 重要: 実際の環境では、APIキーは環境変数やシークレット管理ツールで管理してください ★
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
VIRUSTOTAL_API_KEY = os.environ.get("VIRUSTOTAL_API_KEY", "YOUR_VIRUSTOTAL_API_KEY_HERE")
# VIRUSTOTAL_API_KEY = os.environ.get("VIRUSTOTAL_API_KEY", "")

# グローバル変数
ACTIVE_MODEL = None
TOKENIZER = None
ACTIVE_MODEL_TYPE = None # "base" or "trained"
model_lock = threading.Lock()
asset_data = pd.DataFrame() # RAG用データフレーム

# -------------------------------------------------------------
# Pydanticモデル
# -------------------------------------------------------------
class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 1024

# 既存のInferenceRequestとTrainConfigは元のファイルから流用可能
class InferenceRequest(BaseModel):
    prompt: str
    use_trained_model: bool = True
    max_new_tokens: int = 2045
    do_sample: bool = True  # False -> True
    temperature: float = 1.0    # 0.8 -> 1.0
    top_p: float = 0.95  # 0.9 -> 0.95
    top_k: int = 0 # 50 -> 0
    repetition_penalty: float = 1.0 # 1.2 -> 1.0

# -------------------------------------------------------------
# RAG & エージェント機能
# -------------------------------------------------------------
def load_asset_data():
    """RAG用の資産管理データをCSVから読み込む"""
    global asset_data
    try:
        if os.path.exists(ASSET_DATA_PATH):
            asset_data = pd.read_csv(ASSET_DATA_PATH)
            # IPアドレスをインデックスに設定して高速検索
            asset_data.set_index('ip_address', inplace=True)
            logger.info(f"Successfully loaded asset data from {ASSET_DATA_PATH}. {len(asset_data)} records found.")
        else:
            logger.warning(f"Asset data file not found at {ASSET_DATA_PATH}. RAG will be disabled.")
            asset_data = pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading asset data: {e}")
        asset_data = pd.DataFrame()

def get_rag_context(prompt: str) -> str:
    """プロンプトからIPアドレスを抽出し、資産情報を取得する"""
    if asset_data.empty:
        return ""
    
    # プロンプトからIPアドレスを正規表現で抽出
    ip_match = re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', prompt)
    if not ip_match:
        return ""
        
    ip_address = ip_match.group(0)
    logger.info(f"Found IP address in prompt: {ip_address}")

    try:
        # DataFrameからIPアドレスで情報を検索
        info = asset_data.loc[ip_address]
        context = f"""
### 内部システム（内部資産）情報 (RAG)
- IPアドレス: {ip_address}
- ホスト名: {info.get('hostname', 'N/A')}
- 役割: {info.get('role', 'N/A')}
- 重要度: {info.get('importance', 'N/A')}
- 管理部門: {info.get('department', 'N/A')}
"""
        logger.info(f"Found RAG context for {ip_address}")
        return context
    except KeyError:
        logger.info(f"No RAG context found for IP address: {ip_address}")
        return ""
    except Exception as e:
        logger.error(f"Error retrieving RAG context: {e}")
        return ""

# -------------------------------------------------------------
# --- エージェントツール定義 ---
# -------------------------------------------------------------
def check_virustotal(ip_address: str) -> str:
    """VirusTotal APIでIPアドレスのレピュテーションを調査する（シミュレーション）"""
    logger.info(f"[TOOL] Executing VirusTotal check for {ip_address}")
    # if VIRUSTOTAL_API_KEY == "YOUR_VIRUSTOTAL_API_KEY_HERE":
    if VIRUSTOTAL_API_KEY == "YOUR_VIRUSTOTAL_API_KEY_HERE":
        logger.warning("VirusTotal API key is not set. Returning dummy data.")
        # ダミーの応答
        if ip_address.startswith("8.8"):
            return "VirusTotal Report: IP is well-known (Google DNS). Considered safe."
        return f"VirusTotal Report for {ip_address}: Reputation is neutral. (API Key not set - SIMULATED)"

    headers = {"x-apikey": VIRUSTOTAL_API_KEY}
    try:
        response = requests.get(f"https://www.virustotal.com/api/v3/ip_addresses/{ip_address}", headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        stats = data.get('data', {}).get('attributes', {}).get('last_analysis_stats', {})
        malicious = stats.get('malicious', 0)
        total_vendors = sum(stats.values())
        vt_result = ""
        if malicious > 0:
            vt_result = f"IP アドレス [{ip_address}] の VirusTotal レポート結果: {malicious}/{total_vendors} のベンダーがこのIPを「悪意のあるIP」としてフラグ付けしています。"
        else:
            vt_result = f"IP アドレス [{ip_address}] の VirusTotal レポート結果: {total_vendors} のベンダーがこのIPは「悪意のないIP」として判断しています。（問題なし）"

        # return f"VirusTotal Report for IP address [{ip_address}]: {malicious}/{total_vendors} vendors flagged this IP as malicious."
        return vt_result
    
    except requests.exceptions.RequestException as e:
        logger.error(f"VirusTotal API request failed: {e}")
        return f"Failed to get VirusTotal report for {ip_address}. Error: {e}"

def apply_firewall_rule(ip_address: str) -> str:
    """ファイアウォールルール適用スクリプトを実行する（シミュレーション）"""
    logger.info(f"[TOOL] Executing firewall rule application for {ip_address}")
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★ 警告: これはシミュレーションです。実際の環境では、subprocessで直接スクリプトを   ★
    # ★ 実行する際は、厳重なセキュリティ対策（入力のサニタイズ、権限管理など）が必要です。★
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # command = ["/path/to/your/firewall_script.sh", "add", ip_address]
    # result = subprocess.run(command, capture_output=True, text=True)
    # if result.returncode == 0:
    #     return f"Firewall rule to block {ip_address} applied successfully. stdout: {result.stdout}"
    # else:
    #     return f"Failed to apply firewall rule for {ip_address}. stderr: {result.stderr}"

    # return f"Firewall rule to block {ip_address} has been successfully applied (SIMULATED)."
    return f"IPアドレス「{ip_address}」をブロックするファイアウォール ルールが正常に適用されました。"

# -------------------------------------------------------------
# モデル読み込み・管理 (変更なし)
# -------------------------------------------------------------
def _create_peft_configured_base_model_and_tokenizer():
    """ベースモデルとトークナイザーをロードし、PEFT設定を適用"""
    logger.info(f"Loading base model: {MODEL_NAME} with max_seq_length: {MAX_SEQ_LENGTH}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    logger.info("Applying PEFT (LoRA) settings...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    return model, tokenizer

def switch_active_model(target_model_type: str):
    """モデルをベースモデルとファインチューニング済みモデルで切り替える"""
    global ACTIVE_MODEL, ACTIVE_MODEL_TYPE, TOKENIZER
    with model_lock:
        if ACTIVE_MODEL_TYPE == target_model_type and ACTIVE_MODEL is not None:
            return True

        logger.info(f"Switching active model to: {target_model_type}")
        if ACTIVE_MODEL is not None:
            del ACTIVE_MODEL
            torch.cuda.empty_cache()

        try:
            base_model, tokenizer = _create_peft_configured_base_model_and_tokenizer()
            if TOKENIZER is None:
                TOKENIZER = tokenizer

            if target_model_type == "trained":
                if not os.path.exists(TRAINED_PATH):
                    logger.error(f"Trained model not found at {TRAINED_PATH}. Falling back to base model.")
                    ACTIVE_MODEL = base_model
                    ACTIVE_MODEL_TYPE = "base"
                    return False
                
                logger.info(f"Loading adapter from {TRAINED_PATH} into the base model.")
                # ACTIVE_MODEL = FastLanguageModel.from_pretrained(base_model, TRAINED_PATH)
                base_model.load_adapter(TRAINED_PATH, adapter_name="default")
                ACTIVE_MODEL = base_model
                ACTIVE_MODEL_TYPE = "trained"
                logger.info("Successfully loaded TRAINED model.")

            else: # "base"
                ACTIVE_MODEL = base_model
                ACTIVE_MODEL_TYPE = "base"
                logger.info("Successfully loaded BASE model.")
            
            return True
        except Exception as e:
            logger.error(f"Error switching model to {target_model_type}: {e}", exc_info=True)
            ACTIVE_MODEL, ACTIVE_MODEL_TYPE = None, None
            return False

# -------------------------------------------------------------
# LLM推論パイプライン (同期・非同期)
# -------------------------------------------------------------
def run_llm_inference_sync(prompt: str, max_new_tokens: int) -> str:
    """同期的なLLM推論（ツール選択用）"""
    if not ACTIVE_MODEL or not TOKENIZER:
        raise RuntimeError("Model is not ready.")
    
    pipe = pipeline("text-generation", model=ACTIVE_MODEL, tokenizer=TOKENIZER)

    # Gemma-ITモデルが期待するチャット形式のプロンプトを作成
    messages = [{"role": "user", "content": prompt}]

    formatted_prompt = TOKENIZER.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    logger.info(f"Generating text with max_new_tokens: {max_new_tokens}")
    outputs = pipe(
        formatted_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,    # 0.7 -> 1.0
        top_p=0.95,
        top_k=0, # add
        repetition_penalty=1.0, # 1.1 -> 1.0
        pad_token_id=TOKENIZER.eos_token_id,
    )
    
    full_text = outputs[0]['generated_text']
    # プロンプト部分を除き、モデルが生成した応答のみを抽出
    generated_text = full_text.split("<start_of_turn>model\n")[-1].strip()
    logger.info(f"LLM Raw Output: {generated_text[:200]}...")
    return generated_text

def run_llm_inference_stream(prompt: str, streamer: TextIteratorStreamer, max_new_tokens: int):
    """非同期LLM推論（ストリーミング用）"""
    if not ACTIVE_MODEL or not TOKENIZER:
        raise RuntimeError("Model is not ready.")
    
    pipe = pipeline("text-generation", model=ACTIVE_MODEL, tokenizer=TOKENIZER)

    messages = [{"role": "user", "content": prompt}]

    formatted_prompt = TOKENIZER.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    generation_kwargs = dict(
        text_inputs=formatted_prompt,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=0, # add
        repetition_penalty=1.0, # add
        pad_token_id=TOKENIZER.eos_token_id,
    )

    # スレッドでパイプラインを実行
    thread = threading.Thread(target=pipe, kwargs=generation_kwargs)
    thread.start()

# -------------------------------------------------------------
# FastAPI エンドポイント
# -------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """サーバー起動時にモデルとRAGデータをロードする"""
    logger.info("Application startup...")
    load_asset_data() # RAGデータロード
    # ファインチューニング済みモデルを優先してロード
    if os.path.exists(TRAINED_PATH):
        logger.info(f"Trained model found. Loading '{TRAINED_PATH}' as initial model.")
        if not switch_active_model("trained"):
            raise RuntimeError("Failed to load trained model on startup.")
    else:
        logger.warning("No trained model found. Loading base model.")
        if not switch_active_model("base"):
            raise RuntimeError("Failed to load base model on startup.")


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """RAGとエージェント機能を持ち、ストリーミング応答を返すチャットエンドポイント"""

    user_prompt = request.prompt
    logger.info(f"Received chat request: '{user_prompt}'")

    async def stream_generator() -> AsyncGenerator[str, None]:
        # 1. RAG: コンテキスト情報を取得
        rag_context = get_rag_context(request.prompt)
        if rag_context:
            yield json.dumps({"type": "rag", "data": rag_context}) + "\n"

        # 2. Agent (Tool Selection): LLMにツール使用を判断させる - 同期的に実行
#         tool_prompt = f"""
# あなたはセキュリティアシスタントです。ユーザーの依頼を分析し、以下のツールを実行する必要があるか判断してください。
# 判断結果は必ずJSON形式で出力してください。ツールが不要な場合は `tool` を "none" にしてください。
# 利用可能なツール:
# - "virustotal": IPアドレスの「調査」をします。IPアドレスが必要です。
# - "firewall": IPアドレスを「ブロック」するファイアウォールルールを適用します。IPアドレスが必要です。

# ## ユーザーの依頼: "{request.prompt}"

# {rag_context}

# 判断結果をJSONで出力してください:
# 例1: {{"tool": "virustotal", "arguments": {{"ip_address": "8.8.8.8"}}}}
# 例2: {{"tool": "firewall", "arguments": {{"ip_address": "1.2.3.4"}}}}
# 例3: {{"tool": "none", "arguments": {{}}}}
# """
        # 思考プロセス（Chain of Thought）の導入: LLMに判断の根拠を段階的に考えさせ、その上で最終的なツールを決定させる方法
        tool_prompt = f"""あなたは、ユーザーの依頼を分析し、適切なツールを選択する優秀なAIアシスタントです。
以下のステップに従って、最終的な判断をJSON形式で出力してください。

利用可能なツール:
- `virustotal`: IPアドレスのレピュテーションを「調査」します。
- `firewall`: IPアドレスを「ブロック」します。
- `none`: 適切なツールがない場合。

## ステップ1: ユーザーの意図を分析する
ユーザーの依頼文から、主要なキーワード（動詞）を特定し、その意図を簡潔に説明してください。

## ステップ2: 適切なツールを選択する
ステップ1の分析に基づき、最も適切なツールを一つ選択してください。

## ステップ3: 最終的な判断をJSONで出力する
ステップ2で選択したツールと、依頼文から抽出した引数（IPアドレスなど）を組み合わせて、JSON形式で出力してください。

---
### ユーザーの依頼: "{request.prompt}"

{rag_context}
---

# 判断結果を以下のJSON形式の例に従って出力してください:
 - JSON形式の例: {{"tool": "[利用可能なツール]", "arguments": {{"ip_address": "[IPアドレス]"}}}}
"""
        
        # ツール選択はブロッキング呼び出し
        logger.info(f"Tool prompt: {tool_prompt}")
        tool_decision_str = run_llm_inference_sync(tool_prompt, max_new_tokens=100)
        tool_result = ""
        try:
            # LLMの出力からJSON部分を抽出
            logger.info(f"Tool decision str: {tool_decision_str}")
            json_match = re.search(r'\{.*\}', tool_decision_str, re.DOTALL)
            if json_match:
                tool_decision = json.loads(json_match.group(0))
                tool_name = tool_decision.get("tool")
                arguments = tool_decision.get("arguments", {})
                if tool_name == "virustotal" and "ip_address" in arguments:
                    tool_result = check_virustotal(arguments["ip_address"])
                elif tool_name == "firewall" and "ip_address" in arguments:
                    tool_result = apply_firewall_rule(arguments["ip_address"])
            
            if tool_result:
                 yield json.dumps({"type": "tool", "data": tool_result}) + "\n"

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse tool decision from LLM output. Output: '{tool_decision_str}'. Error: {e}")
            # ツール判断に失敗しても、そのまま次のステップに進む

        # 3. Final Response Generation: 全ての情報を統合して最終応答を生成 (Streaming)
        tool_info = f"## ツール実行結果\n{tool_result}" if tool_result else "## ツール実行結果\nツールは実行されませんでした。"
        rag_info = f"## 内部システム情報\n{rag_context}" if rag_context else "## 内部システム情報\n関連する内部情報はありません。"
        # final_prompt = f"""あなたは優秀なセキュリティアシスタントです。以下の情報を元に、「ユーザーの依頼」に自然な日本語で応答してください。
        final_prompt = f"""あなたは優秀なセキュリティエンジニアです。以下の情報を元に、「ユーザーの依頼」に日本語で応答してください。
# ユーザーの依頼
{request.prompt} 

{rag_info}

{tool_info}

# MITRE ATT&CKの情報がある場合は、その情報を優先してユーザーへ応答してください。
 - technique_id
 - technique_name_eng
 - technique_description_jpn
 - subtechnique_id
 - subtechnique_name_eng
 - subtechnique_description_jpn

# Cyber kill chainの情報がある場合は、その情報を優先してユーザーへ応答してください。
 - cyber_kill_chain_phase_number_1
 - cyber_kill_chain_phase_name_eng_1
 - cyber_kill_chain_phase_description_jp_1
 - cyber_kill_chain_phase_number_2
 - cyber_kill_chain_phase_name_eng_2
 - cyber_kill_chain_phase_description_jp_2

## MITRE ATT&CK情報を元にした現段階の攻撃の仕組みについての情報がある場合、その情報を優先してユーザーへ応答してください。
## MITRE ATT&CK情報を元にした現段階の影響についての情報がある場合、その情報を優先してユーザーへ応答してください。

上記情報を踏まえて、ユーザーへの応答を作成してください:
"""
#↑ 上記情報を踏まえて、ユーザーへの最終的な応答を作成してください:
        
        logger.info(f"Final prompt: {final_prompt}")
        streamer = TextIteratorStreamer(TOKENIZER, skip_prompt=True, skip_special_tokens=True)
        
        # 推論を別スレッドで開始
        # asyncio.to_thread を使ってブロッキングコードを非同期コンテキストで安全に実行
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, 
            run_llm_inference_stream, 
            final_prompt, streamer, request.max_new_tokens
        )
        
        # ストリーマーからトークンを非同期に読み取り、クライアントに送信
        for token in streamer:
            yield json.dumps({"type": "token", "data": token}) + "\n"

    # StreamingResponseを使って、ジェネレータからの出力をストリーミングする
    # media_typeを 'application/x-ndjson' にして、改行区切りのJSONとして扱う
    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")


# 元の /infer エンドポイントも残しておく
@app.post("/infer")
def infer(request: InferenceRequest):
    target_type = "trained" if request.use_trained_model else "base"
    if ACTIVE_MODEL_TYPE != target_type or ACTIVE_MODEL is None:
        if not switch_active_model(target_type):
            raise HTTPException(status_code=503, detail=f"Failed to switch model to {target_type}")
    
    try:
        response = run_llm_inference_stream(request.prompt, request.max_new_tokens)
        return {"result": response}
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
