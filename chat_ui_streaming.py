# chat_ui_streaming.py
import streamlit as st
import requests
import json
from typing import Generator, Any, Dict
import base64
from pathlib import Path

# -------------------------------------------------------------
# アプリケーション設定 (st.set_page_configを最初に呼び出す)
# -------------------------------------------------------------
# st.set_page_config() は、スクリプト内で一番最初に呼び出す必要があり、1ページに1回しか呼び出せません。
# 複数の呼び出しがあったため、1つに統合し、ページのタイトルもUIと合わせます。
st.set_page_config(page_title="サイバー攻撃対策Chatbot", page_icon="🔐")

# --- 背景画像設定 ---
def get_base64_of_bin_file(bin_file):
    """バイナリファイルをBase64エンコードされた文字列として取得する"""
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_page_bg_from_file(image_file):
    """指定された画像ファイルをページの背景として設定し、半透明のオーバーレイを適用する"""
    try:
        # ファイル拡張子からMIMEタイプを決定
        ext = Path(image_file).suffix.lower()
        if ext in [".jpg", ".jpeg"]:
            mime_type = "image/jpeg"
        elif ext == ".png":
            mime_type = "image/png"
        else:
            st.warning(f"Unsupported image format for background: {ext}. Defaulting to jpeg.")
            mime_type = "image/jpeg"

        bin_str = get_base64_of_bin_file(image_file)
        page_bg_img = f'''
        <style>
        .stApp {{
            /* 半透明の白色(rgba)を重ねることで画像を薄く見せます。0.7の部分を0.0(透明)から1.0(不透明)の間で調整してください。 */
            background-image: linear-gradient(rgba(255, 255, 255, 0.0), rgba(255, 255, 255, 0.0)), url("data:{mime_type};base64,{bin_str}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"背景画像ファイルが見つかりません: {image_file}")

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★ 重要: FastAPIサーバーが動作している端末AのIPアドレスに書き換えてください ★
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
FASTAPI_SERVER_URL = "http://192.168.1.168:8000/chat"

# image.pngのパスを正しく指定してください
# このスクリプトと同じディレクトリにあると仮定します
image_path = Path(__file__).parent / "image.jpg"
set_page_bg_from_file(str(image_path))

# -------------------------------------------------------------
# バックエンド通信
# -------------------------------------------------------------
def get_streaming_response(prompt: str) -> Generator[Dict[str, Any], None, None]:
    """バックエンドにリクエストを送り、ストリーミング応答を処理するジェネレータ"""
    payload = {"prompt": prompt}
    try:
        with requests.post(FASTAPI_SERVER_URL, json=payload, stream=True, timeout=300) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try:
                        # 各行は独立したJSONオブジェクト
                        data = json.loads(line.decode('utf-8'))
                        yield data
                    except json.JSONDecodeError:
                        # JSONデコードに失敗した行は無視
                        st.warning(f"Could not decode line: {line}")
                        continue
    except requests.exceptions.RequestException as e:
        yield {"type": "error", "data": f"バックエンドサーバーへの接続に失敗しました: {e}"}
    except Exception as e:
        yield {"type": "error", "data": f"予期せぬエラーが発生しました: {e}"}

# -------------------------------------------------------------
# UI表示
# -------------------------------------------------------------
st.title("🔐 サイバー攻撃対策Chatbot")
st.caption("MITRE ATT&CK + CKC Finetuning済みGemma3によるRAG＋エージェント(ツール使用)を活用したチャットシステム")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 履歴の表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "rag_context" in message and message["rag_context"]:
            with st.expander("RAG Context"):
                st.code(message["rag_context"], language="markdown")
        if "tool_result" in message and message["tool_result"]:
            with st.expander("Tool Result"):
                st.code(message["tool_result"], language="text")

# ユーザー入力
if prompt := st.chat_input("IPアドレスに関する質問や、調査・対応依頼などを入力してください。"):
    # ユーザーメッセージを保存・表示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # アシスタントの応答をストリーミング表示
    with st.chat_message("assistant"):
        
        # ストリーミングデータを処理するためのラッパージェネレータ
        def response_generator():
            rag_context = ""
            tool_result = ""
            
            # get_streaming_responseからデータを受け取る
            for chunk in get_streaming_response(prompt):
                chunk_type = chunk.get("type")
                chunk_data = chunk.get("data")

                if chunk_type == "token":
                    yield chunk_data  # トークンは直接yieldしてst.write_streamに渡す
                elif chunk_type == "rag":
                    rag_context = chunk_data
                elif chunk_type == "tool":
                    tool_result = chunk_data
                elif chunk_type == "error":
                    st.error(chunk_data) # エラーメッセージを表示
                    yield "" # エラー時は空文字を返す
            
            # ストリーム終了後、完全なメッセージを履歴に保存
            # st.session_stateから最後のメッセージ（アシスタントの空のメッセージ）を取得
            # full_response = st.session_state.get('full_response', '') # これはうまく動かない
            # 代わりに、UIに表示されたテキストを収集する必要があるが、st.write_streamでは難しい
            # ここでは、RAGとToolの結果のみを保存するアプローチを取る
            st.session_state.messages.append({
                "role": "assistant",
                "content": "*(ストリーミング応答)*", # 履歴には固定テキスト
                "rag_context": rag_context,
                "tool_result": tool_result,
            })
            
            # ストリーム終了後、収集した情報をExpanderで表示
            if rag_context:
                with st.expander("RAG Context"):
                    st.code(rag_context, language="markdown")
            if tool_result:
                with st.expander("Tool Result"):
                    st.code(tool_result, language="text")

        # st.write_streamを使ってジェネレータからの出力を表示
        full_response_content = st.write_stream(response_generator)
        
        # ストリーム完了後に完全な応答を履歴に追加する
        # 最後のメッセージ（ユーザーのメッセージ）の次がアシスタントのメッセージ
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response_content
            # RAG/Toolの結果はresponse_generator内で表示されるため、ここではcontentのみ
        })
        # UIを再実行して、Expanderを含む最新の履歴を表示
        st.rerun()
