# 修正済み streamlit_llm_compare.py
import streamlit as st
import requests
import time
import pandas as pd
import math

FASTAPI_URL = "http://172.0.0.1:8000"

st.set_page_config(layout="wide")
st.title("LLM 比較・学習・評価 UI")

# --- サイドバー ---
# --- 推論パラメータ調整 ---
st.sidebar.header("⚙️ 推論パラメータ調整")
with st.sidebar.form("infer_form"):
    max_new_tokens = st.sidebar.slider("Max New Tokens", 1, 2048, 1024, help="生成するテキストの最大長を制御します。")
    do_sample = st.sidebar.checkbox("Do Sample (多様な出力)", value=True, help="Trueにすると、temperature, top_p, top_kに基づいた多様なテキストを生成します。")
    temperature = st.sidebar.slider("Temperature", 0.01, 2.0, 1.00, 0.05, help="値が高いほどランダムで創造的な出力になります。低いほど決定的で保守的な出力になります。")
    top_p = st.sidebar.slider("Top P (Nucleus Sampling)", 0.00, 1.0, 0.95, 0.05, help="累積確率がこの値を超えるまでのトークン候補からサンプリングします。")
    top_k = st.sidebar.slider("Top K", 0, 64, 0, help="確率の高い上位K個のトークン候補からサンプリングします。0で無効化。")
    repetition_penalty = st.sidebar.slider("Repetition Penalty", 1.0, 2.0, 1.0, 0.05, help="同じ単語やフレーズの繰り返しを抑制します。1.0でペナルティなし。")

# --- 学習セクション ---
st.sidebar.header("\U0001F527 モデル再学習 / パラメータ調整")
with st.sidebar.form("train_form"):
    batch_size = st.number_input("バッチサイズ", min_value=1, max_value=3, value=2)
    epochs = st.number_input("エポック数", min_value=1, value=3)
    learning_rate = st.number_input("学習率", min_value=1e-4, value=2e-4, step=1e-4, format="%.4f")
    logging_steps = st.number_input("ログ間隔", min_value=1, value=1)
    eval_steps = st.number_input("評価間隔 (steps)", min_value=1, value=1)
    save_steps = st.number_input("保存間隔 (steps)", min_value=1, value=100)
    submit_train = st.form_submit_button("学習開始（再学習）")

    if submit_train:
        st.session_state.training_started = True
        response = requests.post(f"{FASTAPI_URL}/train", json={
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "logging_steps": logging_steps,
            "eval_steps": eval_steps,
            "save_steps": save_steps
        })
        st.success("学習プロセスを開始しました。")

# --- データアップロードセクション ---
st.sidebar.header("📊 学習データアップロード")
uploaded_file = st.sidebar.file_uploader("学習用JSONファイルを選択", type=["json"])
if uploaded_file is not None:
    if st.sidebar.button("このデータをアップロード"):
        with st.spinner("ファイルをアップロード中..."):
            files = {'file': (uploaded_file.name, uploaded_file, 'application/json')}
            response = requests.post(f"{FASTAPI_URL}/uploadfile/", files=files)
            if response.status_code == 200 and "error" not in response.json():
                st.sidebar.success(f"✅ {uploaded_file.name} がアップロードされました。")
            else:
                st.sidebar.error(f"アップロード失敗: {response.text}")


# --- メインコンテンツ ---
# --- 推論セクション ---
st.header("\U0001F4DC LLM 推論比較")
prompt = st.text_area("プロンプト入力", "### T1557.003::DHCP Spoofingについて説明せよ。\n### RESPONSE:")

col1, col2 = st.columns(2)

payload_base = {
    "prompt": prompt,
    "use_trained_model": False,
    "max_new_tokens": max_new_tokens,
    "do_sample": do_sample,
    "temperature": temperature,
    "top_p": top_p,
    "top_k": top_k,
    "repetition_penalty": repetition_penalty,
}

payload_trained = payload_base.copy()
payload_trained["use_trained_model"] = True

with col1:
    if st.button("推論実行（学習前モデル）"):
        with st.spinner("学習前モデル推論中..."):
            res_base = requests.post(f"{FASTAPI_URL}/infer", json=payload_base)
            base_text = res_base.json().get("result", "エラーが発生しました")
            st.subheader("\U0001F4D6 学習前モデルの出力")
            st.code(base_text, wrap_lines=True)

with col2:
    if st.button("推論実行（学習後モデル）"):
        with st.spinner("学習後モデル推論中..."):
            res_trained = requests.post(f"{FASTAPI_URL}/infer", json=payload_trained)
            trained_text = res_trained.json().get("result", "エラーが発生しました")
            st.subheader("\U0001F4D8 学習後モデルの出力")
            st.code(trained_text, wrap_lines=True)

# --- 学習ステータス ---
st.header("\U0001F4CA 学習ステータス & 評価")
if st.session_state.get("training_started", False):
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    status_box = st.empty()
    loss_chart_placeholder = st.empty()

    while True:
        try:
            status = requests.get(f"{FASTAPI_URL}/status").json()
        except Exception as e:
            st.warning(f"ステータスサーバーへの接続に失敗しました: {e}")
            time.sleep(3)
            continue

        training_status = status.get("status", "取得中...")
        loss = status.get('loss')
        val_loss = status.get('val_loss')
        loss_str = f"{loss:.4f}" if loss is not None else "N/A"
        val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"

        # ステータステキストの更新
        if status.get('total_steps', 0) > 0:
            status_text.text(f"Status: {status['status']} | Step: {status['step']}/{status['total_steps']} | Epoch: {status.get('epoch', 0):.2f} | Train Loss: {loss_str} | Val Loss: {val_loss_str}")
        else:
            status_text.text(f"Status: {status['status']} | Step: {status.get('step', 0)} | Train Loss: {loss_str} | Val Loss: {val_loss_str}")

        # プログレスバーの更新
        progress = status.get("progress", 0.0)
        progress_bar.progress(min(max(progress, 0.0), 1.0))

        status_box.info(f"ステータス: {training_status}")

        # グラフの更新
        history = status.get("history", {})
        train_steps = history.get("train_steps", [])
        train_loss = history.get("train_loss", [])
        eval_steps = history.get("eval_steps", [])
        val_loss_history = history.get("val_loss", [])

        if train_steps and train_loss:
            # 訓練ロスと検証ロスを一つのデータフレームにまとめる
            df_train = pd.DataFrame({'step': train_steps, 'Training Loss': train_loss}).set_index('step')
            
            chart_data = df_train
            
            if eval_steps and val_loss_history:
                df_val = pd.DataFrame({'step': eval_steps, 'Validation Loss': val_loss_history}).set_index('step')
                # 訓練と検証のデータフレームを結合。stepが一致しない場合はNaNで埋められる
                chart_data = pd.concat([df_train, df_val], axis=1)
                # グラフ描画のために前方/後方で値を補完
                chart_data['Validation Loss'] = chart_data['Validation Loss'].interpolate(method='index')

            loss_chart_placeholder.line_chart(chart_data)


        if training_status.lower() in ["training complete", "error", "infer", "not started"]:
            progress_bar.progress(1.0)
            if training_status.lower() == "training complete":
                st.success("✅ 学習が正常に完了しました！")
            elif training_status.lower() == "error":
                st.error("❌ 学習中にエラーが発生しました。")
            
            # Keep the final state visible but stop polling
            if st.session_state.get("training_started", False):
                st.session_state.training_started = False
                st.info("ポーリングを停止しました。再度学習を開始するには、サイドバーから操作してください。")
            break

        time.sleep(2)


# --- 精度評価の評価軸 ---
st.sidebar.header("⚖️ 精度評価について")
st.sidebar.markdown("""
**評価指標候補：**
- `Loss`: モデルの誤差（低いほど良い）
- `Accuracy`: 正答率（分類的要素があるデータで有効）
- `F1-score`: PrecisionとRecallの調和平均（バランス評価）
- `Perplexity`: 言語モデルの困難度（低いほど良い）

**推奨判断基準：**
- `Loss < 1.0`: 十分に学習されている
- `Accuracy > 80%`: 有用なモデル
- `Perplexity < 30`: 自然な文章生成能力あり
""")
