# CASPAP_LLM

このプロジェクトは、最新のサイバーセキュリティ脅威インテリジェンス（MITRE ATT&CK / Cyber Kill Chain）とSIEM機器から出力されるセキュリティログを活用し、LLM（大規模言語モデル）やRAG（Retrieval-Augmented Generation）技術を組み合わせて脅威分析を行うための統合実証システムです。

## 全体アーキテクチャ・シーケンス図 (Sequence Diagram)

フロントエンドのUIからAPIサーバーにリクエストが送られ、LLMがRAGと連携して分析・推論を行う大まかなシステム全体の流れです。

```mermaid
sequenceDiagram
    participant User as セキュリティアナリスト(ユーザー)
    participant Streamlit as Web UI (chat_ui_streaming 等)
    participant FastAPI as Backend API (fastapi_llm_server / rag_agent_server)
    participant LocalData as ローカルデータ (ATT&CK / SIEMログ等)
    participant LLM as LLM エンジン (推論/生成)

    User->>Streamlit: ログアップロードおよび分析指示
    Streamlit->>FastAPI: アラートデータ・質問を送信
    FastAPI->>LocalData: 関連するATT&CK情報やログの検索(RAG)
    LocalData-->>FastAPI: コンテキストデータの返却
    FastAPI->>LLM: プロンプトとコンテキストを送信
    LLM-->>FastAPI: 分析結果・脅威スコアなどストリーミング生成
    FastAPI-->>Streamlit: 応答のストリーミングストリーム
    Streamlit-->>User: 画面へリアルタイム表示
```

## ディレクトリ・ファイル構成

プロジェクトのルートに含まれる主要なスクリプトファイルおよびデータ群です。

- **`chat_ui_streaming.py`**: ストリーミングレスポンスに対応したLLMとの対話用StreamlitチャットUIアプリケーション。
- **`fastapi_llm_server_CLM.py`**: CLM (Causal Language Model) 向けの推論エンドポイント用のFastAPIサーバー。
- **`fastapi_llm_server_SFT.py`**: SFT (Supervised Fine-Tuning) 済みのLLM推論エンドポイント用のFastAPIサーバー。
- **`rag_agent_server_streaming.py`**: 解析エージェント機能およびRAGを統合し、ストリーミングレスポンスを返すバックエンドAPIサーバー。
- **`streamlit_llm_train.py`**: LLMの評価やファインチューニングの可視化等のためのStreamlitアプリ。
- **`asset_management.csv`**: 分析システム上で使用されるアセット（資産）管理情報のCSVデータ。
- **`BaseData_ATTACK_and_CKC/`**: MITRE ATT&CK と CKC データの収集・統合用サブプロジェクト（詳細は後述）。
- **`SIEM_LEAF_LOG/`**: SIEM ログのパースと構造化を行うサブプロジェクト（詳細は後述）。

### 実行手順

本システム（FastAPIバックエンドおよびStreamlitフロントエンド）をローカルで起動・動作させるための基本的な手順です。

1. **仮想環境の有効化**
   プロジェクトルートディレクトリで仮想環境（`.venv`等）を有効化します。
   ```bash
   # Windows (PowerShell) の場合
   .\.venv\Scripts\Activate.ps1
   ```

2. **バックエンドサーバー (FastAPI) の起動**
   利用する推論エンジン・用途に応じて、エンドポイントを起動します。
   ```bash
   # 例: RAGエージェントのストリーミングサーバーを起動する場合
   uvicorn rag_agent_server_streaming:app --reload --port 8000
   
   # 例: SFTモデルエンドポイントを起動する場合
   uvicorn fastapi_llm_server_SFT:app --reload --port 8000
   ```

3. **フロントエンド (Streamlit) の起動**
   別のターミナルを開き（再度仮想環境を有効化して）、チャットUIなどのStreamlitアプリを起動します。
   ```bash
   streamlit run chat_ui_streaming.py
   ```
   起動に成功すると、ブラウザが自動的に開き（デフォルト `http://localhost:8501`）、チャットUIにアクセスして操作が可能になります。


---

## サブプロジェクト概要

以下は、脅威インテリジェンス統合およびSIEMログ解析を担う各サブディレクトリの詳細な仕様です。

### 1. BaseData_ATTACK_and_CKC

このディレクトリは、MITRE ATT&CK エンタープライズ版データと Cyber Kill Chain (CKC) データを収集・加工・統合するためのサブプロジェクトです。それぞれのフレームワークの利点を取り入れ、攻撃のエンティティとキルチェーンのフェーズをマッピングした構造化データを提供します。

#### ディレクトリ構成

- **`Data/`**: 収集やスクリプトによる加工を経て生成されたJSON、CSVなどのデータファイルが格納されています。
  - `enterprise-attack-stix-data.json`: オリジナルのMITRE ATT&CKのSTIX形式データ。
  - `mitre_attack_enterprise_data_with_subtechniques_eng.json` / `_jp.json`: ATT&CKデータ（サブテクニック含む）の英語/日本語版。
  - `cyber_kill_chain_data_jp.json`: Cyber Kill Chain (CKC) の各フェーズを定義したデータ。
  - `mitre_attack_ckc_mapping_jp.json`: ATT&CKとCKCを関連付けるためのマッピング定義ファイル。
  - `attack_ent_with_subtec_ckc_mapping_jp.json` / `.csv`: ATT&CKデータとCKCのフェーズを結合（マージ）した完成版のデータ。

- **`Source/`**: データを外部から取得・整形し、相互に関連付けるためのPythonスクリプト群です。
  - `Get_Mitre_Attack_Ent_data_eng.py` / `_jp.py`: MITREのSTIXファイルから必要な情報を抽出し、JSON形式として保存します（英語/日本語対応）。
  - `Cyber_Kill_Chain_Data_Jp.py`: CKCの規定フェーズを出力します。
  - `MitreAttack_CKC_Mapping_Data_Jp.py`: 両フレームワーク間のマッピングリファレンスデータを作成します。
  - `Marge_ATTACK_ENT_with_subtec_CKC_mapping.py`: 各データを統合し、最終的なマッピング済みJSONやCSVを出力します。
  - `Split_Attack_ckc_mapping_jp.py`: 統合データの処理や分割などを行います。

#### ワークフロー・役割

主に `Source/` のスクリプト群を実行することで、MITRE ATT&CK の脅威インテリジェンスと、Cyber Kill Chain のフェーズを関連付けます。これにより、自立型のスクリプトやLLMが読み込みやすい形式(JSON/CSV)に統合されたデータ (`Data/` フォルダ配下) を生成・提供することを主目的としています。

##### データ生成ワークフロー (Data Flow)

以下の図は、各スクリプトがどのようにデータを取得・生成し、最終的な統合マッピングデータへと繋がっていくかを示しています。

```mermaid
graph TD
    %% Source Scripts
    subgraph Source["Source/ (Scripts)"]
        GetAtt["Get_Mitre_Attack_Ent_data_jp.py"]
        GetCkc["Cyber_Kill_Chain_Data_Jp.py"]
        GenMap["MitreAttack_CKC_Mapping_Data_Jp.py"]
        Merge["Marge_ATTACK_ENT_with_subtec_CKC_mapping.py"]
    end

    %% Data Files
    StixData("enterprise-attack-stix-data.json")
    AttData("mitre_attack_enterprise_data..._jp.json")
    CkcData("cyber_kill_chain_data_jp.json")
    MapData("mitre_attack_ckc_mapping_jp.json")
    MergedJson("attack_ent_with_subtec_ckc_mapping_jp.json")

    %% Relations
    StixData --> GetAtt
    GetAtt --> AttData

    GetCkc --> CkcData
    GenMap --> MapData

    AttData --> Merge
    CkcData --> Merge
    MapData --> Merge

    Merge --> MergedJson
```

##### オブジェクト関連図 (Entity Relationship)

生成される統合データの論理的な構造は、以下のような包含・マッピング関係を持ちます。Cyber Kill Chain のフェーズに対して、ATT&CKの各要素がどのように紐づいているかを表します。

```mermaid
erDiagram
    CyberKillChainPhase ||--o{ ATTACK_Tactic : "マッピング"
    ATTACK_Tactic ||--|{ ATTACK_Technique : "包含"
    ATTACK_Technique ||--o{ ATTACK_SubTechnique : "包含"

    CyberKillChainPhase {
        string phase_id
        string phase_name "例: Reconnaissance"
    }
    ATTACK_Tactic {
        string tactic_id "例: TA0043"
        string tactic_name "例: Reconnaissance"
    }
    ATTACK_Technique {
        string technique_id "例: T1592"
        string technique_name "例: Gather Victim Host Information"
    }
    ATTACK_SubTechnique {
        string subtechnique_id "例: T1592.001"
        string subtechnique_name "例: Hardware"
    }
```

---

### 2. SIEM_LEAF_LOG

このディレクトリは、SIEM (Security Information and Event Management) システムから出力される **LEEF (Log Event Extended Format)** や **CEF (Common Event Format)** 形式のログをパース（解析）し、機械学習や分析に利用しやすいようにJSON構造データへ整形するためのツールセットです。

#### ディレクトリ構成

- **`Data/`**: パース処理のテストや実行結果となるデータが格納されています。
  - `Test_SIEM_LEEF_LOG.json`: 解析スクリプトによってパース・構造化されたログのサンプル出力データ。

- **`Source/`**: ログデータを実際に処理するプログラムを含みます。
  - `SIEM_Log_Perther.py`: （※Parser）LEEF形式やCEF形式の文字列を読み込み、正規表現を用いた分離およびキーバリューの抽出を行い、分析しやすい共通のキー（フィールド名）へ構造化するPythonスクリプト。

#### データ処理のシーケンス図 (Sequence Diagram)

スクリプト内で、生のログ文字列からどのように構造化済みJSONに変換されるかの流れを示しています。

```mermaid
sequenceDiagram
    participant RawLog as 生ログ (LEEF/CEF)
    participant Parser as SIEM_Log_Perther.py
    participant ParseFunc as parse_leef_log()
    participant StructFunc as structure_qradar_log()
    participant JSON as 出力 (JSON)

    RawLog->>Parser: ログ1行を読み込み
    Parser->>ParseFunc: ヘッダー抽出・拡張フィールド分割
    ParseFunc-->>Parser: パースされた辞書データ (生フィールド名)
    Parser->>StructFunc: 一般的なキー名へのマッピング
    StructFunc-->>Parser: 構造化済み辞書データ
    Parser->>JSON: JSONとして出力・保存
```

#### オブジェクト関連図 (Entity Relationship Diagram)

ログデータが加工される過程で、それぞれのデータ構造がどのような属性と関係性を持つかを表しています。

```mermaid
erDiagram
    RAW_LOG ||--o{ PARSED_LOG : "パース (Extraction)"
    PARSED_LOG ||--|| STRUCTURED_LOG : "構造化 (Mapping)"

    RAW_LOG {
        string raw_string "例: LEEF:1.0|IBM|QRadar|..." 
    }
    PARSED_LOG {
        string LEEFVersion "ヘッダーより"
        string Vendor "ヘッダーより"
        string EventID "ヘッダーより"
        string payload "拡張フィールドの動的 Key-Value"
    }
    STRUCTURED_LOG {
        dict original_log "PARSED_LOGの内容を保持"
        string device_vendor "Vendorからマッピング"
        string source_ip "src/SourceIp等からマッピング"
        string timestamp_epoch_ms "devTime等からマッピング"
        string severity_label "sev等からマッピング"
    }
```

#### 使用方法概要
`SIEM_Log_Perther.py` を実行するか、外部モジュールとして `parse_leef_log` と `structure_qradar_log` 関数を呼び出すことで、多様なベンダのセキュリティログを正規化されたフィールド体系に変換することができます。
