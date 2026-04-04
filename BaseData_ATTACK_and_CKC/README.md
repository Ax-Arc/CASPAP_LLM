# BaseData_ATTACK_and_CKC

このディレクトリは、MITRE ATT&CK エンタープライズ版データと Cyber Kill Chain (CKC) データを収集・加工・統合するためのサブプロジェクトです。それぞれのフレームワークの利点を取り入れ、攻撃のエンティティとキルチェーンのフェーズをマッピングした構造化データを提供します。

## ディレクトリ構成

### `Data/`
収集やスクリプトによる加工を経て生成されたJSON、CSVなどのデータファイルが格納されています。
- `enterprise-attack-stix-data.json`: オリジナルのMITRE ATT&CKのSTIX形式データ。
- `mitre_attack_enterprise_data_with_subtechniques_eng.json` / `_jp.json`: ATT&CKデータ（サブテクニック含む）の英語/日本語版。
- `cyber_kill_chain_data_jp.json`: Cyber Kill Chain (CKC) の各フェーズを定義したデータ。
- `mitre_attack_ckc_mapping_jp.json`: ATT&CKとCKCを関連付けるためのマッピング定義ファイル。
- `attack_ent_with_subtec_ckc_mapping_jp.json` / `.csv`: ATT&CKデータとCKCのフェーズを結合（マージ）した完成版のデータ。

### `Source/`
データを外部から取得・整形し、相互に関連付けるためのPythonスクリプト群です。
- `Get_Mitre_Attack_Ent_data_eng.py` / `_jp.py`: MITREのSTIXファイルから必要な情報を抽出し、JSON形式として保存します（英語/日本語対応）。
- `Cyber_Kill_Chain_Data_Jp.py`: CKCの規定フェーズを出力します。
- `MitreAttack_CKC_Mapping_Data_Jp.py`: 両フレームワーク間のマッピングリファレンスデータを作成します。
- `Marge_ATTACK_ENT_with_subtec_CKC_mapping.py`: 各データを統合し、最終的なマッピング済みJSONやCSVを出力します。
- `Split_Attack_ckc_mapping_jp.py`: 統合データの処理や分割などを行います。

### `backup/`
プロセッサスクリプト（`data_processor.py` 等）の退避コードや、データ整形テスト用の出力ファイル、その他JSONデータのバックアップが保管されています。

## ワークフロー・役割

主に `Source/` のスクリプト群を実行することで、MITRE ATT&CK の脅威インテリジェンスと、Cyber Kill Chain のフェーズを関連付けます。これにより、自立型のスクリプトやLLMが読み込みやすい形式(JSON/CSV)に統合されたデータ (`Data/` フォルダ配下) を生成・提供することを主目的としています。

### データ生成ワークフロー (Data Flow)

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

### オブジェクト関連図 (Entity Relationship)

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
