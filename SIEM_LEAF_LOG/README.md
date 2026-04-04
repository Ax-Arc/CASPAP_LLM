# SIEM_LEAF_LOG

このディレクトリは、SIEM (Security Information and Event Management) システムから出力される **LEEF (Log Event Extended Format)** や **CEF (Common Event Format)** 形式のログをパース（解析）し、機械学習や分析に利用しやすいようにJSON構造データへ整形するためのツールセットです。

## ディレクトリ構成

### `Data/`
パース処理のテストや実行結果となるデータが格納されています。
- `Test_SIEM_LEEF_LOG.json`: 解析スクリプトによってパース・構造化されたログのサンプル出力データ。

### `Source/`
ログデータを実際に処理するプログラムを含みます。
- `SIEM_Log_Perther.py`: （※Parser）LEEF形式やCEF形式の文字列を読み込み、正規表現を用いた分離およびキーバリューの抽出を行い、分析しやすい共通のキー（フィールド名）へ構造化するPythonスクリプト。

---

## データ処理のシーケンス図 (Sequence Diagram)

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

---

## オブジェクト関連図 (Entity Relationship Diagram)

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

## 使用方法概要
`SIEM_Log_Perther.py` を実行するか、外部モジュールとして `parse_leef_log` と `structure_qradar_log` 関数を呼び出すことで、多様なベンダのセキュリティログを正規化されたフィールド体系に変換することができます。
