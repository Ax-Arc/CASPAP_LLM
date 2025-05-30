import json
import logging
import os

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cyber Kill Chainのデータ定義
# 各フェーズにIDとフェーズ番号を追加
CYBER_KILL_CHAIN_STAGES = [
    {
        "ckc_id": "ckc-001",
        "ckc_phase_number": 1,
        "ckc_phase_name_en": "Reconnaissance",
        "ckc_description_en": "Attacker gathers information about the target to identify vulnerabilities and plan the attack. This involves collecting public data (OSINT), deploying spying tools, and using automated scanners.",
        "ckc_phase_name_jp": "偵察",
        "ckc_description_jp": "攻撃者は、標的に関する情報を収集し、脆弱性を特定し、攻撃を計画します。これには、公開情報（OSINT）の収集、スパイツールの展開、自動スキャナーの使用が含まれます。"
    },
    {
        "ckc_id": "ckc-002",
        "ckc_phase_number": 2,
        "ckc_phase_name_en": "Weaponization",
        "ckc_description_en": "Attacker creates or acquires a cyber weapon (e.g., malware, virus) tailored to exploit vulnerabilities identified during reconnaissance. This involves packaging an exploit with a malicious payload.",
        "ckc_phase_name_jp": "武器化",
        "ckc_description_jp": "攻撃者は、偵察段階で特定された脆弱性を悪用するために調整されたサイバー兵器（例：マルウェア、ウイルス）を作成または入手します。これには、エクスプロイトと悪意のあるペイロードのパッケージ化が含まれます。"
    },
    {
        "ckc_id": "ckc-003",
        "ckc_phase_number": 3,
        "ckc_phase_name_en": "Delivery",
        "ckc_description_en": "Attacker transmits the weaponized payload to the target system. Common methods include email attachments, malicious websites, or USB drives.",
        "ckc_phase_name_jp": "配送",
        "ckc_description_jp": "攻撃者は、武器化されたペイロードを標的システムに送信します。一般的な方法には、電子メールの添付ファイル、悪意のあるWebサイト、USBドライブなどがあります。"
    },
    {
        "ckc_id": "ckc-004",
        "ckc_phase_number": 4,
        "ckc_phase_name_en": "Exploitation",
        "ckc_description_en": "The weaponized payload triggers a vulnerability in the target system to execute malicious code.",
        "ckc_phase_name_jp": "攻撃（エクスプロイト）",
        "ckc_description_jp": "武器化されたペイロードが標的システムの脆弱性をトリガーし、悪意のあるコードを実行します。"
    },
    {
        "ckc_id": "ckc-005",
        "ckc_phase_number": 5,
        "ckc_phase_name_en": "Installation",
        "ckc_description_en": "Attacker installs malware or establishes persistent access on the compromised system. This allows the attacker to maintain control for future activities.",
        "ckc_phase_name_jp": "インストール",
        "ckc_description_jp": "攻撃者は、侵害されたシステムにマルウェアをインストールするか、永続的なアクセスを確立します。これにより、攻撃者は将来の活動のために制御を維持できます。"
    },
    {
        "ckc_id": "ckc-006",
        "ckc_phase_number": 6,
        "ckc_phase_name_en": "Command and Control (C2)",
        "ckc_description_en": "Attacker establishes a communication channel with the compromised system to remotely control it, exfiltrate data, or issue further commands.",
        "ckc_phase_name_jp": "コマンド＆コントロール (C2)",
        "ckc_description_jp": "攻撃者は、侵害されたシステムとの通信チャネルを確立し、リモートで制御したり、データを窃取したり、さらなるコマンドを発行したりします。"
    },
    {
        "ckc_id": "ckc-007",
        "ckc_phase_number": 7,
        "ckc_phase_name_en": "Actions on Objectives",
        "ckc_description_en": "Attacker achieves their ultimate goals, such as data exfiltration, data destruction, or launching further attacks.",
        "ckc_phase_name_jp": "目的の実行",
        "ckc_description_jp": "攻撃者は、データの窃取、データの破壊、さらなる攻撃の開始など、最終的な目的を達成します。"
    }
]

def save_to_json(data, filename="cyber_kill_chain_data_jp.json"):
    """
    指定されたデータをJSONファイルに保存します。

    Args:
        data (list): 保存するデータのリスト（辞書を含む）。
        filename (str): 出力するJSONファイル名。
    """
    try:
        # filepath = os.path.join(os.getcwd(), filename)
        filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data', filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info(f"Cyber Kill Chainデータが正常に '{filepath}' に保存されました。")
    except IOError as e:
        logging.error(f"ファイル '{filepath}' の書き込み中にエラーが発生しました: {e}")
    except Exception as e:
        logging.error(f"予期せぬエラーが発生しました: {e}")

def main():
    """
    メイン処理関数。Cyber Kill ChainデータをJSONファイルとして保存します。
    """
    logging.info("Cyber Kill Chain JSON生成スクリプトを開始します。")
    save_to_json(CYBER_KILL_CHAIN_STAGES)
    logging.info("Cyber Kill Chain JSON生成スクリプトが完了しました。")

if __name__ == "__main__":
    main()