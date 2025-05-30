import re
import json
from datetime import datetime

def parse_leef_log(log_line: str) -> dict:
    """
    1行のLEEF形式のログをパースし、構造化された辞書として返します。

    LEEF形式の一般的な構造:
    LEEF:Version|Vendor|Product|Version|EventID|Key=Value\tKey2=Value2...

    Args:
        log_line (str): LEEF形式のログ文字列。

    Returns:
        dict: パースされたログデータを格納した辞書。
              パースに失敗した場合は空の辞書を返します。
    """
    parsed_data = {}
    log_line = log_line.strip()

    # LEEFヘッダーの正規表現 (LEEF:1.0 または LEEF:2.0)
    # LEEF:Version|Vendor|Product|Version|EventID|Extension
    header_match = re.match(r"LEEF:(\d\.\d)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|?(.*)", log_line, re.IGNORECASE)
    
    if not header_match:
        # CEF形式のログも考慮 (一般的なSIEM出力として)
        # CEF:Version|DeviceVendor|DeviceProduct|DeviceVersion|SignatureID|Name|Severity|Extension
        header_match = re.match(r"CEF:(\d+)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|?(.*)", log_line, re.IGNORECASE)
        if header_match:
            parsed_data["CEFVersion"] = header_match.group(1)
            parsed_data["DeviceVendor"] = header_match.group(2)
            parsed_data["DeviceProduct"] = header_match.group(3)
            parsed_data["DeviceVersion"] = header_match.group(4)
            parsed_data["SignatureID"] = header_match.group(5)
            parsed_data["Name"] = header_match.group(6)
            parsed_data["Severity"] = header_match.group(7)
            extensions_str = header_match.group(8)
            parsed_data["log_format"] = "CEF"
        else:
            print(f"警告: 有効なLEEF/CEFヘッダーが見つかりません: {log_line[:100]}...")
            return {} # ヘッダーがなければ処理しない
    else:
        parsed_data["LEEFVersion"] = header_match.group(1)
        parsed_data["Vendor"] = header_match.group(2)
        parsed_data["Product"] = header_match.group(3)
        parsed_data["Version"] = header_match.group(4)
        parsed_data["EventID"] = header_match.group(5)
        extensions_str = header_match.group(6)
        parsed_data["log_format"] = "LEEF"

    if not extensions_str:
        # 拡張フィールドがない場合もある
        return parsed_data

    # 拡張フィールドのパース
    # Key=Value のペアがタブ、または他の一般的なデリミタ（例: スペース、セミコロン）で区切られていることを想定
    # LEEF標準ではタブ区切りだが、QRadarの実装や他のログソースでは異なる場合があるため、柔軟性を持たせる
    # まずタブで分割を試みる
    
    # デリミタの優先順位: タブ -> スペース (ただし、値にスペースが含まれる場合を考慮)
    # LEEFの拡張フィールドは通常タブ区切り
    # Key=Value の形式で、Valueの最後までを正しく取得する
    
    # 拡張フィールドのキー=値ペアを抽出する正規表現
    # 各キー=値ペアはタブで区切られることを基本とする
    # 値にはエスケープされた = やタブが含まれる可能性があるが、ここでは単純なケースを扱う
    # 例: key1=value1\tkey2=value with spaces\tkey3=another=val
    
    # まず、拡張文字列をタブで分割
    raw_extensions = extensions_str.split('\t')
    
    for item in raw_extensions:
        item = item.strip()
        if not item:
            continue
            
        # '=' でキーと値を分割。最初の '=' のみで分割する
        parts = item.split('=', 1)
        if len(parts) == 2:
            key, value = parts
            # キーのサニタイズ (例: 不適切な文字の除去や置換)
            key = key.strip()
            # 一般的なフィールド名をキャメルケースやスネークケースに正規化することも検討できる
            # (例: sourceIP -> sourceIp, user name -> userName)
            # ここでは単純にそのまま使用
            
            # 値のサニタイズや型変換
            value = value.strip()

            # 主要フィールドの特定と型変換の試み
            if key in ["devTime", "DeviceTime", "startTime", "endTime"]:
                try:
                    # QRadarのdevTimeはエポックミリ秒が多い
                    timestamp_ms = int(value)
                    parsed_data[key + "_datetime_utc"] = datetime.utcfromtimestamp(timestamp_ms / 1000).isoformat() + "Z"
                    parsed_data[key] = timestamp_ms # 元の値も保持
                except ValueError:
                    # 他の形式の日時文字列の場合もある
                    parsed_data[key] = value
            elif key in ["src", "dst", "sourceAddress", "destinationAddress", "SourceIp", "DestinationIp"]:
                # IPアドレスフィールド (特に処理は不要だが、型チェックや正規化の余地あり)
                parsed_data[key] = value
            elif key in ["srcPort", "dstPort", "sourcePort", "destinationPort", "SourcePort", "DestinationPort"]:
                try:
                    parsed_data[key] = int(value)
                except ValueError:
                    parsed_data[key] = value # 数値でない場合はそのまま
            elif key in ["sev", "Severity", "deviceSeverity"]: # SeverityはCEFヘッダーにもある
                # Severity は数値の場合と文字列の場合がある
                if value.isdigit():
                     parsed_data[key] = int(value)
                else:
                    parsed_data[key] = value # 例: "High", "Low"
            elif key in ["usrName", "UserName", "sourceUserName", "destinationUserName"]:
                parsed_data[key] = value
            elif key in ["cat", "HighLevelCategory", "categoryDeviceGroup", "deviceEventCategory"]: # HighLevelCategory
                parsed_data[key] = value
            elif key in ["LCID", "LowLevelCategory", "deviceEventClassId"]: # LowLevelCategory
                parsed_data[key] = value
            elif key == "QID": # QRadar Event ID
                 parsed_data[key] = value # 通常は数値だが文字列として扱う
            elif key in ["proto", "Protocol", "transportProtocol"]:
                parsed_data[key] = value
            else:
                # その他のフィールド
                parsed_data[key] = value
        elif item: # 値なしのキー、または不正な形式
            # 不正な形式やキーのみの項目は、キーとして登録し値をNoneとするか、無視するか選択
            # ここではキーとして登録し、値は元の文字列のまま（エラーとして扱うことも可能）
            parsed_data[item.strip()] = None 
            # print(f"警告: 不正なキー・バリューペア形式: {item}")

    return parsed_data

def structure_qradar_log(parsed_log: dict) -> dict:
    """
    パースされたQRadarログ辞書を、さらに分析しやすいように構造化します。
    レポートの表2「分析のための主要なQRadarログフィールド」で定義された
    フィールド名を参考に、共通のキー名にマッピングすることを試みます。

    Args:
        parsed_log (dict): parse_leef_logから返された辞書。

    Returns:
        dict: 構造化されたログデータを格納した辞書。
    """
    if not parsed_log:
        return {}

    structured_event = {"original_log": parsed_log.copy()} # 元の全フィールドも保持

    # 主要フィールドのマッピング (レポートの表2 および一般的なフィールド名)
    # 複数の可能性のあるキー名を1つの共通キーにマッピング
    field_mapping = {
        "timestamp_utc": ["devTime_datetime_utc", "DeviceTime_datetime_utc", "startTime_datetime_utc", "endTime_datetime_utc"],
        "timestamp_epoch_ms": ["devTime", "DeviceTime", "startTime", "endTime"], # 数値の場合
        "source_ip": ["src", "SourceIp", "sourceAddress", "sourceHostName"], # HostNameも考慮
        "destination_ip": ["dst", "DestinationIp", "destinationAddress", "destinationHostName"],
        "source_port": ["srcPort", "SourcePort", "sport"],
        "destination_port": ["dstPort", "DestinationPort", "dport"],
        "protocol": ["proto", "Protocol", "transportProtocol"],
        "username": ["usrName", "UserName", "sourceUserName", "suser"], # suserもよく使われる
        "severity_score": ["sev", "Severity", "deviceSeverity"], # 数値の場合
        "severity_label": ["sev", "Severity", "deviceSeverity"], # 文字列の場合
        "event_id_qradar": ["QID"],
        "event_name": ["eventName", "Name", "catDesc"], # NameはCEFヘッダーにもある
        "category_high_level": ["cat", "HighLevelCategory", "categoryDeviceGroup", "deviceEventCategory"],
        "category_low_level": ["LCID", "LowLevelCategory", "deviceEventClassId"],
        "payload_data": ["payload", "msg", "message"], # msgやmessageもペイロードとしてよく使われる
        "device_vendor": ["DeviceVendor", "Vendor"],
        "device_product": ["DeviceProduct", "Product"],
        "device_version": ["DeviceVersion", "Version"],
        "signature_id": ["SignatureID", "EventID"], # EventIDはLEEFヘッダーにもある
        "log_format_detected": ["log_format"]
    }

    for common_key, possible_keys in field_mapping.items():
        for p_key in possible_keys:
            if p_key in parsed_log:
                value = parsed_log[p_key]
                # Severityの数値/文字列の分離
                if common_key == "severity_score" and isinstance(value, (int, float)):
                    structured_event[common_key] = value
                    break
                elif common_key == "severity_label" and isinstance(value, str) and not value.isdigit():
                    structured_event[common_key] = value
                    break
                elif common_key not in ["severity_score", "severity_label"]: # Severity以外はそのまま
                    structured_event[common_key] = value
                    break 
    
    # 元のログで共通キーにマッピングされなかったものも追加
    for key, value in parsed_log.items():
        # マッピング済みのキーや、datetimeヘルパーフィールドは除く
        already_mapped = False
        for pk_list in field_mapping.values():
            if key in pk_list:
                already_mapped = True
                break
        if key.endswith("_datetime_utc") and key[:-13] in field_mapping["timestamp_epoch_ms"]: # devTime_datetime_utcなど
            already_mapped = True

        if not already_mapped and key not in structured_event:
            structured_event[f"custom_{key}"] = value # 未知のフィールドはcustom_プレフィックス

    return structured_event

# --- メイン処理とテスト ---
if __name__ == "__main__":
    # サンプルLEEFログ (QRadar風)
    sample_logs = [
        r"LEEF:1.0|IBM|QRadar|1.0|EventID123|src=192.168.1.100\tdst=10.0.0.5\tsrcPort=54321\tdstPort=443\tproto=TCP\tusrName=j.doe\tsev=8\tcat=Authentication\tLCID=Login Failure\tQID=5000123\tdevTime=1678886400123\tpayload=User login failed for j.doe from 192.168.1.100",
        r"LEEF:2.0|Microsoft|Windows|10.0|Security:4625|devTime=1678886500000\tSourceIp=172.16.0.20\tDestinationIp=172.16.0.1\tSourcePort=12345\tDestinationPort=80\tProtocol=UDP\tUserName=system\tSeverity=5\tHighLevelCategory=Access\tLowLevelCategory=Failed Logon\tmsg=An account failed to log on.",
        r"LEEF:1.0|PaloAltoNetworks|PAN-OS|9.1|THREAT|sev=High\tdst=1.2.3.4\tsrc=192.168.1.20\tproto=tcp\tusrName=\tdstPort=80\tpayload=Suspicious traffic detected\tcat=Malware", # usrNameが空
        r"LEEF:1.0|CISCO|ASA|1.0|106023|src=203.0.113.10\tdst=198.51.100.20\tsrcPort=1024\tdstPort=22\tproto=SSH\tsev=3\tcat=Firewall\tLCID=Connection Allowed\tQID=1000001\tdevTime=1678886400123\truleName=Allow_SSH_External",
        # CEF形式のサンプル
        r"CEF:0|TrendMicro|DeepSecurity|12.0|1001|URLReputationBlocked|6|src=192.168.0.15 dst=34.56.78.90 spt=49152 dpt=80 request=http://malicious.example.com/evil.exe cat=Malware cn1Label=ThreatID cn1=12345",
        r"LEEF:1.0|SomeVendor|SomeProduct|1.0|SomeEvent|keyOnly\tanotherKey=val\tkeyWithEquals=value=stillvalue", # 不正な形式を含む
        r"NotALeefOrCefLog", # 無効なログ
        r"LEEF:1.0|MinimalVendor|MinimalProduct|1.0|MinimalEventID|", # 拡張なし
    ]

    print("--- QRadar LEEF/CEFログパーサー テスト ---")
    for i, log_entry in enumerate(sample_logs):
        print(f"\n--- サンプルログ {i+1} ---")
        print(f"RAW: {log_entry}")
        
        parsed = parse_leef_log(log_entry)
        if parsed:
            print("パース結果 (parse_leef_log):")
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
            
            structured = structure_qradar_log(parsed)
            print("\n構造化結果 (structure_qradar_log):")
            print(json.dumps(structured, indent=2, ensure_ascii=False))
        else:
            print("パース失敗。")

    # ファイルからの読み込み例 (コメントアウト)
    """
    try:
        with open("qradar_logs.txt", "r", encoding="utf-8") as f:
            for line in f:
                log_line = line.strip()
                if log_line:
                    parsed = parse_leef_log(log_line)
                    if parsed:
                        structured = structure_qradar_log(parsed)
                        # ここで structured データを使って何か処理を行う
                        # 例: print(json.dumps(structured, indent=2, ensure_ascii=False))
                        pass 
    except FileNotFoundError:
        print("\nログファイル 'qradar_logs.txt' が見つかりません。")
    except Exception as e:
        print(f"\nファイル処理中にエラーが発生しました: {e}")
    """
