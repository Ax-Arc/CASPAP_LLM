import json
import os
from collections import OrderedDict

def merge_attack_data(enterprise_file, ckc_mapping_file, output_file):
    """
    MITRE ATT&CK EnterpriseデータにCKCマッピング情報を指定された順序で追加し、
    新しいJSONファイルとして保存する (Ver2)。

    Args:
        enterprise_file (str): MITRE ATT&CK EnterpriseデータのJSONファイルパス。
                                (mitre_attack_enterprise_data_with_subtechniques_jp.json)
        ckc_mapping_file (str): CKCマッピング情報のJSONファイルパス。
                                (mitre_attack_ckc_mapping_jp.json)
        output_file (str): 出力するJSONファイルパス。
                           (attack_ent_with_subtec_ckc_mapping_jp.json)
    """
    try:
        # MITRE ATT&CK Enterpriseデータの読み込み
        with open(enterprise_file, 'r', encoding='utf-8') as f:
            enterprise_data_list = json.load(f) # リストとして読み込む
        print(f"'{enterprise_file}' を正常に読み込みました。")

        # CKCマッピング情報の読み込み
        with open(ckc_mapping_file, 'r', encoding='utf-8') as f:
            ckc_mappings = json.load(f)
        print(f"'{ckc_mapping_file}' を正常に読み込みました。")

    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません - {e.filename}")
        return
    except json.JSONDecodeError as e:
        print(f"エラー: JSONデコードエラー - {e}")
        return
    except Exception as e:
        print(f"ファイルの読み込み中に予期せぬエラーが発生しました: {e}")
        return

    # CKCマッピング情報をタクティクスIDをキーとする辞書に変換
    ckc_dict = {
        mapping['attack_tactic_id']: {
            'ckc_map_info': mapping.get('ckc_map_info', []),
            'ckc_mapping_rational': mapping.get('ckc_mapping_rational', '')
        }
        for mapping in ckc_mappings
    }
    print("CKCマッピング情報を辞書に変換しました。")

    # 新しい順序でデータを格納するためのリスト
    updated_enterprise_data = []

    # EnterpriseデータにCKC情報を指定された順序で追加
    for original_tactic in enterprise_data_list:
        # OrderedDictを使用してキーの順序を保持
        # Python 3.7+ では通常のdictでも挿入順が保持されますが、
        # 明示的にOrderedDictを使用するか、あるいはキーを順番に追加していくことで
        # 意図した順序を確実にします。
        # ここでは、新しい辞書に必要な順序でキーを追加していきます。
        
        processed_tactic = {}

        # 既存のキーを順番に追加 (description_jpまで)
        keys_before_ckc = ['id', 'name_eng', 'name_jp', 'description_eng', 'description_jp']
        for key in keys_before_ckc:
            if key in original_tactic:
                processed_tactic[key] = original_tactic[key]

        # CKC情報を取得して追加
        tactic_id = original_tactic.get('id')
        ckc_info_to_add = ckc_dict.get(tactic_id, {'ckc_map_info': [], 'ckc_mapping_rational': ''})
        processed_tactic['ckc_map_info'] = ckc_info_to_add['ckc_map_info']
        processed_tactic['ckc_mapping_rational'] = ckc_info_to_add['ckc_mapping_rational']

        # 残りのキーを追加 (techniques以降)
        # 'description_jp' の後にCKC情報を入れたので、それ以外のキーを元の順序で追加
        for key, value in original_tactic.items():
            if key not in processed_tactic: # まだ追加されていないキーのみ
                processed_tactic[key] = value
        
        updated_enterprise_data.append(processed_tactic)

    print("EnterpriseデータへのCKC情報の指定順序での追加が完了しました。")

    # 結果を新しいJSONファイルに保存
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(updated_enterprise_data, f, ensure_ascii=False, indent=4)
        print(f"結合されたデータが '{output_file}' に正常に保存されました。")
    except IOError as e:
        print(f"エラー: ファイルへの書き込み中にエラーが発生しました - {e}")
    except Exception as e:
        print(f"ファイルの書き込み中に予期せぬエラーが発生しました: {e}")

if __name__ == '__main__':
    enterprise_data_file = "mitre_attack_enterprise_data_with_subtechniques_jp.json"
    ckc_mapping_file = "mitre_attack_ckc_mapping_jp.json"
    output_file = "attack_ent_with_subtec_ckc_mapping_jp.json" # 出力ファイル名を変更

    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data')

    enterprise_data_file_path = os.path.join(output_path, enterprise_data_file)
    ckc_mapping_file_path = os.path.join(output_path, ckc_mapping_file)
    output_file_path = os.path.join(output_path, output_file)

    # 関数を実行してデータをマージ
    merge_attack_data(enterprise_data_file_path, ckc_mapping_file_path, output_file_path)
