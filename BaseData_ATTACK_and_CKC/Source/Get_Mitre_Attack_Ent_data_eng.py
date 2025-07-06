# coding: utf-8
# MITRE ATT&CKデータの自動取得と構造化 (構文エラー修正版)
#
# 実行前の確認事項:
# 1. `mitreattack-python` ライブラリがインストールされていることを確認してください。
#    pip install mitreattack-python
# 2. 依存ライブラリ、特に `requests-cache` が適切なバージョンであることを確認してください。
#    過去に `requests-cache` の古いバージョン (0.9.1未満) でインポートエラーが報告されています。
#    pip install --upgrade requests-cache mitreattack-python
#
# このスクリプトは、MITRE ATT&CK EnterpriseのSTIXデータ（JSON形式）をダウンロードし、
# 戦術、テクニック、およびサブテクニックの情報を階層的に構造化してJSONファイルに出力します。

from mitreattack.stix20 import MitreAttackData # インポートパスを修正
import json
import os
import requests # requestsを明示的にインポート

def download_attack_data(url, filename="enterprise-attack.json"):
    """
    指定されたURLからATT&CK STIXデータをダウンロードし、ローカルに保存します。
    ファイルが既に存在する場合はダウンロードをスキップします。
    """
    file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data', filename)
    if not os.path.exists(file_path):
        print(f"ATT&CK STIXデータを {url} からダウンロードしています...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # HTTPエラーがあれば例外を発生させる
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"正常に {filename} として保存しました。")
        except requests.exceptions.RequestException as e:
            print(f"ATT&CKデータのダウンロード中にエラーが発生しました: {e}")
            return None
        except Exception as e:
            print(f"予期せぬエラーが発生しました (ダウンロード処理中): {e}")
            return None
    else:
        print(f"{filename} は既に存在します。ダウンロードをスキップします。")
    return filename

def get_attack_enterprise_data_with_subtechniques(stix_file_path):
    """
    ローカルのSTIXファイルからMITRE ATT&CK Enterpriseの戦術、テクニック、
    およびそれらに紐づくサブテクニックの情報を取得し、構造化します。
    無効化された(revoked/deprecated)オブジェクトは除外します。
    """
    try:
        # MitreAttackDataオブジェクトを初期化
        attack_data_source = MitreAttackData(stix_file_path)
    except Exception as e:
        print(f"MitreAttackDataオブジェクトの初期化中にエラーが発生しました: {e}")
        print("STIXファイルが破損しているか、形式が正しくない可能性があります。")
        return None
    
    # Enterprise ATT&CKの戦術を取得 (無効化されたものを除外)
    try:
        enterprise_tactics = attack_data_source.get_tactics(remove_revoked_deprecated=True)
    except Exception as e:
        print(f"戦術データの取得中にエラーが発生しました: {e}")
        return None
        
    structured_data = []# <--- ここを修正しました

    for tactic in enterprise_tactics:
        # Enterpriseドメインに属する戦術のみを対象とする
        if 'enterprise-attack' not in getattr(tactic, 'x_mitre_domains',): # defaultを空リストに変更
            continue

        tactic_info = {
            # "id": tactic.external_references.external_id if hasattr(tactic, 'external_references') and tactic.external_references else "N/A", # external_referencesがリストであることを考慮
            "id": tactic.external_references[0]['external_id'] if hasattr(tactic, 'external_references') and tactic.external_references and isinstance(tactic.external_references, list) and 'external_id' in tactic.external_references[0] else "N/A",
            "name_eng": getattr(tactic, 'name', "N/A"),
            "description_eng": getattr(tactic, 'description', "No description available.").strip(),
            "techniques": []# ここもリストとして初期化
        }
        
        # この戦術に関連するテクニックを取得 (無効化されたものを除外)
        try:
            # techniques_in_tactic = attack_data_source.get_techniques_by_tactic(tactic.id, domain="enterprise-attack", remove_revoked_deprecated=True)
            techniques_in_tactic = attack_data_source.get_techniques_by_tactic(getattr(tactic, 'x_mitre_shortname', "N/A"), domain="enterprise-attack", remove_revoked_deprecated=True)
        except Exception as e:
            print(f"戦術 '{tactic_info['name_eng']}' のテクニック取得中にエラー: {e}")
            continue # この戦術の処理をスキップ

        for technique_obj in techniques_in_tactic:
            # サブテクニックは親テクニックの下にネストするため、ここでは親テクニックのみを処理
            if getattr(technique_obj, 'x_mitre_is_subtechnique', False):
                continue

            technique_details = {
                # "id": technique_obj.external_references.external_id if hasattr(technique_obj, 'external_references') and technique_obj.external_references else "N/A", # external_referencesがリストであることを考慮
                "id": technique_obj.external_references[0]['external_id'] if hasattr(technique_obj, 'external_references') and technique_obj.external_references and isinstance(technique_obj.external_references, list) and 'external_id' in technique_obj.external_references[0] else "N/A",
                "name_eng": getattr(technique_obj, 'name', "N/A"),
                "description_eng": getattr(technique_obj, 'description', "No description available.").strip(),
                "subtechniques": []# ここもリストとして初期化
            }
            
            # 現在の親テクニックに紐づくサブテクニックを取得 (無効化されたものを除外)
            try:
                # get_subtechniques_of_technique は STIX ID を引数に取る
                # subtechniques_of_technique = attack_data_source.get_subtechniques_of_technique(technique_obj.id, remove_revoked_deprecated=True)
                subtechniques_of_technique = attack_data_source.get_subtechniques_of_technique(technique_obj.id)
            except Exception as e:
                print(f"テクニック '{technique_details['name_eng']}' のサブテクニック取得中にエラー: {e}")
                # サブテクニックが取得できなくても、親テクニックの情報は保持する
                subtechniques_of_technique = []# 空リストで初期化

            for sub_technique_obj in subtechniques_of_technique:

                # kill_chain_phases から phase_name を抽出し、カンマ区切りで結合
                phases_str = ",".join(
                    [
                        p.get("phase_name", "")
                        for p in getattr(sub_technique_obj['object'], 'kill_chain_phases', [])
                    ]
                )

                sub_technique_details = {
                    # "id": sub_technique_obj.external_references.external_id if hasattr(sub_technique_obj, 'external_references') and sub_technique_obj.external_references else "N/A", # external_referencesがリストであることを考慮
                    "id": sub_technique_obj['object'].external_references[0]['external_id'] if hasattr(sub_technique_obj['object'], 'external_references') and sub_technique_obj['object'].external_references and isinstance(sub_technique_obj['object'].external_references, list) and 'external_id' in sub_technique_obj['object'].external_references[0] else "N/A",
                    "name_eng": getattr(sub_technique_obj['object'], 'name', "N/A"),
                    "description_eng": getattr(sub_technique_obj['object'], 'description', "No description available.").strip(),
                    # 抽出した文字列を "phases" キーに追加
                    "phases": phases_str if phases_str else "N/A"
                }
                technique_details["subtechniques"].append(sub_technique_details)
            
            tactic_info["techniques"].append(technique_details)
        
        # テクニックが存在する場合のみ戦術情報を追加
        if tactic_info["techniques"]: 
             structured_data.append(tactic_info)
    
    return structured_data

if __name__ == "__main__":
    # MITRE ATT&CK Enterprise STIXデータのURL (通常は最新版)
    attack_stix_url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
    # ローカルに保存するファイル名
    local_stix_filename = "enterprise-attack-stix-data.json" 

    # STIXデータをダウンロード
    stix_file = download_attack_data(attack_stix_url, local_stix_filename)

    stix_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data', stix_file)
    if stix_file and os.path.exists(stix_file_path):
        print(f"{stix_file} からATT&CKデータを読み込んでいます...")
        attack_ttps_with_subs = get_attack_enterprise_data_with_subtechniques(stix_file_path) 
        
        if attack_ttps_with_subs is not None: # Noneでないことを確認
            if attack_ttps_with_subs: # データが空でないことも確認
                output_filename = "mitre_attack_enterprise_data_with_subtechniques_eng_v2.json"
                output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data', output_filename)
                try:
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(attack_ttps_with_subs, f, ensure_ascii=False, indent=4)
                    print(f"サブテクニックを含むMITRE ATT&CK Enterpriseデータ (英語) を {output_filename} に保存しました。")
                except IOError as e:
                    print(f"ファイル書き込みエラー ({output_filename}): {e}")
                except Exception as e:
                    print(f"JSONデータの書き込み中に予期せぬエラーが発生しました: {e}")
            else:
                print("処理されたATT&CKデータが空です。STIXファイルの内容やフィルタリング条件を確認してください。")
        else:
            print("ATT&CKデータの処理に失敗しました。エラーメッセージを確認してください。")
    else:
        print("STIXファイルの取得またはアクセスに失敗したため、処理を続行できません。")