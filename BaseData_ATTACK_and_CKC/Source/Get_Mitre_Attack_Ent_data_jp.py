# coding: utf-8
# MITRE ATT&CKデータの自動取得と構造化 (DeepL APIによる日本語翻訳機能付き)
#
# 実行前の確認事項:
# 1. `mitreattack-python` ライブラリがインストールされていることを確認してください。
#    pip install mitreattack-python
# 2. `deepl` ライブラリがインストールされていることを確認してください。
#    pip install deepl
# 3. DeepL APIの認証キーを環境変数 `DEEPL_AUTH_KEY` に設定してください。
#    例: export DEEPL_AUTH_KEY="YOUR_API_KEY" (Linux/macOS)
#        set DEEPL_AUTH_KEY=YOUR_API_KEY (Windows Command Prompt)
#        $Env:DEEPL_AUTH_KEY="YOUR_API_KEY" (Windows PowerShell)
# 4. 依存ライブラリ、特に `requests-cache` が適切なバージョンであることを確認してください。
#    pip install --upgrade requests-cache mitreattack-python
#
# このスクリプトは、MITRE ATT&CK EnterpriseのSTIXデータ（JSON形式）をダウンロードし、
# 戦術、テクニック、およびサブテクニックの情報を階層的に構造化し、
# 名称と説明を日本語に翻訳してJSONファイルに出力します。

from mitreattack.stix20 import MitreAttackData
import json
import os
import requests
import time # timeモジュールをインポート

# DeepL API関連のインポート
try:
    import deepl
except ImportError:
    print("DeepLライブラリがインストールされていません。'pip install deepl' を実行してください。")
    exit()

# --- DeepL API 設定 ---
# DEEPL_AUTH_KEY = os.getenv("DEEPL_AUTH_KEY")
DEEPL_AUTH_KEY = 'YOURE DeepL AUTH KEY '
if not DEEPL_AUTH_KEY:
    print("エラー: DeepL APIの認証キーが環境変数 DEEPL_AUTH_KEY に設定されていません。")
    print("スクリプトを終了します。")
    exit()

translator = deepl.Translator(DEEPL_AUTH_KEY)

# APIレート制限を考慮した翻訳関数
def translate_texts_with_deepl(texts_to_translate, target_lang="JA"):
    """
    DeepL APIを使用してテキストのリストを翻訳します。
    APIのレート制限を考慮し、リクエスト間に短い遅延を入れます。
    texts_to_translate: 翻訳する文字列のリスト
    target_lang: ターゲット言語コード (例: "JA" for Japanese)
    戻り値: 翻訳されたテキストのリスト、またはエラー時はNoneのリスト
    """
    if not texts_to_translate:
        return []
    
    translated_texts = []
    # 一度に翻訳するテキストの数（DeepL APIの制限やパフォーマンスに応じて調整）
    # ここでは簡略化のため、1つずつ翻訳し、間にウェイトを入れる
    for i, text in enumerate(texts_to_translate):
        if not text or not isinstance(text, str) or text.strip() == "No description available." or text.strip() == "N/A":
            translated_texts.append(text) # 空または特定文字列の場合は翻訳しない
            continue
        try:
            # APIコール間に短いウェイトを入れる (例: 0.5秒)
            # 無料版APIの場合は特に重要です。適宜調整してください。
            if i > 0: # 最初のリクエスト以外
                time.sleep(0.6) # 600ミリ秒のウェイト (50万文字/月 の無料枠を考慮)

            result = translator.translate_text(text, target_lang=target_lang)
            translated_texts.append(result.text)
        except deepl.DeepLException as e:
            print(f"DeepL APIエラーが発生しました: {e}")
            print(f"エラーが発生したテキスト: {text[:100]}...") # エラー発生時のテキストを一部表示
            translated_texts.append(None) # エラー時はNoneを追加
        except Exception as e:
            print(f"翻訳中に予期せぬエラーが発生しました: {e}")
            print(f"エラーが発生したテキスト: {text[:100]}...")
            translated_texts.append(None)
    return translated_texts

def download_attack_data(url, filename="enterprise-attack.json"):
    """
    指定されたURLからATT&CK STIXデータをダウンロードし、ローカルに保存します。
    ファイルが既に存在する場合はダウンロードをスキップします。
    """
    file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data', filename)
    if not os.path.exists(file_path):
        print(f"ATT&CK STIXデータを {url} からダウンロードしています...")
        try:
            response = requests.get(url, stream=True, timeout=60) # タイムアウトを設定
            response.raise_for_status()  # HTTPエラーがあれば例外を発生させる
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"正常に {filename} として保存しました。")
        except requests.exceptions.Timeout:
            print(f"ATT&CKデータのダウンロード中にタイムアウトが発生しました: {url}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"ATT&CKデータのダウンロード中にエラーが発生しました: {e}")
            return None
        except Exception as e:
            print(f"予期せぬエラーが発生しました (ダウンロード処理中): {e}")
            return None
    else:
        print(f"{filename} は既に存在します。ダウンロードをスキップします。")
    return filename

def get_attack_enterprise_data_with_subtechniques_and_translate(stix_file_path):
    """
    ローカルのSTIXファイルからMITRE ATT&CK Enterpriseの戦術、テクニック、
    およびそれらに紐づくサブテクニックの情報を取得・構造化し、日本語に翻訳します。
    無効化された(revoked/deprecated)オブジェクトは除外します。
    """
    try:
        attack_data_source = MitreAttackData(stix_file_path)
    except Exception as e:
        print(f"MitreAttackDataオブジェクトの初期化中にエラーが発生しました: {e}")
        print("STIXファイルが破損しているか、形式が正しくない可能性があります。")
        return None
    
    try:
        enterprise_tactics = attack_data_source.get_tactics(remove_revoked_deprecated=True)
    except Exception as e:
        print(f"戦術データの取得中にエラーが発生しました: {e}")
        return None
        
    structured_data_final = []

    for tactic in enterprise_tactics:
        if 'enterprise-attack' not in getattr(tactic, 'x_mitre_domains', []):
            continue

        tactic_name_eng = getattr(tactic, 'name', "N/A")
        tactic_desc_eng = getattr(tactic, 'description', "No description available.").strip()
        
        # 戦術名と説明を翻訳
        print(f"戦術を翻訳中: {tactic_name_eng}")
        translated_tactic_texts = translate_texts_with_deepl([tactic_name_eng, tactic_desc_eng])
        tactic_name_jp = translated_tactic_texts[0] if translated_tactic_texts and len(translated_tactic_texts) > 0 else tactic_name_eng
        tactic_desc_jp = translated_tactic_texts[1] if translated_tactic_texts and len(translated_tactic_texts) > 1 else tactic_desc_eng


        tactic_info = {
            "id": tactic.external_references[0]['external_id'] if hasattr(tactic, 'external_references') and tactic.external_references and isinstance(tactic.external_references, list) and 'external_id' in tactic.external_references[0] else "N/A",
            "name_eng": tactic_name_eng,
            "name_jp": tactic_name_jp,
            "description_eng": tactic_desc_eng,
            "description_jp": tactic_desc_jp,
            "techniques": []
        }
        
        try:
            techniques_in_tactic = attack_data_source.get_techniques_by_tactic(getattr(tactic, 'x_mitre_shortname', "N/A"), domain="enterprise-attack", remove_revoked_deprecated=True)
        except Exception as e:
            print(f"戦術 '{tactic_info['name_eng']}' のテクニック取得中にエラー: {e}")
            continue

        for technique_obj in techniques_in_tactic:
            if getattr(technique_obj, 'x_mitre_is_subtechnique', False):
                continue

            tech_name_eng = getattr(technique_obj, 'name', "N/A")
            tech_desc_eng = getattr(technique_obj, 'description', "No description available.").strip()

            # テクニック名と説明を翻訳
            print(f"  テクニックを翻訳中: {tech_name_eng}")
            translated_tech_texts = translate_texts_with_deepl([tech_name_eng, tech_desc_eng])
            tech_name_jp = translated_tech_texts[0] if translated_tech_texts and len(translated_tech_texts) > 0 else tech_name_eng
            tech_desc_jp = translated_tech_texts[1] if translated_tech_texts and len(translated_tech_texts) > 1 else tech_desc_eng

            technique_details = {
                "id": technique_obj.external_references[0]['external_id'] if hasattr(technique_obj, 'external_references') and technique_obj.external_references and isinstance(technique_obj.external_references, list) and 'external_id' in technique_obj.external_references[0] else "N/A",
                "name_eng": tech_name_eng,
                "name_jp": tech_name_jp,
                "description_eng": tech_desc_eng,
                "description_jp": tech_desc_jp,
                "subtechniques": []
            }
            
            try:
                # サブテクニックの取得 (remove_revoked_deprecated=True を追加推奨)
                # subtechniques_of_technique = attack_data_source.get_subtechniques_of_technique(technique_obj.id, remove_revoked_deprecated=True)
                subtechniques_of_technique = attack_data_source.get_subtechniques_of_technique(technique_obj.id)
            except Exception as e:
                print(f"テクニック '{technique_details['name_eng']}' のサブテクニック取得中にエラー: {e}")
                subtechniques_of_technique = []

            for sub_technique_item in subtechniques_of_technique: # 'sub_technique_obj' から 'sub_technique_item' に変更 (stix2 v3.0.0以降の変更対応)
                sub_technique_obj = sub_technique_item['object'] # サブテクニックオブジェクト本体

                sub_tech_name_eng = getattr(sub_technique_obj, 'name', "N/A")
                sub_tech_desc_eng = getattr(sub_technique_obj, 'description', "No description available.").strip()

                # サブテクニック名と説明を翻訳
                print(f"    サブテクニックを翻訳中: {sub_tech_name_eng}")
                translated_sub_tech_texts = translate_texts_with_deepl([sub_tech_name_eng, sub_tech_desc_eng])
                sub_tech_name_jp = translated_sub_tech_texts[0] if translated_sub_tech_texts and len(translated_sub_tech_texts) > 0 else sub_tech_name_eng
                sub_tech_desc_jp = translated_sub_tech_texts[1] if translated_sub_tech_texts and len(translated_sub_tech_texts) > 1 else sub_tech_desc_eng

                sub_technique_details = {
                    "id": sub_technique_obj.external_references[0]['external_id'] if hasattr(sub_technique_obj, 'external_references') and sub_technique_obj.external_references and isinstance(sub_technique_obj.external_references, list) and 'external_id' in sub_technique_obj.external_references[0] else "N/A",
                    "name_eng": sub_tech_name_eng,
                    "name_jp": sub_tech_name_jp,
                    "description_eng": sub_tech_desc_eng,
                    "description_jp": sub_tech_desc_jp,
                }
                technique_details["subtechniques"].append(sub_technique_details)
            
            tactic_info["techniques"].append(technique_details)
        
        if tactic_info["techniques"]: 
             structured_data_final.append(tactic_info)
    
    return structured_data_final

if __name__ == "__main__":
    attack_stix_url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
    local_stix_filename = "enterprise-attack-stix-data.json" 

    stix_file = download_attack_data(attack_stix_url, local_stix_filename)
    stix_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data', stix_file)
    if stix_file and os.path.exists(stix_file_path):
        print(f"\n{stix_file} からATT&CKデータを読み込んでいます...")
        # 関数名を変更したものに差し替え
        attack_ttps_translated = get_attack_enterprise_data_with_subtechniques_and_translate(stix_file_path) 

        if attack_ttps_translated is not None:
            if attack_ttps_translated:
                # 出力ファイル名を変更
                output_filename = "mitre_attack_enterprise_data_with_subtechniques_jp.json"
                output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data', output_filename)
                try:
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(attack_ttps_translated, f, ensure_ascii=False, indent=4)
                    print(f"\nサブテクニックと日本語翻訳を含むMITRE ATT&CK Enterpriseデータを {output_filename} に保存しました。")
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

