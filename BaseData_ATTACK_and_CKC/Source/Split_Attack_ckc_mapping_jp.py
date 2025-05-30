import json
import os

def split_tactics_to_files(input_file_path, base_output_directory="output_tactics"):
    """
    指定されたJSONファイルから各タクティクスデータを読み込み、
    タクティクスIDごとに個別のフォルダとJSONファイルに分割して保存する。

    Args:
        input_file_path (str): 入力となるJSONファイルのパス。
                               (例: "attack_ent_with_subtec_ckc_mapping_jp.json")
        base_output_directory (str, optional): 分割されたファイルを保存する親ディレクトリ。
                                               デフォルトは "output_tactics"。
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            tactics_data_list = json.load(f)
        print(f"'{input_file_path}' を正常に読み込みました。")
    except FileNotFoundError:
        print(f"エラー: 入力ファイル '{input_file_path}' が見つかりません。")
        return
    except json.JSONDecodeError:
        print(f"エラー: '{input_file_path}' は有効なJSONファイルではありません。")
        return
    except Exception as e:
        print(f"ファイルの読み込み中に予期せぬエラーが発生しました: {e}")
        return

    # 親出力ディレクトリが存在しない場合は作成
    if not os.path.exists(base_output_directory):
        try:
            os.makedirs(base_output_directory)
            print(f"親出力ディレクトリ '{base_output_directory}' を作成しました。")
        except OSError as e:
            print(f"エラー: 親出力ディレクトリ '{base_output_directory}' の作成に失敗しました: {e}")
            return
        
    # 各タクティクスを処理
    for tactic in tactics_data_list:
        tactic_id = tactic.get("id")
        if not tactic_id:
            print("警告: 'id' が見つからないタクティクスデータがあります。スキップします。")
            continue

        # タクティクスID名のフォルダパスを作成
        tactic_folder_path = os.path.join(base_output_directory, tactic_id)

        # フォルダが存在しない場合は作成
        if not os.path.exists(tactic_folder_path):
            try:
                os.makedirs(tactic_folder_path)
                print(f"ディレクトリ '{tactic_folder_path}' を作成しました。")
            except OSError as e:
                print(f"エラー: ディレクトリ '{tactic_folder_path}' の作成に失敗しました: {e}")
                continue # このタクティクスの処理をスキップ

        # 出力ファイル名を作成 (例: TA0001.json)
        output_file_name = f"{tactic_id}.json"
        output_file_path = os.path.join(tactic_folder_path, output_file_name)

        # タクティクスデータをJSONファイルとして保存
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(tactic, f, ensure_ascii=False, indent=4)
            print(f"タクティクス '{tactic_id}' を '{output_file_path}' に保存しました。")
        except IOError as e:
            print(f"エラー: ファイル '{output_file_path}' への書き込み中にエラーが発生しました: {e}")
        except Exception as e:
            print(f"ファイルの書き込み中に予期せぬエラーが発生しました: {e}")


if __name__ == '__main__':

    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data')

    # 入力ファイル名を指定
    # このファイルはスクリプトと同じディレクトリに存在するか、
    # もしくは適切なパスを指定する必要があります。
    input_json_file = "attack_ent_with_subtec_ckc_mapping_jp.json"
    input_json_file_path = os.path.join(output_path, input_json_file)
    
    # 出力先の親ディレクトリを指定 (オプション)
    # 省略した場合はスクリプトと同じ階層に "output_tactics" フォルダが作成されます。
    output_directory = "ATTACK_and_CKC_Data" 
    output_directory_path = os.path.join(output_path, output_directory)

    # 関数を実行してデータを分割・保存
    split_tactics_to_files(input_json_file_path, output_directory_path)
