#!/usr/bin/env python3
"""
.blendファイル解析スクリプト - tinyblendを使用
"""

from pathlib import Path

def analyze_blend_file(filepath: str):
    """.blendファイルを解析してボーン名を抽出"""
    try:
        from tinyblend import BlenderFile
        
        print(f"ファイルを開いています: {filepath}")
        blend = BlenderFile(filepath)
        
        print(f"\nファイルヘッダー: {blend.header}")
        
        # 利用可能な構造体を一覧
        structs = blend.list_structures()
        print(f"\n利用可能な構造体数: {len(structs)}")
        
        # Object構造体を探す
        if 'Object' in structs:
            print("\n【Object構造体のツリー】")
            print(blend.tree('Object'))
        
        # Object一覧を取得
        try:
            objects = blend.list('Object')
            print(f"\nオブジェクト数: {len(objects)}")
            
            armatures = []
            for obj in objects:
                try:
                    # Objectのtypeフィールドをチェック (1 = OB_ARMATURE)
                    if hasattr(obj, 'type') and obj.type == 1:
                        armatures.append(obj)
                except:
                    pass
            
            print(f"アーマチュアオブジェクト数: {len(armatures)}")
            
            # 各アーマチュアの詳細を表示
            for i, arm_obj in enumerate(armatures):
                print(f"\n{'='*60}")
                print(f"アーマチュア {i+1}")
                print(f"{'='*60}")
                
                try:
                    if hasattr(arm_obj, 'id') and hasattr(arm_obj.id, 'name'):
                        print(f"名前: {arm_obj.id.name}")
                    
                    # Armatureデータを取得
                    if hasattr(arm_obj, 'data') and arm_obj.data:
                        arm_data = arm_obj.data
                        print(f"データポインタ: {arm_data}")
                        
                        # Armature構造体のツリーを表示
                        if 'Armature' in structs:
                            print("\n【Armature構造体のツリー】")
                            print(blend.tree('Armature'))
                        
                        # bonebaseを探す
                        if hasattr(arm_data, 'bonebase'):
                            print(f"\nbonebase: {arm_data.bonebase}")
                except Exception as e:
                    print(f"  詳細取得エラー: {e}")
                    
        except Exception as e:
            print(f"オブジェクトリスト取得エラー: {e}")
            import traceback
            traceback.print_exc()
        
        blend.close()
        
    except ImportError:
        print("tinyblendがインストールされていません")
        print("pip install tinyblend")
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    filepath = r"C:\Users\commo\Desktop\dev\python\od2blender\Male Lowpoly Mesh.blend"
    analyze_blend_file(filepath)
