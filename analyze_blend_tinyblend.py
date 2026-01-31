#!/usr/bin/env python3
"""
.blendファイル解析スクリプト - tinyblendを使用
"""

from pathlib import Path

def analyze_blend_file(filepath: str):
    """.blendファイルを解析してボーン名を抽出"""
    try:
        import tinyblend
        
        print(f"ファイルを開いています: {filepath}")
        blend = tinyblend.BlendFile(filepath)
        
        print(f"\nファイルバージョン: {blend.version}")
        print(f"ポインタサイズ: {blend.ptr_size}")
        
        # Armatureオブジェクトを探す
        armatures = []
        
        # 'Object'タイプのデータを検索
        for block in blend.blocks:
            if hasattr(block, 'get_type'):
                block_type = block.get_type()
                if block_type == 'Object':
                    # Object構造体からデータを取得
                    try:
                        data = block.get_data()
                        if hasattr(data, 'type'):
                            obj_type = data.type
                            if obj_type == 1:  # OB_ARMATURE = 1
                                armatures.append(block)
                    except:
                        pass
        
        print(f"\n見つかったアーマチュア数: {len(armatures)}")
        
        # 各アーマチュアのボーンを抽出
        for i, arm_block in enumerate(armatures):
            print(f"\n{'='*60}")
            print(f"アーマチュア {i+1}")
            print(f"{'='*60}")
            
            try:
                data = arm_block.get_data()
                if hasattr(data, 'data'):
                    arm_data = data.data
                    if arm_data:
                        # Armatureデータからボーンリストを取得
                        bones = []
                        
                        # bonebaseフィールドを探す
                        if hasattr(arm_data, 'bonebase'):
                            bonebase = arm_data.bonebase
                            if bonebase:
                                for bone in bonebase:
                                    if hasattr(bone, 'name'):
                                        bones.append(bone.name)
                        
                        # または bonebase.firstから linked listをたどる
                        elif hasattr(arm_data, 'bonebase') and hasattr(arm_data.bonebase, 'first'):
                            bone = arm_data.bonebase.first
                            while bone:
                                if hasattr(bone, 'name'):
                                    bones.append(bone.name)
                                if hasattr(bone, 'next'):
                                    bone = bone.next
                                else:
                                    break
                        
                        print(f"ボーン数: {len(bones)}")
                        
                        # ボーンをカテゴリ別に分類
                        ik_bones = [b for b in bones if '_ik' in b.lower()]
                        fk_bones = [b for b in bones if '_fk' in b.lower()]
                        target_bones = [b for b in bones if 'target' in b.lower()]
                        other_bones = [b for b in bones if b not in ik_bones + fk_bones + target_bones]
                        
                        if ik_bones:
                            print(f"\n[IKボーン] ({len(ik_bones)}個):")
                            for b in sorted(ik_bones):
                                print(f"  - {b}")
                        
                        if fk_bones:
                            print(f"\n[FKボーン] ({len(fk_bones)}個):")
                            for b in sorted(fk_bones):
                                print(f"  - {b}")
                        
                        if target_bones:
                            print(f"\n[ターゲットボーン] ({len(target_bones)}個):")
                            for b in sorted(target_bones):
                                print(f"  - {b}")
                        
                        # 重要なパターンをチェック
                        print(f"\n[重要ボーンチェック]:")
                        patterns = ['foot_ik', 'thigh_ik', 'shin_ik', 'leg_ik', 
                                   'knee_target', 'thigh_fk', 'shin_fk',
                                   'root', 'torso', 'hips', 'spine']
                        for pattern in patterns:
                            matches = [b for b in bones if pattern.lower() in b.lower()]
                            status = "✓" if matches else "✗"
                            print(f"  {status} {pattern}: {matches if matches else 'なし'}")
                        
            except Exception as e:
                print(f"  エラー: {e}")
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
