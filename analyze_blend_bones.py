#!/usr/bin/env python3
"""
解析スクリプト: .blendファイルからボーン名を抽出
このスクリプトをBlenderのPythonコンソールで実行してください
"""

import bpy

# ファイルを開く
filepath = r"C:\Users\commo\Desktop\dev\python\od2blender\Male Lowpoly Mesh.blend"
bpy.ops.wm.open_mainfile(filepath=filepath)

print("\n" + "="*60)
print("ファイル内のアーマチュアオブジェクトとボーン一覧")
print("="*60)

# 全アーマチュアを検索
for obj in bpy.data.objects:
    if obj.type == 'ARMATURE':
        print(f"\n【アーマチュア: {obj.name}】")
        print(f"  ボーン数: {len(obj.data.bones)}")
        print("\n  ボーン一覧:")
        
        # ボーンをカテゴリ別に整理
        ik_bones = []
        fk_bones = []
        target_bones = []
        ctrl_bones = []
        other_bones = []
        
        for bone in obj.data.bones:
            name = bone.name
            if '_ik' in name.lower() and not name.startswith('MCH-') and not name.startswith('DEF-') and not name.startswith('ORG-'):
                ik_bones.append(name)
            elif '_fk' in name.lower() and not name.startswith('MCH-') and not name.startswith('DEF-') and not name.startswith('ORG-'):
                fk_bones.append(name)
            elif 'target' in name.lower():
                target_bones.append(name)
            elif name.startswith('CTRL-') or name.startswith('ctrl_'):
                ctrl_bones.append(name)
            else:
                other_bones.append(name)
        
        # 表示
        if ik_bones:
            print("\n  [IKコントロールボーン]:")
            for b in sorted(ik_bones):
                print(f"    - {b}")
        
        if fk_bones:
            print("\n  [FKコントロールボーン]:")
            for b in sorted(fk_bones):
                print(f"    - {b}")
        
        if target_bones:
            print("\n  [ターゲット/ポールボーン]:")
            for b in sorted(target_bones):
                print(f"    - {b}")
        
        if ctrl_bones:
            print("\n  [CTRLプレフィックスボーン]:")
            for b in sorted(ctrl_bones):
                print(f"    - {b}")
        
        # 重要なパターンをチェック
        print("\n  [重要ボーンチェック]:")
        important = [
            'foot_ik', 'thigh_ik', 'shin_ik', 'leg_ik', 
            'knee_target', 'thigh_fk', 'shin_fk',
            'root', 'torso', 'hips', 'spine'
        ]
        
        all_names = [b.name for b in obj.data.bones]
        for pattern in important:
            matches = [n for n in all_names if pattern.lower() in n.lower()]
            if matches:
                print(f"    ✓ {pattern}: {matches}")
            else:
                print(f"    ✗ {pattern}: なし")

print("\n" + "="*60)
print("解析完了")
print("="*60)
