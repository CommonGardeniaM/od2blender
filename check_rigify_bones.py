#!/usr/bin/env python3
"""
Rigifyボーン名確認スクリプト
===========================
このスクリプトをBlenderで実行して、リグのボーン名を確認します

使い方:
1. Blenderでファイルを開く
2. Scriptingワークスペースに切り替え
3. このスクリプトをテキストエディタに貼り付け
4. 「▶ 実行スクリプト」ボタンを押す
5. 結果はシステムコンソールで確認（Window → Toggle System Console）
"""

import bpy


def main():
    """メイン関数"""
    print("\n" + "="*70)
    print(" Rigify ボーン名確認")
    print("="*70)
    
    # アーマチュアを探す
    armatures = [obj for obj in bpy.data.objects if obj.type == 'ARMATURE']
    
    if not armatures:
        print("\n❌ アーマチュアが見つかりません")
        return
    
    for obj in armatures:
        print(f"\n【アーマチュア: {obj.name}】")
        print(f"  ボーン数: {len(obj.data.bones)}")
        
        bones = obj.data.bones
        names = [b.name for b in bones]
        
        # IKボーン（コントロール用）
        print("\n  [IKコントロール]:")
        ik_bones = [n for n in names if '_ik' in n.lower() 
                    and not n.startswith(('MCH-', 'DEF-', 'ORG-'))]
        if ik_bones:
            for b in sorted(ik_bones):
                print(f"    ✓ {b}")
        else:
            print("    ❌ なし")
        
        # FKボーン
        print("\n  [FKコントロール]:")
        fk_bones = [n for n in names if '_fk' in n.lower()
                    and not n.startswith(('MCH-', 'DEF-', 'ORG-'))]
        if fk_bones:
            for b in sorted(fk_bones):
                print(f"    ✓ {b}")
        else:
            print("    ❌ なし")
        
        # ターゲット/ポール
        print("\n  [ターゲット/ポール]:")
        target_bones = [n for n in names if 'target' in n.lower()]
        if target_bones:
            for b in sorted(target_bones):
                print(f"    ✓ {b}")
        else:
            print("    ❌ なし")
        
        # 重要ボーン確認
        print("\n  [重要ボーン確認]:")
        checks = [
            'foot_ik.L', 'foot_ik.R',
            'thigh_ik.L', 'thigh_ik.R',
            'shin_ik.L', 'shin_ik.R',
            'knee_target.L', 'knee_target.R',
            'root', 'torso'
        ]
        
        for check in checks:
            matches = [n for n in names if check.lower() in n.lower()]
            if matches:
                print(f"    ✓ {check}: {matches}")
            else:
                print(f"    ✗ {check}: 未検出")
    
    print("\n" + "="*70)
    print(" 確認完了！")
    print("="*70)
    print("\n結果を見るには: Window → Toggle System Console")
    print("="*70 + "\n")


# 実行
main()
