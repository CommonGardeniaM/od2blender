# Blenderでボーン名を確認する手順

## 簡単版（30秒で完了）

1. Blenderで `Male Lowpoly Mesh.blend` を開く

2. **Scripting** ワークスペースに切り替える
   - 上部のメニューから "Scripting" を選択

3. **Pythonコンソール**（左下のパネル）に以下を貼り付け：

```python
# 全アーマチュアのボーン名を一覧表示
for obj in bpy.data.objects:
    if obj.type == 'ARMATURE':
        print(f"\n【アーマチュア: {obj.name}】")
        print(f"ボーン数: {len(obj.data.bones)}\n")
        
        # IK/FK/Target関連のボーンを抽出
        ik_bones = [b.name for b in obj.data.bones if '_ik' in b.name.lower() and not b.name.startswith(('MCH-', 'DEF-', 'ORG-'))]
        fk_bones = [b.name for b in obj.data.bones if '_fk' in b.name.lower() and not b.name.startswith(('MCH-', 'DEF-', 'ORG-'))]
        target_bones = [b.name for b in obj.data.bones if 'target' in b.name.lower()]
        
        print("【IKコントロールボーン】:")
        for b in sorted(ik_bones):
            print(f"  - {b}")
        
        print("\n【FKコントロールボーン】:")
        for b in sorted(fk_bones):
            print(f"  - {b}")
            
        print("\n【ターゲット/ポールボーン】:")
        for b in sorted(target_bones):
            print(f"  - {b}")
        
        # 特定のパターンをチェック
        print("\n【重要ボーンチェック】:")
        patterns = ['foot_ik', 'thigh_ik', 'shin_ik', 'leg_ik', 'knee_target', 'root', 'torso']
        for p in patterns:
            matches = [b.name for b in obj.data.bones if p.lower() in b.name.lower()]
            if matches:
                print(f"  ✓ {p}: {matches}")
            else:
                print(f"  ✗ {p}: なし")
```

4. **Enter** を押して実行

5. **システムコンソール**（上部メニューの Window → Toggle System Console）で結果を確認

## または、もっと簡単に

ポーズモードにして、**アウトライナー**でボーン名を見るだけでもOKです！
特に以下が存在するか確認してほしいです：

- `foot_ik.L` / `foot_ik.R` ← おそらく存在
- `knee_target.L` / `knee_target.R` ← これが重要！
- `thigh_ik.L` / `thigh_ik.R` ← 存在する？
- `shin_ik.L` / `shin_ik.R` ← 存在する？

結果を教えてください！
