# DiffInk 推理测试数据

最后更新：2026-04-28

## 现存测试数据（2 个目录）

| 目录 | 跑的脚本 | 输入 | 关键参数 | 用途 / 对应文档 | 跑日期 |
|------|---------|------|---------|----------------|-------|
| `test_outputs_sentence_v3` | `test_local.py` | `test_input_sentence.json` | ref `天地黄宇宙洪荒严寒永` (10 字 gfont, T=402) + target `春风又绿江南岸明月何时照我还` (14 字), prefix_ratio=0.42, T=964 | **8.3** sentence: gfont style + 长 target text（mask 修复后） | 2026-04-28 |
| `test_outputs_inksight_6s10t` | `test_local.py` | `test_input_inksight_6s10t.json` | ref `人生得意须尽` (6 字 InkSight 还原, T=450) + target `春夏秋冬东南西北上下` (10 字随机), prefix_ratio=0.38, T=1200 | **8.2** InkSight 还原真实手写: 6 ref + 10 random target | 2026-04-28 |

跑在 `pro-7772d41fb379` ("diffink", west-D, vGPU-32GB)，推理 1.2-1.4s。

## 配套测试输入

| 文件 | 用途 |
|------|------|
| `test_input_sentence.json` | 8.3 sentence 输入（gfont style + 14 字 target） |
| `test_input_inksight_6s10t.json` | 8.2 InkSight 输入（6 ref + 10 random target，2026-04-28 由 `test_outputs_image_real/test_input_image.json` 截前 6 字 + 替换 target_text 生成） |
| `test_input_short.json` | 短样本（5 字 target），用于 quick verify |
| `test_input_long.json` | 长样本 |
| `test_input_gfont.json` | gfont 输入 |
| `test_input_selfrecon.json` | selfrecon 输入 |

## mask 修复

handler.py / test_local.py 在 2026-04-27 13:33 修复 `mask = torch.ones(1, T)` → paper 一致逻辑：

```python
mask = torch.zeros(1, T, dtype=torch.bool, device=device)
mask[0, :orig_len] = True  # pad 位置标 False
```

当前两个 outputs 都是修复后跑的。修复前 / 早期实验全部已清理。

## OSS 上传准备

```bash
cd ~/suchuan/diffink-infer/test
tar czf diffink_test_outputs_2026-04-28.tar.gz test_outputs_* test_input_*.json README.md
```

预计压缩后 <100K（总未压缩 ~120K）。

OSS 上传需要 ossutil + 凭证（见父目录 README 或 `~/suchuan/.secrets/`）。
