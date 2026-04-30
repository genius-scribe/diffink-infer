"""
gfont → DiffInk 五元组格式转换工具。

将 gfont（JSON 预解析格式）中的字符笔画数据转换为 DiffInk 模型所需的
[T, 5] 五元组序列格式，可直接作为 style reference 输入 DiffInk 推理。

用法:
    python gfont_to_diffink.py \
        --json 严寒_kvenjoy_严寒.json \
        --chars "天地黄宇宙洪荒严寒永" \
        --num_style 5 \
        --output test_input.json

    # 也支持直接输入 .gfont 文件
    python gfont_to_diffink.py \
        --gfont 严寒_kvenjoy_严寒.gfont \
        --chars "天地黄宇宙洪荒严寒永"

输出: DiffInk handler 可直接使用的 JSON 文件。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import tempfile
import zipfile

import numpy as np


# ---------------------------------------------------------------------------
# gfont 解析
# ---------------------------------------------------------------------------

def parse_gfont_file(gfont_path: str) -> dict[str, list]:
    """
    解压 .gfont ZIP 文件，解析所有字符的笔画数据。

    Returns:
        dict: {character: [stroke1, stroke2, ...]}
              每个 stroke 是 np.array([N, 2])
    """
    tmpdir = tempfile.mkdtemp()
    with zipfile.ZipFile(gfont_path, "r") as z:
        z.extractall(tmpdir)

    result = {}
    for name in os.listdir(tmpdir):
        if not name.isdigit():
            continue
        fpath = os.path.join(tmpdir, name)
        strokes = _parse_char_binary(fpath)
        if strokes:
            ch = chr(int(name))
            result[ch] = strokes
    return result


def _parse_char_binary(file_path: str) -> list[np.ndarray] | None:
    """解析 gfont 单字符二进制文件。"""
    with open(file_path, "rb") as f:
        data = f.read()
    if len(data) < 28:
        return None

    n_points = int.from_bytes(data[2:6], byteorder="big") // 2
    expected_size = n_points * 9 + 10
    if len(data) != expected_size:
        return None

    body = data[6:]
    points = np.zeros((n_points, 2), dtype=np.float32)
    for i in range(n_points):
        points[i, 0] = struct.unpack(">f", body[i * 8: i * 8 + 4])[0]
        points[i, 1] = struct.unpack(">f", body[i * 8 + 4: i * 8 + 8])[0]

    lc_offset = n_points * 8
    pen_down_flags = body[lc_offset + 4:]

    strokes, current = [], []
    for i in range(n_points):
        current.append(points[i])
        if i == n_points - 1 or pen_down_flags[i + 1] == 0:
            if len(current) > 1:
                strokes.append(np.array(current, dtype=np.float32))
            current = []
    return strokes


def load_gfont_json(json_path: str) -> dict[str, list]:
    """
    加载 gfont 预解析 JSON 文件。

    JSON 格式:
        {"char_count": N, "characters": [{"character": "天", "strokes": [[[x,y],...],...]}, ...]}

    Returns:
        dict: {character: [np.array([N, 2]), ...]}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = {}
    for c in data["characters"]:
        strokes = [np.array(s, dtype=np.float32) for s in c["strokes"] if len(s) >= 2]
        if strokes:
            result[c["character"]] = strokes
    return result


# ---------------------------------------------------------------------------
# 核心转换: gfont strokes → DiffInk 五元组
# ---------------------------------------------------------------------------

def gfont_to_diffink(
    chars: str,
    font_data: dict[str, list],
    char_gap: float = 30.0,
    target_y_range: float = 5.0,
) -> tuple[np.ndarray, list[int]]:
    """
    将 gfont 笔画数据转换为 DiffInk 五元组格式。

    转换步骤:
        1. 遍历每个字符，将笔画坐标转为五元组 flags
        2. y 轴翻转（gfont 屏幕坐标系 → DiffInk 数学坐标系）
        3. 各字水平拼接，字间留 char_gap 间距
        4. 记录 char_points_idx[i] = 第 i+1 字第一点的索引（末字 = 总点数，past-the-end）
        5. 全局等比缩放到目标 y 范围

    char_points_idx 语义（与 CASIA val.h5 实测格式对齐）:
        - cpi[i] 指向第 i+1 字第一个实际笔画点的索引（该点 flag = (1, 0, 0)）。
        - cpi[-1] == total_points（past-the-end sentinel）。
        - mask.py 把它当作右开区间切分点: [0, cpi[num_prefix-1]) = 前 num_prefix 字。

    Args:
        chars: 要转换的字符串，如 "天地黄宇宙"
        font_data: {char: [strokes]} 字典
        char_gap: 原始坐标下的字间间距
        target_y_range: 缩放后的目标 y 轴范围

    Returns:
        point_seq: np.array [T, 5] 五元组序列
        char_points_idx: list[int] 每个字符的边界索引

    Raises:
        KeyError: 字符不在字体数据中
    """
    # --- Step 1: 逐字生成五元组（未拼接） ---
    char_sequences = []
    for ch in chars:
        if ch not in font_data:
            raise KeyError(f"字符 '{ch}' 不在字体数据中")
        strokes = font_data[ch]
        all_pts = np.vstack(strokes)
        x_min = all_pts[:, 0].min()

        points = []
        for si, stroke in enumerate(strokes):
            for pi in range(len(stroke)):
                x = stroke[pi, 0] - x_min        # x 归零（相对于字符左边缘）
                y = -stroke[pi, 1]                # Step 2: y 轴翻转
                is_last_pt = (pi == len(stroke) - 1)
                is_last_stroke = (si == len(strokes) - 1)

                if is_last_pt and is_last_stroke:
                    # 字符结束标记: is_next=0, is_new_stroke=0, is_new_char=1
                    points.append([x, y, 0, 0, 1])
                elif is_last_pt:
                    # 笔画结束 + 抬笔: is_next=0, is_new_stroke=1, is_new_char=0
                    points.append([x, y, 0, 1, 0])
                else:
                    # 正常画线: is_next=1, is_new_stroke=0, is_new_char=0
                    points.append([x, y, 1, 0, 0])

        char_sequences.append(np.array(points, dtype=np.float32))

    # --- Step 3 & 4: 水平拼接，记录边界索引 ---
    all_points = []
    char_points_idx = []
    x_offset = 0.0

    for ci, seq in enumerate(char_sequences):
        char_w = seq[:, 0].max() - seq[:, 0].min()

        for pt in seq:
            all_points.append([pt[0] + x_offset, pt[1], pt[2], pt[3], pt[4]])

        # cpi[i] = 下一字第一点将要落位的索引（末字时 = 总点数，past-the-end）
        char_points_idx.append(len(all_points))
        x_offset += char_w + char_gap

    point_seq = np.array(all_points, dtype=np.float32)

    # --- Step 5: 全局等比缩放 ---
    y_min, y_max = point_seq[:, 1].min(), point_seq[:, 1].max()
    y_range = y_max - y_min + 1e-6
    scale = target_y_range / y_range
    point_seq[:, 0] *= scale
    point_seq[:, 1] *= scale
    # y 居中到 0
    y_mid = (point_seq[:, 1].max() + point_seq[:, 1].min()) / 2
    point_seq[:, 1] -= y_mid

    return point_seq, char_points_idx


# ---------------------------------------------------------------------------
# 生成 DiffInk 输入 JSON
# ---------------------------------------------------------------------------

def make_diffink_input(
    point_seq: np.ndarray,
    char_points_idx: list[int],
    reference_text: str,
    target_text: str,
    num_style_chars: int,
    sampling_timesteps: int = 20,
    cfg_scale: float = 1.0,
    temperature: float = 0.1,
) -> dict:
    """
    生成 DiffInk handler 可直接使用的输入 payload。

    Args:
        point_seq: [T, 5] 五元组序列
        char_points_idx: 字符边界索引列表
        reference_text: 参考文本（所有字符）
        target_text: 要生成的目标文本
        num_style_chars: 用作风格参考的字符数
        sampling_timesteps: DDIM 采样步数
        cfg_scale: classifier-free guidance scale
        temperature: GMM 采样温度

    Returns:
        dict: 可直接序列化为 JSON 的 payload
    """
    strokes_b64 = base64.b64encode(point_seq.tobytes()).decode("utf-8")
    return {
        "input": {
            "reference_text": reference_text,
            "target_text": target_text,
            "style_strokes": strokes_b64,
            "char_points_idx": char_points_idx,
            "num_style_chars": num_style_chars,
            "sampling_timesteps": sampling_timesteps,
            "cfg_scale": cfg_scale,
            "temperature": temperature,
            "greedy": True,
            "output_image": True,
        }
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="gfont → DiffInk 五元组格式转换工具"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--json", help="gfont 预解析 JSON 文件路径")
    group.add_argument("--gfont", help=".gfont 文件路径")

    parser.add_argument("--chars", required=True,
                        help="要转换的字符（如 '天地黄宇宙洪荒严寒永'）")
    parser.add_argument("--num_style", type=int, default=5,
                        help="风格参考字符数（默认 5）")
    parser.add_argument("--output", default="test_input.json",
                        help="输出 JSON 文件路径")
    parser.add_argument("--char_gap", type=float, default=30.0,
                        help="原始坐标下的字间间距（默认 30）")
    parser.add_argument("--target_y", type=float, default=5.0,
                        help="缩放后的目标 y 范围（默认 5.0）")
    args = parser.parse_args()

    # 加载字体数据
    if args.json:
        print(f"加载 JSON: {args.json}")
        font_data = load_gfont_json(args.json)
    else:
        print(f"解压 gfont: {args.gfont}")
        font_data = parse_gfont_file(args.gfont)
    print(f"字体包含 {len(font_data)} 个字符")

    # 检查字符可用性
    chars = args.chars
    missing = [ch for ch in chars if ch not in font_data]
    if missing:
        print(f"错误: 以下字符不在字体中: {''.join(missing)}")
        return

    # 转换
    print(f"\n转换 {len(chars)} 个字符: {chars}")
    print(f"  风格参考: {chars[:args.num_style]}")
    print(f"  生成目标: {chars[args.num_style:]}")

    point_seq, char_points_idx = gfont_to_diffink(
        chars, font_data,
        char_gap=args.char_gap,
        target_y_range=args.target_y,
    )

    # 统计
    print(f"\n转换结果:")
    print(f"  总点数: {point_seq.shape[0]}")
    print(f"  字符边界: {char_points_idx}")
    print(f"  坐标范围: x=[{point_seq[:,0].min():.2f}, {point_seq[:,0].max():.2f}]"
          f"  y=[{point_seq[:,1].min():.2f}, {point_seq[:,1].max():.2f}]")

    # 格式验证
    char_end_markers = [i for i in range(len(point_seq))
                        if point_seq[i, 2] == 0 and point_seq[i, 3] == 0 and point_seq[i, 4] == 1]
    assert len(char_end_markers) == len(chars), \
        f"char_end 标记数 ({len(char_end_markers)}) != 字符数 ({len(chars)})"
    assert all(char_points_idx[i] == char_end_markers[i] + 1 for i in range(len(chars))), \
        "char_points_idx 应该 = char_end marker + 1"
    print(f"  格式验证: 通过 ✓")

    # 生成输出
    target_text = chars[args.num_style:]
    payload = make_diffink_input(
        point_seq, char_points_idx,
        reference_text=chars,
        target_text=target_text,
        num_style_chars=args.num_style,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n输出: {args.output}")


if __name__ == "__main__":
    main()
