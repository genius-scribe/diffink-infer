import numpy as np
import cv2
import torch


def plot_line_cv2(data, save_path, canvas_height=256, padding=20, line_thickness=2, max_dist=200):
    """Render a pen-stroke sequence to a PNG using OpenCV.

    Args:
        data: [T, 5] or [5, T] array/tensor — columns are (x, y, is_next, is_new_stroke, is_new_char).
        save_path: output file path.
        canvas_height: fixed height of the output image in pixels.
        padding: border padding in pixels.
        line_thickness: line width.
        max_dist: maximum pixel jump before a segment is skipped (artifact filter).
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    if data.shape[0] == 5:
        data = data.T  # → [T, 5]

    x = data[:, 0]
    y = data[:, 1]
    is_next = data[:, 2].astype(np.int32)
    is_new_stroke = data[:, 3].astype(np.int32)
    is_new_char = data[:, 4].astype(np.int32)

    # Find character-end markers (xy001 pattern) and trim to second-to-last
    char_end_indices = [
        i for i in range(len(is_next))
        if is_next[i] == 0 and is_new_stroke[i] == 0 and is_new_char[i] == 1
    ]
    if len(char_end_indices) >= 2:
        end = char_end_indices[-2] + 1
        x, y, is_next = x[:end], y[:end], is_next[:end]

    if len(x) == 0:
        return

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    dy = y_max - y_min + 1e-6
    scale = (canvas_height - 2 * padding) / dy

    x_scaled = (x - x_min) * scale
    y_scaled = (y_max - y) * scale  # flip y

    scaled_width = max(1, int(np.ceil((x_max - x_min) * scale)) + 2 * padding)
    x_px = (x_scaled + padding).astype(np.int32)
    y_px = (y_scaled + padding).astype(np.int32)

    canvas = np.ones((canvas_height, scaled_width, 3), dtype=np.uint8) * 255

    for i in range(len(x_px) - 1):
        if is_next[i] == 1:
            pt1 = np.array([x_px[i], y_px[i]])
            pt2 = np.array([x_px[i + 1], y_px[i + 1]])
            if np.linalg.norm(pt1 - pt2) < max_dist:
                cv2.line(canvas, tuple(pt1), tuple(pt2), (0, 0, 0), thickness=line_thickness)

    cv2.imwrite(save_path, canvas)
