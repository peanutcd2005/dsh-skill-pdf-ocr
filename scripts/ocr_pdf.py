# -*- coding: utf-8 -*-
"""PDF/图片 OCR：pypdfium2 渲染 + RapidOCR 识别。"""
import argparse
import os
import sys

import numpy as np
import pypdfium2 as pdfium
from rapidocr_onnxruntime import RapidOCR

ENGINE = None


def engine():
    global ENGINE
    if ENGINE is None:
        ENGINE = RapidOCR()
    return ENGINE


def ocr_image(pil):
    arr = np.array(pil.convert("RGB"))
    result, _ = engine()(arr)
    if not result:
        return ""
    return "\n".join(item[1] for item in result)


def pick_pages(total, mode):
    if mode == "all":
        return list(range(total))
    if mode == "first":
        return [0]
    if mode == "last":
        return [total - 1]
    if mode == "firstlast":
        return sorted({0, total - 1})
    pages = []
    for part in mode.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            pages.extend(range(int(a) - 1, int(b)))
        else:
            pages.append(int(part) - 1)
    return [p for p in pages if 0 <= p < total]


def process(path, mode="firstlast", scale=2.5):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        doc = pdfium.PdfDocument(path)
        total = len(doc)
        pages = pick_pages(total, mode)
        chunks = []
        for i in pages:
            page = doc[i]
            pil = page.render(scale=scale).to_pil()
            txt = ocr_image(pil)
            chunks.append(f"===== 第 {i + 1} 页 =====\n{txt}")
        doc.close()
        return total, pages, "\n".join(chunks)
    else:
        from PIL import Image
        pil = Image.open(path)
        return 1, [0], ocr_image(pil)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--pages", default="firstlast")
    ap.add_argument("--scale", type=float, default=2.5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    total, pages, text = process(args.input, args.pages, args.scale)
    header = f"文件：{os.path.basename(args.input)}\n总页数：{total}\n识别页：{[p + 1 for p in pages]}\n"
    body = header + text

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(body)
        print("SAVED", args.out)
    print(body)


if __name__ == "__main__":
    main()
