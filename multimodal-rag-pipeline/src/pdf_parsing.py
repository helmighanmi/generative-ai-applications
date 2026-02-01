import json
from typing import Any, Dict, List, Optional, Tuple

import pymupdf  # or: import fitz


def export_pdf_layout_to_json(
    doc_path: str,
    out_json: str = "layout.json",
    out_pdf_overlay: Optional[str] = None,  # e.g. "debug_overlay.pdf" or None
    draw_text_spans: bool = False,
    draw_images: bool = False,
    min_img_px: int = 60,  # filter tiny artifact images by pixel size
    sort: bool = True,
) -> Dict[str, Any]:
    """
    Export PDF layout to JSON for ML/layout pipelines.

    Output JSON schema (high-level):
    {
      "doc_path": "...",
      "page_count": N,
      "pages": [
        {
          "page_index": 0,
          "page_number": 1,
          "width": ...,
          "height": ...,
          "spans": [
            {
              "block_no": ...,
              "line_no": ...,
              "span_no": ...,
              "text": "...",
              "bbox": [x0,y0,x1,y1],
              "font": "...",
              "size": ...,
              "flags": ...,
              "color": ...,
              "origin": [x,y]   # if present
            }, ...
          ],
          "images": [
            {
              "xref": 145,
              "name": "Image145",
              "pixel_size": [909, 682],
              "bpc": ...,
              "colorspace": "...",
              "smask": 0/other,
              "rects": [
                [x0,y0,x1,y1], ...
              ]
            }, ...
          ]
        }, ...
      ]
    }

    Returns the dict (also written to out_json).
    """
    doc = pymupdf.open(doc_path)

    out: Dict[str, Any] = {
        "doc_path": doc_path,
        "page_count": doc.page_count,
        "pages": [],
    }

    for page_index, page in enumerate(doc):
        page_rect = page.rect
        page_info: Dict[str, Any] = {
            "page_index": page_index,
            "page_number": page_index + 1,
            "width": float(page_rect.width),
            "height": float(page_rect.height),
            "spans": [],
            "images": [],
        }

        # ---------- TEXT SPANS ----------
        pgdict = page.get_text("dict", sort=sort)

        block_no_counter = 0
        for block in pgdict.get("blocks", []):
            if block.get("type") != 0:
                continue

            block_no = block.get("number", block_no_counter)
            block_no_counter += 1

            for line_no, line in enumerate(block.get("lines", [])):
                for span_no, span in enumerate(line.get("spans", [])):
                    text = span.get("text", "")
                    if not text.strip():
                        continue

                    bbox = span.get("bbox")
                    span_item: Dict[str, Any] = {
                        "block_no": int(block_no) if isinstance(block_no, (int, float)) else block_no,
                        "line_no": line_no,
                        "span_no": span_no,
                        "text": text,
                        "bbox": [float(b) for b in bbox] if bbox else None,
                        "font": span.get("font"),
                        "size": float(span.get("size")) if span.get("size") is not None else None,
                        "flags": span.get("flags"),
                        "color": span.get("color"),
                        "origin": [float(x) for x in span.get("origin")] if "origin" in span else None,
                    }
                    page_info["spans"].append(span_item)

                    # Optional overlay draw: span bbox
                    if out_pdf_overlay and draw_text_spans and bbox:
                        page.draw_rect(pymupdf.Rect(bbox), width=0.6, color=(0, 1, 0))

        # ---------- REAL IMAGES (xref -> rects) ----------
        for img in page.get_images(full=True):
            xref, smask, w, h, bpc, cs, alt_cs, name, filt, refcount = img

            # filter out tiny artifacts by pixel size
            if w < min_img_px or h < min_img_px:
                continue

            rects = page.get_image_rects(xref)
            img_item: Dict[str, Any] = {
                "xref": int(xref),
                "name": name,
                "pixel_size": [int(w), int(h)],
                "bpc": int(bpc) if bpc is not None else None,
                "colorspace": cs,
                "smask": int(smask) if smask is not None else None,
                "rects": [[float(r.x0), float(r.y0), float(r.x1), float(r.y1)] for r in rects],
                "filter": filt,
                "refcount": int(refcount) if refcount is not None else None,
            }
            page_info["images"].append(img_item)

            # Optional overlay draw: image rects
            if out_pdf_overlay and draw_images:
                for r in rects:
                    page.draw_rect(r, width=1.5, color=(1, 0, 0))
                    page.insert_text(
                        pymupdf.Point(r.x0, max(0, r.y0 - 2)),
                        f"xref {xref} {w}x{h}",
                        fontsize=7,
                    )

        out["pages"].append(page_info)

    # Write JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Save overlay PDF if requested
    if out_pdf_overlay:
        doc.save(out_pdf_overlay)

    doc.close()
    return out


# Example usage:
# data = export_pdf_layout_to_json(
#     doc_path,
#     out_json="layout.json",
#     out_pdf_overlay="debug_overlay.pdf",
#     draw_text_spans=False,   # set True if you want green boxes
#     draw_images=True,        # set True if you want red boxes for real images
#     min_img_px=50,
# )
# print("Saved JSON + optional overlay PDF.")
