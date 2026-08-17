#!/usr/bin/env python3
"""Render ANSI-colored terminal text as a self-contained SVG card.

Regenerates the README image directly from report output:

    CADENCE_BUDGET_MS=0.071 CADENCE_COLOR=1 ./build/examples/cadence_example_loop > run.txt
    python3 docs/render_report_svg.py run.txt docs/report.svg "cadence report"

Standard library only. Runs are emitted as adjacent tspans with no explicit x,
so columns line up in whatever monospace font the reader has; textLength then
pins each line to the width the card was sized against.
"""
import html
import re
import sys

PALETTE = {
    "fg":     "#c9d1de",
    "1":      "#ffffff",   # bold
    "2":      "#6b7382",   # dim
    "31":     "#ff6b68",
    "32":     "#7fd88f",
    "33":     "#e5c07b",
    "36":     "#56b6c2",
}

FONT_SIZE = 13.5
ADVANCE = FONT_SIZE * 0.6
LINE_H = FONT_SIZE * 1.45
PAD_X = 22.0
PAD_TOP = 46.0
PAD_BOTTOM = 20.0

SGR = re.compile(r"\x1b\[([0-9;]*)m")


def runs(line):
    """Split one line into (text, css_fill) runs."""
    out, pos, fill, weight = [], 0, PALETTE["fg"], "normal"
    for m in SGR.finditer(line):
        if m.start() > pos:
            out.append((line[pos:m.start()], fill, weight))
        codes = [c for c in m.group(1).split(";") if c] or ["0"]
        for code in codes:
            if code == "0":
                fill, weight = PALETTE["fg"], "normal"
            elif code == "1":
                fill, weight = PALETTE["1"], "bold"
            elif code in PALETTE:
                fill = PALETTE[code]
        pos = m.end()
    if pos < len(line):
        out.append((line[pos:], fill, weight))
    return out


def main(src, dst, title):
    raw = [l.rstrip("\n") for l in open(src, encoding="utf-8")]
    while raw and not raw[-1].strip():
        raw.pop()
    plain = [SGR.sub("", l).rstrip() for l in raw]
    cols = max(len(l) for l in plain)
    width = cols * ADVANCE + PAD_X * 2
    height = len(raw) * LINE_H + PAD_TOP + PAD_BOTTOM

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" font-family="ui-monospace, SFMono-Regular, '
        f'&quot;SF Mono&quot;, Menlo, Consolas, &quot;DejaVu Sans Mono&quot;, monospace" '
        f'font-size="{FONT_SIZE}">',
        f'<rect width="{width:.0f}" height="{height:.0f}" rx="10" fill="#14161c"/>',
        f'<rect x="0.5" y="0.5" width="{width-1:.0f}" height="{height-1:.0f}" rx="10" fill="none" stroke="#2a2f3a"/>',
        f'<line x1="0" y1="30" x2="{width:.0f}" y2="30" stroke="#2a2f3a"/>',
        '<circle cx="20" cy="15.5" r="5" fill="#ff5f57"/>',
        '<circle cx="38" cy="15.5" r="5" fill="#febc2e"/>',
        '<circle cx="56" cy="15.5" r="5" fill="#28c840"/>',
        f'<text x="{width/2:.0f}" y="19.5" fill="#6b7382" font-size="11.5" text-anchor="middle">{html.escape(title)}</text>',
    ]

    for i, line in enumerate(raw):
        text = plain[i]
        if not text:
            continue
        y = PAD_TOP + i * LINE_H
        length = len(text) * ADVANCE
        spans = []
        # Runs are emitted back to back with no explicit x, so columns stay aligned
        # in the selected monospace font; textLength then pins
        # the whole line to the width the layout above was computed against.
        line_runs = runs(line)
        for index, (chunk, fill, weight) in enumerate(line_runs):
            if index == len(line_runs) - 1:
                chunk = chunk.rstrip()   # trailing spaces would stretch textLength
            if not chunk:
                continue
            bold = ' font-weight="bold"' if weight == "bold" else ""
            spans.append(f'<tspan fill="{fill}"{bold} xml:space="preserve">{html.escape(chunk)}</tspan>')
        parts.append(
            f'<text xml:space="preserve" x="{PAD_X}" y="{y:.1f}" textLength="{length:.1f}" lengthAdjust="spacing">'
            + "".join(spans) + "</text>"
        )

    parts.append("</svg>")
    open(dst, "w", encoding="utf-8").write("\n".join(parts) + "\n")
    print(f"{dst}: {cols} cols x {len(raw)} lines -> {width:.0f}x{height:.0f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
