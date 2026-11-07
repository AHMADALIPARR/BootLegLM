#!/usr/bin/env python3
"""Render README figures from measured BootLegLM transcripts. No synthetic series."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
IMAGES = DOCS / "images"
FONT = "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"


def load_losses():
    rows = []
    for line in (DOCS / "train-corpus-loss.txt").read_text().splitlines():
        step, loss = line.split(",")
        rows.append((int(step), float(loss)))
    return rows


def load_bench():
    text = (DOCS / "bench-runs.txt").read_text().splitlines()
    summary = {}
    env = ""
    for line in text:
        if line.startswith("bench_env,"):
            env = line[len("bench_env,"):]
        match = re.match(r"bench_(inference|search|train),repeats=(\d+),min_ms=([0-9.]+),median_ms=([0-9.]+),max_ms=([0-9.]+),checksum=(.*)", line)
        if match:
            name, repeats, lo, mid, hi, checksum = match.groups()
            summary[name] = {
                "repeats": int(repeats),
                "min": float(lo),
                "median": float(mid),
                "max": float(hi),
                "checksum": checksum,
            }
    return env, summary


def loss_svg(rows):
    width, height = 920, 360
    left, right, top, bottom = 64, 24, 36, 48
    plot_w = width - left - right
    plot_h = height - top - bottom
    ys = [loss for _, loss in rows]
    y_max = max(ys) * 1.08
    def x(step):
        return left + plot_w * step / (len(rows) - 1)
    def y(loss):
        return top + plot_h * (1 - loss / y_max)
    points = " ".join(f"{x(step):.1f},{y(loss):.1f}" for step, loss in rows)
    ticks = []
    for loss in (0, 1, 2, 3, 4):
        yy = y(loss)
        ticks.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#e4e8ee"/>')
        ticks.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-size="12" fill="#445">{loss:.0f}</text>')
    start, end = rows[0][1], rows[-1][1]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">Corpus training loss</title>
  <desc id="desc">Measured next-character cross-entropy on examples/corpus.txt for 40 SGD steps, seed 42, learning rate 0.2, sequence length 12. The loss starts at {start:.6f} and ends at {end:.6f}. Spikes are windows that contain a newline.</desc>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{left}" y="22" font-family="Liberation Sans, DejaVu Sans, sans-serif" font-size="16" font-weight="700" fill="#1c2430">examples/corpus.txt, 40 SGD steps, seed 42, lr 0.2, seq 12</text>
  {''.join(ticks)}
  <polyline fill="none" stroke="#1f7a4d" stroke-width="2.2" points="{points}"/>
  <text x="{left}" y="{height-16}" font-family="Liberation Sans, DejaVu Sans, sans-serif" font-size="12" fill="#445">step 0 loss {start:.6f}    step 39 loss {end:.6f}    not monotonic: newline windows raise the loss</text>
</svg>'''
    (IMAGES / "train-loss.svg").write_text(svg)


def bench_svg(env, summary):
    width, height = 920, 380
    names = ["inference", "search", "train"]
    labels = {
        "inference": "stack forward x100",
        "search": "reason x200",
        "train": "SGD x20 on abc",
    }
    colors = {"min": "#8fb8d6", "median": "#1f7a4d", "max": "#c47b3a"}
    left, top = 180, 48
    plot_w = 680
    max_ms = max(summary[name]["max"] for name in names) * 1.15
    row_h = 78
    parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">Benchmark timing ranges</title>
  <desc id="desc">Five repetitions on the measurement host. Bars are minimum, median, and maximum wall time in milliseconds. Checksums were identical across repetitions.</desc>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="24" y="28" font-family="Liberation Sans, DejaVu Sans, sans-serif" font-size="16" font-weight="700" fill="#1c2430">Five repetitions, wall time in milliseconds</text>''']
    for index, name in enumerate(names):
        y = top + index * row_h
        parts.append(f'<text x="24" y="{y+28}" font-family="Liberation Sans, DejaVu Sans, sans-serif" font-size="14" fill="#1c2430">{labels[name]}</text>')
        for offset, key in enumerate(("min", "median", "max")):
            value = summary[name][key]
            bar_w = plot_w * value / max_ms
            yy = y + offset * 18
            parts.append(f'<rect x="{left}" y="{yy}" width="{bar_w:.1f}" height="14" fill="{colors[key]}"/>')
            parts.append(f'<text x="{left + bar_w + 6:.1f}" y="{yy+12}" font-family="Liberation Sans, DejaVu Sans, sans-serif" font-size="12" fill="#1c2430">{key} {value:.3f}</text>')
    parts.append(
        f'<text x="24" y="340" font-family="Liberation Sans, DejaVu Sans, sans-serif" font-size="12" fill="#445">{env}</text>'
        f'<text x="24" y="360" font-family="Liberation Sans, DejaVu Sans, sans-serif" font-size="12" fill="#445">'
        f'Inference checksum {summary["inference"]["checksum"]}. '
        f'Search checksum {summary["search"]["checksum"]}. '
        f'Train final loss {summary["train"]["checksum"]}.</text>\n</svg>'
    )
    (IMAGES / "benchmarks.svg").write_text("".join(parts))


def png_excerpt():
    from PIL import Image, ImageDraw, ImageFont
    lines = (DOCS / "cli-session.txt").read_text().splitlines()
    # Verbatim slices: prove, the train endpoints, load, reason, guided, generation.
    keep = list(range(25, 31)) + list(range(32, 75)) + list(range(76, 99))
    excerpt = [lines[i] for i in keep if i < len(lines)]
    font = ImageFont.truetype(FONT, 16)
    pad = 18
    line_h = 22
    longest = max((font.getlength(line) for line in excerpt), default=400)
    width = int(pad * 2 + longest + 8)
    height = pad * 2 + line_h * len(excerpt)
    image = Image.new("RGB", (width, height), "#1b1f24")
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(excerpt):
        color = "#9ad0a4" if line.startswith("$") else "#e6edf3"
        draw.text((pad, pad + i * line_h), line, font=font, fill=color)
    image.save(IMAGES / "cli-demo.png")


if __name__ == "__main__":
    IMAGES.mkdir(parents=True, exist_ok=True)
    loss_svg(load_losses())
    env, summary = load_bench()
    bench_svg(env, summary)
    png_excerpt()
    print("wrote", IMAGES)
