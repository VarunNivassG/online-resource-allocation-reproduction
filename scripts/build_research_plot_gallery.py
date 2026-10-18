"""Consolidate research figures from smoke and supplementary experiments."""
from __future__ import annotations
import re
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = [ROOT / "results_research_smoke" / "plots", ROOT / "results_research_supplementary" / "plots"]
DEST = ROOT / "results_research_complete"
PLOTS = DEST / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)
for old_plot in PLOTS.glob("*.png"):
    old_plot.unlink()

files = []
for source in SOURCE_DIRS:
    files.extend(source.glob("*.png"))

def sort_key(path: Path):
    match = re.match(r"(\d+)", path.name)
    return (int(match.group(1)) if match else 999, path.name)

files = sorted(files, key=sort_key)
for path in files:
    shutil.copy2(path, PLOTS / path.name)

cols, tile_w, tile_h, margin = 4, 420, 310, 18
rows = (len(files) + cols - 1) // cols
sheet = Image.new("RGB", (margin + cols * (tile_w + margin), margin + rows * (tile_h + margin)), (245, 246, 248))
draw = ImageDraw.Draw(sheet)
try:
    title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
except OSError:
    title_font = ImageFont.load_default(); label_font = ImageFont.load_default()

for index, path in enumerate(files):
    col, row = index % cols, index // cols
    x, y = margin + col * (tile_w + margin), margin + row * (tile_h + margin)
    draw.rounded_rectangle((x, y, x + tile_w, y + tile_h), radius=10, fill=(255, 255, 255), outline=(210, 215, 222), width=1)
    image = Image.open(path).convert("RGB")
    image.thumbnail((tile_w - 16, tile_h - 48), Image.Resampling.LANCZOS)
    px = x + (tile_w - image.width) // 2
    py = y + 8 + (tile_h - 48 - image.height) // 2
    sheet.paste(image, (px, py))
    draw.text((x + 8, y + tile_h - 33), path.stem[:54], font=title_font, fill=(25, 35, 50))
    draw.text((x + 8, y + tile_h - 16), path.parent.parent.name, font=label_font, fill=(95, 105, 120))

sheet.save(DEST / "research_plots_contact_sheet.png", optimize=True)
lines = [
    "# Consolidated Research Plot Gallery", "",
    f"Total figure files: {len(files)}. Figures from the two-seed smoke study and multi-seed supplementary studies are copied into `plots/` for easy review.",
    "", "[Open contact sheet](research_plots_contact_sheet.png)", "", "## Figure index", "",
]
for path in files:
    lines.append(f"- [{path.stem}](plots/{path.name})")
(DEST / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Copied {len(files)} figures and wrote {DEST / 'research_plots_contact_sheet.png'}")
