import os
from PIL import Image, ImageOps, ImageDraw, ImageFont

# ====== Réglages ======
F1_IMG  = "matrice de confusion (1).png"
F05_IMG = "Matrice de confusion.png"
OUT_IMG = "matrice de confusion_F1_F0.5.png"

TITLE = "Lucene v3 — Transformer"
LEFT_LABEL  = "(a) Seuil optimisé selon F0.5"
RIGHT_LABEL = "(b) Seuil optimisé selon F1"  # juste pour l'image; LaTeX gérera mieux

BG_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)

PADDING = 40          # marge externe
GAP = 40              # espace entre les deux panneaux
TITLE_H = 60          # hauteur réservée au titre
LABEL_H = 55          # hauteur réservée aux labels sous chaque panneau

# ====== Utils ======
def load_img(path):
    img = Image.open(path).convert("RGB")
    # ajoute une petite bordure blanche uniforme
    return ImageOps.expand(img, border=10, fill=BG_COLOR)

def get_font(size):
    # Essaye une police "classique" Windows
    for name in ["arial.ttf", "calibri.ttf", "times.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except:
            pass
    return ImageFont.load_default()

def draw_centered_text(draw, text, x_center, y, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    draw.text((x_center - w // 2, y), text, fill=TEXT_COLOR, font=font)

# ====== Main ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
f1_path = os.path.join(BASE_DIR, F1_IMG)
f05_path = os.path.join(BASE_DIR, F05_IMG)

if not os.path.exists(f1_path):
    raise FileNotFoundError(f"Image introuvable: {f1_path}")
if not os.path.exists(f05_path):
    raise FileNotFoundError(f"Image introuvable: {f05_path}")

img_f1 = load_img(f1_path)
img_f05 = load_img(f05_path)

# Redimensionner à la même hauteur (pour un rendu propre)
target_h = min(img_f1.height, img_f05.height)
def resize_to_h(img, h):
    w = int(img.width * (h / img.height))
    return img.resize((w, h), Image.Resampling.LANCZOS)

img_f1 = resize_to_h(img_f1, target_h)
img_f05 = resize_to_h(img_f05, target_h)

# Canvas final
w = PADDING*2 + img_f1.width + GAP + img_f05.width
h = PADDING*2 + TITLE_H + target_h + LABEL_H

canvas = Image.new("RGB", (w, h), BG_COLOR)
draw = ImageDraw.Draw(canvas)

# Titre
title_font = get_font(26)
draw_centered_text(draw, TITLE, w//2, PADDING//2, title_font)

# Coller panneaux
top_y = PADDING + TITLE_H
left_x = PADDING
right_x = PADDING + img_f1.width + GAP

canvas.paste(img_f1, (left_x, top_y))
canvas.paste(img_f05, (right_x, top_y))

# Labels sous panneaux
label_font = get_font(18)
label_y = top_y + target_h + 10
draw_centered_text(draw, LEFT_LABEL, left_x + img_f1.width//2, label_y, label_font)
draw_centered_text(draw, RIGHT_LABEL, right_x + img_f05.width//2, label_y, label_font)

# Export
out_path = os.path.join(BASE_DIR, OUT_IMG)
canvas.save(out_path, quality=95)
print("✅ Figure académique générée :", out_path)
