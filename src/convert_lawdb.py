import json
import math
from PIL import Image, ImageDraw, ImageFont

import re
from html_to_markdown import convert_to_markdown

def process_tables(text: str) -> str:
    pattern = re.compile(r"<<TABLE[:\s]*(.*?)([\\/]+\s*TABLE\s*>>)", flags=re.DOTALL | re.IGNORECASE)
    parts = []
    last_end = 0
    matches = list(pattern.finditer(text))

    for m in matches:
        parts.append(text[last_end:m.start()])      
        content = m.group(1).strip()
        if "<table" in content.lower():
         
            md = convert_to_markdown(content)
            parts.append("<<TABLE\n" + md.strip() + "\n\\TABLE>>")
        else:
            parts.append(m.group(0))
        last_end = m.end()

    parts.append(text[last_end:]) 
    return "".join(parts)

def format_database(text):
    convert_text = ""
    chunk_splited = text.split("<<IMAGE:")
    convert_text +=  chunk_splited[0].strip()
    img_path, caption = [], []
    for chunk in chunk_splited[1:]:
        img_id, post_text = chunk.split("/IMAGE>>")[0].strip(), chunk.split("/IMAGE>>")[1]
        img_path.append(f"law_db/images.fld/{img_id}")
        caption.append(img_id)
        convert_text += f"<Image Caption: {img_id}>" + post_text
    
    return convert_text, img_path, caption

def concat_images_grid(image_paths, captions=None, grid_size=None, font_path=None, font_size=16, box_color="red", box_width=3):
    images = [Image.open(p) for p in image_paths]
    n = len(images)

    if grid_size is None:
        grid_cols = math.ceil(math.sqrt(n))
        grid_rows = math.ceil(n / grid_cols)
    else:
        grid_cols, grid_rows = grid_size

    font = ImageFont.truetype("DejaVuSans.ttf", 16)

    caption_h = font_size + 6 if captions else 0

    col_widths = [0] * grid_cols
    row_heights = [0] * grid_rows

    for idx, img in enumerate(images):
        col = idx % grid_cols
        row = idx // grid_cols
        col_widths[col] = max(col_widths[col], img.width)
        row_heights[row] = max(row_heights[row], img.height + caption_h)

    W = sum(col_widths)
    H = sum(row_heights)

    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)

    y_offset = 0
    for row in range(grid_rows):
        x_offset = 0
        for col in range(grid_cols):
            idx = row * grid_cols + col
            if idx >= n:
                break

            img = images[idx]
            canvas.paste(img, (x_offset, y_offset))

            x1, y1 = x_offset, y_offset
            x2, y2 = x_offset + img.width, y_offset + img.height
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=box_width)

            if captions:
                cap = captions[idx]
                bbox = draw.textbbox((0, 0), cap, font=font)
                text_w = bbox[2] - bbox[0]
                draw.text(
                    (x_offset + (img.width - text_w)//2, y_offset + img.height + 2),
                    cap, fill="black", font=font
                )

            x_offset += col_widths[col]
        y_offset += row_heights[row]

    return canvas

database = json.load(open("law_db/vlsp2025_law_table_replace.json"))
database_new = json.load(open("law_db/vlsp2025_law_new.json"))
database_mapping = {}

for item in database:
    database_mapping[item["id"]] = {}
    for article in item["articles"]:
        database_mapping[item["id"]][article["id"]] = article["text"]

for item in database_new:
    for article in item["articles"]:
        if article["id"] not in database_mapping[item["id"]]:
            print(article["id"])
            database_mapping[item["id"]][article["id"]] = article["text"]

        
FUSION_IMG_COUNT = 29

for law, rule in database_mapping.items():
    for k, text in rule.items():
        if text.count("<<IMAGE") > 1:
            new_text, img, cap = format_database(text)
            concat_image = concat_images_grid(img, cap, box_color="blue", box_width=2)
            img_name = f"concat_{FUSION_IMG_COUNT}.jpg"
            database_mapping[law][k] = f"<<IMAGE: {img_name} /IMAGE>>" + "\n" + new_text
            concat_image.save(f"law_db/images.fld/{img_name}")
            FUSION_IMG_COUNT += 1
            
        database_mapping[law][k] = process_tables(database_mapping[law][k])

with open("law_db/vlsp2025_law_v3.json", "w") as f:
    json.dump(database_mapping, f, indent=4, ensure_ascii=False)