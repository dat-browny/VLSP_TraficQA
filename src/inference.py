import os
import re
import json
import base64
import argparse
from openai import OpenAI
from multiprocessing import Pool
from tqdm import tqdm

from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy.vl import load_image

RULE = ["Biển báo hình tròn, viền đỏ, nền xanh có một gạch chéo là biển cấm đỗ xe. Cấm đỗ xe thì xe được phép dừng trả khách",
"Biển báo cấm đỗ ngày chẵn, các phương tiện không được đỗ vào ngày chẵn trong tháng (ngày 2, 4, 6,...) tương tự như biển cấm đỗ ngày lẻ(biển có chữ I màu trắng, gạch đỏ chéo) (ngày 1, 3, 5,..)",
"Biển cấm có hình: hai ô tô màu đỏ/đen cạnh nhau - hiệu lực cấm tất cả các loại xe cơ giới vượt nhau (kể cả xe được ưu tiên theo quy định) nhưng được phép vượt xe máy 2 bánh, xe gắn máy."]
RULE_ADDED = "\n".join(RULE)

ROOT_PATH = os.getcwd()

SYS_PROMPT = "Bạn là một trợ lý ảo có khả năng nhận diện biển báo và hiểu biết luật về đường bộ Việt Nam."
PREFIX_ARTICLE = f"Dưới đây là một đoạn thông tin trong database\n"

PREFIX_QUESTION = "Dựa vào đoạn thông tin được cung cấp bên trên, chú ý tới các biển báo được cung cấp trong hình ảnh dưới đây, sau đó trả lời câu hỏi:\n"
RULE_ADDED = "\n".join(RULE)


POSTFIX_QUESTION = f"\n\nSuy luận và trả ra đáp án cuối cùng với cụm từ: Đáp án cuối cùng: A, B, C, D... - chỉ được chọn một đáp án.\n\nMột số thông tin cần lưu ý khi trả lời\n\n{RULE_ADDED}. Nếu câu hỏi không liên quan đến thông tin này thì có thể bỏ qua."

test = json.load(open(f"{ROOT_PATH}/data/vlsp2025_submission_task2.json"))

database_mapping = json.load(open(f"{ROOT_PATH}/law_db/vlsp2025_law_converted.json"))

def extract_answer(answer):
    keyword = answer.split("Đáp án cuối cùng:")[-1]
    keyword = re.sub(r'[^\w\s]', '', keyword).strip()
    return keyword        

def parse_answer(item):
    question_type = item.pop('question_type')
    answer_key = extract_answer(item["answer"])

    item["answer_key"] = answer_key
    if question_type == "Yes/No":
        if answer_key == "A":
            item["answer"] = "Đúng"
        else:
            item["answer"] = "Sai"
    else:
        item["answer"] = answer_key
    item.pop('choices')
    return item


def encode_base64(filepath: str) -> str:
    return filepath


def get_database(article):
    conv_img = []
    try:
        text = database_mapping[article["law_id"]][article["article_id"]]
        chunk_splited = text.split("<<IMAGE:")
        text_processed = chunk_splited[0].strip()
        for chunk in chunk_splited[1:]:
            img_id, post_text = chunk.split("/IMAGE>>")[0].strip(), chunk.split("/IMAGE>>")[1]
            img_path = f"{ROOT_PATH}/law_db/images.fld/{img_id}"
            conv_img.append({"type": "image_url", "image_url": {
                "url": encode_base64(img_path)}})
            text_processed += f"{IMAGE_TOKEN}\n{post_text}"
    except:
        return 
    return text_processed, conv_img


def process_question_prompt(item):
    text = PREFIX_ARTICLE
    conv_img = []
    for article in item["relevant_articles"]:
        if get_database(article):
            txt, img = get_database(article)
            text += txt
            conv_img += img

    img_id = item["image_id"]
    text += f"{PREFIX_QUESTION}\n{IMAGE_TOKEN}"
    conv_img += [{"type": "image_url", "image_url": {
        "url": encode_base64(f"{ROOT_PATH}/private_test/private_test_images/{img_id}.jpg")}}]

    if "choices" not in item:
        if "sai" in item["question"].lower():
            item["choices"] = {"A": "Đúng", "B": "Sai"}
        else:
            item["choices"] = {"A": "Có", "B": "Không"}

    question = item['question'] + "\n".join([f"{k}: {v}" for k, v in item["choices"].items()])
    text += f"{question}{POSTFIX_QUESTION}"

    conv_base = [
        {"type": "text", "text": text},
        *conv_img,
    ]
    
    message = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": conv_base}]

    return message

if __name__ == "__main__":

    requests = []
    
    for item in test:
        message = process_question_prompt(item)
        requests.append(message)
        
    pipe = pipeline('OpenGVLab/InternVL3-78B', backend_config=TurbomindEngineConfig(tp=8, chat_template="iternvl2_5"))
    
    gen_config = GenerationConfig(
        n=1,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
        max_new_tokens=1024,
        random_seed=42,
    )
    
    outputs = []
    BATCH_SIZE = 32
    
    num_batch = len(requests) // BATCH_SIZE
    
    for idx in tqdm(range(num_batch+1)):
        batch_requests = requests[BATCH_SIZE*idx:BATCH_SIZE*(idx+1)]
        outputs += pipe(batch_requests, gen_config=gen_config, use_tqdm=True)

    for item, response in zip(test, outputs):
        item["answer"] = response.text
        item = parse_answer(item)
        assert "answer_key" in item

    with open("ouput/submission/submission_task2.json", "w") as f:
        json.dump(test, f, indent=4, ensure_ascii=False)