# VLSP 2025 - Task 2 Submission

Pipeline sinh câu trả lời cho **VLSP 2025 Traffic QA (Task 2)** sử dụng `lmdeploy` và mô hình `InternVL3-78B`.  
Hệ thống sẽ đọc câu hỏi, truy xuất điều luật liên quan (text + hình ảnh), sau đó sinh đáp án và xuất ra file submission đúng format.

---

## Cài đặt
```
conda create -n traficqa -y python=3.10
pip install -r requirements.txt
```

## Cấu trúc dữ liệu
```
.
├── data/
│   └── vlsp2025_submission_task2.json      # Private test
│
├── law_db/
│   ├── vlsp2025_law_new.json               # Database luật gốc
│   ├── vlsp2025_law_converted.json         # Database luật được xử lý
│   └── images.fld/                         # Thư mục chứa ảnh được tham chiếu trong luật
│       ├── <img_id_1>.jpg
│       ├── <img_id_2>.jpg
│       └── ...
│
├── private_test/
│   └── private_test_images/                # Ảnh của bộ test private
│       ├── <image_id>.jpg
│       └── ...
│
├── src/
│   └── convert_lawdb.py
│   └── inference.py
├── output/
│   └── submission/
│       └── submission_task2.json           # File submission cuối cùng
│
├── requirements.txt                                 
└── README.md                               

```
## Sử dụng pipeline
```
# NVIDI A100 8x40/80GB 
python src/convert_lawdb.py
python src/inference.py
```
