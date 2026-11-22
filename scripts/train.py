"""YOLOv8n 전이학습 스크립트 (노트북 2번 셀과 동일한 설정)

사용법: python scripts/train.py --data data.yaml 경로
"""
import argparse
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--data", required=True, help="Roboflow에서 받은 data.yaml 경로")
p.add_argument("--model", default="yolov8n.pt")
p.add_argument("--epochs", type=int, default=50)
p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--batch", type=int, default=16)
a = p.parse_args()

model = YOLO(a.model)                     # COCO 사전학습 가중치
model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
            patience=20, project="runs", name="hw", exist_ok=True)
print("결과 저장 위치:", model.trainer.save_dir)

best = YOLO(f"{model.trainer.save_dir}/weights/best.pt")
m = best.val(data=a.data, split="test")
print(f"[test] P {m.box.mp:.3f} | R {m.box.mr:.3f} | mAP50 {m.box.map50:.3f} | mAP50-95 {m.box.map:.3f}")
