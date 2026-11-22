"""학습한 모델로 영상에서 객체 감지 + 추적, 궤적 시각화 및 CSV 저장

사용법: python scripts/track.py --source input.mp4 --seconds 5
출력:  tracking/track_result.mp4, tracking/trajectory.csv
"""
import argparse, csv, os, subprocess
from collections import defaultdict
import cv2, numpy as np
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--source", required=True)
p.add_argument("--weights", default="weights/best.pt")
p.add_argument("--start", type=float, default=0, help="시작 시각(초)")
p.add_argument("--seconds", type=float, default=5)
p.add_argument("--conf", type=float, default=0.4)
p.add_argument("--out", default="tracking")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

model = YOLO(a.weights)
cap = cv2.VideoCapture(a.source)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.set(cv2.CAP_PROP_POS_MSEC, a.start * 1000)
raw = f"{a.out}/track_raw.mp4"
out = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

history, rows, f = defaultdict(list), [], 0
while f < int(fps * a.seconds):
    ok, frame = cap.read()
    if not ok:
        break
    r = model.track(frame, persist=True, conf=a.conf, tracker="bytetrack.yaml", verbose=False)[0]
    img = r.plot()
    if r.boxes.id is not None:
        for (x, y, w, h), tid, c in zip(r.boxes.xywh.cpu().numpy(),
                                        r.boxes.id.int().cpu().tolist(),
                                        r.boxes.cls.int().cpu().tolist()):
            history[tid] = (history[tid] + [(float(x), float(y))])[-60:]
            rows.append([f, round(f / fps, 3), tid, model.names[c], round(float(x), 1), round(float(y), 1)])
            cv2.polylines(img, [np.array(history[tid], np.int32).reshape(-1, 1, 2)], False, (0, 255, 255), 2)
    out.write(img)
    f += 1
cap.release(); out.release()

with open(f"{a.out}/trajectory.csv", "w", newline="") as fp:
    csv.writer(fp).writerows([["frame", "time_s", "track_id", "class", "cx", "cy"]] + rows)

# 브라우저·GitHub 재생용 H.264 변환
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-vcodec", "libx264",
                "-preset", "veryfast", "-pix_fmt", "yuv420p", f"{a.out}/track_result.mp4"], check=True)
os.remove(raw)
print(f"{f}프레임 처리, 추적 ID {len(history)}개 → {a.out}/")
