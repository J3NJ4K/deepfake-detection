import sys
import cv2
from core.image_io import read_img, save_img
from detection.pipeline import analyze_deepfake

if len(sys.argv) < 2:
    sys.exit("Usage: python test_detector.py <image>")

image = read_img(sys.argv[1])
results = analyze_deepfake(image)

out = image.copy()
for r in results:
    x1, y1, x2, y2 = r["bbox"]
    p = r["manipulation_probability"]
    print(f"{r['bbox']}: {p:.2f}")
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(out, f"{p:.0%}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
save_img(out, "faces_out.jpg")