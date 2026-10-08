import sys
import cv2
from core.image_io import read_img, save_img
from detection.face_detector import detect_faces

image = read_img(sys.argv[1])
boxes = detect_faces(image)
print(f"{len(boxes)} face(s): {boxes}")

out = image.copy()
for x1, y1, x2, y2 in boxes:
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
save_img(out, "faces_out.jpg")