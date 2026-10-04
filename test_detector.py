import sys
import cv2
from detection.face_detector import detect_faces

path = sys.argv[1]
bgr = cv2.imread(path)
if bgr is None:
    sys.exit(f"Can't find the picture: {path}")

image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
boxes = detect_faces(image)
print(f"{len(boxes)} face(s): {boxes}")

for x1, y1, x2, y2 in boxes:
    cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imwrite("faces_out.jpg", bgr)