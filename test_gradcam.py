from core.image_io import read_img, save_img
from detection.face_detector import detect_faces
from detection.preprocess import crop_box, crop_face, to_tensor
from detection.classifier import load_model
from detection.gradcam import grad_cam, overlay

img = read_img("test_picture.jpg")
model = load_model()

out = img.copy()
for bbox in detect_faces(img):
    cam = grad_cam(model, to_tensor(crop_face(img, bbox)))
    out = overlay(out, cam, crop_box(img, bbox))
save_img(out, "gradcam_out.jpg")