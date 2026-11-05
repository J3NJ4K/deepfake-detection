import numpy as np
import torch
import torch.nn.functional as F
from detection.xception_net import Xception
import cv2


def grad_cam(model: Xception, tensor: torch.Tensor) -> np.ndarray:
    #returns a 8x8 heatmap showing where the model looked when it said 'fake'
    fmap = model.features(tensor)
    score = model.classify(fmap)[0, 1]

    # how much the fake score changes if each value in the feature map changes
    grads = torch.autograd.grad(score, fmap)[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    # weighted sum of all channels, keeps only what pushes towards "fake"
    cam = F.relu((weights * fmap).sum(dim=1))[0]
    return (cam / (cam.max() + 1e-8)).detach().numpy()

def overlay(image: np.ndarray, cam: np.ndarray, box: tuple[int, int, int, int], alpha: float = 0.4) -> np.ndarray:
    #blends the heatmap into the image inside `box` (the crop the model saw).
    x1, y1, x2, y2 = box

    # stretches the 8x8 map to the crop size, smooth
    heat = cv2.resize(cam, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)

    # colors, blue low, red high
    colored = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    out = image.copy()
    # transparency follows the heatmap: 0 = invisible, 1 = `alpha` strength
    a = (heat * alpha)[..., None]
    region = out[y1:y2, x1:x2].astype(np.float32)
    out[y1:y2, x1:x2] = (region * (1 - a) + colored * a).astype(np.uint8)
    return out

def heatmap_layer(results: list[dict], height: int, width: int, strength: float = 0.8) -> tuple[
        np.ndarray, np.ndarray]:

    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    alpha = np.zeros((height, width), dtype=np.uint8)

    for res in results:
        x1, y1, x2, y2 = res["crop_box"]
        heat = cv2.resize(res["heatmap"], (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
        colored = cv2.cvtColor(cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET),
                               cv2.COLOR_BGR2RGB)
        # more suspicious face = stronger heatmap
        a = (heat * strength * res["manipulation_probability"] * 255).astype(np.uint8)

        # crop boxes can overlap, keep the strongest heatmap where they do
        stronger = a > alpha[y1:y2, x1:x2]
        rgb[y1:y2, x1:x2][stronger] = colored[stronger]
        alpha[y1:y2, x1:x2][stronger] = a[stronger]

    return rgb, alpha