import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCTR_DIR = PROJECT_ROOT / "DocTr"

INPUT_PATH = PROJECT_ROOT / "output" / "oriented.jpeg"
OUTPUT_PATH = PROJECT_ROOT / "output" / "geometric_unwarped.png"

SEG_MODEL_PATH = DOCTR_DIR / "model_pretrained" / "seg.pth"
GEOTR_MODEL_PATH = DOCTR_DIR / "model_pretrained" / "geotr.pth"

# Allow Python to import DocTr modules
sys.path.insert(0, str(DOCTR_DIR))

from seg import U2NETP
from GeoTr import GeoTr


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# --------------------------------------------------
# GeoTr + segmentation model
# --------------------------------------------------

class GeoTrSeg(nn.Module):

    def __init__(self):
        super().__init__()

        self.msk = U2NETP(3, 1)
        self.GeoTr = GeoTr(num_attn_layers=6)

    def forward(self, x):

        # Document segmentation
        msk, *_ = self.msk(x)

        msk = (msk > 0.5).float()

        x = msk * x

        # Geometric unwarping
        bm = self.GeoTr(x)

        bm = (2 * (bm / 286.8) - 1) * 0.99

        return bm


# --------------------------------------------------
# Load pretrained model
# --------------------------------------------------

def load_model():

    model = GeoTrSeg().to(device)

    # Segmentation model
    seg_checkpoint = torch.load(
        SEG_MODEL_PATH,
        map_location=device
    )

    seg_state = {
        k[6:]: v
        for k, v in seg_checkpoint.items()
        if k[6:] in model.msk.state_dict()
    }

    model.msk.load_state_dict(
        {**model.msk.state_dict(), **seg_state}
    )

    # GeoTr model
    geotr_checkpoint = torch.load(
        GEOTR_MODEL_PATH,
        map_location=device
    )

    geotr_state = {
        k[7:]: v
        for k, v in geotr_checkpoint.items()
        if k[7:] in model.GeoTr.state_dict()
    }

    model.GeoTr.load_state_dict(
        {**model.GeoTr.state_dict(), **geotr_state}
    )

    model.eval()

    print("GeoTr model loaded successfully.")

    return model


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    model = load_model()

    # Read image
    im_ori = np.array(
        Image.open(INPUT_PATH)
    )[:, :, :3] / 255.0

    h, w, _ = im_ori.shape

    print("Input size:", (w, h))

    # GeoTr expects 288 × 288
    im = cv2.resize(
        im_ori,
        (288, 288)
    )

    im = im.transpose(2, 0, 1)

    im = torch.from_numpy(
        im
    ).float().unsqueeze(0).to(device)

    # --------------------------------------------------
    # GeoTr inference
    # --------------------------------------------------

    with torch.no_grad():

        bm = model(im)

    bm = bm.cpu()

    # Resize deformation maps
    bm0 = cv2.resize(
        bm[0, 0].numpy(),
        (w, h)
    )

    bm1 = cv2.resize(
        bm[0, 1].numpy(),
        (w, h)
    )

    # Smooth deformation field
    bm0 = cv2.blur(
        bm0,
        (3, 3)
    )

    bm1 = cv2.blur(
        bm1,
        (3, 3)
    )

    # Create sampling grid
    grid = torch.from_numpy(
        np.stack(
            [bm0, bm1],
            axis=2
        )
    ).unsqueeze(0).float()

    # --------------------------------------------------
    # Warp image
    # --------------------------------------------------

    image_tensor = torch.from_numpy(
        im_ori
    ).permute(2, 0, 1).unsqueeze(0).float()

    output = F.grid_sample(
        image_tensor,
        grid,
        align_corners=True
    )

    # Convert to image
    img_geo = (
        output[0]
        .permute(1, 2, 0)
        .numpy()
        * 255
    ).clip(0, 255).astype(np.uint8)

    # Save
    cv2.imwrite(
        str(OUTPUT_PATH),
        cv2.cvtColor(
            img_geo,
            cv2.COLOR_RGB2BGR
        )
    )

    print("GeoTr output saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()


