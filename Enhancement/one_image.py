import argparse
import torch
import torch.nn.functional as F
import numpy as np
from skimage import img_as_ubyte
from utils import load_img, save_img
from basicsr.models import create_model
from basicsr.utils.options import parse


def process_image(input_path, output_path, use_cuda):
    # Load model configuration and weights
    opt = parse("Options/RetinexFormer_SDSD_indoor.yml", is_train=False)
    model = create_model(opt).net_g
    checkpoint = torch.load("pretrained_weights/SDSD_indoor.pth", map_location="cpu")
    model.load_state_dict(checkpoint["params"])
    if use_cuda:
        model.cuda()
    model.eval()

    # Load and preprocess the image
    img = np.float32(load_img(input_path)) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    if use_cuda:
        img = img.cuda()

    # Padding
    _, _, h, w = img.shape
    H, W = ((h + 3) // 4) * 4, ((w + 3) // 4) * 4  # Assuming factor is 4
    img = F.pad(img, (0, W - w, 0, H - h), "reflect")

    # Process the image
    with torch.no_grad():
        output = model(img)
        output = output[:, :, :h, :w]  # Unpad
        output = torch.clamp(output, 0, 1).cpu().numpy().squeeze(0).transpose(1, 2, 0)

    # Save the output image
    save_img(output_path, img_as_ubyte(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Enhancement CLI Tool")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output JPEG file"
    )
    parser.add_argument("--cuda", action="store_true", help="Use CUDA for processing")
    args = parser.parse_args()

    process_image(args.input, args.output, args.cuda)
