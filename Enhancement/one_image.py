import argparse
import torch
import torch.nn as nn
import numpy as np
from skimage import img_as_ubyte
import utils
from basicsr.models import create_model
from basicsr.utils.options import parse


def process_image(input_path, output_path, weights, use_cuda):
    # Load model configuration
    opt = parse("Options/RetinexFormer_SDSD_indoor.yml", is_train=False)
    opt["dist"] = False  # Ensure distributed setting is correctly configured

    # Create model
    model = create_model(opt).net_g

    # Load weights
    checkpoint = torch.load(weights, map_location="cpu")
    try:
        model.load_state_dict(checkpoint["params"])
    except RuntimeError:
        # Adjust for DataParallel wrapping
        new_checkpoint = {"module." + k: v for k, v in checkpoint["params"].items()}
        model.load_state_dict(new_checkpoint)

    if use_cuda:
        model = nn.DataParallel(model)
        model.cuda()

    model.eval()

    # Load and preprocess the image
    img = np.float32(utils.load_img(input_path)) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    if use_cuda:
        img = img.cuda()

    # Process the image
    with torch.no_grad():
        output = model(img)
        output = torch.clamp(output, 0, 1).cpu().numpy().squeeze(0).transpose(1, 2, 0)

    # Save the output image
    utils.save_img(output_path, img_as_ubyte(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Enhancement CLI Tool")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output JPEG file"
    )
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to model weights"
    )
    parser.add_argument("--cuda", action="store_true", help="Use CUDA for processing")
    args = parser.parse_args()

    process_image(args.input, args.output, args.weights, args.cuda)
