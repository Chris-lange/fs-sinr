import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

def print_config_details(config):
    """
    Print the resolved data configuration in a human-readable format.
    """
    print("Model Data Configuration:")
    print("=" * 40)
    for key, value in config.items():
        print(f"{key:20}: {value}")
    print("=" * 40)

class ImageFolderDataset(Dataset):
    """
    Simply lists all images in 'input_folder' (non-recursive here),
    loads them, applies 'transform', and returns (filename, transformed_image).
    """
    def __init__(self, input_folder, transform=None):
        super().__init__()
        self.input_folder = input_folder
        self.transform = transform

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        self.image_files = [
            f for f in os.listdir(input_folder)
            if os.path.splitext(f.lower())[-1] in valid_exts
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.input_folder, image_name)
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image_name, image


class FeatureExtractor(nn.Module):
    """
    Wraps a timm model and hooks onto a chosen block to extract features.
    For many Vision Transformers in timm, model.blocks is the list of transformer blocks.
    Adjust block_to_extract as needed.
    """
    def __init__(self, model, block_to_extract=23, device='cpu'):
        super().__init__()
        self.model = model.to(device)
        self.model.eval()
        self.block_to_extract = block_to_extract
        self._features = None

        # Hook onto the specified block
        block_module = dict(self.model.named_children())['blocks'][block_to_extract]
        block_module.register_forward_hook(self._hook_fn)

        self.device = device

    def _hook_fn(self, module, input, output):
        # We'll store the output (the entire sequence of embeddings for ViTs, including CLS)
        self._features = output

    def forward(self, x):
        with torch.no_grad():
            x = x.to(self.device)
            _ = self.model(x)
        # By the time we return, self._features should be set.
        return self._features.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str,
                        default='/disk/scratch_fast/chris_2/sinr/all_eval_images',
                        help="Folder containing images to process.")
    parser.add_argument("--output_folder", type=str,
                        default='/disk/scratch_fast/chris_2/sinr/all_iucn_reps',
                        help="Folder in which to store the .pt files.")
    parser.add_argument("--model_name", type=str,
                        default="hf_hub:timm/eva02_large_patch14_clip_336.merged2b_ft_inat21",
                        help="Any model name recognized by timm.")
    parser.add_argument("--block_to_extract", type=int, default=23,
                        help="Which transformer block to hook.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on.")
    args = parser.parse_args()

    # Create the output folder if needed
    os.makedirs(args.output_folder, exist_ok=True)

    # 1. Load the model via timm.
    print(f"Loading model {args.model_name}...")
    model = timm.create_model(args.model_name, pretrained=True)
    model.eval()

    # 2. Create transforms (includes center crop and normalization).
    config = resolve_data_config({}, model=model)
    print_config_details(config)
    transform = create_transform(**config)

    # 3. Make a dataset / dataloader over the given folder of images.
    dataset = ImageFolderDataset(args.input_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 4. Wrap the model in our FeatureExtractor
    extractor = FeatureExtractor(model, block_to_extract=args.block_to_extract,
                                device=args.device)

    print(f"Extracting features from block {args.block_to_extract}...")
    for (filenames, images) in tqdm(dataloader):
        # images: (batch_size, 3, H, W)
        features = extractor(images)
        # 'features' shape depends on the model. In many ViTs, shape is [B, sequence_len, embed_dim].

        # 5. Save each sample individually, matching the original filename
        for i, fname in enumerate(filenames):
            cls_embedding = features[i, 0, :]

            # Construct output path (strip extension, add .pt or just reuse entire fname + .pt)
            base, ext = os.path.splitext(fname)
            output_name = base + ".pt"
            out_path = os.path.join(args.output_folder, output_name)

            torch.save(cls_embedding, out_path)

    print("Done! Extracted representations are in:", args.output_folder)

if __name__ == "__main__":
    main()
