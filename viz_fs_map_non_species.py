import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import json
import argparse
import utils
import models
import datasets
import setup
from gritlm import GritLM

matplotlib.rcParams["figure.dpi"] = 208.75

def extract_grit_token(model, text:str):
    """Helper function to create text embeddings from GritLM."""
    def gritlm_instruction(instruction):
        return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"
    d_rep = model.encode([text], instruction=gritlm_instruction(""))
    d_rep = torch.from_numpy(d_rep)
    return d_rep

def main(eval_params):
    # ------------------------------------------------
    # 1. Load configuration, model, environment mask
    # ------------------------------------------------
    with open('paths.json', 'r') as f:
        paths = json.load(f)

    # load model
    train_params = torch.load(eval_params['model_path'], map_location='cpu')
    default_params = setup.get_default_params_train()
    for key in default_params:
        if key not in train_params['params']:
            train_params['params'][key] = default_params[key]
    model = models.get_model(train_params['params'], inference_only=True)
    model.load_state_dict(train_params['state_dict'], strict=False)
    model = model.to(eval_params['device'])
    model.eval()

    # load environment data if needed
    if train_params['params']['input_enc'] in ['env', 'sin_cos_env']:
        raster = datasets.load_env()
    else:
        raster = None

    enc = utils.CoordEncoder(
        train_params['params']['input_enc'],
        raster=raster,
        input_dim=train_params['params']['input_dim']
    )

    # load ocean mask
    if eval_params['high_res']:
        mask = np.load(os.path.join(paths['masks'], 'ocean_mask_hr.npy'))
    else:
        mask = np.load(os.path.join(paths['masks'], 'ocean_mask.npy'))

    mask_inds = np.where(mask.reshape(-1) == 1)[0]

    # generate input features for the entire global grid (masked)
    locs = utils.coord_grid(mask.shape)
    if not eval_params['disable_ocean_mask']:
        locs = locs[mask_inds, :]
    locs = torch.from_numpy(locs)
    locs_enc = enc.encode(locs).to(eval_params['device'])

    # ------------------------------------------------
    # 2. Determine which .pt files to process
    # ------------------------------------------------
    image_emb_path = eval_params['image_emb_path']
    if os.path.isdir(image_emb_path):
        # If user passes a directory, gather all .pt files inside it
        emb_files = [
            os.path.join(image_emb_path, f)
            for f in os.listdir(image_emb_path)
            if f.endswith('.pt')
        ]
    else:
        # If user passes a single file, process only that file
        emb_files = [image_emb_path]

    # Optionally load GritLM if we need text embeddings
    if eval_params['use_text']:
        grit = GritLM("GritLM/GritLM-7B", torch_dtype="auto", mode="embedding")

    # ------------------------------------------------
    # 3. For each embedding file, do inference & save
    # ------------------------------------------------
    for emb_file in emb_files:
        print(f"\nProcessing: {emb_file}")

        # load image embedding
        image_emb = torch.load(emb_file, map_location='cpu')

        # handle text embedding if needed
        if eval_params['use_text']:
            with torch.no_grad():
                text = input('Enter Species Description (for this embedding): ')
                text_emb = extract_grit_token(grit, text).to(eval_params['device'])
                preds = model.zero_shot(locs_enc, image_emb=image_emb, text_emb=text_emb)
        else:
            with torch.no_grad():
                preds = model.zero_shot(locs_enc, image_emb=image_emb, text_emb=None)

        preds = preds.cpu().squeeze(-1)

        # threshold predictions
        if eval_params['threshold'] > 0:
            print(f'Applying threshold of {eval_params["threshold"]} to the predictions.')
            preds[preds < eval_params['threshold']] = 0.0
            preds[preds >= eval_params['threshold']] = 1.0

        # mask data
        if not eval_params['disable_ocean_mask']:
            op_im = np.ones((mask.shape[0] * mask.shape[1])) * np.nan  # set to NaN
            op_im[mask_inds] = preds
        else:
            op_im = preds

        # reshape for plotting
        op_im = op_im.reshape((mask.shape[0], mask.shape[1]))
        op_im = np.ma.masked_invalid(op_im)

        # set color for masked values
        cmap = plt.cm.plasma
        cmap.set_bad(color='none')
        if eval_params['set_max_cmap_to_1']:
            vmax = 1.0
        else:
            vmax = np.max(op_im)

        # figure out a good name for output
        base_name = os.path.splitext(os.path.basename(emb_file))[0]
        save_name = f"{base_name}_output.png"
        save_loc = os.path.join(eval_params['op_path'], save_name)

        # save the image
        print(f"Saving image to {save_loc}")
        plt.imsave(fname=save_loc, arr=op_im, vmin=0, vmax=vmax, cmap=cmap)

        # Optionally show the plot. (Remove if you don't want a pop-up for every file.)
        fig, ax = plt.subplots(figsize=plt.figaspect(op_im))
        fig.subplots_adjust(0, 0, 1, 1)
        ax.imshow(op_im, interpolation='nearest')
        plt.axis("off")
        plt.axis("tight")
        plt.axis("image")
        plt.show()

def plot_preds(preds, tt, eval_params):
    """Utility for plotting predictions with ocean mask, for a single array of preds."""
    eval_params['high_res'] = False
    eval_params['threshold'] = -1
    eval_params['disable_ocean_mask'] = False
    eval_params['set_max_cmap_to_1'] = False
    eval_params['op_path'] = './images'

    with open('paths.json', 'r') as f:
        paths = json.load(f)

    # load ocean mask
    if eval_params['high_res']:
        mask = np.load(os.path.join(paths['masks'], 'ocean_mask_hr.npy'))
    else:
        mask = np.load(os.path.join(paths['masks'], 'ocean_mask.npy'))
    mask_inds = np.where(mask.reshape(-1) == 1)[0]

    # threshold
    if eval_params['threshold'] > 0:
        print(f'Applying threshold of {eval_params["threshold"]} to the predictions.')
        preds[preds < eval_params['threshold']] = 0.0
        preds[preds >= eval_params['threshold']] = 1.0

    if not eval_params['disable_ocean_mask']:
        op_im = np.ones((mask.shape[0] * mask.shape[1])) * np.nan
        op_im[mask_inds] = preds
        op_im = op_im[:, np.newaxis]
    else:
        op_im = preds

    op_im = op_im.reshape((mask.shape[0], mask.shape[1]))
    op_im = np.ma.masked_invalid(op_im)

    cmap = plt.cm.plasma
    cmap.set_bad(color='none')
    if eval_params['set_max_cmap_to_1']:
        vmax = 1.0
    else:
        vmax = np.max(op_im)
        if vmax <= 0:
            vmax=0.01

    save_loc = os.path.join(eval_params['op_path'], f'zero_shot_range_pos+emb_{tt}.png')
    print(f'Saving image to {save_loc}')
    plt.imsave(fname=save_loc, arr=op_im, vmin=0, vmax=vmax, cmap=cmap)

    fig, ax = plt.subplots(figsize=plt.figaspect(op_im))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.imshow(op_im, interpolation='nearest')
    plt.axis("off")
    plt.axis("tight")
    plt.axis("image")
    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(
        description='Demo that takes a folder (or single file) of image embeddings and generates predicted ranges.'
    )
    parser.add_argument('--model_path', type=str,
                        default='/disk/scratch_fast/chris_2/clean_fs_sinr/experiments/test_fs/model.pt')
    parser.add_argument('--threshold', type=float, default=-1,
                        help='Threshold the range map [0, 1].')
    parser.add_argument('--op_path', type=str, default='./generated_ranges/',
                        help='Location to save all output images.')
    parser.add_argument('--high_res', action='store_true',
                        help='Generate higher resolution output.')
    parser.add_argument('--disable_ocean_mask', action='store_true',
                        help='Skip use of ocean mask.')
    parser.add_argument('--set_max_cmap_to_1', action='store_true',
                        help='Use fixed colormap max=1.')
    parser.add_argument('--device', type=str, default=device,
                        help='cpu or cuda')
    parser.add_argument('--image_emb_path', type=str,
                        default='/disk/scratch_fast/chris_2/sinr/all_iucn_reps/13385771_14064.pt',
                        help='Either a single .pt file or a folder of .pt embeddings.')
    parser.add_argument('--use_text', action='store_false',
                        help='Prompt for text input and use text embeddings.')

    eval_params = vars(parser.parse_args())

    if not os.path.isdir(eval_params['op_path']):
        os.makedirs(eval_params['op_path'])

    main(eval_params)