"""
Demo that takes an iNaturalist taxa ID as input and generates a prediction 
for each location on the globe and saves the ouput as an image.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
import utils
import datasets
import eval

def get_pregenerated_text_emb(taxon_id, text_type):
    text_embedding = None
    if text_type is None or text_type == 'none':
        return text_embedding

    with open('paths.json', 'r') as f:
        paths = json.load(f)
    embs1 = torch.load(paths['eval_text_embs'], map_location='cpu')
    emb_ids1 = embs1['taxon_id'].tolist()
    keys1 = embs1['keys']
    embs1 = embs1['data']

    taxa_of_interest = taxon_id
    taxa_index_of_interest = emb_ids1.index(taxa_of_interest)

    possible_text_embedding_indexes = [i for i, key in enumerate(keys1) if
                                       key[0] == taxa_index_of_interest and key[1] == text_type]

    if len(possible_text_embedding_indexes) != 1:
        print("No text embedding found for taxon {}".format(taxa_of_interest))
        return text_embedding

    text_embedding_index = possible_text_embedding_indexes[0]
    text_embedding = embs1[text_embedding_index].unsqueeze(0).unsqueeze(0)
    print(text_embedding_index)
    return text_embedding

def get_pregenerated_image_emb(taxon_id):
    image_embedding = None
    with open('paths.json', 'r') as f:
        paths = json.load(f)

    image_embs_dict = torch.load(paths['image_embs'], map_location='cpu', weights_only=False)
    image_embs = image_embs_dict['data']
    image_embs_ids = image_embs_dict['taxon_id']
    image_embs_keys = image_embs_dict['keys']

    taxa_of_interest = taxon_id
    possible_image_embedding_indexes = [i for i, key in enumerate(image_embs_keys) if
                                       key[0] == taxa_of_interest]

    if len(possible_image_embedding_indexes) == 0:
        print("No text embedding found for taxon {}".format(taxa_of_interest))
        return image_embedding

    image_embedding_index = possible_image_embedding_indexes[0]
    image_embedding = image_embs[image_embedding_index].unsqueeze(0).unsqueeze(0)

    return image_embedding

def get_eval_context_points(taxa_id, context_data, size):
  all_context_pts = context_data['locs'][context_data['labels'] == np.argwhere(context_data['class_to_taxa'] == taxa_id)[0]][0:]
  context_pts = all_context_pts[0:size]
  normalized_pts = torch.from_numpy(context_pts) * torch.tensor([[1/180,1/90]], device='cpu')
  return normalized_pts

def main(eval_params):
     # load params
    with open('paths.json', 'r') as f:
        paths = json.load(f)

    ckp_name = os.path.split(eval_params['model_path'])[-1]
    experiment_name = os.path.split(os.path.split(eval_params['model_path'])[-2])[-1]

    eval_overrides = {'ckp_name':ckp_name,
                      'experiment_name':experiment_name,
                      'device':eval_params['device']}


    train_overrides = {'dataset': 'eval_flexible'}
    context_data = np.load(os.path.join(paths['data'], 'positive_eval_data.npz'))

    for pt in eval_params['context_pt_trial']:
        number_of_context_points = pt

        if eval_params['use_text']:
            text_emb = get_pregenerated_text_emb(taxon_id=eval_params['test_taxa'], text_type=eval_params['text_type'])
        else:
            text_emb = None

        if eval_params['use_image']:
            image_emb = get_pregenerated_image_emb(taxon_id=eval_params['test_taxa'])
        else:
            image_emb = None

        context_points = get_eval_context_points(taxa_id=eval_params['test_taxa'],
                                                                          context_data=context_data,
                                                                          size=number_of_context_points)

        model, context_locs_of_interest, train_params, class_of_interest = eval.generate_eval_embedding_from_given_points(
                                                                    context_points=context_points,
                                                                    overrides=eval_overrides,
                                                                    taxa_of_interest=eval_params['test_taxa'],
                                                                    train_overrides=train_overrides,
                                                                    text_emb=text_emb,
                                                                    image_emb=image_emb)

        if train_params['params']['input_enc'] in ['env', 'sin_cos_env']:
            raster = datasets.load_env()
        else:
            raster = None
        enc = utils.CoordEncoder(train_params['params']['input_enc'], raster=raster, input_dim=train_params['params']['input_dim'])

        # load ocean mask
        if eval_params['high_res']:
            mask = np.load(os.path.join(paths['masks'], 'ocean_mask_hr.npy'))
        else:
            mask = np.load(os.path.join(paths['masks'], 'ocean_mask.npy'))

        mask_inds = np.where(mask.reshape(-1) == 1)[0]
            
        # generate input features
        locs = utils.coord_grid(mask.shape)
        if not eval_params['disable_ocean_mask']:
            locs = locs[mask_inds, :]
        locs = torch.from_numpy(locs)
        locs_enc = enc.encode(locs).to(eval_params['device'])

        with torch.no_grad():
            preds = model.embedding_forward(x=locs_enc, class_ids=None, return_feats=False, class_of_interest=class_of_interest, eval=True).cpu().numpy()

        # threshold predictions
        if eval_params['threshold'] > 0:
            print(f'Applying threshold of {eval_params["threshold"]} to the predictions.')
            preds[preds<eval_params['threshold']] = 0.0
            preds[preds>=eval_params['threshold']] = 1.0
            
        # mask data
        if not eval_params['disable_ocean_mask']:
            op_im = np.ones((mask.shape[0] * mask.shape[1])) * np.nan  # set to NaN
            op_im[mask_inds] = preds
        else:
            op_im = preds

        # reshape and create masked array for visualization
        op_im = op_im.reshape((mask.shape[0], mask.shape[1]))
        op_im = np.ma.masked_invalid(op_im) 

        # set color for masked values
        cmap = plt.cm.plasma
        cmap.set_bad(color='none')
        if eval_params['set_max_cmap_to_1']:
            vmax = 1.0
        else:
            vmax = np.max(op_im)

        if eval_params['show_map'] == 1:
            # Display the image
            fig, ax = plt.subplots(figsize=(6,3), dpi=334)
            plt.imshow(op_im, vmin=0, vmax=vmax, cmap=cmap, interpolation='nearest')  # Display the image
            plt.axis('off')  # Turn off the axis

            if eval_params['show_context_points'] == 1:
                # Convert the tensor to numpy array if it's not already
                context_locs = context_locs_of_interest.numpy() if isinstance(context_locs_of_interest, torch.Tensor) else context_locs_of_interest
                # Convert context locations directly to image coordinates
                image_x = (context_locs[:, 0] + 1) / 2 * op_im.shape[1]  # Scale longitude from [-1, 1] to [0, image width]
                image_y = (1 - (context_locs[:, 1] + 1) / 2) * op_im.shape[
                    0]  # Scale latitude from [-1, 1] to [0, image height]

                from matplotlib.offsetbox import OffsetImage, AnnotationBbox
                # Plot the context locations
                def getImage(path):
                    return OffsetImage(plt.imread(path), zoom=.04)

                for x0, y0 in zip(image_x, image_y):
                    ab = AnnotationBbox(getImage('black_circle.png'), (x0, y0), frameon=False)
                    ax.add_artist(ab)

            # plt.show(block=True)  # Block execution until the window is closed

        # save image
        if eval_params['use_text']:
            if eval_params['use_image']:
                save_loc = 'images/' + str(eval_params['test_taxa']) + '_' + eval_params['text_type'] + '_' + 'image' + '_' + str(number_of_context_points) +'.png'
            else:
                save_loc = 'images/' + str(eval_params['test_taxa']) + '_' + eval_params['text_type'] + '_' + str(number_of_context_points) + '.png'
        else:
            if eval_params['use_image']:
                save_loc = 'images/' + str(eval_params['test_taxa']) + '_' + 'image' + '_' + str(number_of_context_points) +'.png'
            else:
                save_loc = 'images/' + str(eval_params['test_taxa']) + '_' + str(number_of_context_points) + '.png'


        print(f'Saving image to {save_loc}')
        plt.savefig(save_loc, bbox_inches='tight', pad_inches=0, dpi=334)
    
    return True

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
    info_str = '\nDemo that takes an iNaturalist taxa ID as input and ' + \
               'generates a predicted range for each location on the globe ' + \
               'and saves the output as an image.\n\n' + \
               'Warning: these estimated ranges should be validated before use.'  
               
    parser = argparse.ArgumentParser(usage=info_str)
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--model_path', type=str, default='./experiments/test_fs/model.pt')
    parser.add_argument('--threshold', type=float, default=-1, help='Threshold the range map [0, 1].')
    parser.add_argument('--op_path', type=str, default='./images/', help='Location where the output image will be saved.')
    parser.add_argument('--high_res', action='store_true', help='Generate higher resolution output.')
    parser.add_argument('--disable_ocean_mask', action='store_true', help='Do not use an ocean mask.')
    parser.add_argument('--set_max_cmap_to_1', action='store_true', help='Consistent maximum intensity ouput.')
    parser.add_argument('--show_map', type=int, default=1, help='shows the map if 1')
    parser.add_argument('--show_context_points', type=int, default=1, help='also plots context points if 1')
    parser.add_argument('--test_taxa', type=int, default=3352, help='Taxon ID to test.')
    parser.add_argument('--use_text', action='store_false', help='use text')
    parser.add_argument('--use_image', action='store_true', help='use image')
    parser.add_argument('--text_type', type=str, default='range', help='Type of text for input.')
    parser.add_argument('--context_pt_trial', type=int, nargs='+', default=[0, 1], help='List of context points for trial.')
    eval_params = vars(parser.parse_args())

    if not os.path.isdir(eval_params['op_path']):
        os.makedirs(eval_params['op_path'])
    main(eval_params)

