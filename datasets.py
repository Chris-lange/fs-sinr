import os
import numpy as np
import json
import pandas as pd
from calendar import monthrange
import torch
import utils
import random
import h3
import h3.api.numpy_int
# from h3 import vect
# import h3.api.numpy_int as h3
from torch.nn.utils.rnn import pad_sequence
from functools import partial
import re
from collections import defaultdict

class TransformerDatasetFlexible(torch.utils.data.Dataset):
    def __init__(self, locs, labels, classes, class_to_taxa, text_embs, text_embs_ids, text_embs_keys,
                 image_embs, image_embs_ids, image_embs_keys, input_enc, device, dates=None, input_dim=4, time_dim=0,
                 noise_time=False, transformer_input_enc=None, token_dim=None, jitter=False,
                 data_type_probs='loc:1.0', num_tokens_per_type='loc:50', eval_mode=False):
        # Handle input encoding
        self.input_enc = input_enc
        self.jitter = jitter
        if self.input_enc in ['env', 'sin_cos_env']:
            raster = load_env()
        else:
            raster = None
        self.enc = utils.CoordEncoder(input_enc, raster, input_dim=input_dim)

        # Handle transformer input encoding
        self.transformer_input_enc = transformer_input_enc
        if self.transformer_input_enc in ['env', 'sin_cos_env']:
            transformer_raster = load_env()
        else:
            transformer_raster = None
        if self.transformer_input_enc == 'sinr':
            self.transformer_enc = self.enc
        else:
            self.transformer_enc = utils.CoordEncoder(transformer_input_enc, transformer_raster, input_dim=token_dim)

        # Define properties
        self.locs = locs  # Keep on CPU
        # Normalize locs and make loc_feats
        self.loc_feats = self.enc.encode(self.locs, normalize=True)
        transformer_loc_feats = self.transformer_enc.encode(self.locs, normalize=False)
        self.labels = labels  # Keep on CPU
        self.classes = classes
        self.class_to_taxa = class_to_taxa
        if dates is not None:
            self.dates = dates
            self.enc_time = utils.TimeEncoder()

        # Useful numbers
        self.num_classes = len(np.unique(labels))
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.noise_time = noise_time
        self.token_dim = token_dim
        self.device = device

        # Parse data type probabilities
        self.data_type_probs = self.parse_data_type_probs(data_type_probs)
        self.data_type_combinations = list(self.data_type_probs.keys())
        self.probabilities = list(self.data_type_probs.values())

        # Parse token counts per type
        self.num_tokens_per_type = self.parse_num_tokens_per_type(num_tokens_per_type)

        # Determine which data types are used
        all_data_types = set()
        for combo in self.data_type_combinations:
            all_data_types.update(combo)
        self.use_loc = 'loc' in all_data_types
        self.use_text = 'text' in all_data_types
        self.use_image = 'image' in all_data_types
        self.use_env = 'env' in all_data_types

        # Text embeddings
        if self.use_text:
            self.text_embs = text_embs  # Keep on CPU
            self.text_embs_ids = text_embs_ids.tolist()
            self.text_embs_class_ids = [class_to_taxa.index(taxa) if taxa in class_to_taxa else -1 for taxa in self.text_embs_ids]
            self.text_embs_keys = text_embs_keys

            # Organize text embeddings per class
            class_emb_dict = defaultdict(list)
            for i, (index, description) in enumerate(text_embs_keys):
                class_id = self.text_embs_class_ids[index]
                if class_id == -1:
                    continue
                class_emb_dict[class_id].append(i)
            self.text_class_emb_dict = class_emb_dict
        else:
            self.text_embs = None

        # Image embeddings
        if self.use_image:
            self.image_embs = image_embs  # Keep on CPU
            self.image_embs_ids = image_embs_ids.tolist()
            self.image_embs_class_ids = [class_to_taxa.index(taxa) if taxa in class_to_taxa else -1 for taxa in self.image_embs_ids]
            self.image_embs_keys = image_embs_keys

            # Organize image embeddings per class
            class_emb_dict = defaultdict(list)
            for i, (index, description) in enumerate(image_embs_keys):
                class_id = self.image_embs_class_ids[index]
                if class_id == -1:
                    continue
                class_emb_dict[class_id].append(i)
            self.image_class_emb_dict = class_emb_dict
        else:
            self.image_embs = None

        # Organize location data per class
        if self.use_loc:
            per_class_location_dict = organize_data_by_labels(np.array(labels), np.array(self.locs))
            per_class_loc_feats_dict = organize_data_by_labels(np.array(labels), np.array(transformer_loc_feats))
            for key, value in per_class_location_dict.items():
                per_class_location_dict[key] = torch.tensor(np.array(value))  # Keep on CPU
            for key, value in per_class_loc_feats_dict.items():
                per_class_loc_feats_dict[key] = torch.tensor(np.array(value))  # Keep on CPU
            self.per_class_locs = per_class_location_dict
            self.per_class_loc_feats = per_class_loc_feats_dict

        # Environment features
        if self.use_env:
            env_raster = load_env()
            self.env_enc = utils.CoordEncoder('env', env_raster, input_dim=0)
            env_feats = self.env_enc.encode(self.locs, normalize=False)
            per_class_env_feats_dict = organize_data_by_labels(np.array(labels), np.array(env_feats))
            for key, value in per_class_env_feats_dict.items():
                per_class_env_feats_dict[key] = torch.tensor(np.array(value))  # Keep on CPU
            self.per_class_env_feats = per_class_env_feats_dict
        else:
            self.env_enc = None

        # Sizes of embeddings
        self.text_emb_size = self.text_embs[0].size(0) if self.use_text else 0
        self.image_emb_size = self.image_embs[0].size(0) if self.use_image else 0
        self.env_emb_size = env_feats.shape[1] if self.use_env else 0

        # Evaluation mode
        self.eval_mode = eval_mode
        if eval_mode:
            print('Using eval dataset. One example per class to generate eval embeddings')
            # Select a single example per class
            unique_labels, unique_indices = np.unique(labels, return_index=True)
            self.locs = self.locs[unique_indices]
            self.labels = labels[unique_indices]
            self.loc_feats = self.loc_feats[unique_indices]

    def parse_data_type_probs(self, data_type_probs_str):
        data_type_probs = {}
        total_prob = 0.0
        entries = data_type_probs_str.strip().split(';')
        for entry in entries:
            combo_str, prob_str = entry.strip().split(':')
            prob = float(prob_str)
            data_types = frozenset(combo_str.strip().split('+'))
            data_type_probs[data_types] = prob
            total_prob += prob
        if total_prob < 1.0:
            # Assign remaining probability to default combination (e.g., 'loc')
            remaining_prob = 1.0 - total_prob
            default_combo = frozenset(['loc', 'env', 'text', 'image'])
            if default_combo in data_type_probs:
                data_type_probs[default_combo] += remaining_prob
            else:
                data_type_probs[default_combo] = remaining_prob
        elif total_prob > 1.0:
            raise ValueError('Total probability exceeds 1.0')
        return data_type_probs

    def parse_num_tokens_per_type(self, num_tokens_per_type_str):
        num_tokens_per_type = {}
        entries = num_tokens_per_type_str.strip().split(';')
        for entry in entries:
            data_type, count_str = entry.strip().split(':')
            if '-' in count_str:
                min_count, max_count = map(int, count_str.split('-'))
                num_tokens_per_type[data_type.strip()] = (min_count, max_count)
            else:
                count = int(count_str)
                num_tokens_per_type[data_type.strip()] = count
        return num_tokens_per_type

    def __len__(self):
        return self.loc_feats.shape[0]

    def __getitem__(self, index):
        # Sample a combination of data types based on probabilities
        combo = random.choices(self.data_type_combinations, weights=self.probabilities)[0]

        # Retrieve the feature and class of the original point
        loc_feat = self.loc_feats[index, :]
        loc = self.locs[index, :]
        class_id = self.labels[index]
        class_id_int = class_id.item()

        # Initialize tokens and embeddings
        context_sequence_list = []
        context_mask_list = []
        text_emb_list = []
        image_emb_list = []
        env_emb_list = []

        # Handle text embeddings
        if 'text' in combo and self.use_text:
            if class_id_int in self.text_class_emb_dict:
                if self.eval_mode:
                    text_indices = self.text_class_emb_dict[class_id_int][0]
                else:
                    text_indices = self.text_class_emb_dict[class_id_int]
                num_text_tokens = self.get_num_tokens('text')
                selected_indices = self.sample_indices(text_indices, num_text_tokens)
                for idx in selected_indices:
                    text_emb = self.text_embs[idx]
                    text_emb_list.append(text_emb)
            else:
                pass  # No embeddings for this class

        # Handle image embeddings
        if 'image' in combo and self.use_image:
            if class_id_int in self.image_class_emb_dict:
                if self.eval_mode:
                    image_indices = [self.image_class_emb_dict[class_id_int][0]]
                else:
                    image_indices = self.image_class_emb_dict[class_id_int]
                num_image_tokens = self.get_num_tokens('image')
                selected_indices = self.sample_indices(image_indices, num_image_tokens)
                for idx in selected_indices:
                    image_emb = self.image_embs[idx]
                    image_emb_list.append(image_emb)
            else:
                pass  # No embeddings for this class

        # Handle environment embeddings
        if 'env' in combo and self.use_env:
            all_class_env_feats = self.per_class_env_feats[class_id_int]
            num_env_tokens = self.get_num_tokens('env')
            selected_indices = self.sample_indices(range(len(all_class_env_feats)), num_env_tokens)
            for idx in selected_indices:
                env_emb = all_class_env_feats[idx]
                env_emb_list.append(env_emb)

        # Handle location context tokens
        if 'loc' in combo and self.use_loc:
            all_class_locs = self.per_class_locs[class_id_int]
            all_class_loc_feats = self.per_class_loc_feats[class_id_int]

            # Find the index of the original location
            matches = (all_class_locs == loc).all(dim=1)
            local_index = torch.where(matches)[0]
            if len(local_index) > 1:
                local_index = local_index[0]

            # Exclude the original location's index
            filtered_local_indices = torch.arange(len(all_class_locs)) != local_index

            num_context = self.get_num_tokens('loc')
            available_indices = filtered_local_indices.nonzero().squeeze()
            selected_indices = self.sample_indices(np.array(available_indices), num_context)

            # Get context locations and features
            context_loc_feats = all_class_loc_feats[selected_indices]
            context_locs = all_class_locs[selected_indices]

            if context_loc_feats.dim() == 1:
                context_loc_feats = context_loc_feats.unsqueeze(0)

            if self.jitter:
                noise_std = 0.001
                noise = torch.full_like(context_loc_feats, noise_std)
                context_loc_feats = context_loc_feats + noise

            context_sequence_list.append(context_loc_feats)
        else:
            context_locs = torch.empty((0, loc.size(0)))  # Empty tensor

        # Prepare outputs
        context_sequence = context_sequence_list  # List of tensors
        text_emb = torch.stack(text_emb_list) if text_emb_list else torch.zeros((0, self.text_emb_size))
        image_emb = torch.stack(image_emb_list) if image_emb_list else torch.zeros((0, self.image_emb_size))
        env_emb = torch.stack(env_emb_list) if env_emb_list else torch.zeros((0, self.env_emb_size))

        return loc_feat, loc, class_id, context_sequence, context_locs, text_emb, image_emb, env_emb

    def get_num_tokens(self, data_type):
        if data_type in self.num_tokens_per_type:
            value = self.num_tokens_per_type[data_type]
            if isinstance(value, int):
                return value
            elif isinstance(value, tuple) and len(value) == 2:
                return random.randint(value[0], value[1])
            else:
                raise ValueError(f"Invalid num_tokens_per_type entry for {data_type}")
        else:
            return None  # Use all available tokens

    def sample_indices(self, indices, num_tokens):
        indices = list(indices)
        if num_tokens is None or len(indices) <= num_tokens:
            return indices
        else:
            return random.sample(indices, num_tokens)

    def collate_fn(self, batch):
        # Unzip the batch
        loc_feats, locs, class_ids, context_sequences_list, context_locs_list, text_embs_list, image_embs_list, env_embs_list = zip(*batch)

        # Stack loc_feats and locs
        loc_feats = torch.stack(loc_feats)
        locs = torch.stack(locs)
        class_ids = torch.tensor(class_ids)

        # Prepare context_sequence
        # Flatten context_sequences_list (list of lists of tensors)
        flattened_context_sequences = []
        for seq_list in context_sequences_list:
            if seq_list:
                concatenated_seq = torch.cat(seq_list, dim=0)
                flattened_context_sequences.append(concatenated_seq)
            else:
                flattened_context_sequences.append(torch.zeros((0, self.input_dim)))

        # Pad context sequences
        padded_sequences = pad_sequence(flattened_context_sequences, batch_first=True, padding_value=0.0)

        # Create context mask
        sequence_mask = (padded_sequences.sum(dim=2) == 0)

        # Stack text embeddings and create mask
        text_embs = [embs if embs.dim() == 2 else embs.unsqueeze(0) for embs in text_embs_list]
        padded_text_embs = pad_sequence(text_embs, batch_first=True, padding_value=0.0)
        text_mask = (padded_text_embs.sum(dim=2) == 0)

        # Stack image embeddings and create mask
        image_embs = [embs if embs.dim() == 2 else embs.unsqueeze(0) for embs in image_embs_list]
        padded_image_embs = pad_sequence(image_embs, batch_first=True, padding_value=0.0)
        image_mask = (padded_image_embs.sum(dim=2) == 0)

        # Stack env embeddings and create mask
        env_embs = [embs if embs.dim() == 2 else embs.unsqueeze(0) for embs in env_embs_list]
        padded_env_embs = pad_sequence(env_embs, batch_first=True, padding_value=0.0)
        env_mask = (padded_env_embs.sum(dim=2) == 0)

        return loc_feats, locs, class_ids, padded_sequences, sequence_mask, padded_text_embs, text_mask, padded_image_embs, image_mask, padded_env_embs, env_mask

    def select_text_section(self, text_section):
        # Initialize an empty dictionary to store the result
        text_class_emb_dict = {}
        # Populate the dictionary
        if self.text_embs != None:
            for i, (index, description) in enumerate(self.text_embs_keys):
                # Find the class using the index from the class_list
                class_id = self.text_embs_class_ids[index]
                # Skip this iteration if class_id is -1 - which should correspond to classes not in dataset
                if class_id == -1:
                    continue
                if description != text_section:
                    continue
                # Check if the class_id is already a key in the dictionary
                if class_id not in text_class_emb_dict:
                    # Initialize with empty lists if class_id is not already in the dictionary
                    text_class_emb_dict[class_id] = ([], [])

                # Append the description and the index of embs_keys to the corresponding lists
                text_class_emb_dict[class_id][0].append(i)
                text_class_emb_dict[class_id][1].append(description)
            self.text_class_emb_dict = text_class_emb_dict


def load_env():
    with open('paths.json', 'r') as f:
        paths = json.load(f)
    raster = load_context_feats(os.path.join(paths['env'],'bioclim_elevation_scaled.npy'))
    return raster

def load_context_feats(data_path):
    context_feats = np.load(data_path).astype(np.float32)
    context_feats = torch.from_numpy(context_feats)
    return context_feats

_file_cache = {}
def load_inat_data(ip_file, taxa_of_interest=None):
    if os.path.exists('.datacache.pt'):
        print('\nLoading cached data')
        if '.datacache.pt' not in _file_cache:
            # If not in the cache, read the file and store its content in the cache
            _file_cache['.datacache.pt'] = torch.load('.datacache.pt', weights_only=False)
        locs, taxa, users, dates, years, obs_ids = _file_cache['.datacache.pt']
    else:
        print('\nLoading  ' + ip_file)
        data = pd.read_csv(ip_file)

        # remove outliers
        num_obs = data.shape[0]
        data = data[((data['latitude'] <= 90) & (data['latitude'] >= -90) & (data['longitude'] <= 180) & (data['longitude'] >= -180) )]
        if (num_obs - data.shape[0]) > 0:
            print(num_obs - data.shape[0], 'items filtered due to invalid locations')

        if 'accuracy' in data.columns:
            data.drop(['accuracy'], axis=1, inplace=True)

        if 'positional_accuracy' in data.columns:
            data.drop(['positional_accuracy'], axis=1, inplace=True)

        if 'geoprivacy' in data.columns:
            data.drop(['geoprivacy'], axis=1, inplace=True)

        if 'observed_on' in data.columns:
            data.rename(columns = {'observed_on':'date'}, inplace=True)

        num_obs_orig = data.shape[0]
        data = data.dropna()
        size_diff = num_obs_orig - data.shape[0]
        if size_diff > 0:
            print(size_diff, 'observation(s) with a NaN entry out of' , num_obs_orig, 'removed')

        # keep only taxa of interest:
        if taxa_of_interest is not None:
            num_obs_orig = data.shape[0]
            data = data[data['taxon_id'].isin(taxa_of_interest)]
            print(num_obs_orig - data.shape[0], 'observation(s) out of' , num_obs_orig, 'from different taxa removed')

        print('Number of unique classes {}'.format(np.unique(data['taxon_id'].values).shape[0]))

        locs = np.vstack((data['longitude'].values, data['latitude'].values)).T.astype(np.float32)
        taxa = data['taxon_id'].values.astype(np.int64)

        if 'user_id' in data.columns:
            users = data['user_id'].values.astype(np.int64)
            _, users = np.unique(users, return_inverse=True)
        elif 'observer_id' in data.columns:
            users = data['observer_id'].values.astype(np.int64)
            _, users = np.unique(users, return_inverse=True)
        else:
            users = np.ones(taxa.shape[0], dtype=np.int64)*-1

        # Note - assumes that dates are in format YYYY-MM-DD
        temp = np.array(data['date'], dtype='S10')
        temp = temp.view('S1').reshape((temp.size, -1))
        years = temp[:,:4].view('S4').astype(int)[:,0]
        months = temp[:,5:7].view('S2').astype(int)[:,0]
        days = temp[:,8:10].view('S2').astype(int)[:,0]
        days_per_month = np.cumsum([0] + [monthrange(2018, mm)[1] for mm in range(1, 12)])
        dates  = days_per_month[months-1] + days-1
        dates  = np.round((dates) / 364.0, 4).astype(np.float32)
        if 'id' in data.columns:
            obs_ids = data['id'].values
        elif 'observation_uuid' in data.columns:
            obs_ids = data['observation_uuid'].values
        torch.save((locs, taxa, users, dates, years, obs_ids), '.datacache.pt')

    return locs, taxa, users, dates, years, obs_ids

def choose_aux_species(current_species, num_aux_species, aux_species_seed, taxa_file):
    if num_aux_species == 0:
        return []
    with open('paths.json', 'r') as f:
        paths = json.load(f)
    data_dir = paths['train']
    taxa_file = os.path.join(data_dir, taxa_file)
    with open(taxa_file, 'r') as f:
        inat_large_metadata = json.load(f)
    aux_species_candidates = [x['taxon_id'] for x in inat_large_metadata]
    aux_species_candidates = np.setdiff1d(aux_species_candidates, current_species)
    print(f'choosing {num_aux_species} species to add from {len(aux_species_candidates)} candidates')
    rng = np.random.default_rng(aux_species_seed)
    idx_rand_aux_species = rng.permutation(len(aux_species_candidates))
    aux_species = list(aux_species_candidates[idx_rand_aux_species[:num_aux_species]])
    return aux_species

def get_taxa_of_interest(species_set='all', num_aux_species=0, aux_species_seed=123, taxa_file=None, taxa_file_snt=None):
    if species_set == 'all':
        return None
    if species_set == 'snt_birds':
        assert taxa_file_snt is not None
        with open(taxa_file_snt, 'r') as f: #
            taxa_subsets = json.load(f)
        taxa_of_interest = list(taxa_subsets['snt_birds'])
    else:
        raise NotImplementedError
    # optionally add some other species back in:
    aux_species = choose_aux_species(taxa_of_interest, num_aux_species, aux_species_seed, taxa_file)
    taxa_of_interest.extend(aux_species)
    return taxa_of_interest

def get_idx_subsample_observations(labels, hard_cap=-1, hard_cap_seed=123, subset=None, subset_cap=-1):
    if hard_cap == -1:
        if subset_cap != -1:
            raise NotImplementedError('subset_cap set but not hard_cap')
        return np.arange(len(labels))
    print(f'subsampling (up to) {hard_cap} per class for the training set')
    ids, counts = np.unique(labels, return_counts=True)
    count_ind = np.cumsum(counts)
    count_ind[1:] = count_ind[:-1]
    count_ind[0] = 0
    ss_rng = np.random.default_rng(hard_cap_seed)
    idx_rand = ss_rng.permutation(len(labels))

    ordered_inds = np.argsort(labels[idx_rand], kind='stable')
    caps = hard_cap + np.zeros_like(counts)
    if subset is not None and subset_cap != -1:
        caps[subset] = subset_cap
    idx_ss = idx_rand[np.concatenate([ordered_inds[i:i+min(limit, cap)] for i, limit, cap in zip(count_ind, counts, caps)])]
    print(f'final training set size: {len(idx_ss)}')
    return idx_ss

# for creating the per class dicts
def organize_data_by_labels(labels, locs):
    label_dict = {}  # Initialize an empty dictionary
    for label, loc in zip(labels, locs):  # Loop through labels and locations
        if label in label_dict:
            label_dict[label].append(loc)  # Append the location
        else:
            label_dict[label] = [loc]  # Start a new list with the tuple of location
    return label_dict


dataset_classes = {
                   'flexible': TransformerDatasetFlexible,
                   'eval_flexible': TransformerDatasetFlexible
                   }

def get_dataset_class(dataset_name):
    # First, try to get the class directly from the dictionary
    if dataset_name in dataset_classes:
        return dataset_classes[dataset_name]

def get_train_data(params):

    if 'eval' in params['dataset']:
        print('Creating eval dataset')
        with open('paths.json', 'r') as f:
            paths = json.load(f)
            eval_data_path = os.path.join(paths['data'], 'positive_eval_data.npz')
        thing = np.load(eval_data_path, allow_pickle=True)
        locs = torch.from_numpy(thing['locs'])
        labels = torch.from_numpy(thing['labels'])
        classes = thing['classes'].item()
        class_to_taxa = list(thing['class_to_taxa'])
        dates = None

        for key in ['add_location_noise', 'variable_context_length']:
            if key not in params:
                params[key] = False

        # Load the text embeddings
        text_embs_dict = torch.load(paths['eval_text_embs'], map_location='cpu', weights_only=False)
        text_embs = text_embs_dict['data']
        text_embs_ids = text_embs_dict['taxon_id']
        text_embs_keys = text_embs_dict['keys']

        # Load the image embeddings
        image_embs_dict = torch.load(paths['image_embs'], map_location='cpu', weights_only=False)
        image_embs = image_embs_dict['data']
        image_embs_ids = image_embs_dict['taxon_id']
        image_embs_keys = image_embs_dict['keys']

        ds = TransformerDatasetFlexible(locs, labels, classes, class_to_taxa, text_embs=text_embs,
                                        text_embs_ids=text_embs_ids, text_embs_keys=text_embs_keys,
                                        image_embs=image_embs, image_embs_ids=image_embs_ids,
                                        image_embs_keys=image_embs_keys, input_enc=params['input_enc'],
                                        device=params['device'], dates=dates, input_dim=params['input_dim'],
                                        time_dim=params['input_time_dim'], noise_time=params['noise_time'],
                                        transformer_input_enc=params['transformer_input_enc'],
                                        token_dim=params['species_dim'], jitter=params['add_location_noise'],
                                        data_type_probs=params['data_probs'],
                                        num_tokens_per_type=params['tokens_per_type'],
                                        eval_mode=True)

    else:
        with open('paths.json', 'r') as f:
            paths = json.load(f)
        data_dir = paths['train']
        obs_file = os.path.join(data_dir, params['obs_file'])
        taxa_file = os.path.join(data_dir, params['taxa_file'])
        taxa_file_snt = os.path.join(data_dir, 'taxa_subsets.json')

        taxa_of_interest = get_taxa_of_interest(
            params['species_set'], params['num_aux_species'], params['aux_species_seed'],
            params['taxa_file'], taxa_file_snt
        )

        locs, labels, _, dates, _, _ = load_inat_data(obs_file, taxa_of_interest)
        if params['zero_shot']:
            with open('paths.json', 'r') as f:
                paths = json.load(f)
                eval_taxa_path = os.path.join(paths['data'], 'eval_taxa_list.npy')
            taxa = np.load(eval_taxa_path, allow_pickle=True)
            mask = labels != taxa[0]
            for i in range(1, len(taxa)):
                mask &= (labels != taxa[i])
            locs = locs[mask]
            dates = dates[mask]
            labels = labels[mask]
        unique_taxa, class_ids = np.unique(labels, return_inverse=True)
        class_to_taxa = unique_taxa.tolist()

        # load class names
        class_info_file = json.load(open(taxa_file, 'r'))
        class_names_file = [cc['latin_name'] for cc in class_info_file]
        taxa_ids_file = [cc['taxon_id'] for cc in class_info_file]
        classes = dict(zip(taxa_ids_file, class_names_file))

        subset = None
        if params['subset_cap_name'] is not None:
            if params['subset_cap_name'] == 'iucn':
                with open('paths.json', 'r') as f:
                    paths = json.load(f)
                with open(os.path.join(paths['iucn'], 'iucn_res_5.json'), 'r') as f:
                    data = json.load(f)
                taxa = [int(tt) for tt in data['taxa_presence'].keys()]
                # get classes to eval
                subset = np.zeros((len(taxa),), dtype=int)
                for tt_id, tt in enumerate(taxa):
                    class_of_interest = np.where(np.array(class_to_taxa) == tt)[0]
                    if len(class_of_interest) != 0:
                        subset[tt_id] = class_of_interest
            else:
                raise NotImplementedError(f'Uknown subset name: {params["subset_cap_name"]}')

        idx_ss = get_idx_subsample_observations(
            labels, params['hard_cap_num_per_class'], params['hard_cap_seed'], subset,
            params['subset_cap_num_per_class']
        )

        locs = torch.from_numpy(np.array(locs)[idx_ss])  # convert to Tensor
        labels = torch.from_numpy(np.array(class_ids)[idx_ss])
        dates = 364/365*torch.from_numpy(np.array(dates)[idx_ss]) if params['input_time'] else None

        # Load the text embeddings
        text_embs_dict = torch.load(paths['text_embs'], map_location='cpu', weights_only=False)
        text_embs = text_embs_dict['data']
        text_embs_ids = text_embs_dict['taxon_id']
        text_embs_keys = text_embs_dict['keys']

        # Load the image embeddings
        image_embs_dict = torch.load(paths['image_embs'], map_location='cpu', weights_only=False)
        image_embs = image_embs_dict['data']
        image_embs_ids = image_embs_dict['taxon_id']
        image_embs_keys = image_embs_dict['keys']

        ds = TransformerDatasetFlexible(locs, labels, classes, class_to_taxa, text_embs=text_embs,
                                        text_embs_ids=text_embs_ids, text_embs_keys=text_embs_keys,
                                        image_embs=image_embs, image_embs_ids=image_embs_ids,
                                        image_embs_keys=image_embs_keys, input_enc=params['input_enc'],
                                        device=params['device'], dates=dates, input_dim=params['input_dim'],
                                        time_dim=params['input_time_dim'], noise_time=params['noise_time'],
                                        transformer_input_enc=params['transformer_input_enc'],
                                        token_dim=params['species_dim'], jitter=params['add_location_noise'],
                                        data_type_probs=params['data_probs'],
                                        num_tokens_per_type=params['tokens_per_type'],
                                        eval_mode=False)

    return ds
