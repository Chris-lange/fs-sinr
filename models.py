import torch
import torch.utils.data
import torch.nn as nn

def get_model(params, inference_only=False):
    return MultiInputModel_2(num_inputs=params['input_dim'] + params['input_time_dim'] + (20 if 'env' in params['input_enc'] and 'contrastive' not in params['input_enc'] else 0) + (1 if params['noise_time'] else 0),
                           num_filts=params['num_filts'], num_classes=params['num_classes'] + (20 if 'env' in params['loss'] else 0),
                           depth=params['depth'], ema_factor=params['ema_factor'], nhead=params['num_heads'], num_encoder_layers=params['species_enc_depth'],
                           dim_feedforward=params['species_filts'], dropout=params['transformer_dropout'],
                           batch_first=True, token_dim=(params['species_dim'] + (20 if 'env' in params['transformer_input_enc'] else 0)),
                           sinr_inputs=True if 'sinr' in params['transformer_input_enc'] else False,
                           register=params['use_register'], use_pretrained_sinr=params['use_pretrained_sinr'],
                           use_pretrained_env_sinr=params['use_pretrained_env_sinr'],
                           use_pretrained_image_sinr=params['use_pretrained_image_sinr'],
                           freeze_sinr=params['freeze_sinr'], pretrained_loc=params['pretrained_loc'],
                           pretrained_env_loc=params['pretrained_env_loc'], pretrained_image_loc=params['pretrained_image_loc'],
                           text_inputs=params['use_text_inputs'], image_inputs=params['use_image_inputs'], env_inputs=params['use_env_inputs'],
                           class_token_transformation=params['class_token_transformation'], primary_env=params['primary_env'])

class MultiInputModel_2(nn.Module):
    def __init__(self, num_inputs, num_filts, num_classes, depth=4, nonlin='relu', lowrank=0, ema_factor=0.1,
                 nhead=8, num_encoder_layers=4, dim_feedforward=2048, dropout=0.1, batch_first=True, token_dim=256,
                 sinr_inputs=False, register=False, use_pretrained_sinr=False, use_pretrained_env_sinr=False,
                 use_pretrained_image_sinr=False, freeze_sinr=False, pretrained_loc='', pretrained_env_loc='',
                 pretrained_image_loc = '', text_inputs=False, image_inputs=False, env_inputs=False,
                 class_token_transformation='identity', primary_env=False):
        super(MultiInputModel_2, self).__init__()

        self.headless_model = HeadlessSINR(num_inputs, num_filts, depth, nonlin, lowrank, dropout_p=dropout)
        self.ema_factor = ema_factor
        self.class_token_transformation = class_token_transformation
        self.primary_env = primary_env

        # Load pretrained state_dict if use_pretrained_sinr is set to True
        if use_pretrained_sinr:
            pretrained_state_dict = torch.load(pretrained_loc, map_location=torch.device('cpu'), weights_only=False)['state_dict']
            filtered_state_dict = {k: v for k, v in pretrained_state_dict.items() if not k.startswith('class_emb')}
            self.headless_model.load_state_dict(filtered_state_dict, strict=False)
            print(f'Using pretrained sinr from {pretrained_loc}')

        # Freeze the SINR model if freeze_sinr is set to True
        if freeze_sinr:
            for param in self.headless_model.parameters():
                param.requires_grad = False
            print("Freezing SINR model parameters")

        self.transformer_model = TransformerEncoderModel(d_model=token_dim,
                                                         nhead=nhead,
                                                         num_encoder_layers=num_encoder_layers,
                                                         dim_feedforward=dim_feedforward,
                                                         dropout=dropout,
                                                         batch_first=batch_first,
                                                         output_dim=num_filts)

        self.ema_embeddings = nn.Embedding(num_embeddings=num_classes, embedding_dim=num_filts)
        self.eval_embeddings = nn.Embedding(num_embeddings=num_classes, embedding_dim=num_filts)
        self.ema_embeddings.weight.requires_grad = False
        self.eval_embeddings.weight.requires_grad = False
        self.num_filts = num_filts
        self.token_dim = token_dim
        self.sinr_inputs = sinr_inputs
        if self.sinr_inputs:
            if self.num_filts != self.token_dim and self.class_token_transformation == 'identity':
                raise ValueError("If using sinr inputs to transformer with identity class token transformation"
                                 " then token_dim of transformer must be equal to num_filts of sinr model")

        # Add a class token
        self.class_token = nn.Parameter(torch.empty(1, self.token_dim))
        nn.init.xavier_uniform_(self.class_token)

        if register:
            # Add a register token initialized with Xavier uniform initialization
            self.register = nn.Parameter(torch.empty(1, self.token_dim))
            nn.init.xavier_uniform_(self.register)
        else:
            self.register = None

        self.text_inputs = text_inputs
        if self.text_inputs:
            print("Initializing text model")
            self.text_model = HeadlessSINR(num_inputs=4096, num_filts=512, depth=2, nonlin=nonlin,
                                           lowrank=token_dim, dropout_p=dropout)

        else:
            self.text_model = None

        self.image_inputs = image_inputs
        if self.image_inputs:
            print("Initializing image model")
            self.image_model = HeadlessSINR(num_inputs=1024, num_filts=512, depth=2, nonlin=nonlin,
                                            lowrank=token_dim, dropout_p=dropout)
            if use_pretrained_image_sinr:
                pretrained_image_state_dict = torch.load(pretrained_image_loc, map_location=torch.device('cpu'), weights_only=False)['state_dict']
                filtered_image_state_dict = {k: v for k, v in pretrained_image_state_dict.items() if not k.startswith('class_emb')}
                self.image_model.load_state_dict(filtered_image_state_dict, strict=False)
                print(f'Using pretrained image sinr from {pretrained_image_loc}')
        else:
            self.image_model = None

        self.env_inputs = env_inputs
        if self.env_inputs:
            print("Initializing environment model")
            # self.env_model = HeadlessSINR(num_inputs=20, num_filts=512, depth=2, nonlin=nonlin,
            #                               lowrank=token_dim, dropout_p=dropout)
            self.env_model = HeadlessSINR(num_inputs, num_filts, depth, nonlin, lowrank, dropout_p=dropout)
            if use_pretrained_env_sinr:
                pretrained_env_state_dict = torch.load(pretrained_env_loc, map_location=torch.device('cpu'), weights_only=False)['state_dict']
                filtered_env_state_dict = {k: v for k, v in pretrained_env_state_dict.items() if not k.startswith('class_emb')}
                self.env_model.load_state_dict(filtered_env_state_dict, strict=False)
                print(f'Using pretrained env sinr from {pretrained_env_loc}')
        else:
            self.env_model = None

        # Type-specific embeddings for class, register, location, text, image, and env tokens
        self.class_type_embedding = nn.Parameter(torch.empty(1, self.token_dim))
        nn.init.xavier_uniform_(self.class_type_embedding)
        if register:
            self.register_type_embedding = nn.Parameter(torch.empty(1, self.token_dim))
            nn.init.xavier_uniform_(self.register_type_embedding)
        self.location_type_embedding = nn.Parameter(torch.empty(1, self.token_dim))
        nn.init.xavier_uniform_(self.location_type_embedding)
        if text_inputs:
            self.text_type_embedding = nn.Parameter(torch.empty(1, self.token_dim))
            nn.init.xavier_uniform_(self.text_type_embedding)
        if image_inputs:
            self.image_type_embedding = nn.Parameter(torch.empty(1, self.token_dim))
            nn.init.xavier_uniform_(self.image_type_embedding)
        if env_inputs:
            self.env_type_embedding = nn.Parameter(torch.empty(1, self.token_dim))
            nn.init.xavier_uniform_(self.env_type_embedding)

        # Instantiate the class token transformation module
        if class_token_transformation == 'identity':
            self.class_token_transform = Identity(token_dim, num_filts)
            self.class_token_env_transform = Identity(token_dim, num_filts)
        elif class_token_transformation == 'linear':
            self.class_token_transform = LinearTransformation(token_dim, num_filts)
            self.class_token_env_transform =  LinearTransformation(token_dim, num_filts)
        elif class_token_transformation == 'single_layer_nn':
            self.class_token_transform = SingleLayerNN(token_dim, num_filts, dropout_p=dropout)
            self.class_token_env_transform = SingleLayerNN(token_dim, num_filts, dropout_p=dropout)
        elif class_token_transformation == 'two_layer_nn':
            self.class_token_transform = TwoLayerNN(token_dim, num_filts, dropout_p=dropout)
            self.class_token_env_transform = TwoLayerNN(token_dim, num_filts, dropout_p=dropout)
        elif class_token_transformation == 'sinr':
            self.class_token_transform = HeadlessSINR(token_dim, num_filts, depth, nonlin, lowrank, dropout_p=dropout)
            self.class_token_env_transform = HeadlessSINR(token_dim, num_filts, depth, nonlin, lowrank, dropout_p=dropout)
        else:
            raise ValueError(f"Unknown class_token_transformation: {class_token_transformation}")

        # Instantiate the class token transformation module
        if class_token_transformation == 'identity':
            self.class_token_transform = Identity(token_dim, num_filts)
        elif class_token_transformation == 'linear':
            self.class_token_transform = LinearTransformation(token_dim, num_filts)
        elif class_token_transformation == 'single_layer_nn':
            self.class_token_transform = SingleLayerNN(token_dim, num_filts, dropout_p=dropout)
        elif class_token_transformation == 'two_layer_nn':
            self.class_token_transform = TwoLayerNN(token_dim, num_filts, dropout_p=dropout)
        elif class_token_transformation == 'sinr':
            self.class_token_transform = HeadlessSINR(token_dim, num_filts, depth, nonlin, lowrank, dropout_p=dropout)
        else:
            raise ValueError(f"Unknown class_token_transformation: {class_token_transformation}")

    def forward(self, x, context_sequence, context_mask, class_ids=None, return_feats=False,
                return_class_embeddings=False, class_of_interest=None, use_eval_embeddings=False, text_emb=None,
                image_emb=None, env_emb=None, mask_prob=0.0):

        if self.primary_env:
            feature_embeddings = self.env_model(env_emb)
        else:
            # Process input through the headless model to get feature embeddings
            feature_embeddings = self.headless_model(x)

        if return_feats:
            return feature_embeddings

        if context_sequence.dim() == 2:
            context_sequence = context_sequence.unsqueeze(0)  # Add batch dimension if missing

        # Process context_sequence through headless_model if sinr_inputs is True
        if self.sinr_inputs:
            context_sequence = self.headless_model(context_sequence)

        # Add type-specific embedding to each location token
        context_sequence += self.location_type_embedding

        batch_size = context_sequence.size(0)

        # Initialize lists for tokens and masks
        tokens = []
        masks = []

        # Process class token
        class_token_expanded = self.class_token.expand(batch_size, -1, -1) + self.class_type_embedding
        tokens.append(class_token_expanded)
        # The class token is always present, so mask is False (i.e., not masked out)
        class_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=context_sequence.device)
        masks.append(class_mask)

        # Process register token if present
        if self.register is not None:
            register_expanded = self.register.expand(batch_size, -1, -1) + self.register_type_embedding
            tokens.append(register_expanded)
            register_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=context_sequence.device)
            masks.append(register_mask)

        # Process text embeddings (supporting multiple embeddings per batch item)
        if (self.text_inputs and (text_emb is not None) and torch.is_tensor(text_emb) and
                all(dim > 0 for dim in text_emb.shape)):
            # text_emb shape: [batch_size, num_text_embeds, text_input_dim]
            batch_size, num_text_embeds, _ = text_emb.size()
            # Create mask for text embeddings where all values are zero
            text_mask = (text_emb.sum(dim=2) == 0)  # Shape: [batch_size, num_text_embeds]
            # Flatten embeddings to process through the model
            text_emb_flat = text_emb.view(-1,
                                          text_emb.size(-1))  # Shape: [batch_size * num_text_embeds, text_input_dim]
            # Process through text model
            text_emb_processed = self.text_model(text_emb_flat)  # Shape: [batch_size * num_text_embeds, token_dim]
            # Reshape back to [batch_size, num_text_embeds, token_dim]
            text_emb_processed = text_emb_processed.view(batch_size, num_text_embeds, -1)
            # Add type-specific embedding
            text_emb_processed += self.text_type_embedding
            # Zero out embeddings where mask is True
            text_emb_processed[text_mask.unsqueeze(2).expand_as(text_emb_processed)] = 0
            tokens.append(text_emb_processed)
            masks.append(text_mask)

        # Process image embeddings (supporting multiple embeddings per batch item)
        if self.image_inputs and (image_emb is not None):
            batch_size, num_image_embeds, _ = image_emb.size()
            image_mask = (image_emb.sum(dim=2) == 0)
            image_emb_flat = image_emb.view(-1, image_emb.size(-1))
            image_emb_processed = self.image_model(image_emb_flat)
            image_emb_processed = image_emb_processed.view(batch_size, num_image_embeds, -1)
            image_emb_processed += self.image_type_embedding
            image_emb_processed[image_mask.unsqueeze(2).expand_as(image_emb_processed)] = 0
            tokens.append(image_emb_processed)
            masks.append(image_mask)

        # Process environment embeddings (treated like context_sequence)
        if self.env_inputs and (env_emb is not None):
            # If sinr_inputs is True, process through headless_model
            env_emb = self.env_model(env_emb)
            env_emb += self.env_type_embedding
            # Assume env_emb shape: [batch_size, env_seq_len, emb_dim]
            # Create mask for env embeddings
            env_mask = (env_emb.sum(dim=2) == 0)  # Shape: [batch_size, env_seq_len]
            tokens.append(env_emb)
            masks.append(env_mask)

        # Process location tokens (context_sequence)
        tokens.append(context_sequence)
        masks.append(context_mask)

        # Concatenate all tokens and masks
        context_sequence = torch.cat(tokens, dim=1)  # Shape: [batch_size, total_seq_len, token_dim]
        context_mask = torch.cat(masks, dim=1)  # Shape: [batch_size, total_seq_len]

        # Apply random masking to tokens (except class and register tokens)
        if mask_prob > 0.0 and self.training:
            # Number of special tokens (class token and possibly register token)
            num_special_tokens = 1 + (1 if self.register is not None else 0)
            # Generate random masks
            random_mask = torch.rand(context_sequence.size(0), context_sequence.size(1),
                                     device=context_sequence.device) < mask_prob
            # Do not mask class and register tokens
            random_mask[:, :num_special_tokens] = False
            # Update context_mask with random masks
            context_mask = context_mask | random_mask

        # Proceed with the transformer model
        if not use_eval_embeddings:
            if class_of_interest is None:
                # Pass the sequence through the transformer
                class_token_output = self.transformer_model(src=context_sequence, src_key_padding_mask=context_mask)
                # Transform class token output to get class embeddings
                if self.primary_env:
                    class_embeddings = self.class_token_env_transform(class_token_output)  # Shape: [batch_size, num_filts]
                else:
                    class_embeddings = self.class_token_transform(class_token_output)  # Shape: [batch_size, num_filts]

                if return_class_embeddings:
                    return class_embeddings
                else:
                    # Update EMA embeddings for these class IDs
                    with torch.no_grad():
                        if self.training:
                            self.update_ema_embeddings(class_ids, class_embeddings)

                    # Compute logits and probabilities
                    logits = feature_embeddings @ class_embeddings.T
                    probabilities = torch.sigmoid(logits)
                    return probabilities
            else:
                device = self.ema_embeddings.weight.device
                class_of_interest_tensor = torch.tensor([class_of_interest]).to(device)
                class_embedding = self.get_ema_embeddings(class_of_interest_tensor)
                print(f'Using EMA estimate for class {class_of_interest}')
                if return_class_embeddings:
                    return class_embedding
                else:
                    logits = feature_embeddings @ class_embedding.T
                    probabilities = torch.sigmoid(logits).squeeze()
                    return probabilities
        else:
            self.eval()
            if not hasattr(self, 'eval_embeddings'):
                print('No Eval Embeddings for this class!')
                self.eval_embeddings = self.ema_embeddings
            if class_of_interest is None:
                class_token_output = self.transformer_model(src=context_sequence, src_key_padding_mask=context_mask)
                if self.primary_env:
                    class_embeddings = self.class_token_env_transform(class_token_output)
                else:
                    class_embeddings = self.class_token_transform(class_token_output)
                self.generate_eval_embeddings(class_ids, class_embeddings)

                logits = feature_embeddings @ class_embeddings.T
                probabilities = torch.sigmoid(logits)
                return probabilities
            else:
                device = self.ema_embeddings.weight.device
                class_of_interest_tensor = torch.tensor([class_of_interest]).to(device)
                class_embedding = self.get_eval_embeddings(class_of_interest_tensor)
                print(f'Using eval embedding for class {class_of_interest}')
                if return_class_embeddings:
                    return class_embedding
                else:
                    logits = feature_embeddings @ class_embedding.T
                    probabilities = torch.sigmoid(logits).squeeze()
                    return probabilities

    # Rest of the methods remain unchanged
    def init_eval_embeddings(self, num_classes):
        self.eval_embeddings = nn.Embedding(num_embeddings=num_classes, embedding_dim=self.num_filts)
        nn.init.xavier_uniform_(self.eval_embeddings.weight)

    def get_ema_embeddings(self, class_ids):
        return self.ema_embeddings(class_ids)

    def get_eval_embeddings(self, class_ids):
        return self.eval_embeddings(class_ids)

    def update_ema_embeddings(self, class_ids, current_embeddings):
        if self.training:
            unique_class_ids, inverse_indices, counts = class_ids.unique(return_counts=True, return_inverse=True)
            ema_current = self.ema_embeddings(unique_class_ids)
            current_sum = torch.zeros_like(ema_current)
            current_sum.index_add_(0, inverse_indices, current_embeddings)
            current_avg = current_sum / counts.unsqueeze(1)
            ema_new = self.ema_factor * current_avg + (1 - self.ema_factor) * ema_current
            self.ema_embeddings.weight.data[unique_class_ids] = ema_new.detach()

    def generate_eval_embeddings(self, class_ids, current_embeddings):
        self.eval_embeddings.weight.data[class_ids, :] = current_embeddings.detach()

    def embedding_forward(self, x, class_ids=None, return_feats=False, return_class_embeddings=False,
                          class_of_interest=None, eval=False):
        if self.primary_env:
            feature_embeddings = self.env_model(x)
            print("Here we are expecting env data as our input x - this is different from the regular forward for this"
                  " model, so watch out! Something is probably going to break")
        else:
            feature_embeddings = self.headless_model(x)

        if return_feats:
            return feature_embeddings
        else:
            if class_of_interest is None:
                if not eval:
                    class_embeddings = self.get_ema_embeddings(class_ids=class_ids)
                else:
                    class_embeddings = self.get_eval_embeddings(class_ids=class_ids)
                if return_class_embeddings:
                    return class_embeddings
                else:
                    logits = feature_embeddings @ class_embeddings.T
                    probabilities = torch.sigmoid(logits)
                    return probabilities
            else:
                device = self.ema_embeddings.weight.device
                class_of_interest_tensor = torch.tensor([class_of_interest]).to(device)
                if not eval:
                    class_embedding = self.get_ema_embeddings(class_of_interest_tensor)
                    print(f'Using EMA estimate for class {class_of_interest}')
                else:
                    class_embedding = self.get_eval_embeddings(class_of_interest_tensor)
                    print(f'Using eval estimate for class {class_of_interest}')
                if return_class_embeddings:
                    return class_embedding
                else:
                    logits = feature_embeddings @ class_embedding.T
                    probabilities = torch.sigmoid(logits).squeeze()
                    return probabilities

    def zero_shot(self, locs_enc, image_emb=None, text_emb=None):
        # create reps for the provided locations
        device = self.class_token.device
        if self.primary_env:
            feature_embeddings = self.env_model(locs_enc)
        else:
            # Process input through the headless model to get feature embeddings
            feature_embeddings = self.headless_model(locs_enc)
        # create context_sequence and mask
        # Initialize lists for tokens and masks
        tokens = []
        masks = []

        # Process class token
        class_token_expanded = self.class_token.expand(1, -1, -1) + self.class_type_embedding
        tokens.append(class_token_expanded)
        # The class token is always present, so mask is False (i.e., not masked out)
        class_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
        masks.append(class_mask)

        # Process register token if present
        if self.register is not None:
            register_expanded = self.register.expand(1, -1, -1) + self.register_type_embedding
            tokens.append(register_expanded)
            register_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
            masks.append(register_mask)

        # Process text embeddings (supporting multiple embeddings per batch item)
        if text_emb is not None:
            # Create mask for text embeddings where all values are zero
            text_emb = text_emb.to(device)
            text_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
            # Flatten embeddings to process through the model
            # text_emb_flat = text_emb.view(-1,
            #                               text_emb.size(-1))  # Shape: [batch_size * num_text_embeds, text_input_dim]
            # Process through text model
            text_emb_processed = self.text_model(text_emb)  # Shape: [batch_size * num_text_embeds, token_dim]
            text_emb_processed = text_emb_processed.view(1, 1, -1)
            text_emb_processed += self.text_type_embedding
            tokens.append(text_emb_processed)
            masks.append(text_mask)

        # Process image embeddings (supporting multiple embeddings per batch item)
        if image_emb is not None:
            image_emb = image_emb.to(device)
            image_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
            # image_emb_flat = image_emb.view(-1, image_emb.size(-1))
            image_emb_processed = self.image_model(image_emb)
            image_emb_processed = image_emb_processed.view(1, 1, -1)
            image_emb_processed += self.image_type_embedding
            tokens.append(image_emb_processed)
            masks.append(image_mask)

        # Concatenate all tokens and masks
        context_sequence = torch.cat(tokens, dim=1)  # Shape: [batch_size, total_seq_len, token_dim]
        context_mask = torch.cat(masks, dim=1)  # Shape: [batch_size, total_seq_len]

        # Pass the sequence through the transformer
        class_token_output = self.transformer_model(src=context_sequence, src_key_padding_mask=context_mask)

        if self.primary_env:
            class_embeddings = self.class_token_env_transform(class_token_output)  # Shape: [batch_size, num_filts]
        else:
            class_embeddings = self.class_token_transform(class_token_output)  # Shape: [batch_size, num_filts]

        # Compute logits and probabilities
        logits = feature_embeddings @ class_embeddings.T
        probabilities = torch.sigmoid(logits)

        return probabilities

class Identity(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Identity, self).__init__()
        # No parameters needed for identity transformation

    def forward(self, x):
        return x

class LinearTransformation(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(LinearTransformation, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        return self.linear(x)

class SingleLayerNN(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_p=0.1, bias=True):
        super(SingleLayerNN, self).__init__()
        hidden_dim = (in_dim + out_dim) // 2  # Choose an appropriate hidden dimension
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=bias),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, out_dim, bias=bias)
        )

    def forward(self, x):
        return self.net(x)

class TwoLayerNN(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_p=0.1, bias=True):
        super(TwoLayerNN, self).__init__()
        hidden_dim = (in_dim + out_dim) // 2  # Choose an appropriate hidden dimension
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=bias),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim, bias=bias),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, out_dim, bias=bias)
        )

    def forward(self, x):
        return self.net(x)

class ResLayer(nn.Module):
    def __init__(self, linear_size, activation=nn.ReLU, p=0.5):
        super(ResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = activation()
        self.nonlin2 = activation()
        self.dropout1 = nn.Dropout(p=p)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y
        return out

class HeadlessSINR(nn.Module):
    def __init__(self, num_inputs, num_filts, depth=4, nonlin='relu', lowrank=0, dropout_p=0.5):
        super(HeadlessSINR, self).__init__()
        self.inc_bias = False
        self.low_rank_feats = None
        if lowrank < num_filts and lowrank != 0:
            l1 = nn.Linear(num_filts if depth != -1 else num_inputs, lowrank, bias=self.inc_bias)
            self.low_rank_feats = l1
        # else:
        #     self.class_emb = nn.Linear(num_filts if depth != -1 else num_inputs, num_classes, bias=self.inc_bias)
        if nonlin == 'relu':
            activation = nn.ReLU
        elif nonlin == 'silu':
            activation = nn.SiLU
        else:
            raise NotImplementedError('Invalid nonlinearity specified.')

        # Create the layers list for feature extraction
        layers = []
        if depth != -1:
            layers.append(nn.Linear(num_inputs, num_filts))
            layers.append(activation())
            for i in range(depth):
                layers.append(ResLayer(num_filts, activation=activation, p=dropout_p))
        else:
            layers.append(nn.Identity())
        # Include low-rank features in the sequential model if it is defined
        if self.low_rank_feats:
            # Apply initial layers then low-rank features
            layers.append(self.low_rank_feats)
        # Set up the features as a sequential model
        self.feats = nn.Sequential(*layers)

    def forward(self, x):
        loc_emb = self.feats(x)
        return loc_emb

class TransformerEncoderModel(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=4, dim_feedforward=2048, dropout=0.1, activation='relu',
                 batch_first=True, output_dim=256): # BATCH FIRST MIGHT HAVE TO CHANGE
        super(TransformerEncoderModel, self).__init__()
        self.input_layer_norm = nn.LayerNorm(normalized_shape=d_model)
        # Create an encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first
        )

        # Stack the encoder layers into an encoder module
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )

        # Example output layer (modify according to your needs)
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: the sequence to the encoder (shape: [seq_length, batch_size, d_model])
            src_mask: the mask for the src sequence (shape: [seq_length, seq_length])
            src_key_padding_mask: the mask for the padding tokens (shape: [batch_size, seq_length])

        Returns:
            output of the transformer encoder
        """
        # Pass the input through the transformer encoder
        encoder_input = self.input_layer_norm(src)
        encoder_output = self.transformer_encoder(encoder_input, src_key_padding_mask=src_key_padding_mask, mask=src_mask)

        # # Pass the encoder output through the output layer
        # output = self.output_layer(encoder_output)

        # Assuming the class token is the first in the sequence
        # batch_first so we have (batch, sequence, dim)
        if encoder_output.ndim == 2:
            # in situations where we don't have a batch
            encoder_output = encoder_output.unsqueeze(0)

        class_token_embedding = encoder_output[:, 0, :]

        output = self.output_layer(class_token_embedding)  # Process only the class token embedding
        return output