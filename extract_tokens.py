from transformers import LlamaModel, AutoTokenizer
import transformers
import torch
import csv
from tqdm import tqdm
import time
from gritlm import GritLM
import json
import numpy as np
import os

@torch.no_grad()
def extract_grit(fpath='./data/inat_taxa_info.csv', mode='wiki'):
    assert mode in ['wiki', 'gpt']
    with open('gpt_data.json', 'r') as f:
        data = json.load(f)
    if mode == 'wiki':
        data = [x for x in data if len(x['text']) > 2]
    data = [{x:y for x,y in i.items() if x in ['taxon_id', 'common_name', 'latin_name'] or len(y) > 2} for i in data]
    # Loads the model for both capabilities; If you only need embedding pass `mode="embedding"` to save memory (no lm head)
    model = GritLM("GritLM/GritLM-7B", torch_dtype="auto")

    ### Embedding/Representation ###
    documents = [
        x[y] for x in data for y in x.keys() if y not in ['taxon_id', 'common_name', 'latin_name']
    ]
    keys = [
        (i, y) for i, x in enumerate(data) for y in x.keys() if y not in ['taxon_id', 'common_name', 'latin_name']
    ]

    def gritlm_instruction(instruction):
        return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"

    # No need to add instruction for retrieval documents
    d_rep = model.encode(documents, instruction=gritlm_instruction(""))
    d_rep = torch.from_numpy(d_rep)
    torch.save({'taxon_id': torch.tensor([x['taxon_id'] for x in data]), 'keys': keys, 'data': d_rep}, 'gpt_data.pt')
    #q_rep = model.encode(queries, instruction=gritlm_instruction(instruction))
    print()


@torch.no_grad()
def extract_llama(fpath='./data/inat_taxa_info.csv'):
    # Load the LLaMA 2 7B model and tokenizer
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LlamaModel.from_pretrained(model_name).cuda()
    model.eval()
    feats = []
    with open(fpath, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        t1 = time.time()
        for row in tqdm(spamreader):
            if row[0] == 'taxon_id':
                continue

            # Preprocess the prompt
            prompt_text = f'{{class: {row[5]}, order: {row[6]}, family: {row[7]}, genus: {row[8]}, species: {row[9]}}}'
            encoding = tokenizer(prompt_text, return_tensors="pt").to('cuda')

            # Generate embeddings
            token_embeddings = model(**encoding).last_hidden_state  # Embeddings for each token

            # Apply mean pooling to obtain a fixed-length feature vector
            feature_vector = token_embeddings[0,-1].cpu()  # Mean pooling across tokens
            feats.append(feature_vector)
            if time.time() - t1 > 120:
                torch.save(torch.stack(feats), 'llama-feats.pt')
                t1 = time.time()
    torch.save(torch.stack(feats), 'llama-feats.pt')

text_prompt = '''**Task:**
Using the Wikipedia article provided on a species, generate a JSON formatted output containing four concise paragraphs as described below.

**Content Requirements:**
1. **Range:**
   Describe the geographical range of the species. Include information about the regions where this species is commonly found. Assume a comprehensive understanding as if the species has been studied for several years.
2. **Habitat:**
   Describe the habitat preferences of the species. Include specific environmental conditions it favors, such as climate, altitude, and the type of ecosystem.
3. **Species Description:**
   Provide a detailed description of the species, focusing on its ecological interactions and relationships within its habitat. If mentioned in the article, make sure to include: Describe how the species interacts with other organisms, including any symbiotic relationships, competition, or predation dynamics it may be involved in. Highlight its role within the ecosystem, such as its impact on local populations of other organisms and its contributions to habitat structure through its feeding or foraging habits. Include information on how the species responds to environmental pressures like habitat fragmentation or changes due to human activity. Mention any interactions with invasive species or its role in its ecosystem's food web. This description should offer insight into the ecological niche of the species and how these interactions define its survival and reproductive strategies.
4. **Overview Summary:**
   Summarize the key points from the above three paragraphs in a concise overview. This summary should encapsulate the main information about the species' range, habitat, and ecological behaviors without directly repeating the previous texts. Aim to provide a snapshot that combines the detailed habitat and behavioral data into a cohesive narrative.

**Formatting Instructions:**
Please format the output in JSON as shown below:
```json
{
  "range": "[Insert text from Content Requirement 1 here]",
  "habitat": "[Insert text from Content Requirement 2 here]",
  "species_description": "[Insert text from Content Requirement 3 here]",
  "overview_summary": "[Insert text from Content Requirement 4 here]"
}
```
Ensure that the text within each JSON field reflects the content requirements accurately, and verify the correctness of the JSON structure. Escape any double quotes within the responses.

Wikipedia Text:
'''

def run_llama():
    with open('paths.json', 'r') as f:
        paths = json.load(f)
    with open(os.path.join(paths['iucn'], 'iucn_res_5.json'), 'r') as f:
        data = json.load(f)
    D = np.load(os.path.join(paths['snt'], 'snt_res_5.npy'), allow_pickle=True)
    D = D.item()
    taxa_snt = D['taxa'].tolist()
    taxa = [int(tt) for tt in data['taxa_presence'].keys()]
    taxa = list(set(taxa + taxa_snt))


    with open('wiki_data_v2.json', 'r') as f:
        data = json.load(f)
    data = [x for x in data if len(x['text']) > 2 and x['taxon_id'] in taxa]
    articles = []
    for d in data:
        text = '\n'.join([y for x,y in d.items() if x not in ['taxon_id', 'common_name', 'latin_name']])
        articles.append({'text': text, **{x:y for x,y in d.items() if x in ['taxon_id', 'common_name', 'latin_name']}})

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16},
                                     device_map="auto")

    savefile = 'gpt_data.json'
    if os.path.exists(savefile):
        with open(savefile, 'r') as f:
            output_text = json.load(f)
    else:
        output_text = []
    completed = [x['taxon_id'] for x in output_text]
    t0 = time.time()
    for article in tqdm(articles):
        if article['taxon_id'] in completed:
            continue
        if len(article['text']) > 35000:
            num_splits = 35000//2000
            split_size = len(article['text'])//num_splits + 1
            outs = []
            for i in range(0, len(article['text']), split_size):
                messages = [{'role': 'user', 'content': 'Rewrite the following in 200 words. Surround your response with [start] and [end] tags:' + article['text'][i:i+split_size+(split_size//10)]}]
                prompt = pipeline.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                terminators = [
                    pipeline.tokenizer.eos_token_id,
                    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]

                outputs = pipeline(
                    prompt,
                    max_new_tokens=1024,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                )
                outs.append(outputs[0]["generated_text"][len(prompt):])
            article_text = ''.join([o[o.find('[start]')+7:o.rfind('[end]')] for o in outs])
            print(len(article_text))
        else:
            article_text = article['text']
        messages = [{'role':'user', 'content': text_prompt + article_text}]
        prompt = pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipeline(
            prompt,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        out = outputs[0]["generated_text"][len(prompt):]
        def escape_quotes(s):
            s = s.replace('"', '\\"')
            s = s.replace('\\"range\\": \\', '"range": ').replace('\\"habitat\\": \\', '"habitat": ').replace(
                '\\"species_description\\": \\', '"species_description": ').replace('\\"overview_summary\\": \\',
                                                                                    '"overview_summary": ')
            s = '\n'.join([line[:line.rfind('\\')] + line[line.rfind('\\') + 1:] if '\\' in line else line for line in
                            s.split('\n')])
            return s

        try:
            out = escape_quotes(out[out.find('{'):out.rfind('}')+1])
            out = json.loads(out)
        except:
            print(out)
            continue
        for k in ['taxon_id', 'common_name', 'latin_name']:
            out[k] = article[k]
        output_text.append(out)
        completed.append(article['taxon_id'])
        if time.time() - t0 > 120:
            with open(savefile, 'w') as outfile:
                json.dump(output_text, outfile)
            t0 = time.time()
    with open(savefile, 'w') as outfile:
        json.dump(output_text, outfile)


def load_species_names(fpath):
    rows = []
    with open(fpath, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in tqdm(spamreader):
            if row[0] == 'taxon_id':
                continue
            rows.append(row)
    return rows


@torch.no_grad()
def analyze_features(path, fpath='./data/inat_taxa_info.csv'):
    species = load_species_names(fpath)
    feats = torch.load(path)
    dists = torch.cdist(feats, feats)
    for i in range(dists.shape[0]):
        if species[i][2] != 'birds':
            continue
        inds = [x for x in range(len(species)) if species[x][2] == 'birds' and species[x][7] != species[i][7]]
        print(species[i][-1])
        print([species[inds[x]][-1] for x in (-dists[i,inds]).topk(5).indices])


if __name__ == '__main__':
    #run_llama()
    extract_grit(mode='gpt')