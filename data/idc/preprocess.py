import json, os, re
from collections import Counter as C

max_len = 10
data_path ='/workspace/datasets/VisualGenome/decompressed'
annotations_path = '/workspace/datasets/VisualGenome/decompressed/densecap_splits.json'
images_path = ['VG_100K', 'VG_100K_2']
annotations_base_path = annotations_path
frames_base_paths = [os.path.join(data_path, p)
                            for p in images_path]
id2path = {int(file.split('.')[0]): os.path.join(path, file)
                       for path in frames_base_paths
                       for file in os.listdir(path)
                  }

splits = json.load(open(annotations_base_path))

savepath = "data/idc/preprocessed"
caption_path = os.path.join(data_path, 'region_descriptions.json')

tokenizer = lambda s: [x for x in re.split(r"[\s]|[\W_]", re.sub(r'[0-9]+', '0', s.lower()))
                       if type(x) == str and x != '']
pad = lambda x, m: x+ (( m -len(x)) * ['<PAD>'])
captions = json.load(open(caption_path))

all_caps = {}
all_ids = {i:split for split, ids in splits.items() for i in ids}

for caps in captions:
    if caps['id'] not in all_ids:
        continue
    this_split = all_ids[caps['id']]
    this_caps = []
    for region in caps['regions']:
        toks = tokenizer(region['phrase'])
        bbox = {key: region[key] for key in ['x', 'y', 'width', 'height']}
        if len(toks) <= max_len:
            toks_padded = pad(toks, max_len)
            this_caps.append([region['region_id'], toks_padded, bbox, " ".join(toks)])
    if len(this_caps) > 0 and len(this_caps) <= 50:# and  len(this_caps) > 20:
        all_caps[caps['id']] = this_caps

print(len(all_caps))

vocab = [k for k,v in 
        C([token
           for img_cap in all_caps.values()
           for region in img_cap
           for token in region[1]]
         ).items()
        if v >= 15]
assert "<UNK>" not in vocab
vocab.append("<UNK>")

vocab_to_ix = {k:i for i,k in enumerate(vocab)}
print(len(vocab))

index_it = lambda x: [vocab_to_ix[xx] if xx in vocab_to_ix else vocab_to_ix['<UNK>'] for xx in x ]
print(index_it(['head', 'of', 'person', 'ayushj']))
splits = {k: [vv for vv in v if vv in all_caps] for k,v in splits.items()}

def index(savepath, all_caps, this_split):
    print(type(all_caps))
    print(all_caps[splits['test'][0]])
    this_split_caps = {imid: {'caps': [[single_cap[0], index_it(single_cap[1]),
                            single_cap[2], single_cap[3]]
                            for single_cap in all_caps[imid]
                           ],
                        'path': id2path[imid]
                        }
                    for imid in splits[this_split]
            }
    print("Done " + this_split + " split")
    json.dump(this_split_caps, open(savepath, 'w+'))

index('preprocessed/test_1.json', all_caps, 'test')
index('preprocessed/val_1.json', all_caps, 'val')
index('preprocessed/train.json', all_caps, 'train')

json.dump(vocab, open('preprocessed/vocab.json', 'w+'))

