"""
Kinetics Format: Json file

{
    'zvdtk1CSpao': {'annotations': {
                            'label': 'changing oil',
                            'segment': [31.0, 41.0]
                        },
                    'duration': 10.0,
                    'subset': 'validate',
                    'max_frames': 254,
                    'url': 'https://www.youtube.com/watch?v=zvdtk1CSpao'
                }
    ...

"""

# Each datapoint 
# 1. Video id
# 2. Label

import torch
import torch.utils.data.Dataset as Dataset

class Kinetics(Dataset):

    def __init__(self, config=None, split='train'):

        super(Kinetics, self).__init__()

        # Load from the cache if cache is available

        # unpack the config
        # Paths and Splits
        self.root_path = config.DATASET.ROOT_PATH
        self.data_path = config.DATASET.DATA_PATH
        self.dataset_name = config.DATASET.DATASET_NAME
        self.use_toy_version = config.DATASET.TOY
        self.annotations_path = eval(f'config.DATASET.{split.upper()}_ANNOTATIONS_PATH')
        self.frames_path = config.DATASET.FRAMES_PATH
        self.split = split

        # Technical details
        self.sampling_stride = config.NETWORK.SAMPLING_STRIDE
        self.sampling_count = config.NETWORK.SAMPLING_COUNT

        # find all the annotations
        self.dataset = self.load_annotations()

        # implement caching here

        if self.use_toy_version:
            self.dataset = self.dataset[:config.DATASET.TOY_SAMPLES]

    def load_annotations(self):

        annotations_base_path = os.path.join(self.root_path, self.data_path, self.annotations_path)
        video_base_path = os.path.join(self.root_path, self.data_path, self.frames_path)

        annotations = json.load(open(annotations_base_path))

        self.label_classes = {ann['annotations']['label'] for ann in annotations.values()}
        self.label_class_to_ix = {v:i for i,v in enumerate(self.label_classes)}

        database = []

        # preprocess sampling rates and other stuff
        for k,v in annotations.items():

            # store all the existing information
            data_point = {'video': k,
                          'label': self.label_class_to_ix(
                                            v['annotations']['labels']
                                        ),
                          'path': os.path.join(frames_base_path, video_name)
                        }

            database.append(data_point)

        return database

    def sample_frames(self, frames_base_path, video_name, max_frames):

        video_path = os.path.join(frames_base_path, video_name)

        fn = int(frame_name.split('.')[0])
        ss = self.sampling_stride
        sc = self.sampling_count

        context_frames = [*range(fn-ss*sc, fn+ss*(sc+1), ss)]

        # clipping
        # Left: replace all the negative indices with 1
        for i in range(len(context_frames)):
            if context_frames[i] <= 0:
                context_frames[i] = 1

        # Right: replace all exceeding indices with max_frames
        for i in range(len(context_frames)):
            if context_frames[i] > max_frames:
                context_frames[i] = max_frames

        convert = lambda  x: os.path.join(video_path, "{:06d}".format(x) + ".png")
        context_frames = list(map(convert, context_frames))

        return context_frames, sc

    def load_and_preprocess_frames(self, video_path):
        # 1. load video frames 
        # 2. Sample frames
        # 3. Preprocess selected frames: resize, mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
        pass

    def __getitem__(self, idx):

        data_point = self.database[idx]


        # 4. convert
        # 2. bounding boxes per frames : 
        # 2. captions per frame :

        context_frames_preprocessed = torch.stack([self.load_and_preprocess_frames(frame_path) \ 
                                    for frame_path in data_point['context_frames'])


        # 
