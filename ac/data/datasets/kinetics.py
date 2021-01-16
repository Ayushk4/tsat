"""
Kinetics Format: Json file

{
    'zvdtk1CSpao': {'annotations': {
                            'label': ('changing oil', <label_id>),
                            'segment': [31.0, 41.0]
                        },
                    'duration': 10.0,
                    'subset': 'validate',
                    'url': 'https://www.youtube.com/watch?v=zvdtk1CSpao'
                    ... some more things
                }
    ...

"""


import torch
import torch.utils.data.Dataset as Dataset
import torchvision
import numpy as np


class Kinetics(Dataset):

    def __init__(self, config=None, split='train'):

        super(Kinetics, self).__init__()

        # Load from the cache if cache is available

        # unpack the config
        # Paths and Splits
        self.split = split

        self.data_path = config.DATASET.DATA_PATH
        self.class_to_index_path = config.DATASET.CLASS_TO_INDEX_FILE
        self.annotations_path = eval(f'config.DATASET.{split.upper()}_ANNOTATIONS_PATH')
        self.videos_path = eval(f'config.DATASET.{split.upper()}_VIDEO_RESIZED_PATH')

        self.use_toy_version = config.DATASET.TOY

        # Technical details
        # self.sampling_stride = config.NETWORK.SAMPLING_STRIDE
        self.sampling_stride = config.DATASET.SAMPLING_STRIDE

        # find all the annotations
        self.dataset = self.load_annotations()

        # Preprocessing
        self.transforms_1 = lambda x: x / 255 # bring to [0,1] and cnvrt from dtype=uint8 to float32
        self.transforms_2 = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


        # implement caching here
        if self.use_toy_version:
            self.dataset = self.dataset[:config.DATASET.TOY_SAMPLES]

    def load_annotations(self):

        annotations_path = os.path.join(self.data_path, self.annotations_path)
        annotations = json.load(open(annotations_path))
        self.label_class_to_ix = json.load(open(os.path.join(self.data_path,
                                                    self.class_to_index_path)))

        database = []

        # preprocess sampling rates and other stuff
        for k,v in annotations.items():

            # store all the existing information
            this_label = v['annotations']['label'][1]
            this_vidpath = os.path.join(self.data_path, self.videos_path, k + ".mp4")

            assert type(this_label) == int
            data_point = {'video_id': k,
                          'label': this_label,
                          'vidpath': this_vidpath
                        }

            database.append(data_point)

        return database

    def load_and_preprocess_frames(self, video_path):
        # 1. load video frames 
        frames, _, fps =  torchvision.io.read_video(video_path)
        fps = fps['video_fps']

        # TODO: Sanity check for permute
        frames = frames.permute(0,3,1,2)# Shape: (T, H, W, C) -> (T, C, H, W) 
        T, C, H, W = frames.shape
        assert C == 3 and H == 224 and W == 224

        # 2. Sample frames
        assert 9.8 < fps < 10.2
        desired_frame_ixs = np.arange(0, T, self.sampling_stride)
        frames_sampled = frames[desired_frame_ixs, :, :, :]
        assert frames_sampled.shape == torch.Size([len(desired_frame_ixs), C, H, W])

        # 3. Preprocess selected frames: mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
        return self.transforms_2(self.transforms_1(frames_sampled))

    def __getitem__(self, idx):

        data_point = self.dataset[idx]
        frames_preprocessed = self.load_and_preprocess_frames(data_point['vidpath'])

        return frames_preprocessed, data_point['label']

    def __len__(self):
        return len(self.dataset)
