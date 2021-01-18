"""
STDC Format

{"50N4E.mp4/000680.png":
            {"split": "train",
            "max_frames": 200,
            "video": "50N4E.mp4",
            "frame": "000680.png",
            "captions": [
                {"bbox": [63.4903678894043,
                    8.207611083984375,
                    192.93972778320312,
                    264.02484130859375],
                "caps": [
                    [12, 46, 71, 14, 81, 81, 81, 81, 81, 81, 81, 81]
                    ],
                "type": "person"
                }
                {"bbox": [68.13486896878712,
                    124.97725912325058,
                    90.49948044399594,
                    144.51899461451148],
                "caps": [
                    [14, 29, 57, 36, 66, 46, 81, 81, 81, 81, 81, 81],
                    [14, 68, 12, 46, 24, 71, 81, 81, 81, 81, 81, 81]
                    ],
                "type": "dish"}
            ]
            },
            ...

            ...

            10 * 4
            (10 * num_captions) * len_captions
}

"""

# Each datapoint 
# 1. Keyframe and Context frames (nearby, sparse sampling)
# 2. Bounding boxes # (B,4) torch.floatTensor
# 3. Caption Indices # (B,10) torch.LongTensor

import torch
import torch.utils.data.Dataset as Dataset

class STDC(Dataset):

    def __init__(self, config=None, split='train'):

        super(STDC, self).__init__()

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
        frames_base_path = os.path.join(self.root_path, self.data_path, self.frames_path)

        annotations = json.load(open(annotations_base_path))

        database = []

        # preprocess sampling rates and other stuff
        for k,v in annotations.items():

            # store all the existing information
            data_point = v

            # calculate the information about nearby frames
            # context_frames: a list of frames (strings) including the keyframe
            # keyframe_index: index of the keyframe in the list
            context_frames, keyframe_index = self.sample_frames(
                    frames_base_path, v['video'], v['frame'], v['max_frames'])

            data_point['context_frames'] = context_frames
            data_point['keyframe_index'] = keyframe_index
            
            database.append(data_point)

        return database

    def sample_frames(self, frames_base_path, video_name, frame_name, max_frames):

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

    def load_and_preprocess_frames(self, frame_path):
        pass

    def __getitem__(self, idx):

        data_point = self.database[idx]

        # 1. load and preprocess frames : done
        # 2. bounding boxes per frames : 
        # 2. captions per frame :

        # we have to pass a tensor that contains self.sampling_count * 2 + 1 number of images

        context_frames_preprocessed = torch.stack([self.load_and_preprocess_frames(frame_path) \ 
                                    for frame_path in data_point['context_frames'])


        # 
