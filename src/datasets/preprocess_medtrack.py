import os
import json
import random
import pandas as pd
import csv
import cv2
import tqdm
from tqdm import trange
import argparse

DATASET_NAME = ''

class BasePreprocess:

    def __init__(self,
                 dataset_name: str,
                 base_dir: str,
                 annots_name: list,
                 videos_name: list,
                 valid_frames_range=None,
                 output_dir=None,
                 image_format='jpg',
                 random_seed=202204):
        self.dataset_name = dataset_name
        self.base_dir = base_dir
        self.dataset_path = os.path.join(base_dir, dataset_name)
        self.annots_name = annots_name  # must correspond to cameras id
        self.videos_name = videos_name  # also correspond to annotation files
        self.valid_frames_range = valid_frames_range
        self.image_format = image_format

        self.output_path = output_dir
        self.frames = {}

        random.seed(random_seed)

    def process(self):
        os.makedirs(self.output_path, exist_ok=True)

        self.load_annotations()
        self.filter_frames()
        self.save_gt()
        self.load_MOTgtfile()

    def load_annotations(self):
        """
        Parse annotation files that across all cameras to obtain frames(self).
        Within data format:
            [frame_id]: (top_left_x, top_left_y, width, height, track_id, camera_id)
        """
        raise NotImplementedError

    def filter_frames(self):
        # Filter out frames that the person only appear in one camera view.
        invalid_frames_id = []
        for frame_id, frame in self.frames.items():
            cams = set([sample[-1] for sample in frame])
            if len(cams) <= 1:
                invalid_frames_id.append(frame_id)

        for frame_id in invalid_frames_id:
            self.frames.pop(frame_id)

    def select_frames(self, frames_id: list):
        ret = {}
        for frame_id in frames_id:
            ret[frame_id] = self.frames[frame_id]
        return ret

    def save_json(self, obj, name):
        with open(os.path.join(self.output_path, f'{name}.json'), 'w') as fp:
            json.dump(obj, fp)

    def load_MOTgtfile(self):
        """
        Use test.json to generate MOT groundtruth txt file with following file structure.
        Each row in c0.txt is one groundtruth detection following the spec.:
            ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'x', 'y', 'z']
        """
        gt_path = os.path.join(self.output_path, 'gt_MOT')
        if not os.path.exists(gt_path):
            os.mkdir(gt_path)

        frames_id = list(sorted(self.frames.keys()))        
        attr = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'x', 'y', 'z']
        gts = []
        for _ in range(len(self.annots_name)): # number of cameras
            gts.append(pd.DataFrame(columns=attr))
        for f in tqdm.tqdm(frames_id):
            for det in self.frames[f]:
                cID = det[5]
                frame_style = {'Wildtrack': f'0000{f*5:04d}', 'PETS09': f'{f:03d}', 'CAMPUS': f'{f:04d}', 'CityFlow': f'{f}',
                               'Medtrack': f'{f}'}
                frame = frame_style[DATASET_NAME]
                gts[cID] = gts[cID].append({
                                            'frame': frame,
                                            'id': det[4],
                                            'bb_left': det[0],
                                            'bb_top': det[1],
                                            'bb_width': det[2],
                                            'bb_height': det[3],
                                            'x': 1, 'y': 1, 'z':1
                                        }, ignore_index=True)
        for c in range(len(self.annots_name)):
            gts[c].to_csv(os.path.join(gt_path, f'c{c}.txt'), header=None, index=None, sep=',')

                
class MedtrackPreprocess(BasePreprocess):
    def load_annotations(self):
        """
        Parse annotation files from cameras to obtain frames(self).
        """
        with open(os.path.join(self.base_dir, 'metainfo.json'), 'r') as f:
            meta_info = json.load(f)
        
            annot_pattern = meta_info[self.dataset_name]["annot_fn_pattern"]
            valid_frames = meta_info[self.dataset_name]["valid_frames_range"]

            for frame_id in range(valid_frames[0], valid_frames[1]):
                annot_file = os.path.join(
                    self.base_dir,
                    meta_info[self.dataset_name]["name"],
                    "annotations_positions",
                    f"{frame_id}.json"
                )
                if not os.path.exists(annot_file):
                    continue  # Skip missing annotation files

                with open(annot_file, 'r') as f:
                    annotations = json.load(f)

                for detection in annotations["root"].values():
                    track_id = detection['personID']
                    for cam_id in range(len(detection['views'])):
                        if detection['views'][str(cam_id)]['xmin'] == -1:
                            print(f"Invalid Ddetection. Frame: {frame_id}, Track: {track_id}, Camera: {cam_id}")
                            continue
                        x_min = max(detection['views'][str(cam_id)]['xmin'], 0) # prevent negative value
                        y_min = max(detection['views'][str(cam_id)]['ymin'], 0) # prevent negative value
                        width = detection['views'][str(cam_id)]['xmax'] - x_min
                        height = detection['views'][str(cam_id)]['ymax'] - y_min
                        self.frames.setdefault(frame_id, []).append(
                            (x_min, y_min, width, height, track_id, cam_id)
                        )

    def save_gt(self):
        rec_ids = list(sorted(self.frames.keys()))
        train_frames = self.select_frames(rec_ids)
        sorted_train_frames = dict(sorted(train_frames.items()))
        self.save_json(sorted_train_frames, 'gt')
    


def preprocess(dataset_dir, PreProcess, output_dir=None):
    with open(os.path.join(dataset_dir, 'metainfo.json'), 'r') as fp:
        meta_info = json.load(fp)

    for recording_name, recording_meta in meta_info.items():
        print(f"Processing recording: {recording_name}")
        dataset = PreProcess(
            dataset_name=recording_name,
            base_dir=dataset_dir,
            annots_name=[recording_meta['annot_fn_pattern'].format(cam_id) for cam_id in range(recording_meta['cam_nbr'])],
            videos_name=[recording_meta['video_fn_pattern'].format(cam_id) for cam_id in range(recording_meta['cam_nbr'])],
            valid_frames_range=recording_meta['valid_frames_range'],
            output_dir=os.path.join(output_dir, recording_name)
        )
        dataset.process()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Medtrack", help="pre-process which dataset", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args.dataset


if __name__ == '__main__':
    DATASET_NAME = parse_args()
    if DATASET_NAME not in ['Wildtrack', 'CAMPUS', 'PETS09', 'CityFlow', 'Medtrack']:
        print('Please enter valid dataset.')
    else:
        print(f'Pre-processing {DATASET_NAME}...')

        if DATASET_NAME == 'Medtrack':
            preprocess(
                dataset_dir=f'/media/sktistakis/T7/USZ_STUDY/USZ_Wildtrack',
                PreProcess=MedtrackPreprocess,
                output_dir=f'/media/sktistakis/T7/USZ_STUDY/USZ_Wildtrack/REST/output'
            )