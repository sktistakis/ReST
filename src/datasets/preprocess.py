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
                 eval_ratio: float,
                 test_ratio: float,
                 valid_frames_range=None,
                 output_dir=None,
                 image_format='jpg',
                 random_seed=202204):
        self.dataset_name = dataset_name
        self.base_dir = base_dir
        self.dataset_path = os.path.join(base_dir, dataset_name)
        self.annots_name = annots_name  # must correspond to cameras id
        self.videos_name = videos_name  # also correspond to annotation files
        self.eval_ratio = eval_ratio
        self.test_ratio = test_ratio
        self.valid_frames_range = valid_frames_range
        self.image_format = image_format
        if output_dir is None:
            self.output_path = os.path.join(self.dataset_path, 'output')
        else:
            self.output_path = output_dir
        self.frames_output_path = os.path.join(self.output_path, 'frames')
        self.frames = {}

        random.seed(random_seed)

    def process(self):
        os.makedirs(self.output_path, exist_ok=True)
        if not os.path.exists(self.frames_output_path):
            os.mkdir(self.frames_output_path)

        self.load_annotations()
        self.filter_frames()
        # self.load_videos()
        if DATASET_NAME == 'Medtrack':
            self.load_frames()
        self.train_test_split()
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

    def load_videos(self):
        """
        Extract avail_frames from synchronized videos and save as figures
        """
        frames_id = list(self.frames.keys())
        caps = [cv2.VideoCapture(os.path.join(self.dataset_path, vn))
                for vn in self.videos_name]
        avail_frames = sorted(frames_id)
        cur_frame_id = 0

        for frame_id in tqdm.trange(0, avail_frames[-1] + 1, desc=self.dataset_name):
            # Skip the invalid frame.
            if frame_id != avail_frames[cur_frame_id]:
                for cap in caps:
                    cap.read()
                continue
            # Capture one frame across all cameras.
            for cam_id, cap in enumerate(caps):
                ret, img = cap.read()
                if ret:
                    cv2.imwrite(os.path.join(self.frames_output_path,
                                             f'{frame_id}_{cam_id}.{self.image_format}'), img)
                else:
                    raise ValueError(f"Cannot load frame {frame_id} from "
                                     f"video {self.dataset_name} at cam {cam_id}.")
            cur_frame_id += 1

    def select_frames(self, frames_id: list):
        ret = {}
        for frame_id in frames_id:
            ret[frame_id] = self.frames[frame_id]
        return ret

    def save_json(self, obj, name):
        with open(os.path.join(self.output_path, f'{name}.json'), 'w') as fp:
            json.dump(obj, fp)

    def train_test_split(self):
        frames_id = list(sorted(self.frames.keys()))
        # random.shuffle(frames_id)
        n = len(frames_id)
        n_test = int(self.test_ratio * n)
        n_eval = int(self.eval_ratio * (n - n_test))
        test_frames_id = frames_id[-n_test:]
        rest_frames_id = frames_id[:-n_test]
        eval_frames_id = rest_frames_id[-n_eval:]
        train_frames_id = rest_frames_id[:-n_eval]

        train_frames = self.select_frames(train_frames_id)
        eval_frames = self.select_frames(eval_frames_id)
        test_frames = self.select_frames(test_frames_id)

        sorted_train_frames = dict(sorted(train_frames.items()))
        sorted_eval_frames = dict(sorted(eval_frames.items()))
        sorted_test_frames = dict(sorted(test_frames.items()))

        self.save_json(sorted_train_frames, 'gt_train')
        self.save_json(sorted_eval_frames, 'gt_eval')
        self.save_json(sorted_test_frames, 'gt_test')

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
        n = len(frames_id)
        n_test = int(self.test_ratio * n)
        test_frames_id = frames_id[-n_test+1:]
        attr = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'x', 'y', 'z']
        gts = []
        for _ in range(len(self.annots_name)):
            gts.append(pd.DataFrame(columns=attr))
        for f in tqdm.tqdm(test_frames_id):
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


class WildtrackPreprocess(BasePreprocess):

    def load_annotations(self):
        """
        Parse annotation files that across all cameras to obtain frames(self).
        Within data format:
            [frame_id]: (top_left_x, top_left_y, width, height, track_id, camera_id)
        """
        for frame_id in range(self.valid_frames_range[-1]):
            anno_file = f'{self.base_dir}/{self.dataset_name}/src/annotations_positions/0000{frame_id*5:04d}.json'
            f = open(anno_file)
            data = json.load(f)
            f.close()
            for raw in data:
                track_id = raw['personID']
                for cam_id in range(len(raw['views'])):
                    if raw['views'][cam_id]['xmin'] == -1:
                        continue
                    x_min = max(raw['views'][cam_id]['xmin'], 0) # prevent negative value
                    y_min = max(raw['views'][cam_id]['ymin'], 0) # prevent negative value
                    width = raw['views'][cam_id]['xmax'] - x_min
                    height = raw['views'][cam_id]['ymax'] - y_min
                    self.frames.setdefault(frame_id, []).append(
                        (x_min, y_min, width, height, track_id, cam_id)
                    )

    def load_frames(self):
        # Copy from original dataset
        # 0000~1995 -> 0~400
        if not os.path.exists(self.frames_output_path):
            os.mkdir(self.frames_output_path)
        for cam_id in range(7):
            path = os.path.join(self.base_dir, f'{self.dataset_name}/src', 'Image_subsets', f'C{cam_id+1}')
            for frame_id in trange(self.valid_frames_range[-1]):
                img = cv2.imread(os.path.join(path, f'0000{frame_id*5:04d}.png'))
                cv2.imwrite(os.path.join(self.frames_output_path,
                                            f'{frame_id}_{cam_id}.{self.image_format}'), img)
                
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
                            continue
                        x_min = max(detection['views'][str(cam_id)]['xmin'], 0) # prevent negative value
                        y_min = max(detection['views'][str(cam_id)]['ymin'], 0) # prevent negative value
                        width = detection['views'][str(cam_id)]['xmax'] - x_min
                        height = detection['views'][str(cam_id)]['ymax'] - y_min
                        self.frames.setdefault(frame_id, []).append(
                            (x_min, y_min, width, height, track_id, cam_id)
                        )

    def load_frames(self):
        """
        Extract frames from videos and save as images for processing.
        """
        with open(os.path.join(self.base_dir, 'metainfo.json'), 'r') as f:
            meta_info = json.load(f)

        cam_count = meta_info[self.dataset_name]["cam_nbr"]

        for cam_id in range(cam_count):
            cam_path = os.path.join(
                self.base_dir,
                meta_info[self.dataset_name]["name"],
                "Image_subsets",
                f"C{cam_id}"
            )
            if not os.path.exists(cam_path):
                continue

            for frame_id in trange(self.valid_frames_range[-1]):
                img = cv2.imread(os.path.join(cam_path, f"{frame_id}.png"))
                cv2.imwrite(os.path.join(self.frames_output_path,
                                            f'{frame_id}_{cam_id}.{self.image_format}'), img)

    def train_test_split(self):
        rec_ids = list(sorted(self.frames.keys()))

        #  Depending on if self.train_ratio == 1, self.test_ratio == 0, self.eval_ratio == 0 we will put all frames 
        # into the train or val or test set.
        if self.test_ratio == 0 and self.eval_ratio == 0:
            train_frames = self.select_frames(rec_ids)
            sorted_train_frames = dict(sorted(train_frames.items()))
            self.save_json(sorted_train_frames, 'gt_train')
            return 
        elif self.eval_ratio == 1:
            eval_frames = self.select_frames(rec_ids)
            sorted_eval_frames = dict(sorted(eval_frames.items()))
            self.save_json(sorted_eval_frames, 'gt_eval')
            return
        elif self.test_ratio == 1:
            test_frames = self.select_frames(rec_ids)
            sorted_test_frames = dict(sorted(test_frames.items()))
            self.save_json(sorted_test_frames, 'gt_test')
            return
    
    
class CAMPUSPreprocess(BasePreprocess):

    def load_annotations(self):
        for cam_id, annot_name in enumerate(self.annots_name):
            fp = open(os.path.join(self.dataset_path, annot_name))
            rd = csv.reader(fp, delimiter=' ')
            # Within annotations format:
            # (track_id, x_min, y_min, x_max, y_max, frame_number, lost, occluded, generated, label)
            for row in rd:
                # Filter out the lost one.
                if row[6] == '1':
                    continue
                track_id, x_min, y_min, x_max, y_max, frame_id = tuple(map(int, row[:6]))
                # Filter out the frame that outside the valid frames range.
                if frame_id > self.valid_frames_range[1] or \
                   frame_id < self.valid_frames_range[0]:
                    continue
                self.frames.setdefault(frame_id, []).append(
                    (x_min, y_min, x_max - x_min, y_max - y_min, track_id, cam_id)
                )
            fp.close()

class CityFlowPreprocess(BasePreprocess):

    def load_annotations(self):
        f = open(f'{self.dataset_path}/output/gt.json')
        data = json.load(f)
        for frame_id in range(self.valid_frames_range[-1]):
            val = data[f'{frame_id}']
            self.frames[frame_id] = val


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
            eval_ratio=recording_meta['eval_ratio'],
            test_ratio=recording_meta['test_ratio'],
            valid_frames_range=recording_meta['valid_frames_range'],
            output_dir=os.path.join(output_dir, recording_name) if output_dir is not None else None
        )
        dataset.process()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Wildtrack", help="pre-process which dataset", type=str)
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

        if DATASET_NAME == 'Wildtrack':
            preprocess(dataset_dir=f'./datasets/{DATASET_NAME}', PreProcess=WildtrackPreprocess,
                            output_dir="./datasets/Wildtrack/sequence1/output")
        if DATASET_NAME == 'Medtrack':
            preprocess(
                dataset_dir=f'/media/sktistakis/T7/USZ_STUDY/USZ_Wildtrack',
                PreProcess=MedtrackPreprocess,
                output_dir=f"./datasets/{DATASET_NAME}/output"
            )
        elif DATASET_NAME == 'CityFlow':
            preprocess(f'./datasets/{DATASET_NAME}', PreProcess=CityFlowPreprocess)
        else: # CAMPUS, PETS09
            preprocess(f'./datasets/{DATASET_NAME}', PreProcess=CAMPUSPreprocess)
