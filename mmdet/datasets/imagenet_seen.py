import numpy as np
from pycocotools.coco import COCO

from .custom import CustomDataset
from .registry import DATASETS


@DATASETS.register_module
class ImagenetSeen(CustomDataset):
    CLASSES = ('accordion', 'airplane', 'ant', 'antelope', 'apple', 'armadillo', 'artichoke', 'axe', 'baby bed', 'backpack', 'bagel',
                 'balance beam', 'banana', 'band aid', 'banjo', 'baseball', 'basketball', 'bathing cap', 'beaker', 'bear', 'bee',
                 'bell pepper', 'bicycle', 'binder', 'bird', 'bookshelf', 'bow', 'bowl', 'brassiere', 'bus', 'butterfly', 'camel',
                 'car', 'cart', 'cattle', 'cello', 'centipede', 'chain saw', 'chair', 'chime', 'cocktail shaker', 'coffee maker',
                 'computer keyboard', 'computer mouse', 'corkscrew', 'cream', 'croquet ball', 'crutch', 'cucumber', 'cup or mug',
                 'diaper', 'digital clock', 'dog', 'domestic cat', 'dragonfly', 'drum', 'dumbbell', 'elephant', 'face powder', 'fig',
                 'filing cabinet', 'flower pot', 'flute', 'fox', 'french horn', 'frog', 'frying pan', 'giant panda', 'goldfish',
                 'golfcart', 'guacamole', 'guitar', 'hair dryer', 'hair spray', 'hamburger', 'hammer', 'harp', 'hat with a wide brim',
                 'head cabbage', 'helmet', 'hippopotamus', 'horse', 'hotdog', 'isopod', 'jellyfish', 'koala bear', 'ladle', 'ladybug',
                 'lamp', 'laptop', 'lemon', 'lion', 'lipstick', 'lizard', 'lobster', 'maillot', 'microphone', 'microwave', 'milk can',
                 'miniskirt', 'monkey', 'motorcycle', 'mushroom', 'nail', 'neck brace', 'oboe', 'orange', 'otter', 'pencil sharpener',
                 'perfume', 'person', 'piano', 'ping-pong ball', 'pitcher', 'pizza', 'plastic bag', 'pomegranate', 'popsicle',
                 'porcupine', 'power drill', 'pretzel', 'printer', 'puck', 'punching bag', 'purse', 'rabbit', 'racket', 'red panda',
                 'refrigerator', 'remote control', 'rubber eraser', 'rugby ball', 'ruler', 'salt or pepper shaker', 'saxophone',
                 'screwdriver', 'seal', 'sheep', 'ski', 'skunk', 'snake', 'snowmobile', 'snowplow', 'soap dispenser', 'soccer ball',
                 'sofa', 'spatula', 'squirrel', 'starfish', 'stethoscope', 'stove', 'strainer', 'strawberry', 'stretcher', 'sunglasses',
                 'swine', 'table', 'tape player', 'tennis ball', 'tick', 'tie', 'toaster', 'traffic light', 'trombone', 'trumpet',
                 'turtle', 'tv or monitor', 'vacuum', 'violin', 'volleyball', 'waffle iron', 'washer', 'water bottle', 'watercraft',
                 'whale', 'wine bottle', 'zebra')

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.img_infos[idx], ann_info)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann
