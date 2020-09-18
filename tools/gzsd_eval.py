from argparse import ArgumentParser

import mmcv
import numpy as np

from mmdet import datasets
from mmdet.core import gzsd_eval_map


def gzsd_eval(result_file, dataset, iou_thr=0.5, num_seen_classes=48, classwise=False):
    det_results = mmcv.load(result_file)
    gt_bboxes = []
    gt_labels = []
    for i in range(len(dataset)):
        ann = dataset.get_ann_info(i)
        bboxes = ann['bboxes']
        labels = ann['labels']
        gt_bboxes.append(bboxes)
        gt_labels.append(labels)
    gt_ignore = None
    dataset_name = dataset.CLASSES
    gzsd_eval_map(
        det_results,
        gt_bboxes,
        gt_labels,
        gt_ignore=gt_ignore,
        scale_ranges=None,
        iou_thr=iou_thr,
        dataset=dataset_name,
        num_classes=len(dataset_name),
        num_seen_classes=num_seen_classes,
        class_wise=classwise)


def main():
    parser = ArgumentParser(description='VOC Evaluation')
    parser.add_argument('result', help='result file path')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold for evaluation')
    parser.add_argument(
        '--num-seen',
        type=int,
        default=48,
        help='seen classes num')
    parser.add_argument(
        '--classwise', action='store_true', help='whether eval class wise ap')
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    test_dataset = mmcv.runner.obj_from_dict(cfg.data.test, datasets)
    print("evaluating GZSD performance...")
    gzsd_eval(args.result, test_dataset, args.iou_thr, args.num_seen, args.classwise)


if __name__ == '__main__':
    main()
