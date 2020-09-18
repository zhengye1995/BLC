from argparse import ArgumentParser

import mmcv
import numpy as np

from mmdet import datasets
from mmdet.core import zsd_eval_map


def zsd_eval(result_file, dataset, iou_thr=0.5, classwise=False):
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
    zsd_eval_map(
        det_results,
        gt_bboxes,
        gt_labels,
        gt_ignore=gt_ignore,
        scale_ranges=None,
        iou_thr=iou_thr,
        dataset=dataset_name,
        num_classes=len(dataset_name),
        class_wise=classwise)


def main():
    parser = ArgumentParser(description='VOC Evaluation')
    parser.add_argument('result', help='result file path')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--iou-thrs', nargs='+', type=float, default=[0.4, 0.5, 0.6], help='IoU thresholds for evaluation')
    parser.add_argument(
        '--classwise', action='store_true', help='whether eval class wise ap')
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    test_dataset = mmcv.runner.obj_from_dict(cfg.data.test, datasets)
    print("evaluating ZSD performance...")
    for iou_thr in args.iou_thrs:
        print("eval results of " + str(iou_thr) + " IoU thresholds:")
        zsd_eval(args.result, test_dataset, iou_thr, args.classwise)
        print()


if __name__ == '__main__':
    main()
