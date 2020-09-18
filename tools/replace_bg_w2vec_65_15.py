
import torch
import numpy as np
filename = './work_dirs/blrpn_r50_fpn_65_15_1x/latest.pth'
map_location = 'cpu'
checkpoint = torch.load(filename, map_location=map_location)
rpn_head_weight = checkpoint['state_dict']['rpn_head.vec_bg.weight'].numpy()[0]
print(rpn_head_weight.shape)
rpn_head_weight = rpn_head_weight.reshape([300,])
word_w2v = np.loadtxt('data/coco/word_w2v_withbg.txt', dtype='float32', delimiter=',')
assert word_w2v[:, 0].shape == rpn_head_weight.shape
word_w2v[:, 0] = rpn_head_weight
np.savetxt('data/coco/word_w2v_with_learnable_bg_65_15.txt', word_w2v, delimiter=',')