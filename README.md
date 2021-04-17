
# Code for **Background Learnable Cascade for Zero-Shot Object Detection** 

## Code requirements
+ python: python3.7
+ nvidia GPU
+ pytorch1.1.0
+ GCC >=5.4
+ NCCL 2
+ the other python libs in requirement.txt

## Install 

```
conda create -n BLC python=3.7 -y
conda activate BLC

conda install pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=10.0 -c pytorch

pip install cython && pip --no-cache-dir install -r requirements.txt
   
python setup.py develop
```

## Dataset prepare


- Download the train and test annotations files for BLC from [annotations](https://drive.google.com/drive/folders/1vSCRFMNayiPPRg3ZpQ36fx1pmY0GHUUM?usp=sharing), put all json label file to
    ```
    data/coco/annotations/
    ```

- Download MSCOCO-2014 dataset and unzip the images it to path： 
    ```
    data/coco/train2014/
    data/coco/val2014/
    ```


- training:
    - train BLRPN:
        - 48/17 split:
          ```
          ./tools/dist_train.sh configs/train/BLRPN/blrpn_r50_fpn_semantic_48_17_1x.py 4
          ```   
            
        - 65/15 split:
          ```
          ./tools/dist_train.sh configs/train/BLRPN/blrpn_r50_fpn_semantic_65_15_1x.py 4
          ```
    - generate new Ws:
        - 48/17 split:
          ```
          python tools/replace_bg_w2vec_48_17.py
          ```   
            
        - 65/15 split:
          ```
          python tools/replace_bg_w2vec_65_15.py
          ```
          
    - train Cascade Semantic R-CNN with new Ws：
    
         - 48/17 split:
             ```
              ./tools/dist_train.sh configs/BLC/train/Cascade_Semantic_R-CNN/cascade_semantic_rcnn_information_flow_learnable_bg_48_17_1x.py 4
            ```
            
        - 65/15 split:
          ```
          ./tools/dist_train.sh configs/BLC/train/Cascade_Semantic_R-CNN/cascade_semantic_rcnn_information_flow_learnable_bg_48_17_1x.py 4
          ```
          
- **Inference & Evaluate**:

    + **ZSD task**:

        - 48/17 split ZSD task:
            - download [ms 48/17](https://drive.google.com/file/d/11LA3C3B-LzTISATCdei6fUFvhlb5rqhK/view?usp=sharing) BLC ms model, put it in checkpoints/BLC_ms_48_17.pth
            
            - inference:
                ```
                python tools/test.py configs/BLC/inference/zsd/cascade_semantic_rcnn_information_flow_learnable_bg_48_17_1x.py  checkpoints/BLC_ms_48_17.pth --out results/zsd_48_17.pkl
                ```
            - our results zsd_48_17.pkl can also downloaded from [zsd_48_17.pkl](https://drive.google.com/file/d/1fCzaWe3ErbNCaSSV00si8eS7RyYxb1mL/view?usp=sharing).
            - evaluate:
                ```
                python tools/zsd_eval.py results/zsd_48_17.pkl configs/BLC/inference/zsd/cascade_semantic_rcnn_information_flow_learnable_bg_48_17_1x.py
                ```
        - 65/15 split ZSD task:
            - download [ms 65/15](https://drive.google.com/file/d/1IHCWTKt5kxbCxQ2tskafbbql6Fqj-Rkf/view?usp=sharing) BLC model, put it in checkpoints/BLC_ms_65_15.pth
            
            - inference:
                ```
                python tools/test.py configs/BLC/inference/zsd/cascade_semantic_rcnn_information_flow_learnable_bg_65_15_1x.py  checkpoints/BLC_ms_65_15.pth --out results/zsd_65_15.pkl
                ```
            - our results zsd_65_15.pkl can also downloaded from [zsd_65_15.pkl](https://drive.google.com/file/d/16e1V7wHxVOOpgE4yFjOeqSS_cC9WvqvL/view?usp=sharing).
            - evaluate:
                ```
                python tools/zsd_eval.py results/zsd_65_15.pkl configs/BLC/inference/zsd/cascade_semantic_rcnn_information_flow_learnable_bg_65_15_1x.py
                ```

    + **GZSD task**:

        - 48/17 split GZSD task:
            - download [48/17](https://drive.google.com/file/d/1FM6AQ-ew6o-J-MedqniqhEOpReRRmwXV/view?usp=sharing) BLC model, put it in checkpoints/BLC_48_17.pth
            
            - inference:
                ```
                python tools/test.py configs/BLC/inference/gzsd/gzsd_cascade_semantic_rcnn_information_flow_learnable_bg_48_17_1x.py checkpoints/BLC_48_17.pth --out results/gzsd_48_17.pkl
                ```
            - our results gzsd_48_17.pkl can also downloaded from [gzsd_48_17.pkl](https://drive.google.com/file/d/1J9_JZDQxVa_GdWDwgmOZ8JOrGtPnpdBD/view?usp=sharing).
            - evaluate:
                ```
                python tools/gzsd_eval.py results/gzsd_48_17.pkl configs/BLC/inference/gzsd/gzsd_cascade_semantic_rcnn_information_flow_learnable_bg_48_17_1x.py
                ```
        - 65/15 split ZSD task:
             - download [65/15](https://drive.google.com/file/d/1r5vrr5sYcIWOGZtH6yllz_i7hVBHkq8k/view?usp=sharing) BLC model, put it in checkpoints/BLC_65_15.pth
            
            - inference:
                ```
                python tools/test.py configs/BLC/inference/gzsd/gzsd_cascade_semantic_rcnn_information_flow_learnable_bg_65_15_1x.py checkpoints/BLC_65_15.pth --out results/gzsd_65_15.pkl
                ```
            - our results gzsd_65_15.pkl can also downloaded from [gzsd_65_15.pkl](https://drive.google.com/file/d/1cDMhQG9N6qW3AxQjrpu-I_om2wJqV8PK/view?usp=sharing).
            - evaluate:
                ```
                python tools/gzsd_eval.py results/gzsd_65_15.pkl configs/BLC/inference/gzsd/gzsd_cascade_semantic_rcnn_information_flow_learnable_bg_65_15_1x.py
                ```
# License

ZSD is released under MIT License.


## Citing

If you use BLC in your research or wish to refer to the baseline results published here, please use the following BibTeX entries:

```BibTeX

@InProceedings{Zheng_2020_ACCV,
    author    = {Zheng, Ye and Huang, Ruoran and Han, Chuanqi and Huang, Xi and Cui, Li},
    title     = {Background Learnable Cascade for Zero-Shot Object Detection},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    month     = {November},
    year      = {2020}
}

```
