# coding=utf-8
"""
@Data: 2021/1/4
@Author: 算影
@Email: wangmang.wm@alibaba-inc.com
"""
import os
import time
import json
from pycocotools.coco import COCO

def sel_cat(anno_file, sel_num, only_last=False):
    print('loading annotations into memory...')
    tic = time.time()
    # dataset = json.load(open(anno_file, 'r'))
    coco  = COCO(anno_file)
    # assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
    print('Done (t={:0.2f}s)'.format(time.time()- tic))

    dataset = coco.dataset
    # sort by cat_ids
    dataset['categories'] = sorted(dataset['categories'], key=lambda k: k['id'])

    # select first 40 cats
    sel_cats = dataset['categories'][:sel_num]
    # selected categories
    sel_cats_ids = [cat['id'] for cat in sel_cats]
    sel_anno = []
    # selected images
    img_ids = set()
    for cat_id in sel_cats_ids:
        img_ids |= set(coco.getImgIds(catIds=cat_id))
    sel_images = coco.loadImgs(img_ids)
    # selected annotations
    for imgId in img_ids :
        if imgId in coco.imgToAnns:
            sel_anno.extend([ann for ann in coco.imgToAnns[imgId] if ann['category_id'] in sel_cats_ids])
    sel_dataset = dict()
    sel_dataset['categories'] = sel_cats
    sel_dataset['annotations'] = sel_anno
    sel_dataset['images'] = sel_images
    # writing results
    with open(os.path.splitext(anno_file)[0] + f'_sel_first_{sel_num}_cats.json', 'w') as f:
        f.write(json.dumps(sel_dataset))

    
    # select last 40 cats
    sel_cats = dataset['categories'][sel_num:]
    # selected annotations
    sel_cats_ids = [cat['id'] for cat in sel_cats]
    sel_anno = []
    # selected images
    img_ids = set()
    for cat_id in sel_cats_ids:
        img_ids |= set(coco.getImgIds(catIds=cat_id))
    sel_images = coco.loadImgs(img_ids)
    # selected annotations
    new_img_ids = []
    for imgId in img_ids :
        if imgId in coco.imgToAnns:
            if not only_last:
                sel_anno.extend([ann for ann in coco.imgToAnns[imgId] if ann['category_id'] in sel_cats_ids])
                new_img_ids.append(imgId)
            else:
                if set([ann['category_id'] for ann in coco.imgToAnns[imgId]]).issubset(sel_cats_ids):
                    sel_anno.extend([ann for ann in coco.imgToAnns[imgId] if ann['category_id'] in sel_cats_ids])
                    new_img_ids.append(imgId)
    sel_images = coco.loadImgs(new_img_ids)
    # selected dataset
    sel_dataset = dict()
    sel_dataset['categories'] = sel_cats
    sel_dataset['annotations'] = sel_anno
    sel_dataset['images'] = sel_images
    key = ''
    if only_last:
        key = '_only'
    # writing results
    with open(os.path.splitext(anno_file)[0] + f'{key}_sel_last_{sel_num}_cats.json', 'w') as f :
        f.write(json.dumps(sel_dataset))


if __name__ == "__main__":
    # anno_file = 'data/coco/annotations/instances_val2017.json'
    # anno_file = 'data/coco/annotations/instances_train2017.json'
    anno_file = 'data/VOCdevkit/anns_coco_fmt/voc0712_trainval.json'
    # anno_file = 'data/VOCdevkit/anns_coco_fmt/voc07_test.json'
    sel_num = 10
    sel_cat(anno_file, sel_num, False)
