import os
import numpy as np
import skimage.io
from tqdm import tqdm

import sys
sys.path.append('mmdetection/')
from mmdetection.mmdet import __version__
from mmdet.apis import init_detector, inference_detector,attact_detector
from utils.helper import generate_attack,grad_important,bfs,generate_attact_roi,generate_attact_rc,reduce_roi,generate_atmask
from PIL import Image, ImageDraw, ImageFont
import cv2
import torch
from mmcv import Config, DictAction
from mmdet.models import build_detector
from mmdet.datasets import build_dataset

def infer(config, checkpoint, img_file_dir, output_dir, json_name='bbox_score.json', show_score_thr=0.3):

    model = init_detector(config, checkpoint, device='cuda:0')
    img_dir = img_file_dir
    file_name_list = os.listdir(img_dir)
    img_dir2 = img_dir.replace('_p', '')
    results = {}
    ik = 0
    for i in tqdm(range(len(file_name_list))):
        file_name = file_name_list[i]
        if os.path.splitext(file_name)[1] not in ['.jpg', '.png', '.bmp', '.gif']:
            continue


        img1=cv2.imread(img_dir + file_name)
        img0=cv2.imread(img_dir2+file_name)
        result_c = inference_detector(model, img0)
        result_p = inference_detector(model, img_dir + file_name)
        if isinstance(result_p, tuple):
            bbox_results, _ = result_p
            result_p = bbox_results
            bbox_results, _ = result_c
            result_c = bbox_results
        result_above_confidence_num_p = 0
        result_above_confidence_num_c = 0
        result_p = np.concatenate(result_p)
        result_c = np.concatenate(result_c)

        for ir in range(len(result_c)):
            if result_c[ir, 4] > show_score_thr:
                result_above_confidence_num_c = result_above_confidence_num_c + 1
                # x1, y1, x2, y2 = result_c[ir, :4]
                # if x2 >= 499: x2 = 499
                # if y2 >= 499: y2 = 499
                # if x1 <= 0: x1 = 0
                # if y1 <= 0: y1 = 0
                # cv2.rectangle(img0,(x1,y1),(x2,y2),[255,255,255])
                # cv2.imwrite(img_dir.replace('_p', '_org')+'/'+file_name,img0)


        for ir in range(len(result_p)):
            if result_p[ir, 4] > show_score_thr:
                result_above_confidence_num_p = result_above_confidence_num_p + 1
                # x1, y1, x2, y2 = result_p[ir, :4]
                # if x2 >= 499: x2 = 499
                # if y2 >= 499: y2 = 499
                # if x1 <= 0: x1 = 0
                # if y1 <= 0: y1 = 0
                # cv2.rectangle(img1,(x1,y1),(x2,y2),[255,255,255])
                # cv2.imwrite(img_dir.replace('_p', '_at')+'/'+file_name,img1)

        if result_above_confidence_num_c == 0:  # can't find any object in clean img
            bb_score = 0
            print('i=', ik)
            print(file_name)
            ik += 1
        else:
            bb_score = 1 - min(result_above_confidence_num_c,
                               result_above_confidence_num_p) / result_above_confidence_num_c
        results[file_name] = bb_score
    import json
    with open(os.path.join(output_dir, json_name), 'w') as f_obj:
        json.dump(results, f_obj)
    return results

def gene_atroi(config, checkpoint, img_file_dir, output_dir, show_score_thr=0.3,reduce=0.2):
    # Cfg = Config.fromfile(config)
    # atmodel = build_detector(
    #     Cfg.model, train_cfg=Cfg.train_cfg, test_cfg=Cfg.test_cfg)

    model = init_detector(config, checkpoint, device='cuda')
    img_dir = img_file_dir
    file_name_list = os.listdir(img_dir)
    img_dir2 = img_dir.replace('_p', '')
    results = {}
    ik = 0
    one_scores=0

    for i in tqdm(range(len(file_name_list))):
        file_name = file_name_list[i]
        result_p = inference_detector(model, img_dir + file_name)
        result_c = inference_detector(model, img_dir2 + file_name)
        if isinstance(result_p, tuple):
            bbox_results, _ = result_p
            result_p = bbox_results
            bbox_results, _ = result_c
            result_c = bbox_results

        result_above_confidence_num_p = 0
        result_above_confidence_num_c = 0
        result_p = np.concatenate(result_p)
        result_c = np.concatenate(result_c)

        rois=[]
        areas=[]
        scores=[]
        gt_bboxes=[]
        gt_labels=[]
        # roi_dir = img_dir.replace('p', 'froi')
        # txtfile = file_name.replace('png', 'txt')
        # device=torch.device('cuda:1')
        img0 = Image.open(img_dir2 + file_name).convert('RGB')
        img1 = Image.open(img_dir + file_name).convert('RGB')

        img0 =np.array(img0)
        img0 = np.squeeze(img0)
        img1 = np.array(img1)
        img1 = np.squeeze(img1)
        img0_copy = np.array(img0)
        for ir in range(len(result_c)):
            if result_c[ir, 4] > show_score_thr:
                result_above_confidence_num_c = result_above_confidence_num_c + 1
                gt_labels.append(80)
                x1,y1,x2,y2 = result_c[ir, :4]
                gt_bboxes.append(result_c[ir,:4])
                rois.append(result_c[ir, :4])
                scores.append(result_c[ir,4])
                areas.append((x2-x1)*(y2-y1))
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                if x2 >= 499: x2 = 499
                if y2 >= 499: y2 = 499
                if x1 <= 0: x1 = 0
                if y1 <= 0: y1 = 0
                #cv2.rectangle(img1,(x1,y1),(x2,y2),[255,255,255])
        # reduce_img=reduce_roi(img0,img1,rois=rois,thred=50,reduce=0.2)
        # cv2.imwrite(img_dir.replace('p', 'at')+'/'+file_name,reduce_img[...,::-1])
        #generate_atmask(img0,rois,scores,file_name,yolo=False)
        generate_attack(img0, rois, scores, file_name, yolo=False,reduce=reduce)
        # attack_roi = np.where(img0!=img1)
        # data_grad_,adv_x= attact_detector(model,img0,img1,attack_roi,gt_bboxes,gt_labels,file_name)
        # ix = np.where(adv_x[attack_roi]==img0[attack_roi])
        # adv_x[attack_roi[0][ix],attack_roi[1][ix],attack_roi[2][ix]]=\
        #     255-adv_x[attack_roi[0][ix],attack_roi[1][ix],attack_roi[2][ix]]
        # cv2.imwrite(img_dir.replace('p', 'at') + file_name, adv_x[:, :, ::-1])

        # data_grad_abs,atcked_img = grad_important(data_grad_,img0,direct=True)
        #
        # data_grad_abs = np.sum(data_grad_abs, axis=-1)
        # data_grad_abs = data_grad_abs.reshape([-1, ])
        # grad_sort_ix = np.argsort(data_grad_abs)
        # threshld = data_grad_abs[grad_sort_ix][-5000]
        # data_grad_abs = data_grad_abs.reshape([500, 500])
        #
        # visit = generate_attact_rc(data_grad_abs,rois,areas)
        # attack_roi=np.where(visit==1)
        # #attack_roi = np.where(img0!=img1)
        #
        # iy, ix = np.where(data_grad_abs >= threshld)
        # img1=np.zeros_like(img1)
        # img1[iy,ix]=255
        #
        # img0_inverse = np.array(img0)
        # ix0 = np.where(img0 > 127)
        # ix1 = np.where(img0 <= 127)
        # img0_inverse[ix0] = 0
        # img0_inverse[ix1] = 255
        #
        # #img0[attack_roi]=img0_inverse[attack_roi]
        # img0[attack_roi]=atcked_img[attack_roi]
        # unat = np.where(abs(img0_copy[attack_roi] - img0[attack_roi]) <= 10)
        # img0[attack_roi[0][unat], attack_roi[1][unat], attack_roi[2][unat]] = \
        #     img0_inverse[attack_roi[0][unat], attack_roi[1][unat], attack_roi[2][unat]]
        #
        # cv2.imwrite(img_dir.replace('p','at') + file_name, img1[:, :, ::-1])

        #np.savetxt(roi_dir+txtfile,np.array(rois),fmt='%d')
        # img0 = Image.open(img_dir2 + file_name).convert('RGB')
        # img0 = np.array(img0)

        for ir in range(len(result_p)):
            if result_p[ir, 4] > show_score_thr:
                result_above_confidence_num_p = result_above_confidence_num_p + 1

        if result_above_confidence_num_c == 0:  # can't find any object in clean img
            bb_score = 0
            print('i=', ik)
            print(file_name)
            ik += 1
        else:
            bb_score = 1 - min(result_above_confidence_num_c,
                               result_above_confidence_num_p) / result_above_confidence_num_c
        results[file_name] = bb_score
    return results
