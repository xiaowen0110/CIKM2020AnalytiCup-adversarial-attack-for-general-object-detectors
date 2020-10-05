import torch
from torchvision import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils.utils import *
import json
import numpy as np
import os
from tool.darknet2pytorch import *
from infer import infer,gene_atroi
from tqdm import tqdm
from skimage import measure
from utils.helper import generate_attack
from utils.helper import compute_iou
import os
import numpy as np
import skimage.io
from tqdm import tqdm

import sys
sys.path.append('mmdetection/')
from mmdetection.mmdet import __version__
from mmdetection.mmdet.apis import init_detector, inference_detector,attact_detector
from utils.helper import generate_attack,grad_important,bfs,generate_attact_roi,generate_attact_rc
from PIL import Image, ImageDraw, ImageFont
import cv2
import torch
from mmcv import Config, DictAction
from mmdet.models import build_detector
from mmdet.datasets import build_dataset

torch.cuda.set_device(0)
def joint_attack(img_file_dir,epoch,frcnn_e=50,yolo_e=0.5,random_begin=False,pass_success=False):
    #frcnn_e  50对应像素值变化-10～10
    #yolo_e   0.5对应像素值变化-10～10

    #初始化frcnn模型
    config = './mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint = './models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    # config = './mmdetection/configs/ssd/ssd512_coco.py'
    # checkpoint = './models/ssd512_coco_20200308-038c5591.pth'
    #nas-fpn
    # config = './mmdetection/configs/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco.py'
    # checkpoint = './models/retinanet_r50_nasfpn_crop640_50e_coco-0ad1f644.pth'
    #retinanet
    # config = './mmdetection/configs/retinanet/retinanet_x101_64x4d_fpn_1x_coco.py'
    # checkpoint = './models/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth'
    model = init_detector(config, checkpoint, device='cuda:0')

    #初始化yolo模型
    cfgfile = "models/yolov4.cfg"
    weightfile = "models/yolov4.weights"
    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    darknet_model = darknet_model.eval().cuda()

    #图片目录
    img_dir = img_file_dir
    file_name_list = os.listdir(img_dir)
    img_dir2 = img_dir.replace('_p', '')
    #将经过一阶段攻击后的图片放入到 ./select1000_new_atroi目录下
    at_roi_path = img_dir.replace('p', 'atroi')

    #加载已有的攻击结果
    if pass_success==True:
        import json
        with open('whitebox_fasterrcnn_boundingbox_score.json', 'r', encoding='utf8')as fp:
            frcnn_bbox_scores = json.load(fp)
        with open('whitebox_yolo_boundingbox_score.json', 'r', encoding='utf8')as fp:
            yolo_bbox_scores = json.load(fp)

    #记录白盒的攻击效果
    one_scores = 0
    yolo_score = 0
    frcnn_score = 0

    for i in tqdm(range(len(file_name_list))):

        file_name = file_name_list[i]
        if pass_success == True:
            if file_name in frcnn_bbox_scores.keys():

                score_thred=1#攻击成功阈值，两个白盒结果都已经大于阈值，则跳过
                if frcnn_bbox_scores[file_name] >=score_thred:
                    frcnn_score += 1
                if yolo_bbox_scores[file_name] >= score_thred:
                    yolo_score += 1
                if frcnn_bbox_scores[file_name] >=score_thred and yolo_bbox_scores[file_name] >= score_thred:
                #if frcnn_bbox_scores[file_name] < score_thred or yolo_bbox_scores[file_name] <score_thred:
                    one_scores += 1
                    print(file_name, 'both 100% attacked')
                    continue

        if os.path.splitext(file_name)[1] not in ['.jpg', '.png', '.bmp', '.gif']:
            continue
        #读取图片
        img0 = Image.open(img_dir2 + file_name).convert('RGB')#原图
        img1 = Image.open(img_dir + file_name).convert('RGB')#被攻击的初始图，可以在之前的攻击结果上继续训练
        img_atroi=Image.open(at_roi_path+file_name).convert('RGB')#攻击区域图，每张突破只攻击区域中制定的像素点
        img_atroi = np.array(img_atroi)
        img0 = np.array(img0)
        img0 = np.squeeze(img0)
        img1 = np.array(img1)
        img1 = np.squeeze(img1)
        attack_roi = np.where(img_atroi!=img0)
        input_adv=np.array(img1)
        yolo_adv=np.array(img1)
        img0_608 = cv2.resize(np.array(img0), (608, 608))
        at_savepath = img_dir.replace('p', 'at')
        at_imgpath = os.path.join(at_savepath, file_name)
        orgimg = np.array(img0)
        # if cv2.imread(at_imgpath) is not None:
        #     continue

        #获得白盒的ROI，训练时作为输入的gt_bbox
        boxes0 = do_detect(darknet_model, img0_608, 0.5, 0.4, True)
        result_p = inference_detector(model, img1)
        result_c = inference_detector(model, img0)
        if isinstance(result_p, tuple):
            bbox_results, _ = result_p
            result_p = bbox_results
            bbox_results, _ = result_c
            result_c = bbox_results
        result_above_confidence_num_c = 0
        result_c = np.concatenate(result_c)
        gt_bboxes = []
        gt_labels = []
        for ir in range(len(result_c)):
            if result_c[ir, 4] > 0.3:
                result_above_confidence_num_c = result_above_confidence_num_c + 1
                gt_labels.append(0)
                x1, y1, x2, y2 = result_c[ir, :4]
                # print(result_c[ir])
                # gt_bboxes.append(result_c[ir, :4])
                # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                if x2 >= 499: x2 = 499
                if y2 >= 499: y2 = 499
                if x1 <= 0: x1 = 0
                if y1 <= 0: y1 = 0
                h = y2 - y1
                w = x2 - x1
                x0 = x1 + w / 2.0
                y0 = y1 + h / 2.0
                r = result_c[ir, :4]
                r = x0, y0, w, h
                r=x1,y1,x2,y2
                gt_bboxes.append(r)
                # cv2.rectangle(img1,(x1,y1),(x2,y2),[255,255,255])


        if random_begin==True:
            input_adv = attact_detector(model, img0, input_adv, gt_bboxes, gt_labels, file_name, attack_roi= attack_roi,
                                        at_times=1, e=frcnn_e, image_size=800, random_begin=True)

        #交替进行攻击，frcnn10步->yolo10步->frcnn10步， 迭代10个epoch后即可获得白盒2000+
        for i in range(epoch):
            frcnn_adv = attact_detector(model, img0, input_adv, gt_bboxes, gt_labels, file_name,attack_roi=attack_roi,
                                    at_times=10, e=frcnn_e, image_size=800, random_begin=False,rpn=False)
            yolo_adv = yolo_attack(darknet_model, img0_608, orgimg, frcnn_adv, boxes0, attack_roi,at_imgpath,
                                  at_times=10,e=yolo_e,early_stop=1)
            frcnn_adv = attact_detector(model, img0, yolo_adv, gt_bboxes, gt_labels, file_name,attack_roi=attack_roi,
                                    at_times=10, e=frcnn_e, image_size=800, random_begin=False)
            input_adv=frcnn_adv

        #因为攻击图案特殊，若中间某些点训练完成后与原图相同，可能会导致连通域断开
        yolo_adv=frcnn_adv
        ix = np.where(yolo_adv[attack_roi[0],attack_roi[1],attack_roi[2]]==img0[attack_roi[0],attack_roi[1],attack_roi[2]])
        yolo_adv[attack_roi[0][ix],attack_roi[1][ix],attack_roi[2][ix]]=\
            np.where(yolo_adv[attack_roi[0][ix],attack_roi[1][ix],attack_roi[2][ix]]<127,255,0)
        yolo_adv = input_adv
        cv2.imwrite(at_imgpath, yolo_adv[...,::-1])
    print(one_scores, '两种模型均为1分', 'yolo1分个数',yolo_score,' frcnn:',frcnn_score)

def count_detection_score_fasterrcnn(img_file_dir, bb_json_name, output_dir):
    config = './mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint = './models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    infer(config=config, checkpoint=checkpoint, img_file_dir=img_file_dir + '/',
          output_dir=output_dir, json_name=bb_json_name,
          )
    return


def count_detection_score_yolov4(selected_path, json_name, output_dir):
    cfgfile = "models/yolov4.cfg"
    weightfile = "models/yolov4.weights"
    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    darknet_model = darknet_model.eval().cuda()

    files = os.listdir(selected_path)
    files.sort()
    bb_score_dict = {}
    for img_name_index in tqdm(range(len(files))):
        img_name = files[img_name_index]

        img_file_dir2 = selected_path.replace('_p', '')  # clean
        img_path0 = os.path.join(img_file_dir2, img_name)
        img0 = Image.open(img_path0).convert('RGB')

        img_path1 = os.path.join(selected_path, img_name)
        img1 = Image.open(img_path1).convert('RGB')

        resize_small = transforms.Compose([
            transforms.Resize((608, 608)),
        ])
        img0 = resize_small(img0)
        img1 = resize_small(img1)

        # --------------------BOX score
        boxes0 = do_detect(darknet_model, img0, 0.5, 0.4, True)
        boxes1 = do_detect(darknet_model, img1, 0.5, 0.4, True)

        assert len(boxes0) != 0

        bb_score = 1 - min(len(boxes0), len(boxes1)) / len(boxes0)
        bb_score_dict[img_name] = bb_score

    with open(os.path.join(output_dir, json_name), 'w') as f_obj:
        json.dump(bb_score_dict, f_obj)

def count_connected_domin_score(max_total_area_rate, selected_path, max_patch_number, json_name, output_dir):

    files = os.listdir(selected_path)
    resize2 = transforms.Compose([
        transforms.ToTensor()])
    files.sort()


    connected_domin_score_dict = {}

    zero_patch=0
    zero_area=0
    for img_name_index in tqdm(range(len(files))):

        img_name = files[img_name_index]
        img_path0 = os.path.join(selected_path.replace('_p', ''), img_name)
        img0 = Image.open(img_path0).convert('RGB')
        img_path1 = os.path.join(selected_path, img_name)
        img1 = Image.open(img_path1).convert('RGB')
        img0_t = resize2(img0).cuda()
        img1_t = resize2(img1).cuda()
        img_minus_t = img0_t - img1_t

        connected_domin_score, total_area_rate, patch_number = \
            connected_domin_detect_and_score(img_minus_t, max_total_area_rate, max_patch_number)

        if patch_number > max_patch_number:
            connected_domin_score_dict[img_name] = 0.0
            print(img_name,':patch number is too many')
            zero_patch+=1
            continue

        if patch_number == 0:
            connected_domin_score_dict[img_name] = 0.0
            print(img_name,': **********patch number=0**********')
            continue

        if total_area_rate > max_total_area_rate:
            print(img_name,': patch area is too large')
            zero_area+=1
            connected_domin_score_dict[img_name] = 0.0
            continue

        connected_domin_score_dict[img_name] = connected_domin_score
    print('zero_patch:',zero_patch, ' zero_area:',zero_area)
    with open(os.path.join(output_dir, json_name), 'w') as f_obj:
        json.dump(connected_domin_score_dict, f_obj)

def connected_domin_detect_and_score(input_img, max_total_area_rate, max_patch_number):
    # detection
    input_img_new = (input_img[0]+input_img[1]+input_img[2])
    ones = torch.cuda.FloatTensor(input_img_new.size()).fill_(1)
    zeros = torch.cuda.FloatTensor(input_img_new.size()).fill_(0)

    whole_size = input_img_new.shape[0]*input_img_new.shape[1]
    input_map_new = torch.where((input_img_new != 0), ones, zeros)


    labels = measure.label(input_map_new.cpu().numpy()[:, :], background=0, connectivity=2)
    label_max_number = np.max(labels)
    if max_patch_number > 0:
        if label_max_number > max_patch_number:
            return 0, 0, float(label_max_number)
    if label_max_number == 0:
        return 0, 0, 0

    total_area = torch.sum(input_map_new).item()
    total_area_rate = total_area / whole_size
    
    area_score = 2 - float(total_area_rate/max_total_area_rate)
    return float(area_score), float(total_area_rate), float(label_max_number)

def compute_overall_score(json1, json2, output_dir, output_json):

    with open(os.path.join(output_dir, json1)) as f_obj:
        connected_domin_score_dict = json.load(f_obj)

    with open(os.path.join(output_dir, json2)) as f_obj:
        bbox_score_dict = json.load(f_obj)
    assert len(bbox_score_dict) == len(connected_domin_score_dict)
    score_sum = 0
    overall_score = {}
    for (k, _) in bbox_score_dict.items():
        overall_score[k] = connected_domin_score_dict[k] * bbox_score_dict[k]
        score_sum += connected_domin_score_dict[k] * bbox_score_dict[k]
    print('Overall score: ', score_sum)
    print('Saving into {}...'.format(output_json))
    with open(os.path.join(output_dir, output_json), 'w') as f_obj:
        json.dump(overall_score, f_obj)

#从不同的训练结果中选择最优结果
def select():
    path = '最后冲刺/'
    p0 = path + '减0/'
    p202 = path + '减20间隔8/'
    p20 = path + '减20/'
    p30 = path + '减30/'
    p40 = path + '减40/'
    p50 = path + '减50/'
    p60 = path + '减60/'
    p80 = path + '减80/'
    p10 = path + '减10/'
    p70 = path + '减70/'
    #pN = [p0, p10, p20, p30, p40, p50, p60,p70, p80]
    pN=[p20,p30,p40]
    yolo_score = 'whitebox_yolo_overall_score.json'
    frcnn_score = 'whitebox_fasterrcnn_overall_score.json'
    files = os.listdir(p40 + 'images')
    files.sort()
    yolo_bbox_scores = []
    frcnn_bbox_scores = []
    #result = [ 0,0, 0, 0, 0, 0, 0, 0, 0]
    result=[0,0,0]
    for p in pN:
        with open(p + yolo_score, 'r', encoding='utf8')as fp:
            yolo_bbox_scores.append(json.load(fp))
        with open(p + frcnn_score, 'r', encoding='utf8')as fp:
            frcnn_bbox_scores.append(json.load(fp))

    for img_name_index in tqdm(range(len(files))):
        img_name = files[img_name_index]
        ix = -1
        imax = 0
        for i in range(len(pN)):
            # if img_name not in yolo_bbox_scores[i].keys():
            #     scores=0
            # else:
            scores = yolo_bbox_scores[i][img_name] + frcnn_bbox_scores[i][img_name]
            if imax < scores:
                imax = scores
                ix = i
        result[ix] += 1
        maxname = pN[ix] + 'images/' + img_name
        img = cv2.imread(maxname)
        cv2.imwrite(path + 'select/' + img_name, img)
    print(result)

if __name__ == '__main__':
    MAX_TOTAL_AREA_RATE = 0.02  # 5000/(500*500) = 0.02
    selected_path = './select1000_new_p'
    max_patch_number = 10
    output_dir = './output_data'

    # compute_connected_domin_score
    cd_json_name = 'connected_domin_score.json'
    count_connected_domin_score(MAX_TOTAL_AREA_RATE, selected_path, max_patch_number, cd_json_name, output_dir)

    #一阶段，生成通用的攻击区域
    # config = './mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    # checkpoint = './models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    # gene_atroi(config=config, checkpoint=checkpoint, img_file_dir=selected_path + '/',
    #       output_dir=output_dir,reduce=0.3)#recude  减少的像素百分比

    #二阶段， 两个白盒联合训练
    #frcnn_e=50时，frcnn每次训练步长最大约10个像素值
    #yolo_e=0.5时，yolo每次训练不错最大约10个像素值
    #pass_success=True时，跳过两个白盒都已攻击成功的图片，需将对应的线下测试结果'whitebox_fasterrcnn_boundingbox_score.json'
    # 和'whitebox_yolo_boundingbox_score.json'拷贝到根目录下
    #joint_attack(selected_path+'/',frcnn_e=50,yolo_e=0.5,random_begin=True,epoch=1,pass_success=False)

    #3阶段，优化过程，选择最优结果
    select()

    #线下得分
    # bb_json_name = 'whitebox_fasterrcnn_boundingbox_score.json'
    # whitebox_fasterrcnn_result = 'whitebox_fasterrcnn_overall_score.json'
    # count_detection_score_fasterrcnn(selected_path, bb_json_name, output_dir)
    # compute_overall_score(cd_json_name, bb_json_name, output_dir, whitebox_fasterrcnn_result)
    #
    # bb_json_name = 'whitebox_yolo_boundingbox_score.json'
    # whitebox_yolo_result = 'whitebox_yolo_overall_score.json'
    # count_detection_score_yolov4(selected_path, bb_json_name, output_dir)
    # compute_overall_score(cd_json_name, bb_json_name, output_dir, whitebox_yolo_result)
