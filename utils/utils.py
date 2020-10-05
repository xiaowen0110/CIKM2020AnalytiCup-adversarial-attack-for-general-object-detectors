import sys
import os
import time
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable

import itertools
import struct  # get_image_size
import imghdr  # get_image_size


def sigmoid(x):
    return 1.0 / (np.exp(-x) + 1.)


def softmax(x):
    x = np.exp(x - np.expand_dims(np.max(x, axis=1), axis=1))
    x = x / np.expand_dims(x.sum(axis=1), axis=1)
    return x


def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0] - box1[2] / 2.0, box2[0] - box2[2] / 2.0)
        Mx = max(box1[0] + box1[2] / 2.0, box2[0] + box2[2] / 2.0)
        my = min(box1[1] - box1[3] / 2.0, box2[1] - box2[3] / 2.0)
        My = max(box1[1] + box1[3] / 2.0, box2[1] + box2[3] / 2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea / uarea


def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = torch.min(boxes1[0], boxes2[0])
        Mx = torch.max(boxes1[2], boxes2[2])
        my = torch.min(boxes1[1], boxes2[1])
        My = torch.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = torch.min(boxes1[0] - boxes1[2] / 2.0, boxes2[0] - boxes2[2] / 2.0)
        Mx = torch.max(boxes1[0] + boxes1[2] / 2.0, boxes2[0] + boxes2[2] / 2.0)
        my = torch.min(boxes1[1] - boxes1[3] / 2.0, boxes2[1] - boxes2[3] / 2.0)
        My = torch.max(boxes1[1] + boxes1[3] / 2.0, boxes2[1] + boxes2[3] / 2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea / uarea


def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes

    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1 - boxes[i][4]

    _, sortIds = torch.sort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i + 1, len(boxes)):
                box_j = boxes[sortIds[j]]
                if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                    # print(box_i, box_j, bbox_iou(box_i, box_j, x1y1x2y2=False))
                    box_j[4] = 0
    return out_boxes


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)


def get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1, validation=False):
    anchor_step = len(anchors) // num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert (output.size(1) == (5 + num_classes) * num_anchors)
    h = output.size(2)
    w = output.size(3)

    t0 = time.time()
    all_boxes = []
    output = output.view(batch * num_anchors, 5 + num_classes, h * w).transpose(0, 1).contiguous().view(5 + num_classes,
                                                                                                        batch * num_anchors * h * w)

    grid_x = torch.linspace(0, w - 1, w).repeat(h, 1).repeat(batch * num_anchors, 1, 1).view(
        batch * num_anchors * h * w).type_as(output)  # cuda()
    grid_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().repeat(batch * num_anchors, 1, 1).view(
        batch * num_anchors * h * w).type_as(output)  # cuda()
    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y

    anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
    anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
    anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w).type_as(output)  # cuda()
    anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w).type_as(output)  # cuda()
    ws = torch.exp(output[2]) * anchor_w
    hs = torch.exp(output[3]) * anchor_h

    det_confs = torch.sigmoid(output[4])

    cls_confs = torch.nn.Softmax()(Variable(output[5:5 + num_classes].transpose(0, 1))).data
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)
    t1 = time.time()

    sz_hw = h * w
    sz_hwa = sz_hw * num_anchors
    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs = convert2cpu(xs)
    ys = convert2cpu(ys)
    ws = convert2cpu(ws)
    hs = convert2cpu(hs)
    if validation:
        cls_confs = convert2cpu(cls_confs.view(-1, num_classes))
    t2 = time.time()
    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b * sz_hwa + i * sz_hw + cy * w + cx
                    det_conf = det_confs[ind]
                    if only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]

                    if conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx / w, bcy / h, bw / w, bh / h, det_conf, cls_max_conf, cls_max_id]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind] * tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
        all_boxes.append(boxes)
    t3 = time.time()
    if False:
        print('---------------------------------')
        print('matrix computation : %f' % (t1 - t0))
        print('        gpu to cpu : %f' % (t2 - t1))
        print('      boxes filter : %f' % (t3 - t2))
        print('---------------------------------')
    return all_boxes


def get_region_boxes1(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1, validation=False):
    anchor_step = len(anchors) // num_anchors
    if len(output.shape) == 3:
        output = np.expand_dims(output, axis=0)
    batch = output.shape[0]
    assert (output.shape[1] == (5 + num_classes) * num_anchors)
    h = output.shape[2]
    w = output.shape[3]

    t0 = time.time()
    all_boxes = []
    output = output.reshape(batch * num_anchors, 5 + num_classes, h * w).transpose((1, 0, 2)).reshape(
        5 + num_classes,
        batch * num_anchors * h * w)

    grid_x = np.expand_dims(np.expand_dims(np.linspace(0, w - 1, w), axis=0).repeat(h, 0), axis=0).repeat(
        batch * num_anchors, axis=0).reshape(
        batch * num_anchors * h * w)
    grid_y = np.expand_dims(np.expand_dims(np.linspace(0, h - 1, h), axis=0).repeat(w, 0).T, axis=0).repeat(
        batch * num_anchors, axis=0).reshape(
        batch * num_anchors * h * w)

    xs = sigmoid(output[0]) + grid_x
    ys = sigmoid(output[1]) + grid_y

    anchor_w = np.array(anchors).reshape((num_anchors, anchor_step))[:, 0]
    anchor_h = np.array(anchors).reshape((num_anchors, anchor_step))[:, 1]
    anchor_w = np.expand_dims(np.expand_dims(anchor_w, axis=1).repeat(batch, 1), axis=2) \
        .repeat(h * w, axis=2).transpose(1, 0, 2).reshape(batch * num_anchors * h * w)
    anchor_h = np.expand_dims(np.expand_dims(anchor_h, axis=1).repeat(batch, 1), axis=2) \
        .repeat(h * w, axis=2).transpose(1, 0, 2).reshape(batch * num_anchors * h * w)
    ws = np.exp(output[2]) * anchor_w
    hs = np.exp(output[3]) * anchor_h

    det_confs = sigmoid(output[4])

    cls_confs = softmax(output[5:5 + num_classes].transpose(1, 0))
    cls_max_confs = np.max(cls_confs, 1)
    cls_max_ids = np.argmax(cls_confs, 1)
    t1 = time.time()

    sz_hw = h * w
    sz_hwa = sz_hw * num_anchors
    t2 = time.time()
    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b * sz_hwa + i * sz_hw + cy * w + cx
                    det_conf = det_confs[ind]
                    if only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]

                    if conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx / w, bcy / h, bw / w, bh / h, det_conf, cls_max_conf, cls_max_id]
                        #box = [bcx , bcy , bw , bh , det_conf, cls_max_conf, cls_max_id]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind] * tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
        all_boxes.append(boxes)
    t3 = time.time()
    if False:
        print('---------------------------------')
        print('matrix computation : %f' % (t1 - t0))
        print('        gpu to cpu : %f' % (t2 - t1))
        print('      boxes filter : %f' % (t3 - t2))
        print('---------------------------------')
    return all_boxes


def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None):
    import cv2
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]);

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int((box[0] - box[2] / 2.0) * width)
        y1 = int((box[1] - box[3] / 2.0) * height)
        x2 = int((box[0] + box[2] / 2.0) * width)
        y2 = int((box[1] + box[3] / 2.0) * height)

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            img = cv2.putText(img, class_names[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img


def plot_boxes(img, boxes, savename=None, class_names=None):
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]);

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[0] - box[2] / 2.0) * width
        y1 = (box[1] - box[3] / 2.0) * height
        x2 = (box[0] + box[2] / 2.0) * width
        y2 = (box[1] + box[3] / 2.0) * height

        rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            # box[5] should be the object_conf and it is ignored
            cls_conf = box[5]
            cls_id = box[6]
            # print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            rgb = (red, green, blue)
            draw.text((x1, y1), class_names[cls_id], fill=rgb)
        draw.rectangle([x1, y1, x2, y2], outline=rgb)
    if savename:
        print("save plot results to %s" % savename)
        img.save(savename)
    return img


def read_truths(lab_path):
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(truths.size / 5, 5)  # to avoid single truth problem
        return truths
    else:
        return np.array([])


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=1):
    model.eval()
    t0 = time.time()

    if isinstance(img, Image.Image):
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)
    elif type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    elif type(img) == np.ndarray and len(img.shape) == 4:
        img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
    elif type(img) == torch.Tensor and len(img.shape) == 4:
        img = img
    else:
        print("unknow image type")
        exit(-1)

    t1 = time.time()

    if use_cuda:
        img = img.cuda()
    img = torch.autograd.Variable(img)
    #img.requires_grad = True

    t2 = time.time()

    list_boxes = model(img)

    # criterion = Yolo_loss(device=device, batch=1, n_classes=80)
    # bboxes = torch.zeros((1,5,5),dtype=torch.float)
    # bboxes.to(device)
    #
    # loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = criterion(list_boxes, bboxes)
    # print('loss:',loss)
    # model.zero_grad()
    # loss.backward()
    #
    # data_grad = img.grad.data
    # print('data_grad:', data_grad.shape)

    anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    num_anchors = 9
    anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    strides = [8, 16, 32]
    anchor_step = len(anchors) // num_anchors
    boxes = []
    for i in range(3):
        masked_anchors = []
        for m in anchor_masks[i]:
            masked_anchors += anchors[m * anchor_step:(m + 1) * anchor_step]
        masked_anchors = [anchor / strides[i] for anchor in masked_anchors]
        # temp = get_region_boxes1(list_boxes[i].data.numpy(), conf_thresh, 80, masked_anchors, len(anchor_masks[i]))
        # boxes.append(temp)
        boxes.append(get_region_boxes1(list_boxes[i].data.cpu().numpy(), conf_thresh, 80, masked_anchors, len(anchor_masks[i])))
        # boxes.append(get_region_boxes(list_boxes[i], 0.6, 80, masked_anchors, len(anchor_masks[i])))
    if img.shape[0] > 1:
        bboxs_for_imgs = [
            boxes[0][index] + boxes[1][index] + boxes[2][index]
            for index in range(img.shape[0])]
        t3 = time.time()
        boxes = [nms(bboxs, nms_thresh) for bboxs in bboxs_for_imgs]
    else:
        boxes = boxes[0][0] + boxes[1][0] + boxes[2][0]
        t3 = time.time()
        boxes = nms(boxes, nms_thresh)
    t4 = time.time()

    if False:
        print('-----------------------------------')
        print(' image to tensor : %f' % (t1 - t0))
        print('  tensor to cuda : %f' % (t2 - t1))
        print('         predict : %f' % (t3 - t2))
        print('             nms : %f' % (t4 - t3))
        print('           total : %f' % (t4 - t0))
        print('-----------------------------------')
    return boxes

import cv2
from utils.helper import bfs,grad_important,generate_attact_rc,sigle_step_attack
def yolo_attack(model,img,orgimg,dstimg,target,attack_roi,filename=None,use_cuda=1,at_times=50,e=1.0,early_stop=1):
    model.eval()
    t0 = time.time()
    img0=img
    target=np.array(target)
    label_chanel=target.shape[1]
    target = torch.tensor(target)
    if isinstance(img, Image.Image):
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)
    elif type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    elif type(img) == np.ndarray and len(img.shape) == 4:
        img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
    elif type(img) == torch.Tensor and len(img.shape) == 4:
        img = img
    else:
        print("unknow image type")
        exit(-1)

    # 获得标签
    target = target.view(1, -1, label_chanel)
    if label_chanel==7:
        target[:, :, :4] = target[:, :, :4] * 608
        target[:, :, 4] = 1
    else:
        target[:, :, :4] = target[:, :, :4] /500*608
    target = target[:, :, :5]
    target.to(device)

    img0 = cv2.resize(img0,(500,500))
    img0=img0/255.0
    pertubation = np.zeros_like(img0)

    adv_x = dstimg/255
    all_grad=np.zeros_like(img0)
    loss_last=0
    times=0
    logs=[]
    momentent=np.zeros_like(img0)
    for i in range(at_times):
        #adv_x[attack_roi]=adv_x[attack_roi]+pertubation[attack_roi]
        img = cv2.resize(adv_x,(608,608))
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
        if use_cuda:
            img = img.cuda()

        img = torch.autograd.Variable(img)
        img.requires_grad = True
        list_boxes = model(img)
        criterion = Yolo_loss(device=device, batch=1, n_classes=80)

        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = criterion(list_boxes, target)
        if (i+1)%5==0 or (i+1)==at_times:
            print(filename.split('/')[-1],' yolo step:%d loss:'%(i+1),loss_obj)
            if abs(loss_obj - loss_last) < 0.1:
                times += 1
                # e=e/2
                # if times >= 10:
                #     break
            loss_last = loss_obj
            logs.append(loss_obj)
        # if loss_obj<1:
        #     print(filename,'***********allready attacked')
        #     break

        loss_back = loss_obj
        if loss_back<early_stop:break
        model.zero_grad()
        loss_back.backward()

        pertubation = img.grad.data
        pertubation= pertubation.cpu()
        pertubation= pertubation.numpy()
        pertubation=pertubation.squeeze()
        pertubation = pertubation.transpose(1,2,0)
        pertubation = cv2.resize(pertubation,(500,500))
        if loss_back<1:
            momentent = 0.9 * momentent + e/2 * pertubation
        else:
            momentent = 0.9 * momentent + e *pertubation
        # grad_range=np.sort(np.reshape(momentent[attack_roi],[-1]))
        # print('梯度范围', grad_range[:100],grad_range[-100:])

        x=np.array(adv_x)
        adv_x[attack_roi]=adv_x[attack_roi]-momentent[attack_roi]
        adv_x = np.clip(adv_x,0,1.0)
        pertubation = adv_x-x
        all_grad[attack_roi]=all_grad[attack_roi]+pertubation[attack_roi]


    target = target[:, :, :4].cpu()
    target = target.numpy()
    target = target.reshape(-1, 4)
    rois = []
    areas = []
    for i in range(target.shape[0]):
        r = target[i] / 608 * 500
        x, y, w, h = r
        x1, x2 = x - w / 2, x + w / 2
        y1, y2 = y - h / 2, y + h / 2
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        r = [x1, y1, x2, y2]
        rois.append(r)
        areas.append(w * h)
    all_grad=all_grad*255
    adv_x=adv_x*255
    org_copy = np.array(orgimg)
    org_copy[attack_roi]=adv_x[attack_roi]
    return org_copy

def pgd_attack(model,img,orgimg,dstimg,target,filename=None,use_cuda=1,at_times=50,e=1.0):
    attaked = cv2.imread(filename)
    if attaked is not None:
        print('***********attacked!')
        return dstimg
    model.eval()
    t0 = time.time()
    img0=img
    target=np.array(target)
    label_chanel=target.shape[1]
    target = torch.tensor(target)
    if isinstance(img, Image.Image):
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)
    elif type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    elif type(img) == np.ndarray and len(img.shape) == 4:
        img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
    elif type(img) == torch.Tensor and len(img.shape) == 4:
        img = img
    else:
        print("unknow image type")
        exit(-1)

    # 获得标签
    target = target.view(1, -1, label_chanel)
    if label_chanel==7:
        target[:, :, :4] = target[:, :, :4] * 608
        target[:, :, 4] = 1
    else:
        target[:, :, :4] = target[:, :, :4] /500*608
    target = target[:, :, :5]
    target.to(device)

    attack_roi = np.where(orgimg!=dstimg)
    img0 = cv2.resize(img0,(500,500))
    img0=img0/255.0
    pertubation = np.zeros_like(img0)

    adv_x = dstimg/255
    all_grad=np.zeros_like(img0)
    loss_last=0
    times=0
    logs=[]
    for i in range(at_times):
        #adv_x[attack_roi]=adv_x[attack_roi]+pertubation[attack_roi]
        img = cv2.resize(adv_x,(608,608))
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
        if use_cuda:
            img = img.cuda()

        img = torch.autograd.Variable(img)
        img.requires_grad = True
        list_boxes = model(img)
        criterion = Yolo_loss(device=device, batch=1, n_classes=80)

        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = criterion(list_boxes, target)
        if (i+1)%25==0:
            if i+1==25:
                print('\n')
            print(filename,' %d loss:'%(i+1),loss_obj)
            if abs(loss_obj - loss_last) < 0.1:
                times += 1
                # e=e/2
                if times >= 10:
                    break
            loss_last = loss_obj
            logs.append(loss_obj)
        # if loss_obj<1:
        #     print(filename,'***********allready attacked')
        #     break

        loss_back = loss_obj
        model.zero_grad()
        loss_back.backward()

        pertubation = -img.grad.data
        pertubation= pertubation.cpu()
        pertubation= pertubation.numpy()
        pertubation=pertubation.squeeze()
        pertubation = pertubation.transpose(1,2,0)
        pertubation = cv2.resize(pertubation,(500,500))

        x=np.array(adv_x)
        adv_x[attack_roi]=e*pertubation[attack_roi]+adv_x[attack_roi]

        adv_x = np.clip(adv_x,0,1.0)
        pertubation = adv_x-x
        all_grad[attack_roi]=all_grad[attack_roi]+pertubation[attack_roi]


    target = target[:, :, :4].cpu()
    target = target.numpy()
    target = target.reshape(-1, 4)
    rois = []
    areas = []
    for i in range(target.shape[0]):
        r = target[i] / 608 * 500
        x, y, w, h = r
        x1, x2 = x - w / 2, x + w / 2
        y1, y2 = y - h / 2, y + h / 2
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        r = [x1, y1, x2, y2]
        rois.append(r)
        areas.append(w * h)
    all_grad=all_grad*255
    adv_x=adv_x*255
    org_copy = np.array(orgimg)

    orgimg[attack_roi]=adv_x[attack_roi]


    # ix = np.where(abs(all_grad[attack_roi])<10)
    # all_grad[attack_roi[0][ix],attack_roi[1][ix],attack_roi[2][ix]]=\
    #     1e10*all_grad[attack_roi[0][ix],attack_roi[1][ix],attack_roi[2][ix]]
    # orgimg[attack_roi] = orgimg[attack_roi] + all_grad[attack_roi]
    ix_n = np.where(orgimg[attack_roi]==org_copy[attack_roi])
    orgimg[attack_roi[0][ix_n],attack_roi[1][ix_n],attack_roi[2][ix_n]]\
        =255-org_copy[attack_roi[0][ix_n],attack_roi[1][ix_n],attack_roi[2][ix_n]]

    cv2.imwrite(filename, orgimg[:, :, ::-1])
    logtxt=filename.replace('at','logs')
    logtxt=logtxt.replace('png','txt')
    np.savetxt(logtxt,logs,fmt='%.5e')
    boxes = []
    return boxes

def do_attack(model, img, conf_thresh, nms_thresh, use_cuda=1,target=None,dstimg=None,orgimg=None,attack_roi=None,filename=None):
    model.eval()
    t0 = time.time()
    target = torch.tensor(target)
    if isinstance(img, Image.Image):
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)
    elif type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    elif type(img) == np.ndarray and len(img.shape) == 4:
        img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
    elif type(img) == torch.Tensor and len(img.shape) == 4:
        img = img
    else:
        print("unknow image type")
        exit(-1)


    if use_cuda:
        img = img.cuda()
    #获得标签
    target = target.view(1,-1,7)
    target[:,:,:4]=target[:,:,:4]*608
    target[:,:,4]=1
    target=target[:,:,:5]
    target.to(device)

    img = torch.autograd.Variable(img)
    img.requires_grad = True
    list_boxes = model(img)
    criterion = Yolo_loss(device=device, batch=1, n_classes=80)

    loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = criterion(list_boxes, target)
    print('loss:',loss_obj)
    loss_back=loss_obj
    model.zero_grad()
    loss_back.backward()

    data_grad = img.grad.data

    data_grad_ = data_grad.cpu()
    data_grad_ = data_grad_.numpy()

    data_grad_ = data_grad_.squeeze()
    # std = np.std(data_grad_)
    # means = np.mean(data_grad_)
    # data_grad_ = (data_grad_ - means) / (std + 1e-10)

    data_grad_ = data_grad_.transpose([1, 2, 0])
    data_grad_ = cv2.resize(data_grad_, (500, 500))

    data_grad_abs, atcked_img = grad_important(data_grad_, orgimg, direct=True)


    data_grad_abs = np.sum(data_grad_abs, axis=-1)
    data_grad_abs = data_grad_abs.reshape([-1, ])
    grad_sort_ix = np.argsort(data_grad_abs)
    threshld = data_grad_abs[grad_sort_ix][-5]
    data_grad_abs = data_grad_abs.reshape([500, 500])


    target=target[:,:,:4].cpu()
    target=target.numpy()
    target=target.reshape(-1,4)
    rois=[]
    areas=[]
    for i in range(target.shape[0]):
        r=target[i]/608*500
        x,y,w,h=r
        x1, x2 = x - w / 2, x + w / 2
        y1, y2 = y - h / 2, y + h / 2
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        r= [x1, y1, x2, y2]
        rois.append(r)
        areas.append(w*h)

    visit = generate_attact_rc(data_grad_abs, rois, areas)
    attack_roi = np.where(visit == 1)
    # attack_roi = np.where(img0!=img1)

    # iy, ix = np.where(data_grad_abs >= threshld)
    # img1=np.zeros_like(img1)
    # img1[iy,ix]=255
    img0 = np.array(orgimg)
    img0_inverse=np.array(img0)
    ix0=np.where(img0>127)
    ix1=np.where(img0<=127)
    img0_inverse[ix0]=0
    img0_inverse[ix1]=255

    orgimg[attack_roi] = atcked_img[attack_roi]
    #orgimg[attack_roi] = img0_inverse[attack_roi]

    unat = np.where(abs(orgimg[attack_roi] - img0[attack_roi]) <= 100)
    orgimg[attack_roi[0][unat], attack_roi[1][unat], attack_roi[2][unat]] = \
        img0_inverse[attack_roi[0][unat], attack_roi[1][unat], attack_roi[2][unat]]

    # img1 = np.zeros_like(img1)
    # img1[iy, ix] = 255
    cv2.imwrite(filename,orgimg[:,:,::-1])

    anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    num_anchors = 9
    anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    strides = [8, 16, 32]
    anchor_step = len(anchors) // num_anchors
    boxes = []
    # for i in range(3):
    #     masked_anchors = []
    #     for m in anchor_masks[i]:
    #         masked_anchors += anchors[m * anchor_step:(m + 1) * anchor_step]
    #     masked_anchors = [anchor / strides[i] for anchor in masked_anchors]
    #     # temp = get_region_boxes1(list_boxes[i].data.numpy(), conf_thresh, 80, masked_anchors, len(anchor_masks[i]))
    #     # boxes.append(temp)
    #     boxes.append(get_region_boxes1(list_boxes[i].data.cpu().numpy(), conf_thresh, 80, masked_anchors, len(anchor_masks[i])))
    #     # boxes.append(get_region_boxes(list_boxes[i], 0.6, 80, masked_anchors, len(anchor_masks[i])))
    # if img.shape[0] > 1:
    #     bboxs_for_imgs = [
    #         boxes[0][index] + boxes[1][index] + boxes[2][index]
    #         for index in range(img.shape[0])]
    #     t3 = time.time()
    #     boxes = [nms(bboxs, nms_thresh) for bboxs in bboxs_for_imgs]
    # else:
    #     boxes = boxes[0][0] + boxes[1][0] + boxes[2][0]
    #     t3 = time.time()
    #     boxes = nms(boxes, nms_thresh)
    # t4 = time.time()
    #
    # if False:
    #     print('-----------------------------------')
    #     print(' image to tensor : %f' % (t1 - t0))
    #     print('  tensor to cuda : %f' % (t2 - t1))
    #     print('         predict : %f' % (t3 - t2))
    #     print('             nms : %f' % (t4 - t3))
    #     print('           total : %f' % (t4 - t0))
    #     print('-----------------------------------')
    return boxes

import torch.nn as nn
import torch.nn.functional as F
class Yolo_loss(nn.Module):
    def __init__(self, n_classes=80, n_anchors=3, device=None, batch=2):
        super(Yolo_loss, self).__init__()
        self.device = device
        self.strides = [8, 16, 32]
        image_size = 608
        self.n_classes = n_classes
        self.n_anchors = n_anchors

        self.anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]]
        self.anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.ignore_thre = 0.5

        self.masked_anchors, self.ref_anchors, self.grid_x, self.grid_y, self.anchor_w, self.anchor_h = [], [], [], [], [], []

        for i in range(3):
            all_anchors_grid = [(w / self.strides[i], h / self.strides[i]) for w, h in self.anchors]
            masked_anchors = np.array([all_anchors_grid[j] for j in self.anch_masks[i]], dtype=np.float32)
            ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)
            ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
            ref_anchors = torch.from_numpy(ref_anchors)
            # calculate pred - xywh obj cls
            fsize = image_size // self.strides[i]
            grid_x = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).to(device)
            grid_y = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).permute(0, 1, 3, 2).to(device)
            anchor_w = torch.from_numpy(masked_anchors[:, 0]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(
                device)
            anchor_h = torch.from_numpy(masked_anchors[:, 1]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(
                device)

            self.masked_anchors.append(masked_anchors)
            self.ref_anchors.append(ref_anchors)
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)

    def build_target(self, pred, labels, batchsize, fsize, n_ch, output_id):
        # target assignment
        tgt_mask = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 4 + self.n_classes).to(device=self.device)
        obj_mask = torch.ones(batchsize, self.n_anchors, fsize, fsize).to(device=self.device)
        tgt_scale = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2).to(self.device)
        target = torch.zeros(batchsize, self.n_anchors, fsize, fsize, n_ch).to(self.device)

        # labels = labels.cpu().data
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        truth_x_all = (labels[:, :, 2] + labels[:, :, 0]) / (self.strides[output_id] * 2)
        truth_y_all = (labels[:, :, 3] + labels[:, :, 1]) / (self.strides[output_id] * 2)
        truth_w_all = (labels[:, :, 2] - labels[:, :, 0]) / self.strides[output_id]
        truth_h_all = (labels[:, :, 3] - labels[:, :, 1]) / self.strides[output_id]
        truth_i_all = truth_x_all.to(torch.int16).cpu().numpy()
        truth_j_all = truth_y_all.to(torch.int16).cpu().numpy()

        for b in range(batchsize):
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = torch.zeros(n, 4).to(self.device)
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # calculate iou between truth and reference anchors
            anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors[output_id], CIoU=True)

            # temp = bbox_iou(truth_box.cpu(), self.ref_anchors[output_id])

            best_n_all = anchor_ious_all.argmax(dim=1)
            best_n = best_n_all % 3
            best_n_mask = ((best_n_all == self.anch_masks[output_id][0]) |
                           (best_n_all == self.anch_masks[output_id][1]) |
                           (best_n_all == self.anch_masks[output_id][2]))

            if sum(best_n_mask) == 0:
                continue

            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            pred_ious = bboxes_iou(pred[b].view(-1, 4), truth_box, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = (pred_best_iou > self.ignore_thre)
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            obj_mask[b] = ~ pred_best_iou

            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(
                        truth_w_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(
                        truth_h_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 1] + 1e-16)
                    target[b, a, j, i, 4] = 1
                    target[b, a, j, i, 5 + labels[b, ti, 4].to(torch.int16).cpu().numpy()] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)
        return obj_mask, tgt_mask, tgt_scale, target

    def forward(self, xin, labels=None):
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = 0, 0, 0, 0, 0, 0
        for output_id, output in enumerate(xin):
            batchsize = output.shape[0]
            fsize = output.shape[2]
            n_ch = 5 + self.n_classes

            output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)

            output = output.permute(0, 1, 3, 4, 2)  # .contiguous()

            # logistic activation for xy, obj, cls
            output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:n_ch]])

            pred = output[..., :4].clone()
            pred[..., 0] += self.grid_x[output_id]
            pred[..., 1] += self.grid_y[output_id]
            pred[..., 2] = torch.exp(pred[..., 2]) * self.anchor_w[output_id]
            pred[..., 3] = torch.exp(pred[..., 3]) * self.anchor_h[output_id]

            obj_mask, tgt_mask, tgt_scale, target = self.build_target(pred, labels, batchsize, fsize, n_ch, output_id)

            # loss calculation
            output[..., 4] *= obj_mask
            output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            output[..., 2:4] *= tgt_scale

            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            target[..., 2:4] *= tgt_scale

            loss_xy += F.binary_cross_entropy(input=output[..., :2], target=target[..., :2],
                                              weight=tgt_scale * tgt_scale, size_average=False)
            loss_wh += F.mse_loss(input=output[..., 2:4], target=target[..., 2:4], size_average=False) / 2
            #loss_obj += F.binary_cross_entropy(input=output[..., 4], target=target[..., 4], size_average=False)

            loss_obj += torch.sum(output[..., 4])
            loss_cls += F.binary_cross_entropy(input=output[..., 5:], target=target[..., 5:], size_average=False)
            loss_l2 += F.mse_loss(input=output, target=target, size_average=False)

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True, GIoU=False, DIoU=False, CIoU=False):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    https://github.com/ultralytics/yolov3/blob/eca5b9c1d36e4f73bf2f94e141d864f1c2739e23/utils/utils.py#L262-L282
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        # intersection top left
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # intersection bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min(bboxes_a[:, None, :2], bboxes_b[:, :2])
        con_br = torch.max(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, 0] + bboxes_a[:, None, 2]) - (bboxes_b[:, 0] + bboxes_b[:, 2])) ** 2 / 4 + (
                (bboxes_a[:, None, 1] + bboxes_a[:, None, 3]) - (bboxes_b[:, 1] + bboxes_b[:, 3])) ** 2 / 4

        w1 = bboxes_a[:, 2] - bboxes_a[:, 0]
        h1 = bboxes_a[:, 3] - bboxes_a[:, 1]
        w2 = bboxes_b[:, 2] - bboxes_b[:, 0]
        h2 = bboxes_b[:, 3] - bboxes_b[:, 1]

        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        # intersection top left
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # intersection bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        con_br = torch.max((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, :2] - bboxes_b[:, :2]) ** 2 / 4).sum(dim=-1)

        w1 = bboxes_a[:, 2]
        h1 = bboxes_a[:, 3]
        w2 = bboxes_b[:, 2]
        h2 = bboxes_b[:, 3]

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    area_u = area_a[:, None] + area_b - area_i
    iou = area_i / area_u

    if GIoU or DIoU or CIoU:
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            area_c = torch.prod(con_br - con_tl, 2)  # convex area
            return iou - (area_c - area_u) / area_c  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = torch.pow(con_br - con_tl, 2).sum(dim=2) + 1e-16
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w1 / h1).unsqueeze(1) - torch.atan(w2 / h2), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
    return iou