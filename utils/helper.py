import numpy as np
import cv2
from queue import Queue,LifoQueue,PriorityQueue

def bfs(orgimg,data_grad_abs,threshold_num=4000,k=10,attacked=None):
    data_grad_abs = data_grad_abs.reshape([-1, ])
    grad_sort_ix = np.argsort(data_grad_abs)
    threshld = data_grad_abs[grad_sort_ix][-k]
    threshld_min = data_grad_abs[grad_sort_ix][-100000]
    data_grad_abs = data_grad_abs.reshape([500, 500])
    iy, ix = np.where(data_grad_abs >= threshld)
    visit = np.zeros_like(data_grad_abs)
    if attacked is not None:
        visit[attacked[0],attacked[1]]=1
    if threshold_num<=0:
        return visit
    count=0
    serch_dir=[[0,0,1,1,1,-1,-1,-1],[1,-1,0,1,-1,0,1,-1]]
    #serch_dir = [[0, 0, 1, -1], [1,-1,0,0]]

    Q = Queue(maxsize=-1)
    for i in range(iy.shape[0]-1,-1,-1):
        Q.put([iy[i],ix[i]])

    while(not Q.empty()):
        if count>=threshold_num: break

        size = Q.qsize()
        for i in range(size):
            if count>=threshold_num: break

            [y,x] = Q.get()
            visit[y,x] = 1
            count+=1
            iy = y + serch_dir[0]
            ix = x + serch_dir[1]
            indy=[]
            indx=[]
            for j in range(8):
                if iy[j]<0 or ix[j]<0 or iy[j]>=500 or ix[j]>=500 \
                        or visit[iy[j],ix[j]]==1:
                    continue
                if count>=threshold_num: break
                if data_grad_abs[iy[j],ix[j]]>=0:
                    indx.append(ix[j])
                    indy.append(iy[j])
            neiboor=1
            ind =np.argsort(data_grad_abs[indy,indx])
            for j in range(len(ind)):
                if neiboor<=0:
                    break
                if visit[indy[ind[j]],indx[ind[j]]]==0:
                    Q.put([indy[ind[j]],indx[ind[j]]])
                    neiboor-=1
    print('攻击点数',count)
    visit=[visit,visit,visit]
    visit=np.array(visit)
    visit=np.transpose(visit,[1,2,0])
    return visit

import cv2
def generate_atmask(image,r,scores,file_name,save_path='./select1000_new_atroi/', transxy=True,yolo=False,save=True,
                    thred=50,reduce=0.2):
    allrest=0
    imagecopy = np.array(image)
    image=np.zeros_like(image)
    r=np.reshape(r,[-1,4])
    if yolo==True:
        r = r / 608 * 500
        for i in range(r.shape[0]):
            x,y,w,h=r[i]
            x1,x2=x-w/2,x+w/2
            y1,y2=y-h/2,y+h/2
            r[i]=[x1,y1,x2,y2]

    scores=np.reshape(scores,[-1,1])
    areas = []
    for j in range(scores.shape[0]):
        y1, x1, y2, x2 = r[j]
        areas.append((y2 - y1) * (x2 - x1))
    ix = np.argsort(areas)
    scores = scores[ix]
    rois = r[ix]
    # overlap=[]
    # for j in range(0,len(rois)-1):
    #     iou,overlapROI=compute_iou(rois[j],rois[j+1])
    #     if overlapROI.shape[0]!=1:
    #         overlap.append(overlapROI)
    #     if iou>=0.5:
    #         rois[j][0]=min(rois[j][0],rois[j+1][0])
    #         rois[j][1] = min(rois[j][1], rois[j + 1][1])
    #         rois[j][2] = max(rois[j][2], rois[j + 1][2])
    #         rois[j][3] = max(rois[j][3], rois[j + 1][3])
    #         scores[j+1]=0

    rest_pixels = 5000

    allmask = np.zeros([500, 500], dtype=np.int)
    for j in range(rois.shape[0]):
        if j >= 9 or rest_pixels <= 500: break;
        y1, x1, y2, x2 = rois[j]
        if transxy:
            x1, y1, x2, y2 = int(y1), int(x1), int(y2), int(x2)
        else:
            y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
        if x2 >= 499: x2 = 498
        if y2 >= 499: y2 = 498
        if x1<=0: x1 = 0
        if y1<=0:y1=0

        h = y2 - y1
        w = x2 - x1
        x0 = x1 + w // 2
        y0 = y1 + h // 2
        if w >= thred:
            x1 = x1 + reduce / 2 * w
            x2 = x2 - reduce / 2 * w
        if h >= thred:
            y1 = y1 + reduce / 2 * h
            y2 = y2 - reduce / 2 * h
        y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
        # scale = 10 + 5 * np.log(np.sqrt(h * w) / 250) / np.log(2.0)
        # if scale < 8:
        #     scale = 8
        scale = 8
        scale = int(scale)
        w_rest = 0  # int(0.05*w)
        if j>0 and w * h >300*300:
            break
        # ************竖线**********************
        if h <= w:
            if rest_pixels >= w + h:
                ix = range(x1, x2)

                image[y0, x1:x2] = 255
                rest_pixels -= w
                k = 0
                left = True
                while rest_pixels > h:
                    if x1 + k > x2 - 2:
                        break
                    if left == True:

                        # 间隔攻击
                        overlap = np.where(allmask[y1:y0, x1 + k] == 1)
                        begin = y1
                        if overlap[0].shape[0] != 0:
                            for l in range(y0, y1, -1):
                                if allmask[l, x1 + k] == 0:
                                    begin = l
                                else:
                                    begin = l + 1
                                    break
                            if begin >= y0 - 4:
                                k += scale
                                left = False
                                continue
                        iy0 = np.array(range(begin + w_rest, y0, 4))
                        if iy0.shape[0] > 0:
                            image[iy0, x1 + k] = 255
                        iy1 = np.array(range(begin + 1 + w_rest, y0, 2))
                        if iy1.shape[0] > 0:
                            image[iy1, x1 + k + 1,] = 255
                        iy2 = np.array(range(begin + 2 + w_rest, y0, 4))
                        if iy2.shape[0] > 0:
                            image[iy2, x1 + k + 2] = 255
                        rest_pixels -= (y0 - begin + 1)
                        left = False
                    else:
                        # 间隔攻击
                        overlap = np.where(allmask[y0:y2, x1 + k] == 1)
                        end = y2
                        if overlap[0].shape[0] != 0:
                            for l in range(y1, y2, 1):
                                if allmask[l, x1 + k] == 0:
                                    end = l
                                else:
                                    end = l - 1
                                    break
                            if end <= y0 + 4:
                                k += scale
                                left = True
                                continue

                        iy0 = np.array(range(y0 + 1, end - w_rest, 4))
                        if iy0.shape[0] > 0:
                            image[iy0, x1 + k] = 255
                        iy1 = np.array(range(y0 + 2, end - w_rest, 2))
                        if iy1.shape[0] > 0:
                            image[iy1, x1 + k + 1] = 255
                        iy2 = np.array(range(y0 + 3, end - w_rest, 4))
                        if len(iy2) != 0:
                            image[iy2, x1 + k + 2] = 255
                        # 无间隔攻击
                        # ix0,ic0=np.where(image[y1+k,x0:end-w_rest,:]>127)
                        # ix1,ic1=np.where(image[y1+k,x0:end-w_rest,:]<=127)
                        # image[y1+k,ix0+x0,ic0]=0
                        # image[y1+k,ix1+x0,ic1]=255
                        rest_pixels -= (end - y0 + 1)
                        left = True
                    k += scale
                if x1 + k < x2:
                    sig = True
                    for ix in range(x1 + k, x2):
                        if allmask[y0, ix] == 1:
                            sig = False
                    if sig == True:
                        image[y0, x1 + k:x2] = 0
        # ************横线**********************
        else:
            if rest_pixels >= h + w:
                iy = range(y1,y2)
                image[y1:y2, x0] = 255
                rest_pixels -= h
                k = 0
                left = True
                while rest_pixels > w:
                    if y1 + k > y2 - 2:
                        break
                    if left == True:
                        # 间隔攻击
                        overlap = np.where(allmask[y1 + k, x1:x0] == 1)
                        begin = x1
                        if overlap[0].shape[0] != 0:
                            for l in range(x0, x1, -1):
                                if allmask[y1 + k, l] == 0:
                                    begin = l
                                else:
                                    begin = l + 1
                                    break
                            if begin >= x0 - 4:
                                k += scale
                                left = False
                                continue
                        ix0 = np.array(range(begin + w_rest, x0, 4))
                        image[y1 + k, ix0] = 255
                        ix1 = np.array(range(begin + 1 + w_rest, x0, 2))
                        image[y1 + k + 1, ix1] = 255
                        ix2 = np.array(range(begin + 2 + w_rest, x0, 4))
                        if len(ix2) != 0:
                            image[y1 + k + 2, ix2] = 255

                        rest_pixels -= (x0 - begin + 1)
                        left = False
                    else:
                        # 间隔攻击
                        overlap = np.where(allmask[y1 + k, x0:x2] == 1)
                        end = x2
                        if overlap[0].shape[0] != 0:
                            for l in range(x1, x2, 1):
                                if allmask[y1 + k, l] == 0:
                                    end = l
                                else:
                                    end = l - 1
                                    break
                            if end <= x0 + 4:
                                k += scale
                                left = True
                                continue
                        ix0 = np.array(range(x0 + 1, end - w_rest, 4))
                        image[y1 + k, ix0] = 255
                        ix1 = np.array(range(x0 + 2, end - w_rest, 2))
                        image[y1 + k + 1, ix1] = 255
                        ix2 = np.array(range(x0 + 3, end - w_rest, 4))
                        if len(ix2) != 0:
                            image[y1 + k + 2, ix2] = 255
                        # 无间隔攻击
                        # ix0,ic0=np.where(image[y1+k,x0:end-w_rest,:]>127)
                        # ix1,ic1=np.where(image[y1+k,x0:end-w_rest,:]<=127)
                        # image[y1+k,ix0+x0,ic0]=0
                        # image[y1+k,ix1+x0,ic1]=255
                        rest_pixels -= (end - x0 + 1)
                        left = True
                    k += scale

                if y1 + k < y2:
                    sig = True
                    for ix in range(y1 + k, y2):
                        if allmask[ix, x0] == 1:
                            sig = False
                    if sig == True:
                        image[y1 + k:y2, x0] = 0
        allmask[y1:y2, x1:x2] = 1
    allrest += rest_pixels
    print('rest pixels', allrest)
    if rois.shape[0]==0:
        x=100
        while x<=400:
            image[100:400, x] = 255
            x+=30
        image[250, 100:400] = 255
    if save:
        cv2.imwrite(save_path + file_name, image[..., ::-1])

def generate_attack(image,r,scores,file_name,save_path='./select1000_new_atroi/', transxy=True,yolo=False,save=True,
                    thred=50,reduce=0.2,grade_abs=None):
    allrest=0
    imagecopy = np.array(image)
    r=np.reshape(r,[-1,4])
    if yolo==True:
        r = r / 608 * 500
        for i in range(r.shape[0]):
            x,y,w,h=r[i]
            x1,x2=x-w/2,x+w/2
            y1,y2=y-h/2,y+h/2
            r[i]=[x1,y1,x2,y2]

    scores=np.reshape(scores,[-1,1])
    areas = []
    for j in range(scores.shape[0]):
        y1, x1, y2, x2 = r[j]
        areas.append((y2 - y1) * (x2 - x1))
    ix = np.argsort(areas)
    scores = scores[ix]
    rois = r[ix]
    # overlap=[]
    # for j in range(0,len(rois)-1):
    #     iou,overlapROI=compute_iou(rois[j],rois[j+1])
    #     if overlapROI.shape[0]!=1:
    #         overlap.append(overlapROI)
    #     if iou>=0.5:
    #         rois[j][0]=min(rois[j][0],rois[j+1][0])
    #         rois[j][1] = min(rois[j][1], rois[j + 1][1])
    #         rois[j][2] = max(rois[j][2], rois[j + 1][2])
    #         rois[j][3] = max(rois[j][3], rois[j + 1][3])
    #         scores[j+1]=0

    rest_pixels = 5000

    allmask = np.zeros([500, 500], dtype=np.int)
    for j in range(rois.shape[0]):
        if j >= 9 or rest_pixels <= 500: break;
        y1, x1, y2, x2 = rois[j]
        if transxy:
            x1, y1, x2, y2 = int(y1), int(x1), int(y2), int(x2)
        else:
            y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
        if x2 >= 499: x2 = 497
        if y2 >= 499: y2 = 497
        if x1<=0: x1 = 2
        if y1<=0:y1=2

        h = y2 - y1
        w = x2 - x1
        x0 = x1 + w // 2
        y0 = y1 + h // 2
        if w >= thred:
            x1 = x1 + reduce / 2 * w
            x2 = x2 - reduce / 2 * w
        if h >= thred:
            y1 = y1 + reduce / 2 * h
            y2 = y2 - reduce / 2 * h
        y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
        # scale = 10 + 5 * np.log(np.sqrt(h * w) / 250) / np.log(2.0)
        # if scale < 8:
        #     scale = 8
        scale =12
        scale = int(scale)
        w_rest = 0  # int(0.05*w)
        if j>0 and w * h >300*300:
            scale=16
            #break
        # ************竖线**********************
        if h <= w:
            if rest_pixels >= w + h:
                ix = range(x1, x2)

                ix0_m, ic0_m = np.where(image[y0, ix] > 127)
                ix1_m, ic1_m = np.where(image[y0, ix] <= 127)
                image[y0, ix0_m + x1, ic0_m] = 0
                image[y0, ix1_m + x1, ic1_m] = 255
                rest_pixels -= w

                k = 0
                left = True
                # left_grad = np.sum(grade_abs[y1 + 1 + w_rest:y0:2,x1+k])
                # right_grad = np.sum(grade_abs[y0:y2:2,x1+k])
                while rest_pixels > h:
                    if x1 + k > x2 - 4:
                        break
                    if left == True:

                        # 间隔攻击
                        overlap = np.where(allmask[y1:y0, x1 + k] == 1)
                        begin = y1
                        if overlap[0].shape[0] != 0:
                            for l in range(y0, y1, -1):
                                if allmask[l, x1 + k] == 0:
                                    begin = l
                                else:
                                    begin = l + 1
                                    break
                            if begin >= y0 - 4:
                                k += scale
                                left = False
                                continue

                        # m=0
                        # sig=False
                        # for l in range(begin+w_rest,x0):
                        #     if y1 + k + m>=500:
                        #         break
                        #     for c in range(3):
                        #         if image[y1+k+m,l,c]!=imagecopy[y1+k+m,l,c]:
                        #             break
                        #         if image[y1+k+m,l,c]>127:
                        #             image[y1+k+m,l,c]=0
                        #         elif image[y1+k+m,l,c]<=127:
                        #             image[y1 + k + m, l, c] = 255
                        #     if m==m_i:
                        #         sig=True
                        #     elif m==0:
                        #         sig = False
                        #     if sig==False:
                        #         m+=1
                        #     else:
                        #         m-=1

                        iy0 = np.array(range(begin + w_rest, y0, 4))
                        iyx0 = iy0[np.where(iy0 != y0)]
                        if iy0.shape[0] > 0:
                            ic0_0 = np.where(image[iy0, x1 + k] > 127)
                            ic0_1 = np.where(image[iy0, x1 + k] <= 127)
                            image[iy0[ic0_0[0]], x1 + k, ic0_0[1]] = 0
                            image[iy0[ic0_1[0]], x1 + k, ic0_1[1]] = 255

                        iy1 = np.array(range(begin + 1 + w_rest, y0, 2))
                        iy1 = iy1[np.where(iy1 != y0)]
                        if iy1.shape[0] > 0:
                            ic1_0 = np.where(image[iy1, x1 + k + 1] > 127)
                            ic1_1 = np.where(image[iy1, x1 + k + 1] <= 127)
                            image[iy1[ic1_0[0]], x1 + k + 1, ic1_0[1]] = 0
                            image[iy1[ic1_1[0]], x1 + k + 1, ic1_1[1]] = 255

                        iy2 = np.array(range(begin + 2 + w_rest, y0, 4))
                        iy2 = iy2[np.where(iy2 != y0)]
                        if len(iy2) != 0:
                            ic2_0 = np.where(image[iy2, x1 + k + 2] > 127)
                            ic2_1 = np.where(image[iy2, x1 + k + 2] <= 127)
                            image[iy2[ic2_0[0]], x1 + k + 2, ic2_0[1]] = 0
                            image[iy2[ic2_1[0]], x1 + k + 2, ic2_1[1]] = 255

                        # 无间隔攻击
                        # ix0,ic0=np.where(image[y1+k,begin+w_rest:x0]>127)
                        # ix1,ic1=np.where(image[y1+k,begin+w_rest:x0]<=127)
                        # image[y1+k,ix0+w_rest+begin,ic0]=0
                        # image[y1+k,ix1+w_rest+begin,ic1]=255

                        rest_pixels -= (y0 - begin + 1)
                        left = False
                    else:
                        # 间隔攻击
                        overlap = np.where(allmask[y0:y2, x1 + k] == 1)
                        end = y2
                        if overlap[0].shape[0] != 0:
                            for l in range(y1, y2, 1):
                                if allmask[l, x1 + k] == 0:
                                    end = l
                                else:
                                    end = l - 1
                                    break
                            if end <= y0 + 4:
                                k += scale
                                left = True
                                continue

                        # m=0
                        # sig=False
                        # for l in range(x0,end-w_rest):
                        #     if y1 + k + m>=500:
                        #         break
                        #     for c in range(3):
                        #         if image[y1+k+m,l,c]!=imagecopy[y1+k+m,l,c]:
                        #             break
                        #         if image[y1+k+m,l,c]>127:
                        #             image[y1+k+m,l,c]=0
                        #         elif image[y1+k+m,l,c]<=127:
                        #             image[y1 + k + m, l, c] = 255
                        #     if m==m_i:
                        #         sig=True
                        #     elif m==0:
                        #         sig = False
                        #     if sig==False:
                        #         m+=1
                        #     else:
                        #         m-=1

                        iy0 = np.array(range(y0 + 1, end - w_rest, 4))
                        if iy0.shape[0] > 0:
                            ic0_0 = np.where(image[iy0, x1 + k] > 127)
                            ic0_1 = np.where(image[iy0, x1 + k] <= 127)
                            image[iy0[ic0_0[0]], x1 + k, ic0_0[1]] = 0
                            image[iy0[ic0_1[0]], x1 + k, ic0_1[1]] = 255

                        iy1 = np.array(range(y0 + 2, end - w_rest, 2))
                        if iy1.shape[0] > 0:
                            ic1_0 = np.where(image[iy1, x1 + k + 1] > 127)
                            ic1_1 = np.where(image[iy1, x1 + k + 1] <= 127)
                            image[iy1[ic1_0[0]], x1 + k + 1, ic1_0[1]] = 0
                            image[iy1[ic1_1[0]], x1 + k + 1, ic1_1[1]] = 255

                        iy2 = np.array(range(y0 + 3, end - w_rest, 4))
                        if len(iy2) != 0:
                            ic2_0 = np.where(image[iy2, x1 + k + 2] > 127)
                            ic2_1 = np.where(image[iy2, x1 + k + 2] <= 127)
                            image[iy2[ic2_0[0]], x1 + k + 2, ic2_0[1]] = 0
                            image[iy2[ic2_1[0]], x1 + k + 2, ic2_1[1]] = 255
                        # 无间隔攻击
                        # ix0,ic0=np.where(image[y1+k,x0:end-w_rest,:]>127)
                        # ix1,ic1=np.where(image[y1+k,x0:end-w_rest,:]<=127)
                        # image[y1+k,ix0+x0,ic0]=0
                        # image[y1+k,ix1+x0,ic1]=255
                        rest_pixels -= (end - y0 + 1)
                        left = True

                    k += scale
                if x1 + k < x2:
                    sig = True
                    for ix in range(x1 + k, x2):
                        if allmask[y0, ix] == 1:
                            sig = False
                    if sig == True:
                        image[y0, x1 + k:x2] = imagecopy[y0, x1 + k:x2]
        # ************横线**********************
        else:
            if rest_pixels >= h + w:
                iy = range(y1,y2)
                iy0_m, ic0_m = np.where(image[iy, x0] > 127)
                iy1_m, ic1_m = np.where(image[iy, x0] <= 127)
                image[iy0_m + y1, x0, ic0_m] = 0
                image[iy1_m + y1, x0, ic1_m] = 255
                rest_pixels -= h

                k = 0
                left = True
                while rest_pixels > w:
                    if y1 + k > y2 - 4:
                        break
                    if left == True:
                        # 间隔攻击
                        overlap = np.where(allmask[y1 + k, x1:x0] == 1)
                        begin = x1
                        if overlap[0].shape[0] != 0:
                            for l in range(x0, x1, -1):
                                if allmask[y1 + k, l] == 0:
                                    begin = l
                                else:
                                    begin = l + 1
                                    break
                            if begin >= x0 - 4:
                                k += scale
                                left = False
                                continue

                        # m=0
                        # sig=False
                        # for l in range(begin+w_rest,x0):
                        #     if y1 + k + m>=500:
                        #         break
                        #     for c in range(3):
                        #         if image[y1+k+m,l,c]!=imagecopy[y1+k+m,l,c]:
                        #             break
                        #         if image[y1+k+m,l,c]>127:
                        #             image[y1+k+m,l,c]=0
                        #         elif image[y1+k+m,l,c]<=127:
                        #             image[y1 + k + m, l, c] = 255
                        #     if m==m_i:
                        #         sig=True
                        #     elif m==0:
                        #         sig = False
                        #     if sig==False:
                        #         m+=1
                        #     else:
                        #         m-=1

                        ix0 = np.array(range(begin + w_rest, x0, 4))
                        ix0 = ix0[np.where(ix0 != x0)]
                        if len(ix0) != 0:
                            ic0_0 = np.where(image[y1 + k, ix0] > 127)
                            ic0_1 = np.where(image[y1 + k, ix0] <= 127)
                            image[y1 + k, ix0[ic0_0[0]], ic0_0[1]] = 0
                            image[y1 + k, ix0[ic0_1[0]], ic0_1[1]] = 255

                        ix1 = np.array(range(begin + 1 + w_rest, x0, 2))
                        ix1 = ix1[np.where(ix1 != x0)]
                        if len(ix1) != 0:
                            ic1_0 = np.where(image[y1 + k + 1, ix1] > 127)
                            ic1_1 = np.where(image[y1 + k + 1, ix1] <= 127)
                            image[y1 + k + 1, ix1[ic1_0[0]], ic1_0[1]] = 0
                            image[y1 + k + 1, ix1[ic1_1[0]], ic1_1[1]] = 255

                        ix2 = np.array(range(begin + 2 + w_rest, x0, 4))
                        ix2 = ix2[np.where(ix2 != x0)]
                        if len(ix2) != 0:
                            ic2_0 = np.where(image[y1 + k + 2, ix2] > 127)
                            ic2_1 = np.where(image[y1 + k + 2, ix2] <= 127)
                            image[y1 + k + 2, ix2[ic2_0[0]], ic2_0[1]] = 0
                            image[y1 + k + 2, ix2[ic2_1[0]], ic2_1[1]] = 255

                        # 无间隔攻击
                        # ix0,ic0=np.where(image[y1+k,begin+w_rest:x0]>127)
                        # ix1,ic1=np.where(image[y1+k,begin+w_rest:x0]<=127)
                        # image[y1+k,ix0+w_rest+begin,ic0]=0
                        # image[y1+k,ix1+w_rest+begin,ic1]=255

                        rest_pixels -= (x0 - begin + 1)
                        left = False
                    else:
                        # 间隔攻击
                        overlap = np.where(allmask[y1 + k, x0:x2] == 1)
                        end = x2
                        if overlap[0].shape[0] != 0:
                            for l in range(x1, x2, 1):
                                if allmask[y1 + k, l] == 0:
                                    end = l
                                else:
                                    end = l - 1
                                    break
                            if end <= x0 + 4:
                                k += scale
                                left = True
                                continue

                        # m=0
                        # sig=False
                        # for l in range(x0,end-w_rest):
                        #     if y1 + k + m>=500:
                        #         break
                        #     for c in range(3):
                        #         if image[y1+k+m,l,c]!=imagecopy[y1+k+m,l,c]:
                        #             break
                        #         if image[y1+k+m,l,c]>127:
                        #             image[y1+k+m,l,c]=0
                        #         elif image[y1+k+m,l,c]<=127:
                        #             image[y1 + k + m, l, c] = 255
                        #     if m==m_i:
                        #         sig=True
                        #     elif m==0:
                        #         sig = False
                        #     if sig==False:
                        #         m+=1
                        #     else:
                        #         m-=1

                        ix0 = np.array(range(x0 + 1, end - w_rest, 4))
                        if len(ix0)!=0:
                            ic0_0 = np.where(image[y1 + k, ix0] > 127)
                            ic0_1 = np.where(image[y1 + k, ix0] <= 127)
                            image[y1 + k, ix0[ic0_0[0]], ic0_0[1]] = 0
                            image[y1 + k, ix0[ic0_1[0]], ic0_1[1]] = 255

                        ix1 = np.array(range(x0 + 2, end - w_rest, 2))
                        if len(ix1) != 0:
                            ic1_0 = np.where(image[y1 + k + 1, ix1] > 127)
                            ic1_1 = np.where(image[y1 + k + 1, ix1] <= 127)
                            image[y1 + k + 1, ix1[ic1_0[0]], ic1_0[1]] = 0
                            image[y1 + k + 1, ix1[ic1_1[0]], ic1_1[1]] = 255

                        ix2 = np.array(range(x0 + 3, end - w_rest, 4))
                        if len(ix2) != 0:
                            ic2_0 = np.where(image[y1 + k + 2, ix2] > 127)
                            ic2_1 = np.where(image[y1 + k + 2, ix2] <= 127)
                            image[y1 + k + 2, ix2[ic2_0[0]], ic2_0[1]] = 0
                            image[y1 + k + 2, ix2[ic2_1[0]], ic2_1[1]] = 255
                        # 无间隔攻击
                        # ix0,ic0=np.where(image[y1+k,x0:end-w_rest,:]>127)
                        # ix1,ic1=np.where(image[y1+k,x0:end-w_rest,:]<=127)
                        # image[y1+k,ix0+x0,ic0]=0
                        # image[y1+k,ix1+x0,ic1]=255
                        rest_pixels -= (end - x0 + 1)
                        left = True
                    k += scale

                if y1 + k < y2:
                    sig = True
                    for ix in range(y1 + k, y2):
                        if allmask[ix, x0] == 1:
                            sig = False
                    if sig == True:
                        image[y1 + k:y2, x0] = imagecopy[y1 + k:y2, x0]
        allmask[y1:y2, x1:x2] = 1
    # if rest_pixels>=700:
    #     image[249,100:400]=255-image[249,100:400]
    #     image[100:400,249]=255-image[100:400,249]
    #     rest_pixels-=600
    allrest += rest_pixels
    print('rest pixels', allrest)

    # ix = np.where(allmask == 0)
    # image[ix] = imagecopy[ix]
    if rois.shape[0]==0:
        x=100
        while x<=400:
            iy0_m,ic0_m = np.where(image[100:400:, x] > 127)
            iy1_m, ic1_m = np.where(image[100:400, x] <= 127)
            image[iy0_m + 100, x, ic0_m] = 0
            image[iy1_m + 100, x, ic1_m] = 255
            x+=30
        iy0_m, ic0_m = np.where(image[250,100:400] > 127)
        iy1_m, ic1_m = np.where(image[250,100:400] <= 127)
        image[250, iy0_m+100, ic0_m] = 0
        image[250, iy1_m+100, ic1_m] = 255
    if save:
        cv2.imwrite(save_path + file_name, image[..., ::-1])




def compute_iou(box1,box2):
    y11, x11, y12, x12 = box1
    y21, x21, y22, x22 = box2
    area1=(y12-y11)*(x12-x11)
    area2=(y22-y21)*(x22-x21)
    deltax = min(x22,x12)-max(x21,x11)
    deltay = min(y22,y12)-max(y21,y11)
    if deltax<=0 or deltay<=0:
        return 0,0,area1,area2
    else:
        inter=deltax*deltay
        return inter/(area1+area2-inter),inter,area1,area2

def grad_important(data_grad, img0,direct=True):
    ix0 = np.where(data_grad<0)
    ix1 = np.where(data_grad>0)
    add_grad = np.zeros_like(data_grad)
    minus_grad = np.zeros_like(data_grad)
    img1 = np.array(img0)
    if direct:
        add_grad[ix1] = (255-img1[ix1])*data_grad[ix1]
        minus_grad[ix0] = img1[ix0]*(-data_grad[ix0])
        img1[ix1]=255
        img1[ix0]=0
    else:
        add_grad[ix1] = img1[ix1]*data_grad[ix1]
        minus_grad[ix0] = (255-img1[ix0])*(-data_grad[ix0])
        img1[ix1]=0
        img1[ix0]=255
    return np.sum(add_grad+minus_grad,axis=-1),img1

def generate_attact_roi(data_grad_abs, rois,areas):
    visited = np.zeros_like(data_grad_abs)
    ix_sort=np.argsort(areas)
    count=5000
    #serch_dir=[[0,0,1,1,1,-1,-1,-1],[1,-1,0,1,-1,0,1,-1]]
    serch_dir = [[0,0,1,-1,1,1,-1,-1],[1,-1,0,0,1,-1,1,-1]]
    for i in ix_sort:
        x1,y1,x2,y2=rois[i]
        # x,y,w,h=rois[i]
        # x1 = x - w / 2
        # x2 = x + w / 2
        # y1 = y - h / 2
        # y2 = y + h / 2

        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
        if x2 >= 499: x2 = 499
        if y2 >= 499: y2 = 499
        if x1 <= 0: x1 = 0
        if y1 <= 0: y1 = 0
        i_max = np.max(data_grad_abs[y1:y2,x1:x2])
        ix = np.where(data_grad_abs[y1:y2,x1:x2]==i_max)
        Q = Queue(maxsize=-1)
        Q.put([y1+ix[0][0],x1+ix[1][0]])
        count -= 1
        while (not Q.empty()):
            if count <=100: break

            size = Q.qsize()
            for i in range(size):
                if count <=100: break

                [y, x] = Q.get()
                visited[y, x] = 1
                count -= 1
                iy = y + serch_dir[0]
                ix = x + serch_dir[1]
                indy = []
                indx = []
                for j in range(4):
                    if iy[j] < y1 or ix[j] < x1 or iy[j] >= y2 or ix[j] >= x2 \
                            or visited[iy[j], ix[j]] == 1:
                        continue
                    if count <=100:
                        break
                    if data_grad_abs[iy[j], ix[j]] >= 0:
                        indx.append(ix[j])
                        indy.append(iy[j])

                neiboor = 1
                ind = np.argsort(data_grad_abs[indy, indx])

                for j in range(len(ind)):
                    if neiboor <= 0:
                        break
                    if visited[indy[ind[j]], indx[ind[j]]] == 0:
                        Q.put([indy[ind[j]], indx[ind[j]]])
                        neiboor -= 1
        while(not Q.empty()):
            [y, x] = Q.get()
            visited[y, x] = 1
    print('攻击点数：',5000-count,np.where(visited==1)[0].shape)
    return visited

def generate_attact_rc(data_grad_abs, rois):
    visited = np.zeros_like(data_grad_abs)
    #ix_sort=np.argsort(areas)
    count=5000
    #serch_dir=[[0,0,1,1,1,-1,-1,-1],[1,-1,0,1,-1,0,1,-1]]
    serch_dir = [[0,0,1,-1,1,1,-1,-1],[1,-1,0,0,1,-1,1,-1]]
    objects=0
    for i in range(len(rois)):
        # if objects>=1 and areas[i]>300*300:
        #     break
        objects+=1
        x1,y1,x2,y2=rois[i]
        # x,y,w,h=rois[i]
        # x1 = x - w / 2
        # x2 = x + w / 2
        # y1 = y - h / 2
        # y2 = y + h / 2

        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
        if x2 >= 499: x2 = 498
        if y2 >= 499: y2 = 498
        if x1 <= 0: x1 = 0
        if y1 <= 0: y1 = 0
        i_max = np.max(data_grad_abs[y1:y2,x1:x2])
        ix = np.where(data_grad_abs[y1:y2,x1:x2]==i_max)
        y0,x0=ix[0][0],ix[1][0]
        rows_grad = np.average(data_grad_abs[y1:y2,x1:x2],axis=1)
        cols_grad = np.average(data_grad_abs[y1:y2,x1:x2],axis=0)
        ir = np.argsort(rows_grad)
        ic = np.argsort(cols_grad)
        r=ir.shape[0]-1
        c=ic.shape[0]-1
        w=x2-x1
        h=y2-y1
        row_obs = []
        col_obs = []
        inter=16
        inter = int(inter)
        if count<w+h:break
        cy = y1+ir[r]
        cx = x1+ic[c]
        visited[cy-10:cy+10, cx-10:cx+10] = 1
        # visited[cy,x1:x2]=1
        # visited[y1:y2,cx]=1
        maxtimes=w//16+h//16-2
        for j in range(cy-inter,cy+inter+1):
            row_obs.append(j)
        for j in range(cx - inter, cx + inter + 1):
            col_obs.append(j)
        count=count-w-h
    #print('攻击点数：',5000-count,np.where(visited==1)[0].shape)
    visited=np.array([visited,visited,visited])
    visited=np.transpose(visited,[1,2,0])
    return visited

import torch
import torch.nn as nn
from torch.autograd import Variable

class PGD(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model=model#必须是pytorch的model
        self.device=torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    def generate(self,x,**params):
        self.parse_params(**params)
        labels=self.y

        adv_x=self.attack(x,labels)
        return adv_x
    def parse_params(self,eps=0.3,iter_eps=0.01,nb_iter=40,clip_min=0.0,clip_max=1.0,C=0.0,
                     y=None,ord=np.inf,rand_init=True,flag_target=False):
        self.eps=eps
        self.iter_eps=iter_eps
        self.nb_iter=nb_iter
        self.clip_min=clip_min
        self.clip_max=clip_max
        self.y=y
        self.ord=ord
        self.rand_init=rand_init
        self.model.to(self.device)
        self.flag_target=flag_target
        self.C=C


def sigle_step_attack(model,x,pertubation,labels,attack_roli,device):
    adv_x=x
    adv_x[attack_roli]=adv_x[attack_roli]+pertubation[attack_roli]
    # get the gradient of x
    adv_x=Variable(adv_x)
    adv_x.requires_grad = True
    print(adv_x.shape,x.shape,labels.shape)
    loss_func=Yolo_loss(device=device, batch=1, n_classes=80)
    preds=model(adv_x)

    loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = loss_func(preds,labels)


    model.zero_grad()
    loss.backward()
    grad=adv_x.grad.data.cpu().numpy()
    #get the pertubation of an iter_eps
    pertubation=0.01*np.sign(grad)
    adv_x=adv_x.cpu().detach().numpy()+pertubation
    x=x.cpu().detach().numpy()

    pertubation=np.clip(adv_x,0.0,1.0)-x
    pertubation=np.clip(pertubation,np.inf,0.3)

    return pertubation

def attack(self,x,labels):
    labels = labels.to(self.device)
    print(self.rand_init)
    if self.rand_init:
        x_tmp=x+torch.Tensor(np.random.uniform(-self.eps, self.eps, x.shape)).type_as(x).cuda()
    else:
        x_tmp=x
    pertubation=torch.zeros(x.shape).type_as(x).to(self.device)
    for i in range(self.nb_iter):
        pertubation=self.sigle_step_attack(x_tmp,pertubation=pertubation,labels=labels)
        pertubation=torch.Tensor(pertubation).type_as(x).to(self.device)
    adv_x=x+pertubation
    adv_x=adv_x.cpu().detach().numpy()

    adv_x=np.clip(adv_x,self.clip_min,self.clip_max)

    return adv_x

import torch.nn as nn
import torch.nn.functional as F
import math
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
            loss_obj += F.binary_cross_entropy(input=output[..., 4], target=target[..., 4], size_average=False)
            #loss_obj+=output[..., 4]
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

def reduce_roi(org,dst,rois,thred=100,reduce=0.2):
    org_copy=np.array(org)
    for i in range(len(rois)):
        x1,y1,x2,y2=rois[i]
        w=x2-x1
        h=y2-y1
        if h>w:
            if w>=thred:
                x1=x1+reduce/2*w
                x2=x2-reduce/2*w
        if h<=w:
            if h>=thred:
                y1=y1+reduce/2*h
                y2=y2-reduce/2*h
        y1,y2,x1,x2=int(y1),int(y2),int(x1),int(x2)
        if x2 >= 499: x2 = 499
        if y2 >= 499: y2 = 499
        if x1 <= 0: x1 = 0
        if y1 <= 0: y1 = 0
        patch=np.array(dst[y1:y2,x1:x2])
        org_copy[y1:y2,x1:x2]=patch
    return  org_copy