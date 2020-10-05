import warnings

import matplotlib.pyplot as plt
import mmcv
import torch
from mmcv.ops import RoIAlign, RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector


def init_detector(config, checkpoint=None, device='cuda:0'):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg,train_cfg=config.train_cfg)
    if checkpoint is not None:
        map_loc = 'cpu' if device == 'cpu' else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage(object):
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def inference_detector(model, img):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # Use torchvision ops for CPU mode instead
        for m in model.modules():
            if isinstance(m, (RoIPool, RoIAlign)):
                if not m.aligned:
                    # aligned=False is not implemented on CPU
                    # set use_torchvision on-the-fly
                    m.use_torchvision = True
        warnings.warn('We set use_torchvision=True in CPU mode.')
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data
    
    # forward the model
    with torch.no_grad():
        #print(data['img'][0].shape)
        result = model(return_loss=False, rescale=True, **data)

    return result

import numpy as np
import cv2


def Dpatch_detector(model, img, dstimg,patch, gt_bboxes, gt_labels, filename=None, at_times=50, e=10.0,image_size=800,mode='frcnn'):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    gt_labels = np.array(gt_labels)
    gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
    gt_bboxes = gt_bboxes / 500 *image_size
    #gt_bboxes[:,2:4]=0
    gt_labels = torch.tensor(gt_labels, device=device)
    gt_bboxes = torch.tensor(gt_bboxes, device=device)
    pertubation = np.zeros_like(img)
    adv_x = np.array(dstimg, dtype=np.uint8)
    patch=np.random.randint(0, 256,size=adv_x.shape)
    adv_x[230:270,230:270]=patch[230:270,230:270]
    attack_roi=np.where(adv_x!=img)
    momentom = np.zeros_like(dstimg)
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    # mean=mean[::-1]
    # std=std[::-1]
    adv_x=adv_x[...,::-1]
    data = dict(img=adv_x)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    adv_x = adv_x[..., ::-1]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # Use torchvision ops for CPU mode instead
        for m in model.modules():
            if isinstance(m, (RoIPool, RoIAlign)):
                if not m.aligned:
                    # aligned=False is not implemented on CPU
                    # set use_torchvision on-the-fly
                    m.use_torchvision = True
        warnings.warn('We set use_torchvision=True in CPU mode.')
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data
    data['img'] = torch.tensor(data['img'][0].clone().detach(), device=device)
    data['img_metas'] = data['img_metas'][0]
    data['gt_bboxes']=[gt_bboxes]
    data['gt_labels']=[gt_labels]
    #print(data)
    data_min = torch.min(data['img'])
    data_max = torch.max(data['img'])
    loss_last = 100
    times = 0
    for k in range(at_times):
        data['img'] = torch.autograd.Variable(data['img'])
        data['img'].requires_grad = True
        # data_img=F.interpolate(data['img'],size=[800,800])
        loss = model(return_loss=True, **data)
        # r = model(return_loss=False, img=[data['img']], img_metas=[data['img_metas']],rescale=True)
        loss_rpn_cls = 0
        loss_rpn_bbox = 0
        for i in range(len(loss['loss_rpn_cls'])):
            loss_rpn_cls = loss_rpn_cls + loss['loss_rpn_cls'][i]
            loss_rpn_bbox = loss_rpn_bbox + loss['loss_rpn_bbox'][i]
        if mode == 'ssd':
            loss_back = loss['loss_cls'][0]
        else:
            loss_back =loss['loss_cls']+loss['loss_bbox']

        model.zero_grad()
        loss_back.backward()
        data_grad = data['img'].grad.data

        data_grad = data_grad.cpu().numpy()
        data_grad = data_grad.squeeze()
        data_grad = data_grad.transpose([1, 2, 0])
        data_grad = cv2.resize(data_grad, (500, 500))

        momentom = 0.9 * momentom + e * data_grad

        data['img'] = data['img'].cpu().detach().numpy()
        data['img'] = data['img'].squeeze()
        data['img'] = data['img'].transpose([1, 2, 0])
        data['img'] = cv2.resize(data['img'], (500, 500))
        data['img'][attack_roi] = data['img'][attack_roi] - momentom[attack_roi]
        data['img'][attack_roi] = np.clip(data['img'][attack_roi], data_min.cpu(), data_max.cpu())

        momentom = momentom * std
        if mode == 'ssd':
            data_grad = 100 * e * data_grad * std
        else:
            data_grad = e * data_grad * std
        data_grad = np.clip(data_grad, -10, 10)
        momentom = np.clip(momentom, -10, 10)
        adv_x[attack_roi] = adv_x[attack_roi] - data_grad[attack_roi]

        # grad_range=np.sort(np.reshape(data_grad[attack_roi],[-1]))
        # print('梯度范围', grad_range[:100],grad_range[-100:])

        momentom = momentom / std
        adv_x[attack_roi] = np.clip(adv_x[attack_roi], 0, 255)
        data['img'] = (adv_x - mean) / std
        # data['img'][200:600, 200:600] = data['img'][200:600, 200:600] - 1000 * data_grad[200:600, 200:600]
        data['img'] = cv2.resize(data['img'], (image_size, image_size))
        data['img'] = torch.from_numpy(data['img'].transpose(2, 0, 1)).float().unsqueeze(0)
        data['img'] = torch.clamp(data['img'], data_min, data_max)
        data['img'] = data['img'].cuda()

        # momentom=0.9*momentom+e*data_grad_
        # pertubation=data_grad_*255
        # pertubation[attack_roi]=np.where(pertubation[attack_roi]>0,pertubation[attack_roi]+0.5,pertubation[attack_roi])
        # pertubation[attack_roi] = np.where(pertubation[attack_roi] < 0, pertubation[attack_roi] - 0.5,
        #                                    pertubation[attack_roi])
        # pertubation = np.clip(pertubation, -3, 3)
        # print(np.min(pertubation[attack_roi]),np.max(pertubation[attack_roi]))
        if loss_back < loss_last:
            res = adv_x
            loss_last = loss_back

        if (k + 1) % 5 == 0 or (k + 1) == at_times:
            print(filename, 'frcn step:%d' % (k + 1), loss_back, loss['loss_cls'], loss_rpn_cls)
    return adv_x

import torch.nn.functional as F
from torchvision import transforms
def attact_detector(model, img0,dstimg,gt_bboxes,gt_labels,filename=None,attack_roi=None,
                    at_times=50,e=30.0,image_size=800,mode='frcnn',random_begin=False, return_grad=False,rpn=False):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg=model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    gt_labels=np.array(gt_labels)
    gt_bboxes=np.array(gt_bboxes,dtype=np.float32)
    gt_bboxes=gt_bboxes/500*image_size
    gt_labels=torch.tensor(gt_labels,device=device)
    gt_bboxes=torch.tensor(gt_bboxes,device=device)
    
    #gt_bboxes[:,2:4]=gt_bboxes[:,2:4]/80
    pertubation = np.zeros_like(dstimg)
    momentom=np.zeros_like(dstimg)
    if mode=='frcnn':
        mean = [123.675, 116.28, 103.53]
        std = [58.395, 57.12, 57.375]
    elif mode=='ssd':
        mean = [123.675, 116.28, 103.53]
        std = [1, 1, 1]
    if random_begin==True:
        dstimg[attack_roi]=np.random.randint(0,256,dstimg[attack_roi].shape)
    if return_grad==True:
        at_times=1
        
    adv_x = np.array(dstimg[...,::-1],dtype=np.float)
    res=np.array(adv_x)
    data = dict(img=adv_x)
    adv_x=adv_x[...,::-1]
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # Use torchvision ops for CPU mode instead
        for m in model.modules():
            if isinstance(m, (RoIPool, RoIAlign)):
                if not m.aligned:
                    # aligned=False is not implemented on CPU
                    # set use_torchvision on-the-fly
                    m.use_torchvision = True
        warnings.warn('We set use_torchvision=True in CPU mode.')
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data
    data['img'] = torch.tensor(data['img'][0].clone().detach(), device=device)
    data['img_metas'] = data['img_metas'][0]
    data['gt_bboxes']=[gt_bboxes]
    data['gt_labels']=[gt_labels]
    data_min = torch.min(data['img'])
    data_max = torch.max(data['img'])
    loss_last=100
    times=0
    for k in range(at_times):
        data['img'] = torch.autograd.Variable(data['img'])
        data['img'].requires_grad = True
        #data_img=F.interpolate(data['img'],size=[800,800],mode='bilinear')
        loss = model(return_loss=True,img=data['img'],img_metas=data['img_metas'],gt_bboxes=data['gt_bboxes'],gt_labels=data['gt_labels'])
        #r = model(return_loss=False, img=[data['img']], img_metas=[data['img_metas']],rescale=True)
        loss_rpn_cls = 0
        loss_rpn_bbox = 0
        for i in range(len(loss['loss_rpn_cls'])):
            loss_rpn_cls = loss_rpn_cls + loss['loss_rpn_cls'][i]
            loss_rpn_bbox = loss_rpn_bbox + loss['loss_rpn_bbox'][i]
        if mode=='ssd':
            loss_back = loss['loss_cls'][0]
        else:
            if rpn==True:
                loss_back = -loss_rpn_cls#+loss['loss_cls']#-loss['loss_bbox']
            else:
                loss_back=loss['loss_cls']

        model.zero_grad()
        loss_back.backward()
        data_grad = data['img'].grad.data

        data_grad=data_grad.cpu().numpy()
        data_grad=data_grad.squeeze()
        data_grad=data_grad.transpose([1,2,0])
        data_grad=cv2.resize(data_grad,(500,500),cv2.INTER_AREA)
        if loss_back<=0.1:
            momentom = 0.9 * momentom + e/2 * data_grad
        else:
            momentom=0.9*momentom+e*data_grad

        momentom=momentom*std
        if mode=='ssd':
            data_grad=100*e*data_grad*std
        else:
            data_grad = e * data_grad * std
        #data_grad=np.clip(data_grad,-10,10)
        momentom=np.clip(momentom,-10,10)
        if return_grad == True:
            return data_grad, gt_bboxes
        # grad_range=np.sort(np.reshape(momentom[attack_roi],[-1]))
        # print('梯度范围', grad_range[:100],grad_range[-100:])
        adv_x[attack_roi]=adv_x[attack_roi]-momentom[attack_roi]
        momentom = momentom / std
        adv_x[attack_roi]=np.clip(adv_x[attack_roi],0,255)
        data['img']=(adv_x-mean)/std
        data['img']=cv2.resize(data['img'],(image_size,image_size))
        data['img']=torch.from_numpy(data['img'].transpose(2, 0, 1)).float().unsqueeze(0)
        #data['img']=torch.clamp(data['img'],data_min,data_max)
        data['img']=data['img'].cuda()

        if loss_back < loss_last:
            res=adv_x
            loss_last = loss_back

        if (k+1)%5==0 or (k+1)==at_times:
            print(filename,'frcn step:%d'%(k+1),loss_back,loss['loss_cls'],loss_rpn_cls)
        if k==199:
            e=e/2
            
    # ix = np.where(adv_x[attack_roi]==img[attack_roi])
    # adv_x[attack_roi[0][ix],attack_roi[1][ix],attack_roi[2][ix]]=\
    #     255-img[attack_roi[0][ix],attack_roi[1][ix],attack_roi[2][ix]]
    #data_img=data_img*std+mean
    #adv_x[attack_roi]=data_img[attack_roi]
    
    return  res


async def async_inference_detector(model, img):
    """Async inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        Awaitable detection results.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]

    # We don't restore `torch.is_grad_enabled()` value during concurrent
    # inference since execution can overlap
    torch.set_grad_enabled(False)
    result = await model.aforward_test(rescale=True, **data)
    return result


def show_result_pyplot(model, img, result, score_thr=0.3, fig_size=(15, 10)):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img, result, score_thr=score_thr, show=False)
    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))
    plt.show()
