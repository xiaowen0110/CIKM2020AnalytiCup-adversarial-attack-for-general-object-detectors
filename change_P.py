import numpy as np
import os
from tqdm import tqdm
import PIL.Image as Image
import cv2
if __name__ == '__main__':
    MAX_TOTAL_AREA_RATE = 0.02  # 5000/(500*500) = 0.02
    selected_path = './select1000_new'
    max_patch_number = 10

    save_path='./select1000_new_p'
    files = os.listdir(selected_path)
    files.sort()
    bb_score_dict = {}
    for img_name_index in tqdm(range(len(files))):
        img_name = files[img_name_index]
        img_path1 = os.path.join(selected_path, img_name)
        img1 = Image.open(img_path1).convert('RGB')
        img_array=np.array(img1)
        cv2.imshow('org',img_array)
        # img_array[250,50:450:2]=255-img_array[250,50:450:2]
        # img_array[250,51:450:2]=255-img_array[250,50:450:2]
        img_array[250,50:450:2]=255-img_array[250,51:450:2]
        img_array[250,51:450:2]=255-img_array[250,51:450:2]
        k=200
        for i in range(50,450,40):
            img_array[50:250:2,i]=255-img_array[51:250:2,i]
            img_array[51:250:2,i]=255-img_array[51:250:2,i]
            img_array[250:450:2,i+20]=255-img_array[251:450:2,i+20]
            img_array[251:450:2,i+20]=255-img_array[251:450:2,i+20]
        img_array=img_array[:,:,::-1]
        cv2.imwrite(os.path.join(save_path, img_name),img_array)