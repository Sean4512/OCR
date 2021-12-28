# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import subprocess

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
from tqdm import tqdm, trange
import cv2
import copy
import numpy as np
import time
import logging
from PIL import Image
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from ppocr.utils.logging import get_logger
from tools.infer.utility import draw_ocr_box_txt, get_rotate_crop_image

import color as ReColor
logger = get_logger()
no_use = []
img_show = False
BCET_crop_img_show = False
crop_img_show = False
on_BCET = False
on_otsu = False
Twist = False

def get_test_img(img, box):
    h , w , z = img.shape
    pts_std = np.float32([[0, 0], [w, 0],
                          [w, h],
                          [0, h]])
    M = cv2.getPerspectiveTransform(box, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (w, h),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    

def add_image( img, direction = 1, Proportion = 10):
    h , w , z = img.shape
    box = []
    if Proportion <= 0:
        Proportion = 1
    elif Proportion > 100:
        Proportion = 100
    x = (w*Proportion/100)
    w-=1
    h-=1

    if direction == 1:
        arr = np.float32([[0,0],[w-x,0],[w,h],[x,h]])
    else:
        arr = np.float32([[x,0],[w,0],[w-x,h],[0,h]])
    #arr = np.array(box)
    return arr

def ThanSize2(I,L,R, no_use):
    #比較三個 信心分數 取最高
    #L跟R只有符號的話必須大於I分數0.2以上
    #L跟R比需大於I分數0.1以上 才可取代
    dst = []
    for A, B, C in zip(I,L,R):
        new_A = A[0]
        new_B = B[0]
        new_C = C[0]
        a = A[1]
        b = B[1]
        c = C[1]
        for i in no_use:
            new_A = new_A.replace(i,"")
            new_B = new_B.replace(i,"")
            new_C = new_C.replace(i,"")
        if len(new_A) == 0:
            a = 0.1
        if len(new_B) == 0:
            b = 0.1
        if len(new_C) == 0:
            c = 0.1

        ABC = [(A[0],a),(B[0],b),(C[0],c)]
        idx, max_value = max(ABC, key=lambda item: item[1])
        #print(ABC)
        #print('Maximum value:', max_value, "At index:",idx)
        dst.append((idx,max_value))
    return dst

class TextSystem(object):
    def __init__(self, args):
        if not args.show_log:
            logger.setLevel(logging.INFO)
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

    def print_draw_crop_rec_res(self, img_crop_list, rec_res):
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite("./output/img_crop_%d.jpg" % bno, img_crop_list[bno])
            logger.info(bno, rec_res[bno])

    

    def __call__(self, img, cls=True):
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        '''
        logger.debug("dt_boxes num : {}, elapse : {}".format(
            len(dt_boxes), elapse))
        '''
        if dt_boxes is None:
            return None, None
        img_crop_list = []
        dt_boxes = sorted_boxes(dt_boxes)
        
        #調整裁切高度
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            tmp_box[0][1] -= 0
            tmp_box[1][1] -= 0
            tmp_box[2][1] += 0
            tmp_box[3][1] += 0
            img_crop = get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if crop_img_show:
            for ind,i in enumerate(img_crop_list):
                #print(i.shape)
                cv2.imshow(str(ind),i)
        if self.use_angle_cls and cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            '''
            logger.debug("cls num  : {}, elapse : {}".format(
                len(img_crop_list), elapse))
            '''
        if on_BCET:#BCET 色彩轉換
            img_list = []
            for i in img_crop_list:
                i[:,:,0] = ReColor.BCET(i[:,:,0])
                i[:,:,1] = ReColor.BCET(i[:,:,1])
                i[:,:,2] = ReColor.BCET(i[:,:,2])
                img_list.append(cv2.merge([i[:,:,0],i[:,:,1],i[:,:,2]]))
            img_crop_list = img_list
        if BCET_crop_img_show:
            for index,i in enumerate(img_crop_list):
                cv2.imshow("BCET"+str(index),i)

        rec_res, elapse = self.text_recognizer(img_crop_list)
        if Twist and crop_img_show:
            print("------------------------------------")
            print("rec_res")
            print(rec_res)
            print("------------------------------------")
        if Twist:#從切下來的圖再次拆切
            img_crop_list_L=[]
            img_crop_list_R=[]
            for i in img_crop_list:
                boxL = add_image(i, 0, 7)
                boxR = add_image(i, 1, 7)
                img_crop_list_L.append(get_rotate_crop_image(i,boxL))
                img_crop_list_R.append(get_rotate_crop_image(i,boxR))
            rec_res_L, elapse = self.text_recognizer(img_crop_list_L)
            rec_res_R, elapse = self.text_recognizer(img_crop_list_R)
            rec_res = ThanSize2(rec_res,rec_res_L,rec_res_R,no_use)
        if Twist and crop_img_show:
            print("rec_res_L")
            print(rec_res_L)
            print("------------------------------------")
            print("rec_res_R")
            print(rec_res_R)
            print("------------------------------------")
            for index,i in enumerate(img_crop_list_L):
                cv2.imshow("rec_res_L"+str(index),i)
            for index,i in enumerate(img_crop_list_R):
                cv2.imshow("rec_res_R"+str(index),i)

        '''#類似print()
        logger.debug("rec_res num  : {}, elapse : {}".format(
            len(rec_res), elapse))
        '''
        # self.print_draw_crop_rec_res(img_crop_list, rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)
        return filter_boxes, filter_rec_res


def sorted_boxes(dt_boxes):
    """
    排順序 由上到下 左到右
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes

#畫線 取det輸出的box
def Line(img, box):
    
    for i in box:
        X=[]
        Y=[]
        for x,y in i:
            X.append(x)
            Y.append(y)
        for i in range(len(X)-1):
            cv2.line(img,(X[i], Y[i]),(X[i+1], Y[i+1]),(0,0,255),4)
        cv2.line(img,(X[0], Y[0]),(X[3], Y[3]),(0,0,255),4)
    cv2.namedWindow("321",0);
    cv2.resizeWindow("321", 640, 480);
    cv2.imshow("321",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#寫入txt
def Write_txt(name,dt_boxes,rec_res,no_use):
    import re
    txt_path = "./qwe.txt"
    txt_file = open(txt_path,'a',encoding="utf-8")
    name_num = len(name.split("/"))
    name = name.split("/")[name_num-1]
    name = name.split(".")[0]
    name = name.split("\\")[1]
    num = len(dt_boxes)
    count = 0
    while(count<num):
        X=[]
        Y=[]
        for x,y in dt_boxes[count]:
            X.append(int(x))
            Y.append(int(y))
        text = rec_res[count][0]
        
        for i in no_use:
            if text.count(i)>0:
                try:
                    text = text.replace(i,"")
                except:
                    print(i)
                    print(text)
        if len(text)==0:
            count+=1
            continue
        #if len(text)==0:
            #text = '###'
        
        S = name + "," + str(X[0]) + "," + str(Y[0]) + "," + str(X[1]) + "," + str(Y[1]) + "," + str(X[2]) + "," + str(Y[2]) + "," + str(X[3]) + "," + str(Y[3]) + "," + text
        print(S,file=txt_file)
        count+=1
    txt_file.close()
    
#讀 "不使用的字"檔
def load_no_use(no_use_path):
    import re
    no_use_file = open(no_use_path,'r',encoding="utf-8")
    for i in no_use_file:
        i = re.sub("\n","",i)
        no_use.append(i)
    no_use_file.close()
    return(no_use)

def main(agrs):
    num = len(os.listdir(args.image_dir))
    image_file_list = get_image_file_list(args.image_dir)
    image_file_list = image_file_list[args.process_id::args.total_process_num]
    text_sys = TextSystem(args)
    font_path = args.vis_font_path
    drop_score = args.drop_score
    no_use_path = "./no_use.txt"
    no_use = load_no_use(no_use_path)
    # warm up 10 times
    args.warmup = True#---------------------
    if args.warmup:
        img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
        for i in range(10):
            res = text_sys(img)

    cpu_mem, gpu_mem, gpu_util = 0, 0, 0
    progress = tqdm(total=num)#進度條設定
    for idx, image_file in enumerate(image_file_list):
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        if on_otsu:
            img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret ,o6=  cv2.threshold(img1, 80, 255,  cv2.THRESH_OTSU)
            o = ret/256
            args.det_db_thresh = o
        
        dt_boxes, rec_res = text_sys(img)
        Write_txt(image_file,dt_boxes,rec_res,no_use)
        progress.update(1)#進度條
        if img_show:
            print(rec_res)
            Line(img,dt_boxes)
        else:
            if crop_img_show or BCET_crop_img_show:
                cv2.waitKey(0)
                cv2.destroyAllWindows()


if __name__ == "__main__":
    args = utility.parse_args()
    args.rec_char_dict_path="./ppocr/utils/dict/chinese_cht_dict.txt"
    args.det_model_dir="./output/ch_ppocr_server_v2.0_det_infer/"
    args.cls_model_dir="./output/ch_ppocr_mobile_v2.0_cls_infer/"
    args.rec_model_dir="./output/rec_T_inference/12_18/"
    args.image_dir = "./data/public"
    #args.image_dir = "./tools/CAPTCHA_output"
    #args.use_dilation = True
    args.use_angle_cls=True
    
    #args.det_algorithm = "DB"
    '''
    args.rec_algorithm = "SRN"
    args.rec_image_shape = "1, 64, 256"
    '''
    #args.rec_algorithm = "RARE"
    #args.rec_image_shape = "3, 32, 100"
    
    args.det_db_thresh = 0.3
    args.det_db_unclip_ratio=1.75
    args.det_db_box_thresh= 0.2
    args.drop_score = 0.4

    on_otsu = False
    on_BCET = False
    Twist = False
    img_show = False
    crop_img_show = False
    BCET_crop_img_show = False


    main(args)
    print("Det = ",args.det_model_dir)
    print("Cls = ",args.cls_model_dir)
    print("Rec = ",args.rec_model_dir)