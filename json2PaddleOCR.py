import os
import cv2
import re
from shutil import copyfile

#訓練資料夾
image_path = "./train/img/"
json_path = "./train/json/"

#儲存分割後的路徑
train_image_path = "./Classification_OK/train/"
test_image_path = "./Classification_OK/test/"
train_txt_path = "./Classification_OK/train/Label.txt"
test_txt_path = "./Classification_OK/test/Label.txt"

def load_json(T,file_path):
    S_str = ""
    label_name = ""
    file = open(file_path,'r',encoding="utf-8")
    
    #讀.json 開始切割不要的資料
    #一定還有優化空間
    for i in file:
        label = i.split("[{")[1]
        label_name = label.split("}]")[1]
        label_name = label_name.split("\": \"")[1]
        label_name = label_name.split("\", \"imageData\"")[0]
        label = label.split("}]")[0]
        label = "{" + label + "}"
        label = re.sub('"label":', '"transcription":', label)#從label換成transcription
        for k in label.split("},"):
            k = k.split(", \"group_id\"")[0] + "}"
            S_str += ", " + k
            #print(k)
        #print(label)

    #S_str = image_path + label_name + "\t" + S_str[2:]
    S_str = T + label_name + "\t" + "[" + S_str[2:] + "]"
    #img_1.jpg	[{"transcription": "髮型工作室", "points": [[805, 617], [1052, 613], [1059, 662], [809, 665]]},  {"transcription": "漫", "points": [[694, 557], [773, 558], [797, 667], [692, 666]]},  {"transcription": "", "points": [[701, 1018], [708, 1016], [708, 1024], [701, 1024]]},  {"transcription": "SPA", "points": [[694, 1010], [708, 1008], [711, 1015], [695, 1019]]},  {"transcription": "黛安娜", "points": [[694, 956], [707, 956], [708, 1008], [692, 1011]]},  {"transcription": "髮", "points": [[806, 620], [841, 620], [851, 666], [810, 664]]},  {"transcription": "作", "points": [[957, 617], [1001, 615], [1004, 664], [961, 662]]},  {"transcription": "工", "points": [[908, 625], [943, 619], [950, 655], [914, 658]]},  {"transcription": "室", "points": [[1012, 616], [1052, 616], [1058, 660], [1012, 663]]},  {"transcription": "型", "points": [[852, 623], [892, 618], [895, 663], [860, 664]]},  {"transcription": "娜", "points": [[693, 994], [706, 992], [709, 1007], [695, 1011]]},  {"transcription": "安", "points": [[694, 976], [707, 974], [708, 991], [694, 993]]},  {"transcription": "黛", "points": [[694, 957], [707, 957], [708, 973], [695, 974]]}]
    S_str = re.sub("}",", \"difficult\": false}",S_str)
    
    return S_str,label_name
    

if __name__ == '__main__':
    print("總數 = ",len(os.listdir(json_path)))
    train_quantity = int(len(os.listdir(json_path)) * 0.7)
    print("train\t= ",train_quantity)
    print("test\t= ",len(os.listdir(json_path))-train_quantity)
    
    #訓練data
    train_txt = open(train_txt_path, "w",encoding="utf-8")
    for i in range(0,train_quantity):
        file_path = json_path + os.listdir(json_path)[i]
        print(file_path)
        S, label_name = load_json("train/",file_path)
        print(S,file = train_txt)#寫入train.txt
        copyfile(image_path+label_name,train_image_path+label_name) #複製圖檔
    train_txt.close()
    
    #測試data
    test_txt = open(test_txt_path, "w",encoding="utf-8")
    for i in range(train_quantity,len(os.listdir(json_path))):
        file_path = json_path + os.listdir(json_path)[i]
        print(file_path)
        S, label_name = load_json("test/",file_path)
        print(S,file = test_txt)#寫入test.txt
        copyfile(image_path+label_name, test_image_path+label_name) #複製圖檔
    test_txt.close()