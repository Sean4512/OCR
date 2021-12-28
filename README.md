繁體中文

<p align="center">
 <img src="./doc/PaddleOCR_log.png" align="middle" width = "600"/>
<p align="center">

<p align="left">
	<a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
	<a href=""><img src="https://img.shields.io/badge/os-win-pink.svg"></a>
</p>

PaddleOCR網址-https://github.com/PaddlePaddle/PaddleOCR
## 環境
  - python 3.7.11
  - windowns 10
  - CUDA 11.1
  - cuDNN 8.1.1
  - 請pip install -r requirements.txt
  - 安裝paddlepaddle-gpu
    - 安裝請按照[這裡](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/windows-pip.html)
	- 每個CUDA版本都有對應的cuDNN版本
  - 以下用到的圖片，都是從[T-Brain繁體中文場景文字辨識競賽－高階賽：複雜街景之文字定位與辨識Download Dataset](https://tbrain.trendmicro.com.tw/Competitions/Details/19) 下載

## 資料處理
### json格式轉換到PPOCRLabel需要的格式
  - json2PaddleOCR.py設定
     - image_path = "./train/img/" #Train資料集圖片資料夾路徑
	 - json_path = "./train/json/" #Train資料集的json檔案資料夾路徑
  - 轉換完成後會出現train與test資料夾在Classification_OK資料夾內
  - python PPOCRLabel/PPOCRLabel.py
  - 個別讀取train跟test的圖片資料夾，然後一一點擊Check，都完成後點左上角File>Export Recognition進行輸出訓練Text Recognition時需要的圖片與txt檔案

### BCET
#### color.py是BCET主程式
是對圖片使用平衡對比度增強技術(Balance Contrast Enhancement Technique)達到色彩空間轉換的效果，使用於資料擴增與推理時使用
  - python useGPU_BCET.py
     - ImageFolder_path = "輸入PPOCRLabel.py輸出的圖片資料夾路徑"
	 - save_path = "輸入色彩轉換後圖檔的存檔路徑"

## 訓練流程
可以用來訓練Text Detection、Text Direction Classification和Text Recognition...等
  - 把PPOCRLabel.py輸出的test內的圖檔複製到train資料夾內，相對的txt檔的內容也一起複製到train的txt內容後
  - 我們是先用乾淨沒有資料擴增的圖片訓練，直到loss在1.0上下徘徊與acc在0.8之間上下徘徊，才使用BCET資料擴增，但原先的圖片全部不訓練，只訓練BCET調色過的圖片
  - python tools/train.py -c configs/rec/rec_chinese_cht_train.yml
     - 設置yml
        - use_gpu: True #設置使用GPU
        - pretrained_model: ./pretrain_models/chinese_cht_mobile_v2.0_rec_train/best_accuracy #設置加載預訓練模型路徑
        - checkpoints: #設置加载模型路径，用于中断后加载模型继续训练
        - character_dict_path: ./ppocr/utils/dict/chinese_cht_dict.txt #設置字典路徑
        - character_type: chinese_cht #設置文字種類
        - max_text_length: 30 #設置字串最大長度
        - use_space_char: True #設置是否識別空格
        - data_dir: ./data/ #設置PPOCRLabel.py輸出的圖片資料夾路徑
        - label_file_list: ./data/crop_img_train/rec_gt_train.txt #設置PPOCRLabel.py輸出的txt檔案路徑
        - 詳細請看https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/doc/doc_ch/config.md

## 預測
### 模型導出
inference模型一般是模型訓練，把模型結構和模型參數保存在文件中的固化模型，多用於預測部署場景。訓練過程中保存的模型是checkpoints模型，保存的只有模型的參數，多用於恢復訓練等。與checkpoints模型相比，inference 模型會額外保存模型的結構信息，在預測部署、加速推理上性能優越，靈活方便，適合於實際系統集成。
  - python tools/export_model.py -c configs/rec/rec_chinese_cht_train.yml -o Global.pretrained_model="./output/rec_chinese_cht/latest" Global.save_inference_dir="./output/rec_chinese_cht_inference/"
     - -c 訓練時用的yml檔
     - -o Global.pretrained_model 是設定train.py訓練後輸出的模型路徑
     - Global.save_inference_dir 是設定輸出轉檔存檔路徑

### run.py
是用來推理圖片，並輸出答案至當前資料夾的qwe.txt，是由tools/infer/predict_system.py改寫而來。
  - 以下加載的模型都是inference模型，以及除了Text Recognition模型另外兩個模型都是下載PaddleOCR已訓練好的推理模型
  - Text Detection = [ch_ppocr_server_v2.0_det](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar)
  - Text Direction Classification = [ch_ppocr_mobile_v2.0_cls](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar)
  - Text Recognition = [這裡下載](https://drive.google.com/drive/folders/1KLwdk7-ZURb-rpdP-e9EFMcaXPHnRmDH?usp=sharing)
  - 請至[這裡下載](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/doc/doc_en/models_list_en.md)
  - run.py裡面參數有
     - args.rec_char_dict_path="字典路徑"
     - args.det_model_dir= "Text Detection模型資料夾路徑"
     - args.cls_model_dir= "Text Direction Classification模型資料夾路徑"
     - args.rec_model_dir= "Text Recognition模型資料夾路徑"
     - args.image_dir = "待推圖片的路徑" 這裡可以切換至public 或 private
     - args.use_angle_cls = True 是否開啟方向識別功能
     - args.det_db_thresh = 0.3
     - args.det_db_unclip_ratio=1.75
     - args.det_db_box_thresh= 0.2
     - args.drop_score = 0.4
     - on_otsu = False 無用途!!
     - on_BCET = False 在Text Recognition前對圖片使用BCET
     - Twist = False 對圖片重新裁切 目的是改善對歪斜字體的識別 目前效果不佳
	 - img_show = False 如果True的話，顯示有檢測框的圖
	 - crop_img_show = False 如果True的話，顯示從檢測框裁切下經過拉正後的圖
	 - BCET_crop_img_show = False 如果BCET_crop_img_show = Tru e跟 on_BCET = True，顯示從檢測框裁切下來拉正後在經過BCET的圖
     - 其他參數請參考以下網址
     - https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/doc/doc_ch/inference.md
