import cv2
import os
from threading import Thread, enumerate
import color as ReColor
from tqdm import tqdm, trange

ImageFolder_path = "./tools/tw_data/train/crop_img_train/"
save_path = "./tools/tw_data/train/BCET_crop_img_train/"
#ReColor.BCET(i[:,:,0])

def main(Start, Stop):
	num = Stop - Start + 1
	progress = tqdm(total=num)#進度條設定
	for i in range(Start, Stop):
		file_name = os.listdir(ImageFolder_path)[i]
		if file_name.split(".")[-1] == "jpg" or file_name.split(".")[-1] == "png":
			file_path = ImageFolder_path + file_name
			img = cv2.imread(file_path)
			img[:,:,0] = ReColor.BCET(img[:,:,0])
			img[:,:,1] = ReColor.BCET(img[:,:,1])
			img[:,:,2] = ReColor.BCET(img[:,:,2])
			save = save_path + file_name
			cv2.imwrite(save, img)
		progress.update(1)

if __name__ == "__main__":
	if os.path.isdir(save_path) != True:
		os.makedirs(save_path)
	total = len(os.listdir(ImageFolder_path))
	mid = int(total/2)
	midL = int(mid/2)
	midR = mid + midL
	Thread(target = main,args=(0, midL)).start()
	Thread(target = main,args=(midL, mid)).start()
	Thread(target = main,args=(mid, midR)).start()
	Thread(target = main,args=(midR, total)).start()
	print("Finish")

