import cv2
import os
from threading import Thread, enumerate
import color as ReColor
from tqdm import tqdm, trange
import numba as nb
import numpy as np

ImageFolder_path = "./tools/tw_data/train/crop_img_train/"
save_path = "./tools/tw_data/train/BCET_crop_img_train/"
#ReColor.BCET(i[:,:,0])


@nb.jit
def BCET(gray_image):
	min_val = np.min(gray_image.ravel())
	max_val = np.max(gray_image.ravel())
	x = (gray_image.astype('float') - min_val) / (max_val - min_val)
	#x = im2double(gray_image) # INPUT IMAGE
	Lmin = np.min(x.ravel()) # MINIMUM OF INPUT IMAGE
	Lmax =np.max(x.ravel()) # MAXIMUM OF INPUT IMAGE
	Lmean = np.mean(x) # MEAN OF INPUT IMAGE 
	LMssum = np.mean(pow(x,2)) # MEAN SQUARE SUM OF INPUT IMAGE 
	Gmin = 0 # MINIMUM OF OUTPUT IMAGE 
	Gmax = 255 # MAXIMUM OF OUTPUT IMAGE 
	Gmean =110 # MEAN OF OUTPUT IMAGE 80 (Recomended) 
	bnum = pow(Lmax,2) * (Gmean - Gmin) - LMssum * (Gmax - Gmin) + pow(Lmin,2) * (Gmax - Gmean) 
	bden = 2 * (Lmax * (Gmean - Gmin) - Lmean * (Gmax - Gmin) + Lmin * (Gmax - Gmean)) 
	b = bnum / bden 
	a = (Gmax - Gmin) / ((Lmax - Lmin) * (Lmax + Lmin - 2 * b)) 
	c = Gmin - a * pow((Lmin - b), 2) 
	y = a *pow((x - b),2)+ c # PARABOLIC FUNCTION 
	y = y.astype(np.uint8)
	y = a *pow((x - b),2)+ c 
	height = y.shape[0]
	width = y.shape[1]
	for row in range(height):
		for col in range(width):
			if y[row, col] > 255:
				y[row, col] = 255 
	y = y.astype(np.uint8) 
	return y 

def main(Start, Stop):
	num = Stop - Start + 1
	progress = tqdm(total=num)#進度條設定
	for i in range(Start, Stop):
		file_name = os.listdir(ImageFolder_path)[i]
		if file_name.split(".")[-1] == "jpg" or file_name.split(".")[-1] == "png":
			file_path = ImageFolder_path + file_name
			img = cv2.imread(file_path)
			img[:,:,0] = BCET(img[:,:,0])
			img[:,:,1] = BCET(img[:,:,1])
			img[:,:,2] = BCET(img[:,:,2])
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
	Thread(target = main,args=(0, mid)).start()
	Thread(target = main,args=(mid, total)).start()
	#main(0,total)
	print("Finish")
	

	'''
	Thread(target = main,args=(0, midL)).start()
	Thread(target = main,args=(midL, mid)).start()
	Thread(target = main,args=(mid, midR)).start()
	Thread(target = main,args=(midR, total)).start()
	'''
