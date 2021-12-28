import cv2
import os
import numpy as np

def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

#調色
def BCET(gray_image):
    x = im2double(gray_image) # INPUT IMAGE
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

def performDetect(imagePath, cut):
    
    #image = cv2.imread(imagePath)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if cut:
        image_cropped = image_rgb[350:862, 8:1224]        
    #io.imshow(image_rgb)
    #io.show()
    else:
        image_cropped = image_rgb
    
    #io.imshow(image_cropped)
    #io.show()
    image_cropped[:,:,0] = BCET(image_cropped[:,:,0])#調色c=1
    image_cropped[:,:,1] = BCET(image_cropped[:,:,1])#調色c=2
    image_cropped[:,:,2] = BCET(image_cropped[:,:,2])#調色c=3

    #io.imshow(image)
    #io.show()
    return image_cropped

def rotate_image(imagePath):
    savepath = 'C:/Users/SuPoTing/Desktop/rot/'
    filename = imagePath[:-4]
    image = cv2.imread(imagePath)
    h, w, _ = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    image = cv2.warpAffine(image, M, (w, h))
    #cv2.imwrite(savepath+filename+'.jpg', image)
    return image

if __name__ == "__main__":
    cut = True
    
    path='C:/Users/SuPoTing/Desktop/1/'
    fileList=os.listdir(path)   
    a=0
    for i in fileList:
        filename=path+os.sep+fileList[a]
        #image = rotate_image(filename)
        image = performDetect(filename, cut)
        print(filename+',OK')
        cv2.imwrite('C:/Users/SuPoTing/Desktop/1/'+fileList[a], image)
        a+=1