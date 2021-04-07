import cv2
import numpy as np
import glob

img_array = []
for filename in glob.glob('/media/bangquanxie/7E76FE92A73388C3/FusionForTracking/1.mmMOT/ammf/ammf-master/ammf/data/outputs/pyramid_cars_example/predictions/images_2d/predictions/val/20000/0.1/*.png'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


out = cv2.VideoWriter('/media/bangquanxie/7E76FE92A73388C3/FusionForTracking/1.mmMOT/ammf/ammf-master/ammf/data/outputs/pyramid_cars_example/predictions/images_2d/predictions/val/20000/project_1.avi',cv2.VideoWriter_fourcc(*'DIVX'), 1, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()


'''

def make_video(images, outimg=None, fps=5, size=None,
               is_color=True, format="XVID"):
    """
    Create a video from a list of images.
 
    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
 
    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    import os

    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid

make_video('/media/bangquanxie/7E76FE92A73388C3/FusionForTracking/1.mmMOT/ammf/ammf-master/ammf/data/outputs/pyramid_cars_example/predictions/images_2d/predictions/val/20000/0.1/','/media/bangquanxie/7E76FE92A73388C3/FusionForTracking/1.mmMOT/ammf/ammf-master/ammf/data/outputs/pyramid_cars_example/predictions/images_2d/predictions/val/20000/e2.avi')

import cv2
import os
from tqdm import tqdm
import glob
#TODO
image_folder = '/media/bangquanxie/7E76FE92A73388C3/FusionForTracking/1.mmMOT/ammf/ammf-master/ammf/data/outputs/pyramid_cars_example/predictions/images_2d/predictions/val/20000/0.1/*'
video_name = '/media/bangquanxie/7E76FE92A73388C3/FusionForTracking/1.mmMOT/ammf/ammf-master/ammf/data/outputs/pyramid_cars_example/predictions/images_2d/predictions/val/20000/e.avi'#save as .avi
#is changeable but maintain same h&w over all  frames
width=640 
height=400 
#this fourcc best compatible for avi
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
video=cv2.VideoWriter(video_name,fourcc, 2.0, (width,height))



for i in tqdm((sorted(glob.glob(image_folder),key=os.path.getmtime))):
     x=cv2.imread(i)
     video.write(x)

cv2.destroyAllWindows()
video.release()



import cv2
from cv2 import VideoWriter,VideoWriter_fourcc,imread,resize
import os
img_root="/media/bangquanxie/7E76FE92A73388C3/FusionForTracking/1.mmMOT/ammf/ammf-master/ammf/data/outputs/pyramid_cars_example/predictions/images_2d/predictions/val/20000/0.1"
#Edit each frame's appearing time!
fps=5
fourcc=VideoWriter_fourcc(*"MJPG")
videoWriter=cv2.VideoWriter("/media/bangquanxie/7E76FE92A73388C3/FusionForTracking/1.mmMOT/ammf/ammf-master/ammf/data/outputs/pyramid_cars_example/predictions/images_2d/predictions/val/20000/0.1_vio",fourcc,fps,(1200,1200))

im_names=os.listdir(img_root)
for im_name in range(len(im_names)):
	frame=cv2.imread(img_root+str(im_name)+'.png')
	print(im_name)
	videoWriter.write(frame)
	
videoWriter.release()
print("end")


# -*- coding: UTF-8 -*-
import os
import cv2
import time
 
# 图片合成视频
def picvideo(path,size):
    path = "/media/bangquanxie/7E76FE92A73388C3/FusionForTracking/1.mmMOT/ammf/ammf-master/ammf/data/outputs/pyramid_cars_example/predictions/images_2d/predictions/val/20000/0.1/"#文件路径
    filelist = os.listdir(path) #获取该目录下的所有文件名

    fps:
    帧率：1秒钟有n张图片写进去[控制一张图片停留5秒钟，那就是帧率为1，重复播放这张图片5次] 
    如果文件夹下有50张 534*300的图片，这里设置1秒钟播放5张，那么这个视频的时长就是10秒
  
    fps = 12
    size = (591,705) #图片的分辨率片
    file_path = "/media/bangquanxie/7E76FE92A73388C3/FusionForTracking/1.mmMOT/ammf/ammf-master/ammf/data/outputs/pyramid_cars_example/predictions/images_2d/predictions/val/20000/I" + str(int(time.time())) + ".mp4"#导出路径
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')#不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）
 
    video = cv2.VideoWriter(file_path, fourcc, fps, size)
 
    for item in filelist:
        if item.endswith('.png'):   #判断图片后缀是否是.png
            item = path + '/' + item 
            img = cv2.imread(item)  #使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255。
            video.write(img)        #把图片写进视频
 
    video.release() #释放
 
picvideo("/media/bangquanxie/7E76FE92A73388C3/FusionForTracking/1.mmMOT/ammf/ammf-master/ammf/data/outputs/pyramid_cars_example/predictions/images_2d/predictions/val/20000/0.1/",(591,705))



import cv2
import glob
 
fps = 10    #保存视频的FPS，可以适当调整
 
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoWriter = cv2.VideoWriter('/media/bangquanxie/7E76FE92A73388C3/FusionForTracking/1.mmMOT/ammf/ammf-master/ammf/data/outputs/pyramid_cars_example/predictions/images_2d/predictions/val/20000/saveVideo.avi',fourcc,fps,(480,360))#最后一个是保存图片的尺寸
imgs=glob.glob('/media/bangquanxie/7E76FE92A73388C3/FusionForTracking/1.mmMOT/ammf/ammf-master/ammf/data/outputs/pyramid_cars_example/predictions/images_2d/predictions/val/20000/0.1/*.png')
for imgname in imgs:
    frame = cv2.imread(imgname)
    videoWriter.write(frame)
videoWriter.release()

import cv2
import os

#图片路径
im_dir = '/media/bangquanxie/7E76FE92A73388C3/FusionForTracking/1.mmMOT/ammf/ammf-master/ammf/data/outputs/pyramid_cars_example/predictions/images_2d/predictions/val/20000/0.1/'
#输出视频路径
video_dir = '/media/bangquanxie/7E76FE92A73388C3/FusionForTracking/1.mmMOT/ammf/ammf-master/ammf/data/outputs/pyramid_cars_example/predictions/images_2d/predictions/val/20000/8.avi'
#帧率
fps = 30  
#图片数 
num = 426
#图片尺寸
img_size = (841,1023)

#fourcc = cv2.cv.CV_FOURCC('M','J','P','G')#opencv2.4
fourcc = cv2.VideoWriter_fourcc('M','J','P','G') #opencv3.0
videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

for i in range(1,num):
    im_name = os.path.join(im_dir, str(i).zfill(6)+'.png')
    frame = cv2.imread(im_name)
    videoWriter.write(frame)
    print(im_name)

videoWriter.release()
print ('finish')

'''