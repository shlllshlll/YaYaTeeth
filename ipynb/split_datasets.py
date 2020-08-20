import numpy as np
import os
from shutil import move

def get_all_file_name(filedir, extnames):
	allfilenames = []
	for dirs, dirnames, files in os.walk(filedir):
		for f in files:
			ext = os.path.splitext(f)[-1]
			if ext.lower() in extnames:
				filename = os.path.abspath(os.path.join(dirs, f))
				allfilenames.append(filename)
	return allfilenames

def get_all_image_name(filder):
	return get_all_file_name(filder,['.jpg'])

"""
source_path:原始多类图像的存放路径
gt_source_path:原始多类图像gt的存放路径
"""
#def divideTrainTest(source_path,train_path,test_path):
def divideTrainTest(source_path, gt_path):
    
    #rename the gt; change jpg to png
    gt_jpg_list = (get_all_image_name(gt_path))
    for jpg_path in gt_jpg_list:
        (floder_path, base_path) = os.path.split(jpg_path)
        #print('jpg_path: ', jpg_path)
        #print('floder_path: ',floder_path)
        print('base_path: ', base_path)
        if 'groundtruth' in base_path:
            new_path_list = base_path[28:len(base_path)]
            print('new_path_list:', new_path_list)
            print(floder_path+'/JPEGImages_original_'+ new_path_list[:-4] +'.png')
            os.rename(jpg_path, floder_path+'/JPEGImages_original_'+new_path_list[:-4]+'.png')

    #get the trainval.txt
    jpg_list = (get_all_image_name(source_path))
    trainval_txt = 'trainval.txt'
    for jpg_path in jpg_list:
        base_path = jpg_path.split('/')[-1]
        #print('jpg_path: ', jpg_path)
        print('base_path: ', base_path)
        f = open(trainval_txt,'a')
        f.write('\n'+base_path[:-4])
        f.close()
    image_num = len(jpg_list)
    print('total ', image_num)
    index_array = np.arange(image_num)
    print(index_array)
    np.random.shuffle(index_array)
    print(index_array)
    test_array = index_array[0:int(image_num*0.2):1]
    print('test_array: ', test_array)
    print('len of test_array: ', len(test_array))
    train_array = index_array[int(image_num*0.2):image_num:1]
    print('train_array: ', train_array)
    print('len of train_array: ', len(train_array))


    #get the val.txt
    val_txt = 'val.txt' 
    for i in test_array:
        image_path = os.path.split(jpg_list[i])
        image_ori_path = image_path[0]
        image_image_path = image_path[1]
        #print(image_ori_path)
        print(image_image_path)
        f = open(val_txt,'a')
        f.write('\n'+image_image_path[:-4])
        f.close()

    #get the train.txt
    train_txt = 'train.txt'
    for i in train_array:
        image_path = os.path.split(jpg_list[i])
        image_ori_path = image_path[0]
        image_image_path = image_path[1]
        #print(image_ori_path)
        print(image_image_path)
        f = open(train_txt,'a')
        f.write('\n'+image_image_path[:-4])
        f.close()

if __name__ == "__main__":
	print('start split...')

	source_path = './JPEGImages/'
	gt_source_path = './SegmentationClass/'
	#source_path = './JPEGImages_test/'
	#gt_source_path = './SegmentationClass_test/'
	#train_save_dir = './train_1/'
	#val_save_dir = './val16/'
	#if not os.path.exists(train_save_dir):
	#	os.makedirs(train_save_dir)

	#if not os.path.exists(val_save_dir):
	#	os.makedirs(val_save_dir)

	#divideTrainTest(source_path, train_save_dir, val_save_dir)
	divideTrainTest(source_path, gt_source_path)


