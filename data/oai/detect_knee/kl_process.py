from utils.models import *  
#from utils.datasets import *
from utils.utils import *
import matplotlib.pyplot as plt
from skimage.transform import resize
from utils.seg_function import *
import torch
import pydicom
import pandas as pd
from PIL import Image, ImageFont, ImageDraw
from torchvision import transforms
from torch.nn import DataParallel
import numpy as np
import sys
import joblib 

def detect(img0):
    '''
        return the index of the detected knees
        first line is the R knee
        second line is the L knee
    '''
    weights = 'model_files/best_finalversion.pt'
    cfg = 'model_files/yolov3-tiny.cfg'
    conf_thres = 0.001
    iou_thres = 0.2
    # Initialize
    device = torch_utils.select_device(device='cpu')
    imgsz = 512
    detection_model = Darknet(cfg, imgsz)
    name = ['joint']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(name))]
    
    # load the weight
    if weights.endswith('.pt'):
        detection_model.load_state_dict(torch.load(weights, map_location=device)['model'])
    detection_model.to(device).eval()
    #Read img
    img0 = np.stack((img0,)*3,axis = -1)
    #img0 = cv2.imread(img_path) # BGR
    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]] # the gain in width and height
    #padded resze
    img = letterbox(img0, new_shape = imgsz)[0]
    #covert
    img = img[:, :, :].transpose(2, 0, 1) # change BGR to RGB, 3*imgsz*imgsz
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /=255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
        
    # to calculate the time
    pred = detection_model(img, augment=False)[0]
    #print(pred)
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres,
                                   multi_label=False, classes=None, agnostic=False)
    det = pred[0]
    #process detections
    if det is not None and len(det):
        # Rescale boxes from imgsz to im0 size
        s = ''
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        # Print results
        n = (det[:, -1] == 0).sum()  # detections per class
        s += '%g %ss, ' % (n, name[int(0)])  # add to string
        
        # only save two highest confidence box   
        det = reversed(det[det[:,4].argsort()])[:2,:]
        result_box=[]
        center_index = []
        for *xyxy, conf, cls in reversed(det):
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
            result_box = xywh
            label = '%s %.2f' % (name[int(cls)], conf)
            #plot_one_box(xyxy, img0, label=label, color=colors[int(cls)])
            center_index.append([int(xywh[0]*img0.shape[1]), int(xywh[1]*img0.shape[0])])
        #plt.imshow(img0)
        center_index = np.array(center_index)
        center_index = center_index[center_index[:,0].argsort()]
        return center_index[:,0], center_index[:,1]

def get_input(args, img_name,side, file_type):
    ''' This function is to input image and preprocess for future use
        return: 
        seg_img: the image for segmentation
        class_img: the image for classification
        cropped_img: the image which has just been cropped
    '''


    # read image: distinguish with png or dicom type
    img_path = img_name
    if file_type == 'png':
        try:
            img = Image.open(img_path)
            img = np.asarray(img)
            print(img.shape)
        except:
            print("Image does not exist. (or) This type is not supported.")
            sys.exit()
    elif file_type=='dicom':
        try:
            img = read_dicom(img_path)
        except:
            print("Image does not exist. (or) This type is not supported.")
            sys.exit()
    else:
        raise ValueError(f"file fype {file_type} not supported")

    # initialize the arrays
    cropped_img = np.zeros([2,args.box_size, args.box_size])
    #detect the joint and crop the img
    for i,s in enumerate(side):
        joint_x, joint_y = detect(img)
        cropped_img[i] = crop(joint_y[i], joint_x[i], img, args.box_size)
    return cropped_img, img


def dect_function(args):
    ''' combine the classifciation and segmentation results
    '''
    OAI_table = pd.read_csv('oai_0e1.csv')
    for i in range(0,len(OAI_table)):
        img_name = os.path.join('../',OAI_table['DICOM_PATH'].iloc[i])
        side = ['R', 'L']
        if os.path.splitext(img_name)[-1]=='.png':
            file_type = 'png'
        else:
            file_type = 'dicom'

        # read image and preprocess
        cropped_img, raw_img = get_input(args, img_name, side, file_type)
        label = [OAI_table['R_KL'].iloc[i],OAI_table['L_KL'].iloc[i]]
        for k,s in enumerate(side):
            if label[k]>=0 and label[k]<=1:
                save_path_full = os.path.join(args.save_folder,'0', str(OAI_table['ID'].iloc[i])+ '_' + s + '.png')
            elif label[k]>=2 and label[k]<=4:
                save_path_full = os.path.join(args.save_folder,'1',str(OAI_table['ID'].iloc[i])+'_' + s + '.png')
            Image.fromarray(cropped_img[k]).convert('RGB').save(save_path_full)
        print('result %s done, ID %s'%(str(i), str(OAI_table['ID'].iloc[i])))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KL classifier")
    parser.add_argument("--narrow-type", default="lower_upper_mean", type=str, help = "the method to calculate narrowing distance")

    parser.add_argument(
        "--box-size",
        type=int,
        default=672,
        help="box size",
    )
    parser.add_argument(
        "--save-folder",
        type=str,
        default='../OAI_img/',
        help="save path",
    )
    parser.add_argument(
        "--shape",
        type=tuple,
        default=(672, 672), 
        help="image size",
    )

    parser.add_argument(
        "--mode",
        type=int,
        default=2,
        help="bone mode",
    )
    args = parser.parse_args()
    #print(f"Running in mode: {mode}")
    dect_function(args)