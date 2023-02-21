import os
import cv2
import matplotlib.pyplot as plt
import numpy as np           
import xml.etree.ElementTree as Et
from collections import Counter

def image_read(img_path, annot_path, number):
    annot_file = os.listdir(annot_path)[number]
    filename = annot_file.split(".")[0]+".jpg"
    print(filename)
    image = cv2.imread(os.path.join(img_path,filename))
    
    img = image.copy()
    xml =  open(os.path.join(annot_path, annot_file), "r")
    tree = Et.parse(xml)
    root = tree.getroot()
    
    # size = root.find('size')
    # width = size.find('width').text
    # height = size.find('height').text
    # channels = size.find('depth').text
    objects = root.findall("object")
    for _object in objects:
        name = _object.find('name').text
        bndbox = _object.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255,0,0), 2)
    plt.figure()
    plt.imshow(img)
    plt.show()

    return image


def get_iou(bb1, bb2):
    assert bb1['xmin'] < bb1['xmax']
    assert bb1['ymin'] < bb1['ymax']
    assert bb2['xmin'] < bb2['xmax']
    assert bb2['ymin'] < bb2['ymax']

    x_left = max(bb1['xmin'], bb2['xmin'])
    y_top = max(bb1['ymin'], bb2['ymin'])
    x_right = min(bb1['xmax'], bb2['xmax'])
    y_bottom = min(bb1['ymax'], bb2['ymax'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1['xmax'] - bb1['xmin']) * (bb1['ymax'] - bb1['ymin'])
    bb2_area = (bb2['xmax'] - bb2['xmin']) * (bb2['ymax'] - bb2['ymin'])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def around_context(image, x, y, w, h, p):
    imout = image.copy()
    image_mean = np.mean(imout, axis =(0,1), dtype= np.int)
    
    y_width, x_width,_ = imout.shape

    padded_image = np.full((y_width+2*p,x_width+2*p, 3), image_mean, dtype=np.uint8)
    padded_image[p:(y_width+p), p:(x_width+p), : ] =imout

    context_img = padded_image[y:(y+h+32), x:(x+w+32), :]

    return context_img

def non_max_suppression(box, pred_score, overlapThresh,class_list):
    # pred score (e,21)
    n = len(box) 
    if n == 0: 
        return []
    pick = []
    predict = np.array(list(map(lambda x: class_list[np.argmax(x)], pred_score)))
    
    for k,cl_name in enumerate(class_list):
        
        cl_score_total = pred_score[:,k]  # k번째 클래스의 점수들 (n,21) - > (n)
        cl_idx = [i  for i in range(n) if predict[i]==cl_name]   
       

        if cl_name =='background':
            continue

        Rem_idx = cl_idx   # ex [3,5,1] 같은 label을 갖는 index들
        
        while len(Rem_idx)>0:
            cl_score = cl_score_total[Rem_idx]   #class별 점수 리스트들

            temp_best_idx = Rem_idx[np.argmax(cl_score)]  # 높은 점수를 갖는 idx -> best idx
            pick.append(temp_best_idx)

            Rem_idx = np.setdiff1d(Rem_idx, temp_best_idx)
            for i in Rem_idx:
                 #pivot box와 비교할 박스의 차이가 overlapThresh보다 크면 버린다.
                if get_iou(box[temp_best_idx],box[i]) >overlapThresh: 
                    Rem_idx = np.setdiff1d(Rem_idx, i)
        
    return pick


def mean_average_precision(pred_boxes, true_boxes, obj_class, iou_threshold=0.5):
    # pred_boxes (list) : [[train_idx, class_idx, prob_score, {x1, y1, x2, y2} ], ... ]
    average_precisions = []
    epsilon = 1e-6
    num_classes=len(obj_class)
    # 각각의 클래스에 대한 AP를 구합니다.
    for c in range(1,num_classes):
        
        detections = []
        ground_truths = []

        # 모델이 c를 검출한 bounding box를 detections에 추가합니다.
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)
    
        # 실제 c 인 bounding box를 ground_truths에 추가합니다.
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        

        # amount_bboxes에 class에 대한 bounding box 개수를 저장합니다.
        # 예를 들어, img 0은 3개의 bboxes를 갖고 있고 img 1은 5개의 bboxes를 갖고 있으면
        # amount_bboexs = {0:3, 1:5} 가 됩니다.
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # class에 대한 bounding box 개수 만큼 0을 추가합니다.
        # amount_boxes = {0:np.tensor([0,0,0]), 1:np.tensor([0,0,0,0,0])}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = np.zeros(val)

        # detections를 정확도 높은 순으로 정렬합니다.
        detections.sort(key=lambda x: x[2], reverse=True)
        

        TP = np.zeros((len(detections)))
        FP = np.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)  # FP+TP

		# TP와 FP를 구합니다.
        for detection_idx, detection in enumerate(detections):
           
            # ground_truth에서 detection class와 일치하는(c) ground truth만 뽑아냄
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = get_iou(detection[3],gt[3]) 

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            else:
                FP[detection_idx] = 1
        
        # cumsum은 누적합을 의미합니다.
        # [1, 1, 0, 1, 0] -> [1, 2, 2, 3, 3]
        TP_cumsum = np.cumsum(TP)
        FP_cumsum = np.cumsum(FP)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = np.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = np.concatenate((np.array([1]), precisions))
        recalls = np.concatenate((np.array([0]),recalls))
        # np.trapz(y,x) : x-y 그래프를 적분합니다.
        average_precisions.append(np.trapz(precisions, recalls))

    ap_class=dict()
    for e,name in enumerate(obj_class[1:]):
        ap_class[name]=average_precisions[e]



    mAP=sum(average_precisions) / (len(average_precisions)-1)
    mAP=mAP*100

    return mAP,ap_class