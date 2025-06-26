import time
import cv2
import torch
import numpy as np
import torchvision

from PIL import Image
from numpy import random


IMG_SIZE = 640
AUGMENT = False
CONF_THRES = 0.25
IOU_THRES = 0.45
CLASSES = None
AGNOSTIC_NMS = False


# 1. PyTorch 모델 파일
model_path = "./yolov5s.pt"


# 2. CUDA 연산 시간을 측정하기 위한 함수
def time_synchronized():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


# 3. 탐지된 객체에 대해 Bound Box와 Class명(Label) 출력
def plot_one_box(xyxy, img, color=(0, 255, 0), label=None, line_thickness=1):    
    x1, y1, x2, y2 = map(int, xyxy)     # 좌표 값을 정수로 변환
    cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)  # 바운딩 박스 그리기
    
    if label:
        font_scale = 0.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_color = (255, 0, 0)
        thickness = 1
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_x = x1
        text_y = y1 - 5
        cv2.putText(img, label, (text_x, text_y), font, font_scale, font_color, thickness)


# 4. 탐지된 객체에 대한 Bounding Box 좌표가 실제 이미지 범위를 넘지 않도록 수정 
def clip_coords(boxes, shape):
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


# 5. 실제 카메라 영상 이미지 크기에 맞춰 탐지된 객체에 대한 Bounding Box 좌표값 수정(Rescale)
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):    
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


# 6. Bounding Box 좌표 포맷 변환 
#    From (X_center, Y_center, Width, Height) to (top_left_X, top_left_Y, bottom_right_X, bottom_right_Y) 
def xywh2xyxy(x):    
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


# 7. 모델의 추론 결과에는 탐지된 객체 1개에 대해 여러개의 Bounding Box 정보가 출력됨.
#    이들 중에서 가장 신뢰도가 높은 Bounding Box만 추출(나머지는 제거)  
#    신뢰도 임계치(conf_thres), IoU 임계치(iou_thres)를 통해서 결정 
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    nc = prediction.shape[2] - 5            # number of classes
    xc = prediction[..., 4] > conf_thres    # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
    max_nms = 30000           # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0         # seconds to quit after
    redundant = True          # require redundant detections
    multi_label &= nc > 1     # multiple labels per box (adds 0.5ms/img)
    merge = False             # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


# 8. 탐지 함수 (main) 
def detect():
    imgsz = IMG_SIZE

# 9. PyTorch 초기화 및 CUDA 사용 가능 여부 확인 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    half = device.type != 'cpu'  # CUDA로 float16(half) 사용 
    print('device:', device)
    
# 10. 모델 파일 로드 
    model = torch.load(model_path, map_location=device)["model"].float()
    if half:
        model.half()  # to FP16

# 11. 모델이 탐지할수 있는 class 개수와 class names 출력 
    # Get Detectable object names from model
    names = model.module.names if hasattr(model, 'module') else model.names #names = model.names
    print("Detectable number of objects from model :", len(names)   )
    print("Detectable object names from model :", names) 

    # Randomly generate color datas for bounding box.
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
# 12. 실제 추론을 하기 전 모델을 한 번 "워밍업(warm-up)"
#     CUDA 처음 사용 시, CUDA 커널 로딩, 메모리 할당, 초기 컴파일 등으로 첫 추론이 느리거나 오류가 발생할수 있음
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

# 13. 카메라 설정 (640x480 해상도로 설정)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
# 14. 카메라 영상 이미지 read
        ret, frame = cap.read()
        if not ret:
            break

        img = frame[:]

# 15. 카메라 영상 이미지를 BGR to RGB로 그리고 HWC에서 CHW로 변환후 정규화 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR to RGB
        img = Image.fromarray(img)
        img = np.array(img).transpose(2, 0, 1)  # HWC -> CHW
        if half:
            img = torch.from_numpy(img).to(device).half() / 255.0
        else:
            img = torch.from_numpy(img).to(device).float() / 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t0 = time_synchronized()
        start = time.time()
# 16. 모델에 이미지 데이터를 입력하여 추론 
        pred = model(img, augment=AUGMENT)[0]
        print('pred shape:', pred.shape)

# 17. 추론 결과에 대해서 NMS 실행 
        pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)
        #print("pred :", pred)
        end = time.time()
# 18. 추론 시간 계산 (FPS)
        fps = 1.0 / (end - start)

# 19. NMS 처리후 추론 결과값에서 탐지 객체 정보 확인 
        det = pred[0]
        print('det shape:', det.shape)

        s = ''
        s += '%gx%g ' % img.shape[2:]  # print string

# 20. 탐지 객체들의 Bounding Box 좌표값을 실제 영상 이미지 크기에 맞춰 조정(Rescale)  
        if len(det):
            # Rescale boxes from img_size to img0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results on the image
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=2)
                print(label)

            print(f'Inferencing and Processing Done. ({time.time() - t0:.3f}s)')
            # Print out on image shape and detected class information
            print(s)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

# 21. 최종 객체 탐지 영상 이미지 출력      
            cv2.imshow("My_Object-Detection", frame)

# 22. ‘q’키가 눌려지면 while문 종료 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# 23. Camera 캡쳐 종료 및 OpenCV window 해제     
    cap.release()
    cv2.destroyAllWindows()


# 24. 프로그램 시작 
if __name__ == '__main__':

    # PyTorch의 자동 미분(gradient calculation) 기능 중지
    # 모델을 통해 추론만 하므로 메모리 사용량을 줄이고 연산 속도 향상 
    with torch.no_grad():
            detect()

