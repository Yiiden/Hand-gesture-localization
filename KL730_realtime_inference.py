import os
import sys
import cv2
import kp
import numpy as np
import argparse
import time
import platform
import threading
from numpy import random
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# 1. 基礎工具 (開鏡頭、畫圖)
# ------------------------------
def open_cap(src):
    # 移植自 camshift.py，保證 Linux 鏡頭開啟最穩定
    is_windows = platform.system().lower().startswith("win")
    if src.isdigit():
        idx = int(src)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if is_windows else cv2.CAP_V4L2)
    else:
        backend = 0
        if not is_windows and src.startswith("/dev/video"):
            backend = cv2.CAP_V4L2
        cap = cv2.VideoCapture(src, backend)
    return cap

def plot_one_box(x, img, color=None, label=None, line_thickness=2):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

# ------------------------------
# 2. 後處理邏輯 (Post-process)
# ------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def xywh2xyxy_numpy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def non_max_suppression_v8(prediction, conf_thres=0.25, iou_thres=0.45):
    outputs = []
    for i, image_pred in enumerate(prediction):
        bbox = image_pred[:, :4]
        cls_scores = image_pred[:, 4:]
        
        class_conf = np.max(cls_scores, axis=1)
        class_pred = np.argmax(cls_scores, axis=1)
        
        mask = class_conf > conf_thres
        x_box = bbox[mask]
        x_scores = class_conf[mask]
        x_class = class_pred[mask]
        
        if x_box.shape[0] == 0:
            outputs.append(np.array([]))
            continue
            
        boxes = xywh2xyxy_numpy(x_box)
        # NMS 需要 x,y,w,h (top-left)
        boxes_for_cv2 = np.copy(x_box)
        boxes_for_cv2[:, 0] = x_box[:, 0] - x_box[:, 2] / 2
        boxes_for_cv2[:, 1] = x_box[:, 1] - x_box[:, 3] / 2
        
        indices = cv2.dnn.NMSBoxes(
            boxes_for_cv2.tolist(), 
            x_scores.tolist(), 
            conf_thres, 
            iou_thres
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            selected_boxes = boxes[indices]
            selected_scores = x_scores[indices]
            selected_classes = x_class[indices]
            pred = np.column_stack((selected_boxes, selected_scores, selected_classes))
            outputs.append(pred)
        else:
            outputs.append(np.array([]))
    return outputs

def post_process(model_ndarray, conf_thres=0.25, iou_thres=0.45):
    nd = model_ndarray
    # 自動轉置 [1, 15, 8400] -> [1, 8400, 15]
    if nd.shape[1] < nd.shape[2]:
        nd = np.transpose(nd, (0, 2, 1))
    
    # Sigmoid 轉換
    nd[:, :, 4:] = sigmoid(nd[:, :, 4:])
    
    det_list = non_max_suppression_v8(nd, conf_thres=conf_thres, iou_thres=iou_thres)
    return det_list

# ------------------------------
# 3. 多執行緒與共用變數
# ------------------------------
shared_frame = None
shared_dets = []
lock = threading.Lock()
is_running = True

def inference_worker(device_group, model_nef_descriptor, conf, iou):
    global shared_frame, shared_dets, is_running
    print("[Thread] AI Worker Started (Mode: GenericImageInference / RGB565)")
    
    while is_running:
        img_src = None
        with lock:
            if shared_frame is not None:
                img_src = shared_frame.copy()
        
        if img_src is None:
            time.sleep(0.01)
            continue

        try:
            # 🟢 極速前處理：直接 Resize 到 640x640 (不留黑邊，速度最快)
            img_resized = cv2.resize(img_src, (640, 640))
            
            # 🟢 關鍵優化：轉成 RGB565 格式 (參考官方範例)
            # 這比 float32 小非常多，傳輸超快！
            img_565 = cv2.cvtColor(img_resized, cv2.COLOR_BGR2BGR565)

            # 🟢 使用 GenericImageInference (官方範例的極速模式)
            generic_inference_input_descriptor = kp.GenericImageInferenceDescriptor(
                model_id=model_nef_descriptor.models[0].id,
                inference_number=0,
                input_node_image_list=[
                    kp.GenericInputNodeImage(
                        image=img_565,
                        image_format=kp.ImageFormat.KP_IMAGE_FORMAT_RGB565
                    )
                ]
            )

            # 發送
            kp.inference.generic_image_inference_send(
                device_group=device_group,
                generic_inference_input_descriptor=generic_inference_input_descriptor
            )
            
            # 接收
            generic_raw_result = kp.inference.generic_image_inference_receive(device_group=device_group)
            
            # 取出結果
            outputs = []
            for node_idx in range(generic_raw_result.header.num_output_node):
                inference_float_node_output = kp.inference.generic_inference_retrieve_float_node(
                    node_idx=node_idx,
                    generic_raw_result=generic_raw_result,
                    channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_DEFAULT
                )
                outputs.append(inference_float_node_output)

            nd = outputs[0].ndarray 
            det_list = post_process(nd, conf_thres=conf, iou_thres=iou)
            det = det_list[0]
            
            # 座標還原 (640x640 -> 320x240)
            # 因為我們是用直接 resize (拉伸)，所以 x 和 y 的縮放比例不同
            if len(det) > 0:
                sx = img_src.shape[1] / 640.0 # 320 / 640 = 0.5
                sy = img_src.shape[0] / 640.0 # 240 / 640 = 0.375
                
                det[:, 0] *= sx
                det[:, 1] *= sy
                det[:, 2] *= sx
                det[:, 3] *= sy
                
                det[:, 0::2] = np.clip(det[:, 0::2], 0, img_src.shape[1])
                det[:, 1::2] = np.clip(det[:, 1::2], 0, img_src.shape[0])
            
            with lock:
                shared_dets = det

        except Exception as e:
            # print(f"Inference Error: {e}")
            pass

# ------------------------------
# 4. 主程式
# ------------------------------
def main():
    global shared_frame, shared_dets, is_running

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port_id', default=0, type=int)
    parser.add_argument('-m', '--model', default='res/models/KL730/yolov5/models_730.nef', type=str)
    parser.add_argument('--src', default='0', type=str)
    parser.add_argument('-fw', '--firmware_path', default='res/firmware/KL730/kp_firmware.tar', type=str)
    parser.add_argument('--conf', default=0.45, type=float) # 稍微調高門檻
    parser.add_argument('--iou', default=0.45, type=float)
    
    names = ['5', 'Down-V', 'Front-V', 'Ok', 'One-hand-Heart', 'Rejection', 'Seven', 'Shaka', 'Thumb-and-Heart', 'Thumbs-up', 'Two-hands-Heart']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    args = parser.parse_args()

    # 初始化
    print('[Connect Device]')
    try:
        device_group = kp.core.connect_devices(usb_port_ids=[args.port_id])
        kp.core.load_firmware_from_file(device_group=device_group, scpu_fw_path=args.firmware_path, ncpu_fw_path="")
        model_nef_descriptor = kp.core.load_model_from_file(device_group=device_group, file_path=args.model)
        print(' - Device Ready')
    except Exception as e:
        print('Error init device:', e)
        return

    # 開啟 Webcam (320x240)
    print(f'[Opening Camera] Source: {args.src}')
    cap = open_cap(args.src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # 啟動 AI 線程
    ai_thread = threading.Thread(
        target=inference_worker, 
        args=(device_group, model_nef_descriptor, args.conf, args.iou),
        daemon=True
    )
    ai_thread.start()

    print('[Start Ultimate Demo] - High Speed, Low Latency!')
    print('Press "q" to exit')

    fps_avg = 0
    t_start = time.time()

    while True:
        t_frame_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
            
        with lock:
            shared_frame = frame.copy()
            current_dets = shared_dets

        if len(current_dets) > 0:
            for *xyxy, conf, cls in reversed(current_dets):
                ci = int(cls)
                label_name = names[ci] if ci < len(names) else str(ci)
                label = f"{label_name} {float(conf):.2f}"
                plot_one_box(xyxy, frame, label=label, color=colors[ci % len(names)], line_thickness=2)
        
        # 計算畫面 FPS (應該要很穩 30)
        fps = 1.0 / (time.time() - t_frame_start)
        fps_avg = 0.9 * fps_avg + 0.1 * fps
        
        cv2.putText(frame, f"FPS: {fps_avg:.1f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow('Kneo Pi Ultimate', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            is_running = False
            break

    cap.release()
    cv2.destroyAllWindows()
    time.sleep(0.5)

if __name__ == '__main__':
    main()