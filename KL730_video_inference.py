import os
import sys
import cv2
import kp
import numpy as np
import argparse
import time
from numpy import random
import warnings
warnings.filterwarnings('ignore')

# utils
from utils.ExampleHelper import convert_onnx_data_to_npu_data

# ------------------------------
# Drawing helpers (與之前相同)
# ------------------------------
def plot_one_box(x, img, color=None, label=None, line_thickness=3):
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
# Preprocess (與之前相同)
# ------------------------------
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    h0, w0 = img.shape[:2]
    r = min(new_shape[0] / h0, new_shape[1] / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if (w0, h0) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def preprocess(img, model_input_w=640, model_input_h=640):
    img_lb, r, (dw, dh) = letterbox(img, (model_input_h, model_input_w))
    img_lb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    img_lb = img_lb.astype(np.float32) / 255.0
    img_lb = np.transpose(img_lb, (2, 0, 1)).copy()
    return np.expand_dims(img_lb, 0), r, (dw, dh)

# ------------------------------
# Post-process (與之前相同)
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
    if nd.shape[1] < nd.shape[2]:
        nd = np.transpose(nd, (0, 2, 1))
    nd[:, :, 4:] = sigmoid(nd[:, :, 4:])
    det_list = non_max_suppression_v8(nd, conf_thres=conf_thres, iou_thres=iou_thres)
    return det_list

# ------------------------------
# Main (Video Logic)
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description='KL730 YOLOv5su Video Inference')
    parser.add_argument('-p', '--port_id', default=0, type=int, help='USB port id')
    parser.add_argument('-m', '--model', default='res/models/KL730/yolov5/models_730.nef', type=str, help='NEF model path')
    parser.add_argument('-v', '--video_path', default='test_video.mp4', type=str, help='input video path')
    parser.add_argument('-o', '--output_path', default='output_video.mp4', type=str, help='output video path')
    parser.add_argument('-fw', '--firmware_path', default='res/firmware/KL730/kp_firmware.tar', type=str, help='firmware path')
    parser.add_argument('--conf', default=0.25, type=float, help='score threshold')
    parser.add_argument('--iou', default=0.45, type=float, help='NMS IoU threshold')
    
    # 你的 11 個手勢名稱
    names = ['5', 'Down-V', 'Front-V', 'Ok', 'One-hand-Heart', 'Rejection', 'Seven', 'Shaka', 'Thumb-and-Heart', 'Thumbs-up', 'Two-hands-Heart']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    args = parser.parse_args()

    # 1. 初始化 Kneo Pi
    print('[Connect Device]')
    try:
        device_group = kp.core.connect_devices(usb_port_ids=[args.port_id])
        kp.core.load_firmware_from_file(device_group=device_group, scpu_fw_path=args.firmware_path, ncpu_fw_path="")
        model_nef_descriptor = kp.core.load_model_from_file(device_group=device_group, file_path=args.model)
        print(' - Device Ready')
    except Exception as e:
        print('Error init device:', e)
        return

    # 2. 開啟影片
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video_path}")
        return

    # 取得影片資訊以設定輸出
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video Info: {width}x{height} @ {fps}fps, Total Frames: {frame_count}")

    # 設定 VideoWriter (輸出 MP4)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    # out = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))

    # 設定 VideoWriter (改用 AVI + MJPG 比較穩)
    # output_path 如果使用者沒給副檔名，強制改成 .avi
    save_path = args.output_path
    if not save_path.endswith('.avi'):
        save_path = os.path.splitext(save_path)[0] + '.avi'

    fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    print('[Start Video Inference] - Press Ctrl+C to stop')
    
    frame_idx = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # 影片結束

        frame_idx += 1

        #🟢 修改 1：在開始算之前就先印出正在處理第幾張
        # end='\r' 讓它在同一行更新，不會一直洗版
        print(f"Processing Frame {frame_idx}/{frame_count}...", end='\r', flush=True)

        t1 = time.time()

        # Preprocess
        img, ratio, (dw, dh) = preprocess(frame, 640, 640)

        # Send to NPU
        npu_input_buffer = convert_onnx_data_to_npu_data(
            tensor_descriptor=model_nef_descriptor.models[0].input_nodes[0],
            onnx_data=img
        )
        generic_inference_input_descriptor = kp.GenericDataInferenceDescriptor(
            model_id=model_nef_descriptor.models[0].id,
            inference_number=0,
            input_node_data_list=[kp.GenericInputNodeData(buffer=npu_input_buffer)]
        )

        try:
            kp.inference.generic_data_inference_send(
                device_group=device_group,
                generic_inference_input_descriptor=generic_inference_input_descriptor
            )
            raw = kp.inference.generic_data_inference_receive(device_group=device_group)
        except Exception as e:
            print(f'Frame {frame_idx} failed: {e}')
            continue

        # Retrieve & Post-process
        outputs = []
        for node_idx in range(raw.header.num_output_node):
            node_output = kp.inference.generic_inference_retrieve_float_node(
                node_idx=node_idx, generic_raw_result=raw,
                channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_DEFAULT
            )
            outputs.append(node_output)

        nd = outputs[0].ndarray 
        det_list = post_process(nd, conf_thres=args.conf, iou_thres=args.iou)
        det = det_list[0]

        # 畫圖 Visualize
        if len(det) > 0:
            # Rescale coords back to original video size
            det[:, [0, 2]] -= dw
            det[:, [1, 3]] -= dh
            det[:, :4] /= ratio
            det[:, 0::2] = np.clip(det[:, 0::2], 0, width)
            det[:, 1::2] = np.clip(det[:, 1::2], 0, height)

            for *xyxy, conf, cls in reversed(det):
                ci = int(cls)
                label_name = names[ci] if ci < len(names) else str(ci)
                label = f"{label_name} {float(conf):.2f}"
                plot_one_box(xyxy, frame, label=label, color=colors[ci % len(names)], line_thickness=2)
        
        # 寫入輸出影片
        out.write(frame)

        # 進度顯示
        if frame_idx % 10 == 0:
            elapsed = time.time() - start_time
            current_fps = frame_idx / elapsed
            # print(f"Processing Frame {frame_idx}/{frame_count} | FPS: {current_fps:.2f}", end='\r')

    # 結束清理
    cap.release()
    out.release()
    print(f"\n✅ Video saved to {args.output_path}")

if __name__ == '__main__':
    main()