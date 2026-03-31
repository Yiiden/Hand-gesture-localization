import os
import sys
import cv2
import kp
import numpy as np
import argparse
from numpy import random
import warnings
warnings.filterwarnings('ignore')

# utils
from utils.ExampleHelper import convert_onnx_data_to_npu_data

# ------------------------------
# Drawing helpers
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
# Preprocess
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
# Post-process (Fixed for YOLOv5su / YOLOv8 structure)
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
    """
    prediction: (1, 8400, 4+nc) numpy array
    Format: [cx, cy, w, h, class0_score, class1_score, ...]
    """
    outputs = []
    
    for i, image_pred in enumerate(prediction):
        # image_pred shape: (8400, 15)
        
        # 1. 拆分 Box 和 Class Scores
        # 0-3 是座標，4-14 是類別分數 (沒有獨立的 objectness)
        bbox = image_pred[:, :4]
        cls_scores = image_pred[:, 4:]
        
        # 2. 找出每個 Box 最高分的類別
        class_conf = np.max(cls_scores, axis=1)
        class_pred = np.argmax(cls_scores, axis=1)
        
        # 3. 用分數過濾 (Threshold)
        # 在 v5su/v8 架構中，class_conf 就是最終信心度
        mask = class_conf > conf_thres
        
        x_box = bbox[mask]
        x_scores = class_conf[mask]
        x_class = class_pred[mask]
        
        if x_box.shape[0] == 0:
            outputs.append(np.array([]))
            continue
            
        # 4. NMS 準備
        # 轉成 xyxy
        boxes = xywh2xyxy_numpy(x_box)
        
        # 轉成 cv2 NMS 需要的 (x, y, w, h) 左上角格式
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
            
            # 組合: [x1, y1, x2, y2, conf, cls]
            pred = np.column_stack((selected_boxes, selected_scores, selected_classes))
            outputs.append(pred)
        else:
            outputs.append(np.array([]))
            
    return outputs

def post_process(model_ndarray, conf_thres=0.25, iou_thres=0.45):
    nd = model_ndarray
    
    # Auto Transpose [1, 15, 8400] -> [1, 8400, 15] if needed
    if nd.shape[1] < nd.shape[2]:
        nd = np.transpose(nd, (0, 2, 1))
        print(f'[AutoFix] Transposed to {nd.shape}')
        
    print(f'[Debug] Output shape: {nd.shape}') # Should be (1, 8400, 15)

    # YOLOv5su/v8: 
    # Box coordinates (0-3) are usually raw output (sometimes need sigmoid, usually mostly linear-ish in ONNX)
    # Class scores (4-14) are Logits (because we removed sigmoid).
    # So we apply sigmoid ONLY to classes.
    nd[:, :, 4:] = sigmoid(nd[:, :, 4:])
    
    # Debug: Print max score found
    max_score = nd[:, :, 4:].max()
    print(f'[Debug] Max Class Score detected: {max_score:.4f}')

    det_list = non_max_suppression_v8(nd, conf_thres=conf_thres, iou_thres=iou_thres)
    return det_list

# ------------------------------
# Main
# ------------------------------

def main():
    parser = argparse.ArgumentParser(description='KL730 YOLOv5su Fix Inference')
    parser.add_argument('-p', '--port_id', default=0, type=int, help='USB port id')
    parser.add_argument('-m', '--model', default='res/models/KL730/yolov5/models_730.nef', type=str, help='NEF model path')
    parser.add_argument('-img', '--image_path', default='IMG_5353.jpeg', type=str, help='input image')
    parser.add_argument('-fw', '--firmware_path', default='res/firmware/KL730/kp_firmware.tar', type=str, help='firmware path')
    parser.add_argument('--conf', default=0.25, type=float, help='score threshold')
    parser.add_argument('--iou', default=0.45, type=float, help='NMS IoU threshold')
    
    # 請確認這裡的順序跟你的 data.yaml 一模一樣
    names = ['5', 'Down-V', 'Front-V', 'Ok', 'One-hand-Heart', 'Rejection', 'Seven', 'Shaka', 'Thumb-and-Heart', 'Thumbs-up', 'Two-hands-Heart']
    
    args = parser.parse_args()

    # Connect
    print('[Connect Device]')
    try:
        device_group = kp.core.connect_devices(usb_port_ids=[args.port_id])
        print(' - Success')
    except Exception as e:
        print('Error connecting:', e)
        return

    # Firmware
    print('[Upload Firmware]')
    try:
        kp.core.load_firmware_from_file(device_group=device_group, scpu_fw_path=args.firmware_path, ncpu_fw_path="")
        print(' - Success')
    except Exception as e:
        print('Error loading firmware:', e)
        return

    # Model
    print(f'[Upload Model] {args.model}')
    try:
        model_nef_descriptor = kp.core.load_model_from_file(device_group=device_group, file_path=args.model)
        print(' - Success')
    except Exception as e:
        print(f'Error loading model: {e}')
        return

    # Image
    im0 = cv2.imread(args.image_path)
    if im0 is None:
        print(f'❌ Error: Image not found {args.image_path}')
        return
    
    # Preprocess
    img, ratio, (dw, dh) = preprocess(im0, 640, 640)

    # Inference
    npu_input_buffer = convert_onnx_data_to_npu_data(
        tensor_descriptor=model_nef_descriptor.models[0].input_nodes[0],
        onnx_data=img
    )

    generic_inference_input_descriptor = kp.GenericDataInferenceDescriptor(
        model_id=model_nef_descriptor.models[0].id,
        inference_number=0,
        input_node_data_list=[kp.GenericInputNodeData(buffer=npu_input_buffer)]
    )

    print('[Start Inference]')
    try:
        kp.inference.generic_data_inference_send(
            device_group=device_group,
            generic_inference_input_descriptor=generic_inference_input_descriptor
        )
        raw = kp.inference.generic_data_inference_receive(device_group=device_group)
    except Exception as e:
        print('Inference failed:', e)
        return

    # Retrieve
    outputs = []
    for node_idx in range(raw.header.num_output_node):
        node_output = kp.inference.generic_inference_retrieve_float_node(
            node_idx=node_idx, generic_raw_result=raw,
            channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_DEFAULT
        )
        outputs.append(node_output)

    nd = outputs[0].ndarray 

    # Post-process
    print('[Post Process]')
    det_list = post_process(nd, conf_thres=args.conf, iou_thres=args.iou)
    det = det_list[0]

    # Visualize
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    if len(det) > 0:
        # Rescale coords
        det[:, [0, 2]] -= dw
        det[:, [1, 3]] -= dh
        det[:, :4] /= ratio
        det[:, 0::2] = np.clip(det[:, 0::2], 0, im0.shape[1])
        det[:, 1::2] = np.clip(det[:, 1::2], 0, im0.shape[0])

        for *xyxy, conf, cls in reversed(det):
            ci = int(cls)
            label_name = names[ci] if ci < len(names) else str(ci)
            label = f"{label_name} {float(conf):.2f}"
            plot_one_box(xyxy, im0, label=label, color=colors[ci % len(names)], line_thickness=2)
        print(f"✅ Found {len(det)} objects!")
    else:
        print('[Info] No boxes found after NMS.')

    save_path = 'result.jpg'
    cv2.imwrite(save_path, im0)
    print(f'✅ Result saved to {save_path}')

if __name__ == '__main__':
    main()