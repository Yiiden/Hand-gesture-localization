import os
import sys
import cv2
import kp
import numpy as np
import torch
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


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

# ------------------------------
# NMS (torchvision with pure-torch fallback)
# ------------------------------

def _box_iou(a, b):
    area_a = (a[:, 2] - a[:, 0]).clamp(0) * (a[:, 3] - a[:, 1]).clamp(0)
    area_b = (b[:, 2] - b[:, 0]).clamp(0) * (b[:, 3] - b[:, 1]).clamp(0)
    lt = torch.max(a[:, None, :2], b[:, :2])
    rb = torch.min(a[:, None, 2:], b[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b - inter
    return inter / (union + 1e-7)


def _pure_torch_nms(boxes, scores, iou_thres):
    idxs = scores.argsort(descending=True)
    keep = []
    while idxs.numel() > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.numel() == 1:
            break
        ious = _box_iou(boxes[i].unsqueeze(0), boxes[idxs[1:]]).squeeze(0)
        remain = (ious <= iou_thres).nonzero(as_tuple=False).squeeze(1)
        idxs = idxs[remain + 1]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def _torchvision_nms(boxes, scores, iou_thres):
    try:
        import torchvision
        return torchvision.ops.nms(boxes, scores, iou_thres)
    except Exception:
        return _pure_torch_nms(boxes, scores, iou_thres)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    """prediction: [B, N, 5+nc] with [x,y,w,h,obj,cls...] -> returns list of [M,6] [x1,y1,x2,y2,conf,cls]"""
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]

    for bi, x in enumerate(prediction):
        x = x[xc[bi]]
        if not x.shape[0]:
            continue
        # conf = obj * class
        if nc == 1:
            x[:, 5:] = x[:, 4:5]
        else:
            x[:, 5:] *= x[:, 4:5]
        # xywh -> xyxy
        box = xywh2xyxy(x[:, :4])
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        if not x.shape[0]:
            continue
        c = x[:, 5:6] * 4096  # class offset trick
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = _torchvision_nms(boxes, scores, iou_thres)
        output[bi] = x[i]
    return output

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
    # Letterbox to target size
    img_lb, r, (dw, dh) = letterbox(img, (model_input_h, model_input_w))
    # BGR (cv2) -> RGB (Ultralytics training convention)
    img_lb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    # Normalize to 0-1
    img_lb = img_lb.astype(np.float32) / 255.0
    # HWC -> CHW
    img_lb = np.transpose(img_lb, (2, 0, 1)).copy()
    # NCHW
    return np.expand_dims(img_lb, 0), r, (dw, dh)

# ------------------------------
# Post-process (sigmoid + NMS + debug + auto shape fix)
# ------------------------------

def post_process(model_ndarray, conf_thres=0.25, iou_thres=0.45):
    nd = model_ndarray
    if nd.ndim != 3 or nd.shape[0] != 1:
        raise RuntimeError(f"Unexpected output shape: {nd.shape}")
    # auto transpose to (1, N, C)
    if nd.shape[1] < nd.shape[2]:
        nd = np.transpose(nd, (0, 2, 1))
        print('[AutoFix] Transposed output to (1, N, C)')
    print('[Debug] output shape:', nd.shape)

    out = torch.from_numpy(nd[0]).float()  # [N, 5+nc]
    # sigmoid for obj & cls (tail sigmoid was stripped for NEF export)
    out[:, 4:] = torch.sigmoid(out[:, 4:])
    print('[Debug] obj min/max:', out[:, 4].min().item(), out[:, 4].max().item())
    if out.shape[1] > 5:
        print('[Debug] cls min/max:', out[:, 5:].min().item(), out[:, 5:].max().item())

    pred = out.unsqueeze(0)
    det_list = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)

    if det_list[0].numel() == 0:
        conf2, iou2 = 0.10, 0.30
        print(f"[Debug] No boxes; retry with conf={conf2}, iou={iou2}")
        det_list = non_max_suppression(pred, conf_thres=conf2, iou_thres=iou2)
    return det_list

# ------------------------------
# Main
# ------------------------------

def main():
    parser = argparse.ArgumentParser(description='KL730 YOLOv5su Inference (Fixed & Debug)')
    parser.add_argument('-p', '--port_id', default=0, type=int, help='USB port id')
    parser.add_argument('-m', '--model', default='res/models/KL730/yolov5su/models_730.nef', type=str, help='NEF model path')
    parser.add_argument('-img', '--image_path', default='123.jpg', type=str, help='input image')
    parser.add_argument('-fw', '--firmware_path', default='res/firmware/KL730/kp_firmware.tar', type=str, help='firmware path')
    parser.add_argument('--conf', default=0.001, type=float, help='score threshold')
    parser.add_argument('--iou', default=0.30, type=float, help='NMS IoU threshold')
    parser.add_argument('--names', nargs='*', default=['fis', 'thumbs_up'], help='class names')
    args = parser.parse_args()

    # Connect
    print('[Connect Device]')
    device_group = kp.core.connect_devices(usb_port_ids=[args.port_id])
    print(' - Success')

    # Firmware
    print('[Upload Firmware]')
    kp.core.load_firmware_from_file(device_group=device_group, scpu_fw_path=args.firmware_path, ncpu_fw_path="")
    print(' - Success')

    # Model
    print('[Upload Model]')
    model_nef_descriptor = kp.core.load_model_from_file(device_group=device_group, file_path=args.model)
    print(' - Success')

    # Image & preprocess
    im0 = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    assert im0 is not None, f'Image not found: {args.image_path}'
    img, ratio, (dw, dh) = preprocess(im0, 640, 640)

    # Pack input to NPU buffer
    npu_input_buffer = convert_onnx_data_to_npu_data(
        tensor_descriptor=model_nef_descriptor.models[0].input_nodes[0],
        onnx_data=img
    )

    # Inference descriptor (✔ correct arg name)
    generic_inference_input_descriptor = kp.GenericDataInferenceDescriptor(
        model_id=model_nef_descriptor.models[0].id,
        inference_number=0,
        input_node_data_list=[kp.GenericInputNodeData(buffer=npu_input_buffer)]
    )

    print('[Start Inference]')
    kp.inference.generic_data_inference_send(
        device_group=device_group,
        generic_inference_input_descriptor=generic_inference_input_descriptor
    )
    raw = kp.inference.generic_data_inference_receive(device_group=device_group)
    print(' - Success')

    # Retrieve outputs
    print('[Retrieve Output Node]')
    outputs = []
    for node_idx in range(raw.header.num_output_node):
        node_output = kp.inference.generic_inference_retrieve_float_node(
            node_idx=node_idx,
            generic_raw_result=raw,
            channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_DEFAULT
        )
        outputs.append(node_output)
    print(' - Success')

    # Usually YOLOv5su only has one output node
    nd = outputs[0].ndarray  # shape like (1,8400,5+nc) or (1,5+nc,8400)

    # Post-process
    print('[Post Process]')
    det_list = post_process(nd, conf_thres=args.conf, iou_thres=args.iou)
    det = det_list[0]

    # Visualize
    names = args.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    if len(det):
        # Undo letterbox padding & scale back
        det[:, [0, 2]] -= dw
        det[:, [1, 3]] -= dh
        det[:, :4] /= ratio
        det[:, 0::2] = det[:, 0::2].clamp(0, im0.shape[1])
        det[:, 1::2] = det[:, 1::2].clamp(0, im0.shape[0])

        for *xyxy, conf, cls in reversed(det):
            ci = int(cls)
            label = f"{names[ci] if ci < len(names) else ci} {float(conf):.2f}"
            plot_one_box(xyxy, im0, label=label, color=colors[ci % len(names)], line_thickness=2)
    else:
        print('[Info] No boxes after NMS (even after retry).')

    save_path = os.path.abspath('./kneopi_yolov5su_result.png')
    cv2.imwrite(save_path, im0)
    print(f'✅ Result saved to {save_path}')


if __name__ == '__main__':
    main()
