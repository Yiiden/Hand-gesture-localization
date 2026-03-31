import ktc, onnx, shutil, os
import numpy as np
from PIL import Image

# --- Config ---
ONNX_PATH = "model_no_sigmoid.onnx"   # YOLOv5SU ONNX 模型路徑
DATA_DIR  = "images"            # 校正圖資料夾
MODEL_ID, MODEL_VER, CHIP = 20008, "0001", "730"
OUT_DIR   = "yolov5su_output"   # 輸出目錄（BIE/NEF 會複製到這裡）
IMG_SIZE  = 640                 # YOLOv5SU 輸入尺寸

# --- Helpers ---
def ensure_dir(p):
    if not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def list_images(root):
    return [os.path.join(r, f) for r, _, fs in os.walk(root)
            for f in fs if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]

def letterbox(im, new_shape=(640,640), color=(114,114,114)):
    """
    YOLOv5 風格的 letterbox 預處理
    保持長寬比並填充灰色邊框
    """
    w0, h0 = im.size
    r = min(new_shape[1]/w0, new_shape[0]/h0)
    nw, nh = int(w0*r), int(h0*r)
    im = im.resize((nw, nh), Image.BILINEAR)
    new_im = Image.new("RGB", new_shape, color)
    new_im.paste(im, ((new_shape[1]-nw)//2, (new_shape[0]-nh)//2))
    return new_im

def preprocess(path):
    """
    YOLOv5SU 預處理：
    1. 讀取並轉換為 RGB
    2. Letterbox 縮放至 640x640（保持長寬比）
    3. 歸一化到 [0, 1]
    4. 轉換為 CHW 格式
    """
    im = Image.open(path).convert("RGB")
    im = letterbox(im, new_shape=(IMG_SIZE, IMG_SIZE))
    arr = np.array(im, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None]  # HWC -> CHW -> NCHW
    return arr

def safe_copy_to_dir(src_path, dst_dir):
    ensure_dir(dst_dir)
    dst_path = os.path.join(dst_dir, os.path.basename(src_path))
    # 避免 SameFileError：來源與目的相同就略過
    if os.path.abspath(src_path) == os.path.abspath(dst_path):
        print(f"[copy] Skip: {src_path} already at {dst_dir}")
        return dst_path
    shutil.copy(src_path, dst_path)
    print(f"[copy] -> {dst_path}")
    return dst_path

# --- Pipeline ---
print(f"[1/4] Loading and optimizing ONNX: {ONNX_PATH}")
assert os.path.isfile(ONNX_PATH), f"ONNX not found: {ONNX_PATH}"
m = onnx.load(ONNX_PATH)
m = ktc.onnx_optimizer.onnx2onnx_flow(m)
print(f"   ONNX model loaded successfully")

print("[2/4] Create ModelConfig for KL730...")
km = ktc.ModelConfig(MODEL_ID, MODEL_VER, CHIP, onnx_model=m)
print(f"   Model ID: {MODEL_ID}, Version: {MODEL_VER}, Chip: {CHIP}")

print(f"[3/4] Quantizing (BIE generation) from {DATA_DIR}...")
imgs = list_images(DATA_DIR)
assert imgs, f"No images found under {DATA_DIR}"
print(f"   Found {len(imgs)} images for calibration")
arrs = [preprocess(p) for p in imgs]
input_name = m.graph.input[0].name
print(f"   Input name: {input_name}, Input shape: {arrs[0].shape}")
bie_path = km.analysis({input_name: arrs})
safe_copy_to_dir(bie_path, OUT_DIR)
print(f"   ✅ BIE saved: {bie_path}")

print("[4/4] Compiling to NEF...")
nef_path = ktc.compile([km])
safe_copy_to_dir(nef_path, OUT_DIR)
print(f"   ✅ NEF saved: {nef_path}")
print(f"\n✅ Done! All files saved to {OUT_DIR}/")
