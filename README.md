# Task 2 - Hand Gesture Localization & Optimization 

**Author:** 王亦丞 (Eden Wang) 

## 專案簡介
本專案旨在訓練一個高精度的手勢辨識模型，並透過模型優化與量化技術，成功將其部署於 Kneron KL730 (Kneo Pi) 邊緣運算加速裝置上，進行即時的影像推論。

本模型總共支援辨識 **11 種手勢**：
* One-hand-Heart
* Thumbs-up
* 5
* Down-V
* Thumb-and-Heart
* Two-hands-Heart
* Ok
* Rejection
* Seven
* Shaka
* Front-V

---

## 工作流程 (Workflow)

### 1. 資料準備 (Data Preparation)
* **工具管理**：使用 Roboflow 進行資料集的管理與批次上傳。
* **資料清洗**：移除帶有 typo 的錯誤標籤，並重新映射 (Remap) 不同資料集的標記名稱。
* **資料增強 (Augmentation)**：為了增加模型對不同環境的適應力，加入了水平翻轉 (Horizontal flip) 以及旋轉 (Rotation +/- 15度)。
* **資料集分割**：Train 65% / Valid 25% / Test 10%。

### 2. 模型訓練 (Model Training)
* **模型架構**：YOLOv5su (Ultralytics 版本)。
* **訓練環境**：Google Colab (T4 GPU)。
* **參數設定**：100 Epochs, Image Size 640x640, Batch Size 16。

### 3. 模型轉換與優化 (Model Conversion for Toolchain)
部署至 Kneron 裝置前，需經過以下轉換流程：
1. **Pt to ONNX**：將 PyTorch 模型匯出為 opset 11 版本的 ONNX 格式。
2. **移除 Sigmoid**：透過腳本移除模型末端的 Sigmoid 層，以符合 Kneron Toolchain 的硬體運算需求。
3. **模型量化 (Quantization)**：使用 `onnx2nef` 將模型轉為 Int8 格式，並搭配校正圖片集進行精度校正。
4. **產出模型**：生成最終可在硬體上執行的 `models_730.nef`。

---

## 訓練成果 (Performance Metrics)

模型在測試集上的整體表現如下：
* **mAP50**: 94.7%
* **Precision (P)**: 95.1%
* **Recall (R)**: 90.3%

**表現分析**：
* **最佳表現 (Best Performers)**：Ok (98.9%)、Front-V (97.8%)、Thumbs-up (97.5%)。
* **最具挑戰 (The Challenge)**：Down-V (83.2%)。推測是因為該手勢特徵在某些特定角度下，較難以與背景進行區分。

---

## 推論測試與發現 (Inference & Demo)

我們在 Kneo Pi 上執行 `KL730_v5su_fix.py` 進行即時影像與靜態圖片推論。

### 關鍵觀察
1. **分數 M 型化現象**：在硬體推論時，信心分數呈現兩極化 (通常為 0.50 或 0.99)。這是由於 NPU 進行 Int8 量化運算，配合移除 Sigmoid 函數後的陡峭特性所產生的正常現象，證明 NPU 硬體加速正在正常運作。
2. **距離限制**：在即時推論 (Real-time) 或是部分靜態測試 (如 Rejection 手勢) 時，鏡頭太過靠近會導致無法偵測，需要保持一段適當的距離才能達到最佳辨識效果。
3. **即時追蹤穩定度**：在動態變換手勢 (如從 Thumbs-up 變換為 Two-hands-Heart) 時，模型依舊能精準辨識，且偵測框能穩定跟隨手勢移動。
4. **複雜場景的誤判**：在生活照測試中，發現偶爾會將 Front-V 與 Seven 搞混 (可能受角度影響)，或是發生同一個區塊重複標記的現象。未知的無效手勢大多能被正確過濾而不被標記。
