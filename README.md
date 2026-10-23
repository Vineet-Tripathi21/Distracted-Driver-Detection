# Driver Distraction Detector

Classifies a dashboard-camera frame into 10 driver behaviours (safe driving, texting L/R,
phone call L/R, radio, drinking, reaching behind, hair/makeup, talking to passenger) using a
fine-tuned ImageNet-pretrained CNN, with a Streamlit demo and a live webcam script.

Dataset: [State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection) (Kaggle).

## Why this fixes the original notebook
| Problem in the old notebook | Here |
|---|---|
| ResNet-50 from scratch, 64x64 | Pretrained EfficientNet-B0 (transfer learning), 224x224 |
| `K.set_learning_phase(1)`: BatchNorm in train mode at eval time | `model.eval()` for evaluation/inference |
| No augmentation / regularisation | Crop, rotation, colour jitter, label smoothing, weight decay, cosine LR |
| 26-fold CV = ~6 h | Driver-wise 5-fold split (all frames of a driver stay together) |
| Only accuracy + loss | Per-class precision/recall/F1 + confusion matrix |
| Optimistic reporting | Final-epoch metrics only (no picking the best epoch on validation) |
| No demo | Streamlit app + webcam/video script |

No horizontal-flip augmentation on purpose: flipping swaps left/right hand and corrupts labels (c1<->c3, c2<->c4).

## Quickest way: everything in one command
```bash
python run_all.py --data_dir data --epochs 5 --workers 4
```
It (1) audits and cleans the dataset (drops duplicated CSV rows / missing files), (2) runs 5-fold
driver-wise cross-validation, (3) trains the final model on all drivers, and (4) writes
`results/summary.md`, `summary.json` and `confusion_matrix_all_folds.png`.
The demo model is `results/final/model.pt`. It is **resumable**: re-running skips finished folds.
Add `--skip_cv` or `--skip_final` to do only part of it.

## Step-by-step alternative
## Run it (Google Colab / Kaggle notebook with a free GPU)
```bash
pip install -r requirements.txt
# download the competition data (needs a Kaggle account + accepting the rules), e.g.
kaggle competitions download -c state-farm-distracted-driver-detection -f imgs.zip
kaggle competitions download -c state-farm-distracted-driver-detection -f driver_imgs_list.csv.zip
# unzip so you have data/driver_imgs_list.csv and data/imgs/train/c0..c9/
# (you do NOT need imgs/test - its labels are hidden)

# 1) honest estimate on UNSEEN drivers (do this first, report these numbers)
python train.py --data_dir data --arch efficientnet_b0 --epochs 10 --val_fold 0 --out_dir runs/fold0
#    repeat with --val_fold 1..4 and report mean +/- std across folds

# 2) final model on all drivers, used for the demo
python train.py --data_dir data --epochs 10 --full --out_dir runs/final

# 3) demo
streamlit run app.py                                   # upload / webcam snapshot
python realtime.py --ckpt runs/final/model.pt          # live webcam with smoothing
python realtime.py --ckpt runs/final/model.pt --source clip.mp4 --out annotated.mp4
```
Outputs per run: `model.pt`, `history.csv`, `metrics.json`, `confusion_matrix.png`.
On a Colab T4 expect roughly 2-4 minutes per epoch at 224px (use `--arch mobilenet_v3_large` for faster/lighter).

## Smoke test without the dataset
```bash
python make_dummy_data.py dummy
python train.py --data_dir dummy --arch mobilenet_v3_large --img_size 96 --epochs 25 --lr 2e-3 --no_pretrained --workers 0
```
(synthetic data - verifies the code only; never report these numbers.)

## Honest limitations (put these in your report)
* 26 drivers, one camera angle, one vehicle, staged behaviours -> real-world accuracy will be lower than dataset accuracy.
* Frames are video-derived, so always split by driver, never by image.
* Left/right-hand classes are the likely confusions; check the confusion matrix.
* Research demo, not a certified safety system.

## Ideas to extend (for extra marks)
Grad-CAM heat-maps, ONNX/TFLite export for mobile, a binary "distracted vs safe" head,
comparing architectures (MobileNetV3 vs EfficientNet vs ResNet-50), and testing on your own
webcam footage to quantify the domain gap.
