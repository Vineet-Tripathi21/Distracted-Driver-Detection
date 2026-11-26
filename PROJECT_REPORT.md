# Driver Distraction Detection – Project Report

Image classification of driver behaviour (safe driving, texting, phone call and 7 other distractions) with a
fine-tuned convolutional neural network, a Streamlit GUI with explainability (Grad-CAM) and a monitored deployment.

All numbers come from files in this repository: `eda/eda_stats.json`, `results/summary.md`,
`results_multi/REPORT.md`, `results_multi/tuning/summary.csv` and `results_multi/cv/summary.json`.

---

## 1. Dataset description

Three public datasets were combined, with about 71,000 images in total after cleaning.

| Dataset | Images | Drivers | Labels | Setting |
|---|---|---|---|---|
| **State Farm Distracted Driver Detection** (Kaggle) | 17,462 | 20 | 10 classes c0–c9 | 1 car, 1 side camera, daylight, 320×240 in our copy |
| **Ain Shams "Novel Driver Distractions Dataset with Low Lighting Support"** (Mendeley) | 52,350 | 70 | the same 10 classes | 5 cars, 1280×720, IR camera (purple cast), day and night |
| **Multi-Class Driver Behavior Image Dataset** (Zenodo) | 7,276 raw → 1,940 kept | – | 5 coarse classes (safe, texting, talking, turning, other) | mostly State Farm frames plus phone photos of other cars and buses |

The 10 classes are: safe driving, texting (right hand), phone call (right hand), texting (left hand), phone call
(left hand), operating the radio, drinking, reaching behind, hair/make-up, and talking to a passenger.
For the headline answer they are merged into **4 groups**: Safe, Texting, Phone call and Other distraction.

**Challenges and biases found, and how they were handled**

| Problem | Evidence | Fix |
|---|---|---|
| Duplicate rows | The State Farm CSV has 34,924 rows but only 17,462 unique ones; every image is listed twice. | Duplicates dropped, and the remaining files checked on disk (`data.load_index`). |
| Cross-dataset duplicates and leakage | 5,336 of the 7,276 Multi-Class images are copies of State Farm images (perceptual-hash distance ≤ 4). Their labels agree with State Farm 96% of the time. | Copies dropped. The other 1,293 State-Farm-style frames are used for training only (source `sfx`). The 647 genuinely new photos are kept as an external **wild** test set (`prepare_multi.py`). |
| Near-identical video frames | Each driver has 346–1,237 frames (State Farm) or 675–750 frames (Ain Shams) of the same scene. | All splits are made **by driver**, never by image. Splitting by image would leak almost identical frames into the test set. |
| Camera and car bias | The average image of each dataset shows its fixed camera position (`eda/04_mean_image_per_dataset.png`). A model trained on State Farm alone learns that one car. | Training on several datasets, crop/rotation/colour augmentation, and mirroring with a label swap. |
| Lighting bias | Ain Shams has a cluster of dark night frames (13.5% with brightness < 0.15) and a strong colour cast. Wild photos are brighter. | Colour and brightness jitter, plus random grayscale. |
| Coarse labels | The Multi-Class dataset does not say which hand is used. | A **partial-label loss** accepts either hand (see §3). |
| Group imbalance | At the 4-group level, "Other" covers 5 classes (49% of images) and Safe is only 10%. | The model is trained on the 10 balanced classes and summed into groups. We also report safe-vs-distracted accuracy and the false-alarm rate, not just accuracy. |
| Viewpoint bias that remains | All training cameras look at the driver from the side. | **Not fully solved.** Frontal-camera photos still give many false alarms (§5). This is stated in the GUI and below. |

## 2. Exploratory data analysis

The EDA is in `eda.py`. Figures are saved in `eda/` and also shown in the GUI tab *Data & training*.

* **Class balance** (`01_class_balance.png`): State Farm has a max/min class ratio of 1.29, and Ain Shams 1.01.
  Accuracy is therefore a fair metric, and class weighting is not needed.
* **Images per driver** (`02_images_per_driver.png`): State Farm drivers range from 346 to 1,237 frames (median
  829). Ain Shams is almost uniform at about 750 per driver. Driver-wise splitting matters because a few State
  Farm drivers dominate.
* **Resolution**: State Farm is 320×240 in our copy, Ain Shams 1280×720, and the wild photos range from 683×384
  to 3072×4096 phone pictures. Everything is cached at a long side of 320 px and resized to 224×224.
* **Brightness and colour** (`03_brightness_colour.png`): Ain Shams is bimodal, with a separate night/IR cluster
  around 0.1 brightness. Its mean saturation (0.45) is higher than State Farm's (0.31) because of the purple IR
  cast. Wild photos are the brightest (mean 0.48). This shift motivates the colour augmentation.
* **Mean image per dataset** (`04_mean_image_per_dataset.png`): the camera position and car interior are clearly
  visible in the average image. This explains the domain gap: a model trained only on State Farm scores just 53%
  on Ain Shams.
* **Samples per class** (`05_samples_per_class.png`, `06_wild_samples.png`): left/right texting differ only by
  which hand holds the phone. This is why the 4-group answer is more reliable than the 10-class one.

## 3. Data pre-processing and feature engineering

1. **Cleaning**: duplicate CSV rows, missing files and unreadable images were removed. Perceptual hashing
   (64-bit dHash) found 5,336 cross-dataset duplicates.
2. **Unified index** (`data_multi/index.csv`): one row per image with its source, driver id, the set of allowed
   labels, the fine label and the train/test split.
3. **Image cache**: images are resized once to a long side of 320 px with EXIF rotation fixed, so the large
   1280×720 and phone images do not have to be decoded at full size in every epoch.
4. **Splits**: State Farm uses driver-wise fold 0 as test (4 drivers) and folds 1–4 for training. Ain Shams uses
   the dataset's own driver-wise test split (14 drivers). The wild photos are used for testing only.
5. **Normalisation**: resize to 224×224 and apply ImageNet mean/std, as the pre-trained backbone expects.
6. **Augmentation** (training only): RandomResizedCrop (scale 0.6–1, aspect ratio 0.75–1.8), ±10° rotation,
   colour jitter, 15% random grayscale, and **mirroring with label swap** (texting right ↔ left, call right ↔
   left). A plain flip would corrupt the left/right labels.
7. **Balanced multi-source sampling**: each epoch draws 35% State Farm, 15% extra State Farm, and 50% Ain Shams
   images with a weighted sampler, so the largest dataset does not dominate.
8. **Feature engineering**:
   * *Label-set encoding*: a coarse label such as "texting" becomes the allowed set {c1, c3}. The loss is
     `-log Σ_{k∈allowed} p_k`, so a coarse dataset can still train a 10-class model.
   * *Group features*: the 4 group probabilities are sums of the 10 class probabilities, which makes them robust
     to left/right confusion.
   * Features are learned by the CNN, so no hand-made features are needed. The EfficientNet backbone acts as a
     learned feature extractor (transfer learning).

## 4. Model execution

**Models tried**

| Model | Where |
|---|---|
| ResNet-50 from scratch at 64×64 (original notebook) | baseline, failed (BatchNorm used in training mode at evaluation) |
| EfficientNet-B0, ImageNet pre-trained, State Farm only | `train.py`, 5-fold CV |
| EfficientNet-B0, MobileNetV3-Large and ResNet-50 on all datasets | `tune_multi.py sweep` |
| EfficientNet-B0 multi-dataset, with and without mirror augmentation, with and without wild photos | `train_multi.py`, `run_multi_experiments.sh` |

**Training challenges and how they were solved**

* *Domain gap*: the State-Farm-only model scored 53% on Ain Shams and flagged 100% of safe drivers there.
  Multi-dataset training raised this to 93.6%.
* *Coarse labels*: handled with the partial-label loss.
* *Data leakage*: duplicate audit and driver-wise splits.
* *Windows memory errors* ("paging file too small", CPU out-of-memory in data-loader workers): fewer workers,
  heavy imports kept out of worker processes, and fewer evaluation workers.
* *Left/right confusion*: mirror augmentation with label swap, and the 4-group headline output.

**Hyperparameter optimisation**: a grid search (`tune_multi.py sweep`) over backbone (EfficientNet-B0,
MobileNetV3-Large, ResNet-50), learning rate (1e-4, 3e-4, 1e-3) and mirror augmentation (on/off). Each run is
scored on a **validation split made of held-out training drivers** (16 drivers, 12,108 images). The test drivers
are never used for tuning. Each run is short: 4 epochs × 16,000 images.

| Config | Val 4-group | Val 10-class | Val macro-F1 | Val loss | Minutes |
|---|---|---|---|---|---|
| EfficientNet-B0, lr 1e-4 | 86.1% | 77.7% | 0.775 | 0.651 | 4.7 |
| EfficientNet-B0, lr 3e-4 | 91.5% | 88.0% | 0.880 | 0.447 | 4.1 |
| EfficientNet-B0, lr 1e-3 | 94.1% | 92.0% | 0.920 | 0.332 | 3.9 |
| EfficientNet-B0, lr 3e-4, no mirror | 91.8% | 89.9% | 0.900 | 0.380 | 3.8 |
| MobileNetV3-Large, lr 3e-4 | 90.9% | 88.4% | 0.885 | 0.477 | 3.6 |
| MobileNetV3-Large, lr 1e-3 | 91.6% | 89.7% | 0.898 | 0.403 | 3.6 |
| **ResNet-50, lr 3e-4** | **95.2%** | **93.7%** | **0.937** | **0.284** | 6.9 |

Findings:
* **Learning rate matters most.** Going from 1e-4 to 1e-3 adds 14 points of 10-class accuracy in short runs.
* **ResNet-50 scores highest**, but it has about 5× the parameters of EfficientNet-B0 (25.6M vs 5.3M) and
  trains 1.8× slower. EfficientNet-B0 at 1e-3 is within 2 points, so it stays the demo model because it is
  light enough for CPU deployment.
* **Mirror augmentation** costs about 2 points on this validation set. That is expected: every validation car is
  left-hand drive with a side camera. It helps on the wild photos, where 4-group accuracy rose from 56.6% to
  59.2%, so it is kept.
* **Next step**: retrain the final model with lr 1e-3, the best setting found here. The current final model
  uses 3e-4, which was chosen before the sweep.

**Convergence**: AdamW (weight decay 1e-2), a 1-epoch linear warm-up, then cosine decay to 0, mixed precision,
and 0.1 label smoothing. The final model's training loss falls from 1.41 to 0.51 over 8 epochs and flattens in
the last two (`results_multi/final/history.csv`). In the sweep, the validation loss follows the training loss
without diverging (`results_multi/tuning/curves.png`). We always report the final epoch, never the best epoch
picked on test data.

## 5. Model evaluation

**Metrics**: 4-group accuracy (the headline), safe-vs-distracted accuracy, distracted recall, **false-alarm rate**
(safe drivers flagged), 10-class accuracy, macro-F1 (treats every class equally), and confusion matrices.
Accuracy alone would hide whether errors are false alarms or missed distractions.

**Held-out test results** (unseen drivers):

| Model | State Farm 4-group | Ain Shams 4-group | Wild photos 4-group | Ain Shams false alarms |
|---|---|---|---|---|
| Old model (State Farm only) | 91.9% | 53.0% | 43.4% | 100% |
| + Ain Shams + extra State Farm | 95.4% | 92.9% | 56.6% | 7.2% |
| + mirror with label swap | 95.0% | 92.9% | 59.2% | 4.6% |
| + wild sessions (cross-session test) | 93.4% | 93.1% | 64.8% | 7.4% |
| **Final model** | **95.1%** | **93.6%** | – (trained on all) | 5.4% |

Final model in detail: State Farm 96.0% safe-vs-distracted, 88.1% 10-class, macro-F1 0.872. Ain Shams 97.2%
safe-vs-distracted, 93.3% 10-class, macro-F1 0.932.

**Cross-validation**

* State-Farm-only model: **5-fold driver-wise CV, 85.64% ± 7.04%** (macro-F1 0.857). Fold 3 scored only 73%,
  which shows that results depend heavily on *which* drivers are held out. One train/test split would have
  been misleading. The weakest classes were safe driving, hair/make-up and talking to a passenger.
* Multi-dataset set-up: **3-fold driver-wise CV** on the 73 training drivers (`tune_multi.py cv`, 5 epochs per
  fold), with State Farm and Ain Shams drivers spread over the folds:

  | Fold | Val drivers | Val images | 4-group | 10-class | Macro-F1 |
  |---|---|---|---|---|---|
  | 0 | 25 | 18,451 | 90.9% | 89.5% | 0.896 |
  | 1 | 24 | 19,050 | 94.7% | 93.1% | 0.931 |
  | 2 | 23 | 18,082 | 91.9% | 90.4% | 0.905 |
  | **Mean ± std** | | | **92.5% ± 2.0%** | **91.0% ± 1.8%** | **0.911 ± 0.018** |

  **Impact on the evaluation**: the spread between folds fell from ±7.0 points (State Farm only) to ±1.8 points.
  The multi-dataset model is not only more accurate but also far less dependent on which drivers it meets. The
  CV mean (91.0%) is close to the held-out Ain Shams test score (93.3%), which confirms that the test numbers
  were not a lucky split.

**Honest limitation**: on wild photos from new cameras, 94–98% of safe drivers are still flagged. All training
cameras are side-mounted, so frontal views look "distracted" to the model.

## 6. GUI design and development

Built with **Streamlit** (`demo_app.py`) for a live demonstration audience: one page, no set-up, one click to a
result.

* **Analyse a photo**: a sample gallery (test-set photos with their known true answer, plus internet photos),
  upload, or webcam. The result is a coloured verdict card in plain language, with confidence, the chance of any
  distraction, and a ✔/✘ check against the true answer for test samples. Next to it are the original photo, the
  **Grad-CAM heat-map** and colour-coded probability bars. An expander shows all 10 behaviours.
* **Batch test**: several photos at once, with a results table (thumbnails and confidence bars) and CSV
  download.
* **Model performance**: KPI cards, a colour-coded comparison table with a metric switch, a glossary of the
  metrics, and confusion matrices.
* **Data & training**: the EDA figures with one-line insights, plus the tuning, CV and training curves.
* **Monitoring**: live statistics of the predictions made (see §7).
* **How it works**: a pipeline diagram and an explanation of the model and its limits.
* **Sidebar settings**: heat-map on/off, an adjustable "not sure" confidence threshold, and a logging switch.

How the GUI helps the user: non-experts read a single sentence and a colour. The heat-map builds trust, or
reveals a wrong reason, by showing where the model looked. The "not sure" state prevents overconfident answers.
The performance tabs let an examiner verify the claims.

## 7. Model deployment

* **Technology**: Streamlit web app serving the PyTorch model on GPU if available, else CPU. Inference takes well
  under a second per image. Packaged with a `Dockerfile` (CPU PyTorch, non-root user, health check) and
  `requirements.txt` (CPU-only PyTorch). The image can run on any container host (Render, Railway, Azure Container Apps,
  Hugging Face Spaces) or on Streamlit Community Cloud straight from the GitHub repo. The model file is 16 MB,
  so it is versioned in git.
* **Scalability**: the model is loaded once per server (`st.cache_resource`). EfficientNet-B0 is small (16 MB,
  about 0.4 GFLOPs per image), so CPU instances are enough. For more traffic, run several containers behind a load balancer, or export to
  ONNX/TorchScript and serve it with a dedicated inference server.
* **Security and privacy**: uploads are limited to 10 MB and image types (`.streamlit/config.toml`), with XSRF
  protection on and usage statistics off. **Photos are never stored**: only the answer and its confidence are
  logged. The container runs as a non-root user. Driver photos are personal data, so a public deployment
  should add authentication and HTTPS (provided by the hosting platform).
* **Monitoring and maintenance**: every prediction is appended to `logs/predictions.csv` (time, answer,
  confidence, "not sure" flag). The *Monitoring* tab shows the answer distribution, rolling confidence and the
  share of "not sure" answers, and **warns about data drift** when more than 40% of the last 20 photos are
  uncertain. Maintenance plan: collect and label the uncertain photos, add them as a new data source, retrain
  with `train_multi.py` (fully scripted), and compare against the same held-out test sets before replacing the
  model.

## 8. Additional components

* **Grad-CAM explainability**: a heat-map of the evidence the model used, built into the GUI.
* **Partial-label learning**: training a fine-grained model from coarse labels.
* **Perceptual-hash duplicate audit**: found that 73% of one public dataset duplicates another, which would have
  leaked test data.
* **Mirror augmentation with label permutation**: covers right-hand-drive cars without corrupting labels.
* **Cross-dataset ("domain-shift") evaluation**: every model is tested on other cars and cameras, not only on the
  dataset it was trained on.
* **Uncertainty handling**: a "not sure" answer below an adjustable confidence threshold.
* **Reproducible pipeline**: `prepare_multi.py` → `train_multi.py` / `run_multi_experiments.sh` →
  `summarize_multi.py` → `eda.py` / `tune_multi.py`, each with a fixed seed, writing its results to disk.

**Future work**: front-camera training data (for example 100-Driver or DMD), a CLIP or ViT backbone pre-trained
on internet-scale images, and person-detection cropping before classification.
