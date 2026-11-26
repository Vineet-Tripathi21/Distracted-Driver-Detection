# EfficientNet-B0 trained on State Farm (unseen drivers) + State Farm Kaggle-test frames (coarse labels) + Ain Shams (unseen drivers, other cars, IR) + Wild phone photos (new cars, buses, cameras) (wild session(s) used for training: B), mirror+label-swap p=0.5

Held-out TEST splits (unseen drivers / unseen photo groups).

| Test set | Images | 4-group acc (Safe/Text/Call/Other) | Safe-vs-distracted acc | Distracted recall | False-alarm rate | 10-class acc | 10-class macro-F1 |
|---|---|---|---|---|---|---|---|
| State Farm (unseen drivers) | 3729 | 93.9% | 96.0% | 97.1% | 10.4% | 88.6% | 0.877 |
| Ain Shams (unseen drivers, other cars, IR) | 10500 | 92.8% | 96.5% | 97.2% | 10.2% | 91.8% | 0.918 |
| Wild phone photos (new cars, buses, cameras) | 183 | 69.9% | 85.8% | 98.7% | 100.0% | n/a | n/a (coarse labels) |
