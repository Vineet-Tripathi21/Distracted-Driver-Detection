# EfficientNet-B0 trained on State Farm (unseen drivers) + State Farm Kaggle-test frames (coarse labels) + Ain Shams (unseen drivers, other cars, IR) + Wild phone photos (new cars, buses, cameras) (wild session(s) used for training: A), mirror+label-swap p=0.5

Held-out TEST splits (unseen drivers / unseen photo groups).

| Test set | Images | 4-group acc (Safe/Text/Call/Other) | Safe-vs-distracted acc | Distracted recall | False-alarm rate | 10-class acc | 10-class macro-F1 |
|---|---|---|---|---|---|---|---|
| State Farm (unseen drivers) | 3729 | 93.0% | 95.3% | 95.7% | 7.3% | 88.8% | 0.878 |
| Ain Shams (unseen drivers, other cars, IR) | 10500 | 93.3% | 97.0% | 97.2% | 4.6% | 92.6% | 0.926 |
| Wild phone photos (new cars, buses, cameras) | 464 | 59.7% | 74.4% | 97.1% | 87.9% | n/a | n/a (coarse labels) |
