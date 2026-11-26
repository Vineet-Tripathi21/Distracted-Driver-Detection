# EfficientNet-B0 trained on State Farm (unseen drivers) + State Farm Kaggle-test frames (coarse labels) + Ain Shams (unseen drivers, other cars, IR) + Wild phone photos (new cars, buses, cameras) (wild session(s) used for training: all), mirror+label-swap p=0.5

Held-out TEST splits (unseen drivers / unseen photo groups).

| Test set | Images | 4-group acc (Safe/Text/Call/Other) | Safe-vs-distracted acc | Distracted recall | False-alarm rate | 10-class acc | 10-class macro-F1 |
|---|---|---|---|---|---|---|---|
| State Farm (unseen drivers) | 3729 | 95.1% | 96.0% | 96.8% | 8.5% | 88.1% | 0.872 |
| Ain Shams (unseen drivers, other cars, IR) | 10500 | 93.6% | 97.2% | 97.5% | 5.4% | 93.3% | 0.932 |
