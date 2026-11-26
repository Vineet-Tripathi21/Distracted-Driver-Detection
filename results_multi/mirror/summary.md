# EfficientNet-B0 trained on State Farm (unseen drivers) + State Farm Kaggle-test frames (coarse labels) + Ain Shams (unseen drivers, other cars, IR), mirror+label-swap p=0.5

Held-out TEST splits (unseen drivers / unseen photo groups).

| Test set | Images | 4-group acc (Safe/Text/Call/Other) | Safe-vs-distracted acc | Distracted recall | False-alarm rate | 10-class acc | 10-class macro-F1 |
|---|---|---|---|---|---|---|---|
| State Farm (unseen drivers) | 3729 | 95.0% | 96.4% | 96.7% | 5.2% | 90.3% | 0.892 |
| Ain Shams (unseen drivers, other cars, IR) | 10500 | 92.9% | 98.0% | 98.3% | 4.6% | 92.4% | 0.922 |
| Wild phone photos (new cars, buses, cameras) | 647 | 59.2% | 77.7% | 99.4% | 95.3% | n/a | n/a (coarse labels) |
