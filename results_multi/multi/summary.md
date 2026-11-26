# EfficientNet-B0 trained on State Farm (unseen drivers) + State Farm Kaggle-test frames (coarse labels) + Ain Shams (unseen drivers, other cars, IR)

Held-out TEST splits (unseen drivers / unseen photo groups).

| Test set | Images | 4-group acc (Safe/Text/Call/Other) | Safe-vs-distracted acc | Distracted recall | False-alarm rate | 10-class acc | 10-class macro-F1 |
|---|---|---|---|---|---|---|---|
| State Farm (unseen drivers) | 3729 | 95.4% | 96.6% | 96.9% | 5.4% | 90.4% | 0.892 |
| Ain Shams (unseen drivers, other cars, IR) | 10500 | 92.9% | 97.3% | 97.8% | 7.2% | 92.6% | 0.925 |
| Wild phone photos (new cars, buses, cameras) | 647 | 56.6% | 77.4% | 99.8% | 98.0% | n/a | n/a (coarse labels) |
