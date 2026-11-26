# Evaluation of results/fold0/model.pt

Held-out TEST splits (unseen drivers / unseen photo groups).

| Test set | Images | 4-group acc (Safe/Text/Call/Other) | Safe-vs-distracted acc | Distracted recall | False-alarm rate | 10-class acc | 10-class macro-F1 |
|---|---|---|---|---|---|---|---|
| State Farm (unseen drivers) | 3729 | 91.9% | 95.3% | 98.4% | 23.5% | 86.8% | 0.865 |
| Ain Shams (unseen drivers, other cars, IR) | 10500 | 53.0% | 90.0% | 100.0% | 100.0% | 18.4% | 0.130 |
| Wild phone photos (new cars, buses, cameras) | 647 | 43.4% | 76.7% | 98.4% | 96.6% | n/a | n/a (coarse labels) |
