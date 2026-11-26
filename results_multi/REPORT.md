# Multi-dataset experiments

All numbers are on held-out data: unseen drivers (State Farm fold 0, Ain Shams test drivers) and wild phone photos from capture sessions never used for training by that model.

4 groups = Safe / Texting / Phone call / Other distraction.

## 4-group accuracy

| Model | State Farm | Ain Shams | Wild photos | What changed |
|---|---|---|---|---|
| 1. Old model: State Farm only | 91.9% | 53.0% | 43.4% | baseline - one car, one camera |
| 2. + Ain Shams + extra SF frames | 95.4% | 92.9% | 56.6% | multi-dataset training, partial-label loss |
| 3. + mirror with label swap | 95.0% | 92.9% | 59.2% | adds right-hand-drive / mirrored viewpoints |
| 4. + wild sessions (cross-session) | 93.4% | 93.1% | 64.8% | trained on one wild session, tested on the other; average of both directions |
| 5. Final demo model | 95.1% | 93.6% | - | everything above, all wild sessions in training (so no wild test) |

## Safe-vs-distracted accuracy

| Model | State Farm | Ain Shams | Wild photos | What changed |
|---|---|---|---|---|
| 1. Old model: State Farm only | 95.3% | 90.0% | 76.7% | baseline - one car, one camera |
| 2. + Ain Shams + extra SF frames | 96.6% | 97.3% | 77.4% | multi-dataset training, partial-label loss |
| 3. + mirror with label swap | 96.4% | 98.0% | 77.7% | adds right-hand-drive / mirrored viewpoints |
| 4. + wild sessions (cross-session) | 95.7% | 96.7% | 80.1% | trained on one wild session, tested on the other; average of both directions |
| 5. Final demo model | 96.0% | 97.2% | - | everything above, all wild sessions in training (so no wild test) |

## False-alarm rate (safe drivers flagged)

| Model | State Farm | Ain Shams | Wild photos | What changed |
|---|---|---|---|---|
| 1. Old model: State Farm only | 23.5% | 100.0% | 96.6% | baseline - one car, one camera |
| 2. + Ain Shams + extra SF frames | 5.4% | 7.2% | 98.0% | multi-dataset training, partial-label loss |
| 3. + mirror with label swap | 5.2% | 4.6% | 95.3% | adds right-hand-drive / mirrored viewpoints |
| 4. + wild sessions (cross-session) | 8.9% | 7.4% | 94.0% | trained on one wild session, tested on the other; average of both directions |
| 5. Final demo model | 8.5% | 5.4% | - | everything above, all wild sessions in training (so no wild test) |

## 10-class accuracy

| Model | State Farm | Ain Shams | Wild photos | What changed |
|---|---|---|---|---|
| 1. Old model: State Farm only | 86.8% | 18.4% | - | baseline - one car, one camera |
| 2. + Ain Shams + extra SF frames | 90.4% | 92.6% | - | multi-dataset training, partial-label loss |
| 3. + mirror with label swap | 90.3% | 92.4% | - | adds right-hand-drive / mirrored viewpoints |
| 4. + wild sessions (cross-session) | 88.7% | 92.2% | - | trained on one wild session, tested on the other; average of both directions |
| 5. Final demo model | 88.1% | 93.3% | - | everything above, all wild sessions in training (so no wild test) |
