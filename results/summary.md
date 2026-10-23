# Results (efficientnet_b0, 224px, 5 epochs, 5-fold driver-wise CV)

Dataset: 17462 unique images, 20 drivers, 10 classes.

**Accuracy on unseen drivers: 85.64% +/- 7.04%** (macro-F1 0.857); out-of-fold accuracy 86.24%.

| Fold | Accuracy | Macro-F1 |
|---|---|---|
| 0 | 87.15% | 0.868 |
| 1 | 87.52% | 0.868 |
| 2 | 90.91% | 0.906 |
| 3 | 73.33% | 0.747 |
| 4 | 89.30% | 0.895 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Safe driving | 0.74 | 0.72 | 0.73 | 1953 |
| Texting - right hand | 0.98 | 0.87 | 0.92 | 1754 |
| Phone call - right hand | 0.94 | 0.95 | 0.95 | 1783 |
| Texting - left hand | 0.93 | 0.95 | 0.94 | 1809 |
| Phone call - left hand | 0.95 | 0.90 | 0.92 | 1817 |
| Operating the radio | 0.96 | 0.98 | 0.97 | 1804 |
| Drinking | 0.93 | 0.87 | 0.90 | 1811 |
| Reaching behind | 0.97 | 0.94 | 0.96 | 1532 |
| Hair / makeup | 0.63 | 0.83 | 0.72 | 1518 |
| Talking to passenger | 0.63 | 0.62 | 0.62 | 1681 |