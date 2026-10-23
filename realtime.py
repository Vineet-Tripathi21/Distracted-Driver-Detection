"""Live webcam / video demo with temporal smoothing.

  python realtime.py --ckpt runs/final/model.pt                 # webcam
  python realtime.py --ckpt runs/final/model.pt --source clip.mp4 --out annotated.mp4
Press q to quit.
"""
import argparse

import cv2
import numpy as np
from PIL import Image

from inference import Predictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--source", default="0", help="webcam index or video path")
    ap.add_argument("--out", default="")
    ap.add_argument("--alpha", type=float, default=0.3, help="EMA weight of the newest frame")
    ap.add_argument("--thr", type=float, default=0.6)
    ap.add_argument("--no_window", action="store_true")
    a = ap.parse_args()

    pred = Predictor(a.ckpt)
    cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    writer, ema = None, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        probs = pred.predict(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        ema = probs if ema is None else a.alpha * probs + (1 - a.alpha) * ema  # smooth flicker
        k, name, conf = pred.describe(ema)
        bad = k != 0 and conf >= a.thr
        color = (0, 0, 255) if bad else (0, 160, 0)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(frame, f"{'DISTRACTED: ' if bad else ''}{name} {conf:.0%}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if a.out:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            writer.write(frame)
        if not a.no_window:
            cv2.imshow("Driver distraction", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
