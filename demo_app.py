"""Driver Distraction Detector - demo app.

    streamlit run demo_app.py

Model: EfficientNet-B0 trained on three datasets (State Farm + Ain Shams + Multi-Class Driver Behavior).
Headline answer = 4 groups (Safe / Texting / Phone call / Other); the 10 detailed behaviours are shown underneath.
"""
import base64
import csv
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps

from data import CLASSES, build_transforms
from model import load_checkpoint
from multi_data import GROUP_NAMES, GROUP_OF, group_probs
from summarize_multi import comparison_table

HERE = Path(__file__).parent
MODEL_CANDIDATES = [HERE / "results_multi/final/model.pt", HERE / "results_multi/mirror/model.pt",
                    HERE / "results_multi/multi/model.pt", HERE / "results/final/model.pt"]
SAMPLES = HERE / "demo_samples"
RESULTS = HERE / "results_multi"
EDA = HERE / "eda"
LOG = HERE / "logs" / "predictions.csv"

GROUPS = {  # icon, colour, plain-language meaning
    "Safe driving": ("✅", "#15803d", "Hands on the wheel, attention on the road."),
    "Texting": ("📱", "#dc2626", "Holding a phone and typing or reading."),
    "Phone call": ("📞", "#c026d3", "Holding a phone to the ear."),
    "Other distraction": ("⚠️", "#d97706", "Radio, drinking, reaching behind, hair/make-up or talking to a passenger."),
}

st.set_page_config(page_title="Driver Distraction Detector", page_icon="🚗", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] {font-family: 'Inter', sans-serif;}
.block-container {padding-top: 1.6rem; max-width: 1250px;}
.hero {background: linear-gradient(120deg, #0f172a 0%, #1e3a8a 60%, #2563eb 100%); border-radius: 18px;
       padding: 26px 30px; color: #fff; margin-bottom: 18px;}
.hero h1 {color: #fff; margin: 0 0 6px 0; font-size: 2.1rem; font-weight: 800;}
.hero p {color: #dbeafe; margin: 0; font-size: 1.02rem;}
.steps {display: flex; gap: 10px; margin-top: 16px; flex-wrap: wrap;}
.step {background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.25); border-radius: 999px;
       padding: 6px 14px; font-size: .9rem; color: #fff;}
.verdict {border-radius: 16px; padding: 20px 24px; color: #fff; margin-bottom: 14px;}
.verdict .t {font-size: 1.9rem; font-weight: 800; margin: 0;}
.verdict .s {font-size: 1.02rem; opacity: .95; margin-top: 4px;}
.chip {display: inline-block; background: rgba(255,255,255,.22); border-radius: 999px; padding: 3px 11px;
       font-size: .85rem; margin: 8px 6px 0 0;}
.bar-row {margin: 7px 0;}
.bar-lbl {display: flex; justify-content: space-between; font-size: .93rem; margin-bottom: 3px;}
.bar-bg {background: rgba(128,128,128,.18); border-radius: 8px; height: 14px; overflow: hidden;}
.bar-fg {height: 14px; border-radius: 8px;}
.note {color: #6b7280; font-size: .86rem;}
.section-title {font-size: 1.15rem; font-weight: 700; margin: 6px 0 4px 0;}
.pill {display: inline-block; border-radius: 999px; padding: 1px 9px; font-size: .78rem; font-weight: 600;
       background: rgba(37,99,235,.12); color: #2563eb;}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- model + Grad-CAM
class Detector:
    def __init__(self, ckpt):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.img_size = load_checkpoint(ckpt, self.device)
        self.tf = build_transforms(self.img_size, train=False)
        layer = self.model.features[-1] if hasattr(self.model, "features") else self.model.layer4
        self._act = None
        layer.register_forward_hook(lambda m, i, o: setattr(self, "_act", o))

    def predict(self, img: Image.Image):
        """-> 10-class probabilities and a Grad-CAM heat-map (H x W in [0, 1]) for the top class."""
        x = self.tf(img.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.enable_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, 1)[0]
            k = int(probs.argmax())
            grad = torch.autograd.grad(logits[0, k], self._act)[0]
            heat = torch.relu((grad.mean(dim=(2, 3), keepdim=True) * self._act).sum(1))[0]
            heat = (heat / heat.max().clamp_min(1e-8)).detach().cpu().numpy()
        return probs.detach().cpu().numpy(), heat


@st.cache_resource(show_spinner="Loading the model...")
def get_detector(path):
    return Detector(path)


def overlay(img, heat, alpha=0.45):
    import matplotlib.cm as cm
    img = img.convert("RGB")
    h = Image.fromarray(np.uint8(heat * 255)).resize(img.size, Image.BILINEAR)
    colored = Image.fromarray(np.uint8(cm.jet(np.asarray(h) / 255.0)[..., :3] * 255))
    return Image.blend(img, colored, alpha)


def model_path():
    return next((p for p in MODEL_CANDIDATES if p.is_file()), None)


def load_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def log_prediction(name, source, group, conf, top_class, uncertain):
    """Privacy-friendly monitoring log: only the answer and confidence, never the image."""
    LOG.parent.mkdir(exist_ok=True)
    new = not LOG.is_file()
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "input", "source", "group", "confidence", "top_class", "uncertain"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), name, source, group, f"{conf:.4f}", top_class,
                    int(uncertain)])


def bars_html(names, values, colours):
    rows = []
    for n, v, c in zip(names, values, colours):
        rows.append(f"<div class='bar-row'><div class='bar-lbl'><span>{n}</span><b>{v:.0%}</b></div>"
                    f"<div class='bar-bg'><div class='bar-fg' style='width:{v * 100:.1f}%;background:{c}'></div>"
                    f"</div></div>")
    return "".join(rows)


def verdict(probs, threshold):
    gp = group_probs(probs)[0]
    g = int(gp.argmax())
    return gp, g, GROUP_NAMES[g], bool(gp[g] < threshold)


# ----------------------------------------------------------------------------- setup
mp = model_path()
if mp is None:
    st.error("No trained model found. Run `python train_multi.py` first.")
    st.stop()
det = get_detector(str(mp))
labels = load_json(SAMPLES / "labels.json") or {}
final_metrics = load_json(RESULTS / "final/test_metrics.json") or {}
cv_summary = load_json(RESULTS / "cv/summary.json")

with st.sidebar:
    st.markdown("## 🚗 Driver Distraction Detector")
    st.markdown("<span class='pill'>EfficientNet-B0</span> <span class='pill'>3 datasets</span> "
                "<span class='pill'>~70k images</span>", unsafe_allow_html=True)
    st.divider()
    st.markdown("**Settings**")
    show_cam = st.toggle("Show attention heat-map", value=True,
                         help="Grad-CAM: highlights the pixels that pushed the model towards its answer.")
    threshold = st.slider("Say 'not sure' below this confidence", 0.30, 0.90, 0.50, 0.05,
                          help="If the top answer is less likely than this, the app shows 'Not sure' instead.")
    do_log = st.toggle("Log predictions for monitoring", value=True,
                       help="Stores only the answer and confidence (no images) in logs/predictions.csv.")
    st.divider()
    st.markdown("**Tips for good results**")
    st.markdown("- Photo taken **inside the car**\n- Driver's **upper body and hands** visible\n"
                "- Side or front-side view works best\n- Not a road-facing dash-cam")
    st.caption(f"Model file: `{mp.relative_to(HERE)}` · running on {det.device.upper()}")

st.markdown("""
<div class='hero'>
  <h1>🚗 Driver Distraction Detector</h1>
  <p>A deep-learning model that looks at a photo of a driver and tells you whether they are driving safely,
  texting, on a phone call, or distracted by something else &mdash; and shows <b>where it looked</b> to decide.</p>
  <div class='steps'>
    <span class='step'>① Pick a sample, upload a photo or use the camera</span>
    <span class='step'>② The model analyses it in a fraction of a second</span>
    <span class='step'>③ Read the verdict and check the heat-map</span>
  </div>
</div>""", unsafe_allow_html=True)

tabs = st.tabs(["🔍 Analyse a photo", "🗂️ Batch test", "📊 Model performance", "🔬 Data & training",
                "📈 Monitoring", "ℹ️ How it works"])

# ============================================================================ 1. analyse
with tabs[0]:
    if "img" not in st.session_state:
        st.session_state.update(img=None, img_name=None, img_src=None, logged=None)

    def pick_sample(p):
        st.session_state.update(img=Image.open(p), img_name=p.name, img_src="sample")

    def pick_file(key, src):
        f = st.session_state.get(key)
        if f is not None:
            st.session_state.update(img=ImageOps.exif_transpose(Image.open(f)),
                                    img_name=getattr(f, "name", src), img_src=src)

    picker = st.expander("**Step 1 · Choose a photo**" if st.session_state.img is None
                         else "**Step 1 · Choose another photo** (click to open)",
                         expanded=st.session_state.img is None)
    with picker:
        how = st.segmented_control("Image source", ["🖼️ Sample gallery", "⬆️ Upload", "📷 Camera"],
                                   default="🖼️ Sample gallery", label_visibility="collapsed")
    if how == "⬆️ Upload":
        picker.file_uploader("Upload a photo of a driver", type=["jpg", "jpeg", "png", "webp"], key="up",
                             on_change=pick_file, args=("up", "upload"))
    elif how == "📷 Camera":
        picker.camera_input("Take a photo (sit as if you were driving)", key="cam",
                            on_change=pick_file, args=("cam", "camera"))
    else:
        files = sorted(p for p in SAMPLES.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        groups = [("From the test datasets (drivers the model never saw) — the correct answer is known",
                   [p for p in files if p.name in labels]),
                  ("From the internet (new cars, cameras and people)", [p for p in files if p.name not in labels])]
        for title, items in groups:
            if not items:
                continue
            picker.markdown(f"<span class='note'>{title}</span>", unsafe_allow_html=True)
            cols = picker.columns(8)
            for i, p in enumerate(items):
                with cols[i % 8]:
                    st.image(ImageOps.fit(Image.open(p).convert("RGB"), (320, 240)), width="stretch",
                             caption=labels[p.name]["dataset"] if p.name in labels else "Internet")
                    st.button("Analyse", key=f"s_{p.name}", width="stretch", on_click=pick_sample, args=(p,))
        if not files:
            picker.info("No sample images in `demo_samples/` yet.")

    img = st.session_state.img
    if img is None:
        st.info("👆 Choose a photo above to see the model's answer.")
    else:
        probs, heat = det.predict(img)
        gp, g, gname, unsure = verdict(probs, threshold)
        icon, colour, meaning = GROUPS[gname]
        k = int(probs.argmax())
        p_dis = float(1 - probs[0])

        # monitoring log (once per distinct image)
        key = hashlib.md5(img.convert("RGB").resize((64, 64)).tobytes()).hexdigest()
        if do_log and st.session_state.logged != key:
            log_prediction(st.session_state.img_name, st.session_state.img_src, gname, float(gp[g]), CLASSES[k],
                           unsure)
            st.session_state.logged = key

        st.markdown("<div class='section-title'>Step 2 · The model's answer</div>", unsafe_allow_html=True)
        if unsure:
            title, sub, colour = f"❓ Not sure — maybe {gname.lower()}", \
                "The model is not confident about this photo. It may be an unusual angle or not show the hands.", \
                "#64748b"
        elif g == 0:
            title, sub = f"{icon} Safe driving", meaning
        else:
            title, sub = f"{icon} Distracted — {gname}", meaning
        chips = [f"Confidence {gp[g]:.0%}", f"Chance of any distraction {p_dis:.0%}",
                 f"Detail: {CLASSES[k]}"]
        truth = labels.get(st.session_state.img_name or "")
        if truth:
            tg = GROUP_NAMES[GROUP_OF[truth["label"]]]
            ok = tg == gname and not unsure
            chips.append(("✔ Correct" if ok else "✘ Wrong") + f" — true answer: {CLASSES[truth['label']]}")
        st.markdown(f"<div class='verdict' style='background:{colour}'><div class='t'>{title}</div>"
                    f"<div class='s'>{sub}</div>" + "".join(f"<span class='chip'>{c}</span>" for c in chips)
                    + "</div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns([1, 1, 0.9]) if show_cam else st.columns([1.3, 0.01, 1])
        with c1:
            st.image(img.convert("RGB"), width="stretch", caption="Your photo")
        if show_cam:
            with c2:
                st.image(overlay(img, heat), width="stretch",
                         caption="Heat-map: red = where the model looked")
        with c3:
            st.markdown("**How sure is it?**")
            st.markdown(bars_html(GROUP_NAMES, gp, [GROUPS[n][1] for n in GROUP_NAMES]), unsafe_allow_html=True)
            st.markdown("<span class='note'>The four groups add up to 100%. A good heat-map sits on the hands, "
                        "phone or face — not on the background.</span>", unsafe_allow_html=True)
        with st.expander("🔎 Detailed behaviour (all 10 classes)"):
            order = np.argsort(-probs)
            st.markdown(bars_html([CLASSES[i] for i in order], probs[order],
                                  [GROUPS[GROUP_NAMES[GROUP_OF[i]]][1] for i in order]), unsafe_allow_html=True)
            st.caption("Left/right hand and the subtle behaviours depend on the camera angle, so trust the "
                       "overall answer above more than this detail.")

# ============================================================================ 2. batch
with tabs[1]:
    st.markdown("<div class='section-title'>Analyse many photos at once</div>", unsafe_allow_html=True)
    st.caption("Upload several photos to get a table of answers you can download as CSV.")
    many = st.file_uploader("Photos", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True,
                            key="batch")
    if many:
        rows = []
        bar = st.progress(0.0, text="Analysing...")
        for i, f in enumerate(many):
            im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
            pr, _ = det.predict(im)
            gp, g, gname, unsure = verdict(pr, threshold)
            thumb = im.copy(); thumb.thumbnail((160, 160))
            buf = io.BytesIO(); thumb.save(buf, format="JPEG", quality=80)
            rows.append({"Photo": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
                         "File": f.name, "Answer": ("Not sure: " if unsure else "") + gname,
                         "Confidence": float(gp[g]), "Detail": CLASSES[int(pr.argmax())],
                         **{n: float(v) for n, v in zip(GROUP_NAMES, gp)}})
            bar.progress((i + 1) / len(many), text=f"Analysed {i + 1}/{len(many)}")
        bar.empty()
        res = pd.DataFrame(rows)
        m1, m2, m3 = st.columns(3)
        m1.metric("Photos", len(res))
        m2.metric("Flagged as distracted", int((~res.Answer.str.endswith("Safe driving")).sum()))
        m3.metric("Not sure", int(res.Answer.str.startswith("Not sure").sum()))
        st.dataframe(res, hide_index=True, row_height=70, column_config={
            "Photo": st.column_config.ImageColumn("Photo", width="small"),
            "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="percent"),
            **{n: st.column_config.NumberColumn(n, format="percent") for n in GROUP_NAMES}})
        st.download_button("⬇️ Download results (CSV)", res.drop(columns="Photo").to_csv(index=False),
                           "predictions.csv", "text/csv")

# ============================================================================ 3. performance
with tabs[2]:
    st.markdown("<div class='section-title'>How well does it work?</div>", unsafe_allow_html=True)
    st.caption("Every number is measured on photos of drivers the model never saw during training.")
    sf, as_ = final_metrics.get("sf", {}), final_metrics.get("as", {})
    k1, k2, k3, k4 = st.columns(4)
    if sf and as_:
        k1.metric("4-group accuracy · State Farm", f"{sf['group_acc']:.1%}")
        k2.metric("4-group accuracy · Ain Shams", f"{as_['group_acc']:.1%}")
        k3.metric("Safe-vs-distracted · Ain Shams", f"{as_['binary_acc']:.1%}",
                  help="Is the driver distracted at all? (yes/no)")
    if cv_summary:
        k4.metric("Cross-validation (10-class)",
                  f"{cv_summary['val_fine_acc']['mean']:.1%} ± {cv_summary['val_fine_acc']['std']:.1%}",
                  help="3-fold driver-wise cross-validation on the training drivers")
    elif sf:
        k4.metric("10-class accuracy · State Farm", f"{sf['fine_acc']:.1%}")

    with st.expander("📖 What do these metrics mean?"):
        st.markdown("""
* **4-group accuracy** – how often the headline answer (Safe / Texting / Phone call / Other) is right.
* **Safe-vs-distracted accuracy** – just the yes/no question "is the driver distracted?".
* **False-alarm rate** – share of *safe* drivers the model wrongly flags. Lower is better.
* **10-class accuracy / macro-F1** – the detailed behaviour (incl. left vs right hand). Macro-F1 averages the
  F1-score of every class, so rare classes count as much as common ones.
* **Confusion matrix** – rows = true answer, columns = model's answer; a perfect model has only the diagonal.""")

    metric_names = {"4 groups": "group_acc", "Safe vs distracted": "binary_acc",
                    "False alarms (lower = better)": "false_alarm", "10 detailed behaviours": "fine_acc"}
    pick = st.segmented_control("Metric", list(metric_names), default="4 groups")
    table = comparison_table(metric_names[pick or "4 groups"])
    if table.empty:
        st.info("Run the experiments (`run_multi_experiments.sh`) to fill this page.")
    else:
        import matplotlib
        cols = [c for c in ["State Farm", "Ain Shams", "Wild photos"] if c in table]
        cmap = matplotlib.colormaps["RdYlGn_r" if metric_names[pick or "4 groups"] == "false_alarm" else "RdYlGn"]
        num = table[cols].astype(float)
        shown = table.copy()
        shown[cols] = num.map(lambda v: "–" if pd.isna(v) else f"{v:.1%}")

        def colour(col):
            if col.name not in cols:
                return [""] * len(col)
            return ["" if pd.isna(v) else f"background-color: {matplotlib.colors.to_hex(cmap(v))}; color: black"
                    for v in num[col.name]]

        st.dataframe(shown.style.apply(colour), hide_index=True, width="stretch")
        st.markdown("""
* **Row 1 – the problem:** trained on one car and one camera, the old model fails on other cars.
* **Rows 2–4** add data and augmentation one step at a time; each change is measured on unseen data.
* **Wild photos** are the hardest test (other countries, buses, right-hand drive, frontal cameras). The final model
  trains on all of them, so its wild score cannot be measured (–).""")
        left, right = st.columns([1.2, 1])
        if (RESULTS / "comparison.png").is_file():
            left.image(str(RESULTS / "comparison.png"), width="stretch")
        with right:
            cm_tabs = st.tabs(["Groups · Ain Shams", "Groups · State Farm", "10 classes · Ain Shams",
                               "10 classes · State Farm"])
            for t, n in zip(cm_tabs, ["cm_groups_as", "cm_groups_sf", "cm_fine_as", "cm_fine_sf"]):
                p = RESULTS / "final" / f"{n}.png"
                if p.is_file():
                    t.image(str(p), width="stretch")

# ============================================================================ 4. data & training
with tabs[3]:
    st.markdown("<div class='section-title'>The data</div>", unsafe_allow_html=True)
    eda_stats = load_json(EDA / "eda_stats.json") or {}
    if eda_stats:
        sp = eda_stats["splits"]
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("State Farm", f"{sp['sf/train']['images'] + sp['sf/test']['images']:,} imgs",
                  f"{sp['sf/train']['subjects'] + sp['sf/test']['subjects']} drivers", delta_color="off")
        d2.metric("Ain Shams", f"{sp['as/train']['images'] + sp['as/test']['images']:,} imgs",
                  f"{sp['as/train']['subjects'] + sp['as/test']['subjects']} drivers", delta_color="off")
        d3.metric("Extra State Farm (coarse)", f"{sp['sfx/train']['images']:,} imgs", "train only", delta_color="off")
        d4.metric("Wild phone photos", f"{sp['wild/test']['images']:,} imgs", "8 capture groups", delta_color="off")
    figs = [
        ("05_samples_per_class.png", "One example per class. Same 10 behaviours, but State Farm is daylight in one "
                                     "car while Ain Shams uses several cars and an IR camera with a purple tint."),
        ("01_class_balance.png", "Classes are well balanced (max/min ≈ 1.3 for State Farm, 1.0 for Ain Shams), so "
                                 "plain accuracy is meaningful; the Multi-Class set only has 5 coarse labels."),
        ("02_images_per_driver.png", "Hundreds of near-identical video frames per driver → splits must be by "
                                     "driver, otherwise the test set leaks."),
        ("03_brightness_colour.png", "Ain Shams has a separate cluster of very dark night frames and stronger "
                                     "colour casts; wild photos are brighter → colour/brightness augmentation."),
        ("04_mean_image_per_dataset.png", "The average image shows each dataset's fixed camera position — the "
                                          "model can learn the car instead of the driver (domain bias)."),
        ("06_wild_samples.png", "Wild phone photos: different cars, buses and camera angles. Used to test "
                                "generalisation."),
    ]
    for name, text in figs:
        if (EDA / name).is_file():
            with st.container(border=True):
                st.markdown(f"**{text}**")
                st.image(str(EDA / name), width="stretch")

    st.markdown("<div class='section-title'>Hyper-parameter tuning</div>", unsafe_allow_html=True)
    tune = RESULTS / "tuning" / "summary.csv"
    if tune.is_file():
        t = pd.read_csv(tune)
        st.caption("Short runs scored on a validation split of held-out *training* drivers (the test drivers "
                   "are never used for tuning).")
        st.dataframe(t, hide_index=True, width="stretch", column_config={
            c: st.column_config.NumberColumn(c, format="percent") for c in
            ["Val 4-group acc", "Val 10-class acc"]})
        if (RESULTS / "tuning" / "curves.png").is_file():
            st.image(str(RESULTS / "tuning" / "curves.png"), width="stretch")
    else:
        st.info("Run `python tune_multi.py sweep` to fill this section.")

    st.markdown("<div class='section-title'>Cross-validation</div>", unsafe_allow_html=True)
    folds = RESULTS / "cv" / "folds.csv"
    if folds.is_file():
        f = pd.read_csv(folds)[["fold", "val_drivers", "val_images", "val_group_acc", "val_fine_acc",
                                "val_macro_f1"]]
        st.dataframe(f, hide_index=True, column_config={
            c: st.column_config.NumberColumn(c, format="percent") for c in ["val_group_acc", "val_fine_acc"]})
    else:
        st.info("Run `python tune_multi.py cv` to fill this section.")

    hist = RESULTS / "final" / "history.csv"
    if hist.is_file():
        st.markdown("<div class='section-title'>Training of the final model</div>", unsafe_allow_html=True)
        h = pd.read_csv(hist).set_index("epoch")
        a, b = st.columns(2)
        a.caption("Training loss per epoch"); a.line_chart(h[["train_loss"]])
        b.caption("Training accuracy per epoch"); b.line_chart(h[["train_acc_in_set"]])

# ============================================================================ 5. monitoring
with tabs[4]:
    st.markdown("<div class='section-title'>Monitoring the deployed model</div>", unsafe_allow_html=True)
    st.caption("Every analysed photo adds one line (answer + confidence, no image) to `logs/predictions.csv`. "
               "A rising share of 'not sure' answers or falling confidence is an early sign of data drift.")
    if LOG.is_file():
        lg = pd.read_csv(LOG, parse_dates=["time"])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predictions logged", len(lg))
        m2.metric("Average confidence", f"{lg.confidence.mean():.0%}")
        m3.metric("'Not sure' answers", f"{lg.uncertain.mean():.0%}")
        m4.metric("Flagged distracted", f"{(lg.group != 'Safe driving').mean():.0%}")
        a, b = st.columns(2)
        a.caption("Answers given"); a.bar_chart(lg.group.value_counts())
        b.caption("Confidence over time (rolling mean of 10)")
        b.line_chart(lg.set_index("time").confidence.rolling(10, min_periods=1).mean())
        if lg.uncertain.tail(20).mean() > 0.4:
            st.warning("More than 40% of the last 20 photos were 'not sure' — the incoming photos may differ from "
                       "the training data. Consider collecting and labelling them for re-training.")
        with st.expander("Raw log"):
            st.dataframe(lg.iloc[::-1], hide_index=True)
        if st.button("🗑️ Clear log"):
            LOG.unlink(); st.rerun()
    else:
        st.info("No predictions logged yet — analyse a photo first.")

# ============================================================================ 6. how it works
with tabs[5]:
    st.markdown("<div class='section-title'>Pipeline</div>", unsafe_allow_html=True)
    st.graphviz_chart("""
digraph {
  rankdir=LR; bgcolor="transparent";
  node [shape=box style="rounded,filled" fillcolor="#eff6ff" color="#2563eb" fontname="Helvetica" fontsize=11];
  edge [color="#64748b"];
  a [label="Photo of driver"]; b [label="Resize 224x224\\nnormalise"];
  c [label="EfficientNet-B0\\n(ImageNet pre-trained,\\nfine-tuned)"]; d [label="10 behaviour\\nprobabilities"];
  e [label="4 groups\\nSafe / Text / Call / Other", fillcolor="#dcfce7", color="#15803d"];
  f [label="Grad-CAM\\nheat-map", fillcolor="#fef3c7", color="#d97706"];
  a -> b -> c -> d -> e; c -> f;
}""")
    st.markdown("""
**Model.** A convolutional neural network (EfficientNet-B0, pre-trained on ImageNet and fine-tuned — *transfer
learning*) outputs probabilities for 10 behaviours. These are summed into 4 groups, which stay reliable across
camera angles; left/right-hand detail does not.

**Training data.** *State Farm Distracted Driver* (one car, daylight), *Ain Shams Novel Driver Distractions*
(5 cars, 70 drivers, IR/night) and the *Multi-Class Driver Behavior* dataset (coarse labels, used with a
**partial-label loss**: "texting" = texting with either hand).

**Augmentation.** Random crops, rotation, colour/brightness jitter, occasional grayscale, and **mirroring with a
left/right label swap** so right-hand-drive cars are covered.

**Heat-map.** Grad-CAM highlights the regions that pushed the network to its answer — a quick sanity check that
it looks at hands/phone/face rather than the background.

**Limits.** Works best when the driver's upper body and hands are visible from the side or front-side. Frontal
cameras, road-facing dash-cams, very dark or tiny images are harder. This is a research demo, not a safety system.""")
