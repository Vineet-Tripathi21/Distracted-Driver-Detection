"""Driver Distraction Detector - demo app.

    streamlit run demo_app.py

Model: EfficientNet-B0 trained on three datasets (State Farm + Ain Shams + Multi-Class phone photos).
Headline answer = 4 camera-robust groups; the 10 detailed behaviours are shown underneath.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps

from data import CLASSES, build_transforms
from model import load_checkpoint
from multi_data import GROUP_NAMES, group_probs
from summarize_multi import comparison_table

HERE = Path(__file__).parent
MODEL_CANDIDATES = [HERE / "results_multi/final/model.pt", HERE / "results_multi/mirror/model.pt",
                    HERE / "results_multi/multi/model.pt", HERE / "results/final/model.pt"]
SAMPLES = HERE / "demo_samples"
GROUP_STYLE = {  # icon, colour
    "Safe driving": ("✅", "#1b8a3a"),
    "Texting": ("📱", "#c62828"),
    "Phone call": ("📞", "#c62828"),
    "Other distraction": ("⚠️", "#d97706"),
}

st.set_page_config(page_title="Driver Distraction Detector", page_icon="🚗", layout="wide")
st.markdown("""
<style>
.verdict {border-radius: 14px; padding: 22px 26px; color: white; margin-bottom: 12px;}
.verdict h1 {color: white; margin: 0; font-size: 2.1rem;}
.verdict p {margin: 4px 0 0 0; font-size: 1.05rem; opacity: .95;}
.small-note {color: #6b7280; font-size: .85rem;}
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

    def predict(self, img: Image.Image, cam: bool = True):
        x = self.tf(img.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.set_grad_enabled(cam):
            logits = self.model(x)
            probs = torch.softmax(logits, 1)[0]
            heat = None
            if cam:
                k = int(probs.argmax())
                act = self._act
                grad = torch.autograd.grad(logits[0, k], act)[0]
                w = grad.mean(dim=(2, 3), keepdim=True)
                heat = torch.relu((w * act).sum(1))[0]
                heat = (heat / heat.max().clamp_min(1e-8)).detach().cpu().numpy()
        return probs.detach().cpu().numpy(), heat


@st.cache_resource(show_spinner="Loading model...")
def get_detector(path):
    return Detector(path)


def overlay(img, heat, alpha=0.45):
    import matplotlib.cm as cm
    img = img.convert("RGB")
    h = Image.fromarray(np.uint8(heat * 255)).resize(img.size, Image.BILINEAR)
    colored = Image.fromarray(np.uint8(cm.jet(np.asarray(h) / 255.0)[..., :3] * 255))
    return Image.blend(img, colored, alpha)


def model_path():
    for p in MODEL_CANDIDATES:
        if p.is_file():
            return p
    return None


# ----------------------------------------------------------------------------- UI
st.title("🚗 Driver Distraction Detector")
st.caption("Upload a photo of a driver taken from inside the car (or pick a sample). The model says whether "
           "the driver is distracted and shows **where it looked** to decide.")

mp = model_path()
if mp is None:
    st.error("No trained model found. Run `python train_multi.py` first.")
    st.stop()
det = get_detector(str(mp))
trained_multi = "results_multi" in str(mp)

tab_try, tab_results, tab_about = st.tabs(["🔍 Try it", "📊 How well does it work?", "ℹ️ About"])

with tab_try:
    if "img" not in st.session_state:
        st.session_state.img, st.session_state.img_name = None, None

    src_col, opt_col = st.columns([3, 1])
    with src_col:
        how = st.radio("Image source", ["Sample gallery", "Upload", "Webcam"], horizontal=True,
                       label_visibility="collapsed")
    with opt_col:
        show_cam = st.toggle("Show attention heat-map", value=True)

    if how == "Upload":
        f = st.file_uploader("Driver image", type=["jpg", "jpeg", "png", "webp"])
        if f:
            st.session_state.img, st.session_state.img_name = ImageOps.exif_transpose(Image.open(f)), f.name
    elif how == "Webcam":
        c = st.camera_input("Take a photo")
        if c:
            st.session_state.img, st.session_state.img_name = Image.open(c), "webcam"
    else:
        files = sorted(p for p in SAMPLES.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        if not files:
            st.info("No sample images in `demo_samples/` yet.")
        cols = st.columns(6)
        for i, p in enumerate(files):
            with cols[i % 6]:
                st.image(Image.open(p), width="stretch")
                if st.button("Analyse", key=f"s{i}", width="stretch"):
                    st.session_state.img, st.session_state.img_name = Image.open(p), p.name

    img = st.session_state.img
    if img is not None:
        st.divider()
        probs, heat = det.predict(img, cam=show_cam)
        gp = group_probs(probs)[0]
        g = int(gp.argmax()); gname = GROUP_NAMES[g]
        p_dis = float(1 - probs[0])
        icon, colour = GROUP_STYLE[gname]
        if gp[g] < 0.5:
            colour, title = "#6b7280", f"❓ Not sure - maybe {gname.lower()}"
        elif g == 0:
            title = f"{icon} Safe driving"
        else:
            title = f"{icon} DISTRACTED: {gname}"
        k = int(probs.argmax())

        left, right = st.columns([1.15, 1])
        with left:
            st.image(overlay(img, heat) if (show_cam and heat is not None) else img, width="stretch",
                     caption="Red = regions that drove the decision" if show_cam else st.session_state.img_name)
        with right:
            st.markdown(f"<div class='verdict' style='background:{colour}'><h1>{title}</h1>"
                        f"<p>Confidence {gp[g]:.0%} &nbsp;·&nbsp; Probability distracted {p_dis:.0%}</p></div>",
                        unsafe_allow_html=True)
            st.markdown("**Overall**")
            for name, v in zip(GROUP_NAMES, gp):
                st.progress(float(v), text=f"{name}: {v:.0%}")
            with st.expander("Detailed behaviour (10 classes)"):
                st.markdown(f"Most likely: **{CLASSES[k]}** ({probs[k]:.0%})")
                st.bar_chart(pd.DataFrame({"probability": probs}, index=CLASSES), horizontal=True, sort=False)
                st.markdown("<span class='small-note'>Left/right hand and subtle behaviours depend on the camera "
                            "angle, so trust the overall answer above more than this detail.</span>",
                            unsafe_allow_html=True)

with tab_results:
    root = HERE / "results_multi"
    metric_names = {"4 groups (Safe / Texting / Call / Other)": "group_acc",
                    "Safe vs distracted": "binary_acc",
                    "False alarms (safe drivers flagged)": "false_alarm",
                    "10 detailed behaviours": "fine_acc"}
    pick = st.radio("Metric", list(metric_names), horizontal=True)
    table = comparison_table(metric_names[pick])
    if table.empty:
        st.info("Run the experiments (`run_multi_experiments.sh`) to fill this page.")
    else:
        st.markdown("Every number is measured on images the model **never saw in training**: new drivers, and for "
                    "*Wild photos* whole capture sessions in other cars and buses.")
        cols = [c for c in ["State Farm", "Ain Shams", "Wild photos"] if c in table]
        import matplotlib
        cmap = matplotlib.colormaps["RdYlGn_r" if metric_names[pick] == "false_alarm" else "RdYlGn"]
        num = table[cols].astype(float)
        shown = table.copy()
        shown[cols] = num.map(lambda v: "-" if pd.isna(v) else f"{v:.1%}")

        def colour(col):
            if col.name not in cols:
                return [""] * len(col)
            return ["" if pd.isna(v) else f"background-color: {matplotlib.colors.to_hex(cmap(v))}; color: black"
                    for v in num[col.name]]

        st.dataframe(shown.style.apply(colour), hide_index=True)
        st.markdown("""
* **Row 1 - the domain gap.** Trained on one car and one camera, the old model fails on other cars: it calls almost
  every safe driver distracted.
* **Rows 2-4** add more data and augmentation one step at a time - each change is measured on unseen data.
* **Wild photos** are the hardest test: different countries, cars, buses, right-hand drive and frontal camera
  angles. The final demo model is trained on all sessions, so its wild score cannot be measured (shown as -).""")
        if (root / "comparison.png").is_file():
            st.image(str(root / "comparison.png"), width="stretch")
        d = root / "final"
        pics = [p for p in [d / "cm_groups_as.png", d / "cm_fine_sf.png"] if p.is_file()]
        if pics:
            st.markdown("**Confusion matrices - final demo model**")
            for c, p in zip(st.columns(len(pics)), pics):
                c.image(str(p), width="stretch")

with tab_about:
    st.markdown(f"""
**What it does.** A convolutional neural network (EfficientNet-B0, ImageNet-pretrained, fine-tuned) looks at one
photo of the driver and outputs probabilities for 10 behaviours, which are summed into 4 groups:
**Safe driving · Texting · Phone call · Other distraction** (radio, drinking, reaching behind, hair/makeup, talking
to a passenger).

**Trained on** {'multiple datasets' if trained_multi else 'State Farm only'}:
* *State Farm Distracted Driver* - 16 training drivers, one car, one camera (10 behaviours)
* *Ain Shams Novel Driver Distractions* - 56 training drivers, 5 cars, infrared/night images (same 10 behaviours)
* extra State Farm frames with coarse labels from the *Multi-Class Driver Behavior* dataset, used with a
  partial-label loss ("texting" = texting with either hand)

**Wild phone photos** (647, other cars, buses, frontal camera angles, right-hand drive) are used as an external
test: models are trained on one capture session and tested on the other. The final demo model uses all of them.

**Mirror augmentation.** Training images are randomly mirrored *and* the left/right-hand labels swapped, so the
model also learns right-hand-drive layouts.

**Heat-map.** Grad-CAM highlights the image regions that pushed the network towards its answer - useful for
checking it looks at hands/phone/face rather than the background.

**Limits.** Works best when the driver's upper body and hands are visible from the side/front.
Road-facing dashcams, very dark or tiny images are out of scope. Research demo - not a safety system.

<span class='small-note'>Model file: {mp}</span>""", unsafe_allow_html=True)
