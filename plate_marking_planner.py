# marking_planner.py
import io
import zipfile
import re

import streamlit as st
import pandas as pd
from rectpack import newPacker
import plotly.graph_objects as go
import ezdxf

st.set_page_config(layout="wide", page_title="BH Plate Marking Planner — DXF, ZIP, Interactive")

DENSITY_STEEL = 7850.0  # kg/m^3


# =========================
# Helpers & core utilities
# =========================
def _ensure_pos(*vals):
    try:
        return all(float(v) > 0 for v in vals)
    except Exception:
        return False


PROFILE_RE = re.compile(
    r"^\s*(?:BH|NPB|WPB)?\s*"
    r"(\d+(?:\.\d+)?)\s*[Xx]\s*"
    r"(\d+(?:\.\d+)?)\s*[Xx]\s*"
    r"(\d+(?:\.\d+)?)\s*[Xx]\s*"
    r"(\d+(?:\.\d+)?)\s*$"
)


def extract_dims(profile: str):
    """
    Parse PROFILE and return (web_thk, flange_thk, H, W) in mm.
    Keeps the ordering used earlier: wt, ft, H, W = c, d, a, b
    """
    s = str(profile).strip()
    m = PROFILE_RE.match(s)
    if not m:
        raise ValueError(f"Bad PROFILE format: {profile!r}. Expected like 'BH8x20x500x200'.")
    a, b, c, d = map(float, m.groups())
    wt, ft, H, W = c, d, a, b
    if not _ensure_pos(wt, ft, H, W):
        raise ValueError(f"Non-positive dimensions in PROFILE: {profile}")
    return wt, ft, H, W


def calc_row_weights(row, density=DENSITY_STEEL):
    wt, ft, H, W = extract_dims(row['PROFILE'])
    Lm = float(row['LENGTH(mm)']) / 1000.0
    qty = int(row['QTY.'])
    if qty < 0:
        raise ValueError("QTY. must be >= 0")
    web_h = max(0.0, H - 2.0 * ft)
    web_area_mm2 = (web_h * wt)
    flange_area_mm2 = (W * ft)
    web_weight = (web_area_mm2 / 1e6) * Lm * qty * density
    flg_weight = (flange_area_mm2 / 1e6) * Lm * qty * 2 * density
    return pd.Series({
        'web_thk': wt, 'flange_thk': ft, 'height': H, 'width': W,
        'web_height': web_h, 'web_weight': web_weight, 'flange_weight': flg_weight
    })


def build_cuts(df_calc):
    """
    Expand the input rows into individual parts (each rectangle gets a unique rid).
    Produces both web strips (width=web_thk) and flanges (qty*2).
    """
    cuts = []
    part_uid = 0
    for _, r in df_calc.iterrows():
        qty = int(r['QTY.']); L = float(r['LENGTH(mm)'])
        if qty <= 0 or L <= 0:
            continue

        # Webs
        if r['web_thk'] > 0 and r['web_height'] > 0:
            total_web_wt = float(r['web_weight'])
            wt_each = total_web_wt / qty if qty else 0.0
            for _ in range(qty):
                cuts.append({
                    'rid': part_uid, 'type': 'web',
                    'thickness': float(r['web_thk']),
                    'width': float(r['web_thk']),      # strip width
                    'length': float(L),                # along member length
                    'weight_each': wt_each
                }); part_uid += 1

        # Flanges (two per member)
        if r['flange_thk'] > 0 and r['width'] > 0:
            flg_qty = qty * 2
            total_flg_wt = float(r['flange_weight'])
            wt_each = total_flg_wt / flg_qty if flg_qty else 0.0
            for _ in range(flg_qty):
                cuts.append({
                    'rid': part_uid, 'type': 'flange',
                    'thickness': float(r['flange_thk']),
                    'width': float(r['width']),
                    'length': float(L),
                    'weight_each': wt_each
                }); part_uid += 1
    return cuts


def splice_segments(width_mm, length_mm, max_len_mm, overlap_mm):
    """
    Split a long piece along its length into segments.
    Overlap is distributed as +o/2 to both segments around each joint.
    """
    if length_mm <= max_len_mm:
        return [(width_mm, length_mm)]
    full = int(length_mm // max_len_mm)
    rem = length_mm - full * max_len_mm
    segs = [max_len_mm] * full + ([rem] if rem > 1e-6 else [])
    if overlap_mm > 0 and len(segs) > 1:
        add = overlap_mm / 2.0
        for j in range(len(segs) - 1):
            segs[j] += add
            segs[j + 1] += add
    return [(width_mm, Ls) for Ls in segs]


def filter_parts_by_thickness(cuts, thk):
    t = float(thk)
    return [p for p in cuts if abs(float(p['thickness']) - t) < 1e-6]


def pack_with_positions(parts, plate_w, plate_l, kerf=6, rotation=True,
                        allow_splice=True, max_plate_len=12000, splice_overlap=50):
    """
    Pack parts and return precise placements (with kerf-inflated layout sizes).
    Returns:
      placements: list of dicts
        {plate_no, x, y, w_true, h_true, w_infl, h_infl, rid, type, thickness}
      plate_count: int
    """
    if not parts:
        return [], 0

    # Build rects (with splicing) & a RID map
    rid_to_meta = {}
    rid_counter = 0
    for p in parts:
        w = float(p['width'])
        L = float(p['length'])
        t = float(p['thickness'])
        typ = p['type']
        segs = splice_segments(w, L, max_plate_len, splice_overlap) if allow_splice else [(w, L)]
        for (ws, Ls) in segs:
            rid_to_meta[rid_counter] = dict(
                type=typ, thickness=t, w_true=ws, h_true=Ls, rid_orig=p['rid']
            )
            rid_counter += 1

    if not rid_to_meta:
        return [], 0

    # Inflate by kerf for layout
    add = max(0.0, float(kerf))
    rects = []  # (W_infl, H_infl, rid)
    for rid, meta in rid_to_meta.items():
        w_true = meta['w_true']; h_true = meta['h_true']
        W = max(1, int(round(w_true + add)))
        H = max(1, int(round(h_true + add)))
        rects.append((W, H, rid))

    # Heuristic: add many bins once, pack once (fast)
    bin_guess = max(50, len(rects) // 2)
    packer = newPacker(rotation=rotation)
    for W, H, rid in rects:
        packer.add_rect(W, H, rid)
    for _ in range(bin_guess):
        packer.add_bin(int(plate_w), int(plate_l))
    packer.pack()

    packed = packer.rect_list()  # (bin_id, x, y, w, h, rid)
    if not packed:
        return [], 0

    # Map bin IDs to plate numbers (1..N)
    order = []
    for b, *_ in packed:
        if b not in order:
            order.append(b)
    bin_to_plate = {b: i + 1 for i, b in enumerate(order)}

    # Build placement dicts
    placements = []
    for (b, x, y, w_infl, h_infl, rid) in packed:
        meta = rid_to_meta[rid]
        placements.append({
            'plate_no': bin_to_plate[b],
            'x': int(x), 'y': int(y),
            'w_true': float(meta['w_true']), 'h_true': float(meta['h_true']),
            'w_infl': int(w_infl), 'h_infl': int(h_infl),
            'rid': int(rid),
            'type': meta['type'],
            'thickness': float(meta['thickness'])
        })

    plate_count = len(set(p['plate_no'] for p in placements))
    placements.sort(key=lambda d: (d['plate_no'], d['y'], d['x']))
    return placements, plate_count


# ===========================
# Viewers & export functions
# ===========================
def draw_plate_plotly(
    plate_no,
    placements,
    plate_w,
    plate_l,
    kerf,
    show_ids=True,
    show_kerf_envelope=True,
    min_label_area_mm2=60_000,   # rectangles smaller than this won't get a label
    label_font_size=9            # px
):
    import plotly.graph_objects as go

    fig = go.Figure()

    # Plate boundary
    fig.add_shape(type="rect", x0=0, y0=0, x1=plate_w, y1=plate_l, line=dict(width=2))

    for p in placements:
        if p['plate_no'] != plate_no:
            continue

        x, y = p['x'], p['y']
        W_env, H_env = p['w_infl'], p['h_infl']     # kerf envelope for layout
        W_true, H_true = p['w_true'], p['h_true']   # true size for hover

        # Choose which dims to draw: envelope (recommended) or true
        W_draw, H_draw = (W_env, H_env) if show_kerf_envelope else (W_true, H_true)

        # Rectangle outline
        fig.add_shape(type="rect", x0=x, y0=y, x1=x + W_draw, y1=y + H_draw, line=dict(width=1))

        # Hover polygon (keeps canvas clean)
        hover = (
            f"<b>{p['type']}</b> | {p['thickness']} mm<br>"
            f"True size: {int(W_true)} × {int(H_true)} mm<br>"
            f"Start (X,Y): {int(x)}, {int(y)} mm<br>"
            f"Kerf: {kerf} mm<br>"
            f"RID: {p['rid']}"
        )
        fig.add_trace(go.Scatter(
            x=[x, x + W_draw, x + W_draw, x, x],
            y=[y, y, y + H_draw, y + H_draw, y],
            mode="lines",
            line=dict(width=0),
            fill="toself",
            fillcolor="rgba(0,0,0,0)",
            hovertemplate=hover,
            showlegend=False
        ))

        # RID label as an annotation (supports rotation)
        if show_ids:
            area = W_draw * H_draw
            if area >= min_label_area_mm2:
                angle = 0 if W_draw >= H_draw else 90
                cx = x + W_draw / 2
                cy = y + H_draw / 2
                fig.add_annotation(
                    x=cx, y=cy,
                    text=str(p['rid']),
                    showarrow=False,
                    font=dict(size=label_font_size),
                    textangle=angle,
                    xanchor="center", yanchor="middle",
                    align="center"
                )

    # Clamp axes to plate with small padding; nice ticks; true 1:1 scale
    PAD = max(int(kerf), 20)

    # pick spacing so we get ~6–10 ticks across the axis
    x_dtick = max(1000, round(plate_w / 8, -2))   # e.g. 2000 → 250, 2500 → 300
    y_dtick = max(500, round(plate_l / 10, -2))  # e.g. 12000 → 1200

    fig.update_xaxes(
        range=[0, plate_w + PAD], showgrid=True, gridwidth=1,
        dtick=x_dtick, tick0=0, ticks="outside",
        mirror=True, zeroline=False
    )   
    fig.update_yaxes(
        range=[0, plate_l + PAD], showgrid=True, gridwidth=1,
        dtick=y_dtick, tick0=0, ticks="outside",
        mirror=True, zeroline=False,
        scaleanchor="x", scaleratio=1
    )
    fig.update_shapes(xref="x", yref="y")
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        title=f"Plate {plate_no} — {plate_w}×{plate_l} mm (kerf {kerf} mm)",
        dragmode="pan",
        uirevision="plate"
    )
    return fig



def build_cutlist_dataframe_all(placements, kerf):
    rows = []
    for p in placements:
        rows.append({
            'Plate No': p['plate_no'],
            'RID': p['rid'],
            'Type': p['type'],
            'Thickness (mm)': p['thickness'],
            'Cut Width (mm)': round(p['w_true'], 1),
            'Cut Length (mm)': round(p['h_true'], 1),
            'X (mm)': int(p['x']),
            'Y (mm)': int(p['y']),
            'Kerf (mm)': kerf
        })
    df = pd.DataFrame(rows)
    return df.sort_values(['Plate No', 'Y (mm)', 'X (mm)'])


def build_cutlist_dataframe_for_plate(placements, plate_no, kerf):
    rows = []
    for p in placements:
        if p['plate_no'] != plate_no:
            continue
        rows.append({
            'Plate No': plate_no,
            'RID': p['rid'],
            'Type': p['type'],
            'Thickness (mm)': p['thickness'],
            'Cut Width (mm)': round(p['w_true'], 1),
            'Cut Length (mm)': round(p['h_true'], 1),
            'X (mm)': int(p['x']),
            'Y (mm)': int(p['y']),
            'Kerf (mm)': kerf
        })
    return pd.DataFrame(rows).sort_values(['Y (mm)', 'X (mm)'])


def export_plate_dxf(plate_no, placements, plate_w, plate_l) -> bytes:
    """
    Returns DXF bytes for a single plate.
    Uses StringIO (text) then encodes to bytes (fix for ezdxf write()).
    """
    text_buf = io.StringIO()

    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    if "PLATE" not in doc.layers:
        doc.layers.add("PLATE", color=1)
    if "PARTS" not in doc.layers:
        doc.layers.add("PARTS", color=3)

    # Plate boundary
    msp.add_lwpolyline(
        [(0, 0), (plate_w, 0), (plate_w, plate_l), (0, plate_l), (0, 0)],
        dxfattribs={"layer": "PLATE", "closed": True}
    )

    # Parts (true sizes)
    for p in placements:
        if p['plate_no'] != plate_no:
            continue
        x, y = p['x'], p['y']
        W, H = p['w_true'], p['h_true']
        msp.add_lwpolyline(
            [(x, y), (x + W, y), (x + W, y + H), (x, y + H), (x, y)],
            dxfattribs={"layer": "PARTS", "closed": True}
        )

    doc.write(text_buf)  # writes text
    return text_buf.getvalue().encode("utf-8")


def export_all_plates_zip(placements, plate_w, plate_l, thk, kerf) -> bytes:
    """
    Build a ZIP containing:
      - plate_<n>_<WxL>_thk<thk>.dxf
      - plate_<n>_<WxL>_thk<thk>.csv
      - index.csv summarizing all plates
      - all_plates.csv (combined)
    """
    plate_nos = sorted({p['plate_no'] for p in placements})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        index_rows = []
        all_rows = []

        for n in plate_nos:
            # DXF
            dxf_bytes = export_plate_dxf(n, placements, plate_w, plate_l)
            dxf_name = f"plate_{n}_{plate_w}x{plate_l}_thk{int(thk)}.dxf"
            z.writestr(dxf_name, dxf_bytes)

            # Per-plate CSV
            df_plate = build_cutlist_dataframe_for_plate(placements, n, kerf)
            csv_name = f"plate_{n}_{plate_w}x{plate_l}_thk{int(thk)}.csv"
            z.writestr(csv_name, df_plate.to_csv(index=False))

            # For index + all_plates
            index_rows.append({
                "Plate No": n,
                "Thickness (mm)": thk,
                "Plate Size (mm)": f"{plate_w}x{plate_l}",
                "Items": len(df_plate),
            })
            all_rows.append(df_plate)

        index_df = pd.DataFrame(index_rows).sort_values("Plate No")
        z.writestr("index.csv", index_df.to_csv(index=False))

        if all_rows:
            all_df = pd.concat(all_rows, ignore_index=True)
            z.writestr("all_plates.csv", all_df.to_csv(index=False))

    buf.seek(0)
    return buf.read()


# ===================
#         UI
# ===================
st.title("BH Plate Marking Planner — Plate-wise Cut Lists, Interactive Diagram & DXF/ZIP")

@st.cache_data(show_spinner=False)
def read_excel(file):
    return pd.read_excel(file)

uploaded = st.file_uploader("Upload Excel (columns: PROFILE, LENGTH(mm), QTY.)", type=["xlsx", "xls"])
if not uploaded:
    st.info("Upload Excel with PROFILE, LENGTH(mm), QTY.")
    st.stop()

try:
    df = read_excel(uploaded)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

need = {'PROFILE', 'LENGTH(mm)', 'QTY.'}
if not need.issubset(df.columns):
    st.error(f"Missing columns; need {need}. Found: {list(df.columns)}")
    st.stop()

with st.expander("Preview input", expanded=False):
    st.dataframe(df.head(50), use_container_width=True)

@st.cache_data(show_spinner=False)
def compute_augmented(_df):
    df_calc = pd.concat([_df, _df.apply(calc_row_weights, axis=1)], axis=1)
    cuts = build_cuts(df_calc)
    return df_calc, cuts

df_calc, cuts = compute_augmented(df)

# Gather thickness options present in cuts
thicknesses = sorted({round(float(p['thickness']), 3) for p in cuts})
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    thk = st.selectbox("Select thickness (mm)", thicknesses, index=0)
with c2:
    plate_w = st.number_input("Plate WIDTH (mm)", min_value=500, value=2000, step=50)
with c3:
    plate_l = st.number_input("Plate LENGTH (mm)", min_value=2000, value=12000, step=100)
with c4:
    kerf = st.number_input("Kerf between parts (mm)", min_value=0, value=6, step=1)
with c5:
    rotation = st.checkbox("Allow rotation during packing", value=True,
                           help="Uncheck if rolling/grain direction must be preserved.")

with st.expander("Splice settings", expanded=True):
    allow_splice = st.checkbox("Allow splicing for > max plate length", value=True)
    max_plate_len = st.number_input("Max plate length available (mm)", min_value=1000, value=int(plate_l), step=100)
    splice_overlap = st.number_input("Overlap allowance per splice joint (mm)", min_value=0, value=50, step=5,
                                     help="Use 0–25 for butt bevel/trim; larger for lap joints.")

# Filter parts for this thickness
parts_this_thk = filter_parts_by_thickness(cuts, thk)
if not parts_this_thk:
    st.warning("No parts found for selected thickness.")
    st.stop()

# Pack & place
with st.spinner("Packing parts into plates and generating placements..."):
    placements, plate_count = pack_with_positions(
        parts_this_thk, plate_w, plate_l, kerf=kerf, rotation=rotation,
        allow_splice=allow_splice, max_plate_len=max_plate_len, splice_overlap=splice_overlap
    )

if plate_count == 0:
    st.error("Packing failed (no placements). Try allowing rotation, reducing kerf, or increasing plate size.")
    st.stop()

# Global cut list (all plates)
st.subheader(f"Cut List (ALL Plates) — Thickness {thk} mm | Plate size {plate_w}×{plate_l} mm | Plates: {plate_count}")
cutlist_all_df = build_cutlist_dataframe_all(placements, kerf)
st.dataframe(cutlist_all_df, use_container_width=True, height=420)
st.download_button(
    "⬇️ Download ALL Plates Cut List (CSV)",
    cutlist_all_df.to_csv(index=False),
    file_name=f"cutlist_all_thk{int(thk)}_{plate_w}x{plate_l}.csv"
)

# Batch ZIP of DXF + per-plate CSV + index
zip_bytes = export_all_plates_zip(placements, plate_w, plate_l, thk, kerf)
st.download_button(
    "📦 Download ALL plates (DXF + per-plate CSV + index.csv) as ZIP",
    data=zip_bytes,
    file_name=f"plates_{plate_w}x{plate_l}_thk{int(thk)}.zip",
    mime="application/zip"
)

# Interactive viewer (one plate at a time) + side legend table (no in-figure labels)
st.subheader("Interactive Marking / Nesting Diagram (with side legend)")
plate_to_view = st.number_input("View Plate No.", min_value=1, max_value=plate_count, value=1, step=1)
show_kerf_env = st.checkbox("Show kerf envelope (recommended)", value=True)

left, right = st.columns([2.2, 1.0])
with left:
    fig = draw_plate_plotly(plate_to_view, placements, plate_w, plate_l, kerf, show_kerf_envelope=show_kerf_env)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

with right:
    legend_df = build_cutlist_dataframe_for_plate(placements, plate_to_view, kerf)[
        ['RID', 'Type', 'Thickness (mm)', 'Cut Width (mm)', 'Cut Length (mm)', 'X (mm)', 'Y (mm)']
    ]
    st.caption("Plate legend (IDs & sizes)")
    st.dataframe(legend_df, use_container_width=True, height=500)

# Per-plate CSV & DXF (quick)
plate_df = build_cutlist_dataframe_for_plate(placements, plate_to_view, kerf)
cA, cB = st.columns(2)
with cA:
    st.download_button(
        f"⬇️ Download Plate {plate_to_view} Cut List (CSV)",
        plate_df.to_csv(index=False),
        file_name=f"plate_{plate_to_view}_{plate_w}x{plate_l}_thk{int(thk)}.csv"
    )
with cB:
    dxf_one = export_plate_dxf(plate_to_view, placements, plate_w, plate_l)
    st.download_button(
        f"⬇️ Download Plate {plate_to_view} DXF",
        dxf_one,
        file_name=f"plate_{plate_to_view}_{plate_w}x{plate_l}_thk{int(thk)}.dxf",
        mime="application/dxf"
    )

