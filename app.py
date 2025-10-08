# app.py
import streamlit as st
import pandas as pd
import re
from math import ceil
from rectpack import newPacker

st.set_page_config(layout="wide", page_title="BH Profile Plate Optimizer — No/With Splice")

DENSITY_STEEL = 7850.0  # kg/m^3

# -----------------------------
# Formatting / small utilities
# -----------------------------
def _fmt_kg(x):
    try:
        x = float(x)
    except Exception:
        return x
    if x >= 1000:
        return f"{x/1000:.2f} t"
    return f"{x:.0f} kg"

def _ensure_pos(*vals):
    try:
        return all(float(v) > 0 for v in vals)
    except Exception:
        return False

# -----------------------------------
# PROFILE parsing (robust with regex)
# Returns (web_thk, flange_thk, H, W)
# -----------------------------------
PROFILE_RE = re.compile(
    r"^\s*(?:BH|NPB|WPB)?\s*"
    r"(\d+(?:\.\d+)?)\s*[Xx]\s*"
    r"(\d+(?:\.\d+)?)\s*[Xx]\s*"
    r"(\d+(?:\.\d+)?)\s*[Xx]\s*"
    r"(\d+(?:\.\d+)?)\s*$"
)

def extract_dims(profile: str):
    """
    Parse PROFILE string and return (wt, ft, H, W) in mm.
    Keeps the same downstream ordering used by the original app:
    wt, ft, H, W = parts[2], parts[3], parts[0], parts[1]
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

# -----------------------------------
# Row → web/flange weights & geometry
# -----------------------------------
def calc_row_weights(row, density=DENSITY_STEEL):
    wt, ft, H, W = extract_dims(row['PROFILE'])
    try:
        Lm = float(row['LENGTH(mm)']) / 1000.0  # m
    except Exception:
        raise ValueError(f"Bad LENGTH(mm) value: {row['LENGTH(mm)']}")
    qty = int(row['QTY.'])
    if qty < 0:
        raise ValueError("QTY. must be >= 0")

    web_h = max(0.0, H - 2.0 * ft)  # clear web height

    web_area_mm2    = (web_h * wt)
    flange_area_mm2 = (W * ft)  # one flange
    web_weight  = (web_area_mm2 / 1e6) * Lm * qty * density
    flg_weight  = (flange_area_mm2 / 1e6) * Lm * qty * 2 * density  # two flanges

    return pd.Series({
        'web_thk': wt,
        'flange_thk': ft,
        'height': H,
        'width': W,
        'web_height': web_h,
        'web_weight': web_weight,
        'flange_weight': flg_weight
    })

# -----------------------------------
# Build part rectangles from rows
# -----------------------------------
def build_cuts(df_calc):
    cuts = []
    for _, r in df_calc.iterrows():
        qty = int(r['QTY.'])
        L = float(r['LENGTH(mm)'])
        if qty <= 0 or L <= 0:
            continue

        # Web strip: width MUST be web_height, not web_thk
        if r['web_thk'] > 0 and r['web_height'] > 0:
            total_web_wt = float(r['web_weight'])
            web_each = total_web_wt / qty if qty else 0.0
            cuts.append({
                'thickness': float(r['web_thk']),
                'type': 'web',
                'width': float(r['web_height']),  # ✅ fixed
                'length': float(L),
                'qty': qty,
                'weight_each': web_each
            })

        # Flanges: two per member
        if r['flange_thk'] > 0 and r['width'] > 0:
            total_flg_wt = float(r['flange_weight'])
            flg_qty = qty * 2
            flg_each = total_flg_wt / flg_qty if flg_qty else 0.0
            cuts.append({
                'thickness': float(r['flange_thk']),
                'type': 'flange',
                'width': float(r['width']),
                'length': float(L),
                'qty': flg_qty,
                'weight_each': flg_each
            })
    return cuts

def make_rects(parts):
    """
    Builds (w, l, rid) rects.
    Compatible with:
      - aggregated parts having 'qty'
      - already-expanded parts (no 'qty' -> assume 1)
    """
    rects = []
    id_to_weight = {}
    rid = 0
    for p in parts:
        w = float(p['width'])
        l = float(p['length'])
        q = int(p.get('qty', 1))  # robust: default 1 if missing
        we = float(p.get('weight_each', 0.0))
        for _ in range(q):
            rects.append((w, l, rid))
            id_to_weight[rid] = we
            rid += 1
    return rects, id_to_weight

# -----------------------------------
# Splicing logic: split long pieces
# -----------------------------------
def split_long_rects(rects, max_len_mm, overlap_mm):
    """
    For each (w, L, rid), if L > max_len_mm, split into multiple segments.
    Overlap is applied per joint; distributed equally to adjacent segments (+overlap/2 to both).
    Returns: (new_rects, splice_counts_by_id)
    """
    if max_len_mm <= 0:
        return rects, {}

    new_rects = []
    splice_counts = {}

    for (w, L, rid) in rects:
        if L <= max_len_mm:
            new_rects.append((w, L, rid))
            continue

        full_segs = int(L // max_len_mm)
        rem = L - full_segs * max_len_mm
        seg_count = full_segs + (1 if rem > 1e-6 else 0)
        splice_counts[rid] = max(0, seg_count - 1)

        base_lengths = [max_len_mm] * full_segs
        if rem > 1e-6:
            base_lengths.append(rem)

        if overlap_mm > 0 and seg_count > 1:
            add = overlap_mm / 2.0
            for j in range(seg_count - 1):
                base_lengths[j]     += add
                base_lengths[j + 1] += add

        for seg_len in base_lengths:
            new_rects.append((w, seg_len, rid))

    return new_rects, splice_counts

# -----------------------------------
# Fast packing: single pass, generous bins
# -----------------------------------
def pack_thickness_rects(rects, plate_w, plate_l, kerf=6, rotation=True):
    """
    Single-shot packing (fast). Inflate each rect by 'kerf' in W & H.
    Returns: (plates_used, true_area_without_kerf)
    """
    if not rects:
        return 0, 0.0

    add = max(0.0, float(kerf))
    inflated = [(max(1, int(round(w + add))), max(1, int(round(h + add))), rid, w, h)
                for (w, h, rid) in rects]
    true_area = sum(w * h for (w, h, _) in rects)

    n = len(inflated)
    bin_guess = max(50, n // 2)

    packer = newPacker(rotation=rotation)
    for W, H, ri, *_ in inflated:
        packer.add_rect(W, H, ri)
    for _ in range(bin_guess):
        packer.add_bin(int(plate_w), int(plate_l))
    packer.pack()

    packed = packer.rect_list()
    used_bins = {b for (b, *_rest) in packed}
    return len(used_bins), true_area

# -----------------------------------
# Evaluate a single (width, length) option
# -----------------------------------
def eval_for_size(parts, thk, plate_w, plate_l, kerf,
                  allow_splice, max_plate_len, splice_overlap,
                  include_kerf_in_area=False, rotation=True):
    rects, id_weights = make_rects(parts)
    total_weight = round(sum(id_weights.values()), 3)  # kg
    total_qty = len(rects)

    splice_counts = {}
    if allow_splice:
        rects, splice_counts = split_long_rects(rects, max_plate_len, splice_overlap)

    plate_wt = (plate_w/1000.0) * (plate_l/1000.0) * (thk/1000.0) * DENSITY_STEEL

    min_by_weight = ceil(total_weight / max(plate_wt, 1e-9)) if total_weight > 0 else 0

    plates_used, true_area = pack_thickness_rects(rects, plate_w, plate_l, kerf=kerf, rotation=rotation)

    if plates_used < min_by_weight:
        plates_used = min_by_weight

    total_plate_area = plates_used * (plate_w * plate_l) if plates_used else 0.0
    area_used_for_wastage = true_area
    wastage_area = 100.0 * (1.0 - area_used_for_wastage / total_plate_area) if total_plate_area else 0.0
    wastage_area = round(max(0.0, min(100.0, wastage_area)), 2)

    order_weight = plates_used * plate_wt
    wastage_mass = 100.0 * (1.0 - total_weight / max(order_weight, 1e-9)) if order_weight else 0.0
    wastage_mass = round(max(0.0, min(100.0, wastage_mass)), 2)

    return {
        'plate_w': plate_w,
        'plate_l': plate_l,
        'plates': int(plates_used),
        'plate_weight': plate_wt,
        'order_weight': order_weight,
        'total_qty': total_qty,
        'total_weight': total_weight,
        'wastage_mass': wastage_mass,
        'wastage_area': wastage_area,
        'splices_created': sum(splice_counts.values())
    }

# -----------------------------------
# Sweep widths × lengths and pick best per thickness
# -----------------------------------
def sweep_and_pick(cuts, widths, lengths, kerf, goal,
                   allow_splice, max_plate_len, splice_overlap,
                   include_kerf_in_area, rotation):
    by_thk = {}
    for c in cuts:
        by_thk.setdefault(c['thickness'], []).append(c)

    compare_rows = []
    recommended = []

    for thk, parts in sorted(by_thk.items()):
        options = []
        for W in widths:
            for L in lengths:
                if not _ensure_pos(W, L):
                    continue

                if not allow_splice and any(p['length'] > max_plate_len for p in parts):
                    st.warning(f"[{thk} mm] Some parts exceed max plate length {max_plate_len} mm and splicing is OFF: "
                               f"results may be infeasible.", icon="⚠️")

                opt = eval_for_size(
                    parts, thk, W, L, kerf,
                    allow_splice, max_plate_len, splice_overlap,
                    include_kerf_in_area=include_kerf_in_area, rotation=rotation
                )
                options.append(opt)
                compare_rows.append({
                    'Plate Thickness (mm)': thk,
                    'Plate Width (mm)': W,
                    'Plate Length (mm)': L,
                    'Plate weight (kg)': round(opt['plate_weight'], 1),
                    'Plates Needed': opt['plates'],
                    'Order weight (kg)': round(opt['order_weight'], 1),
                    'Total Qty': opt['total_qty'],
                    'Total Weight (kg)': round(opt['total_weight'], 3),
                    'Wastage (mass %)': opt['wastage_mass'],
                    'Wastage (area %)': opt['wastage_area'],
                    '# Splices': opt['splices_created']
                })

        if not options:
            continue

        if goal == 'Min wastage (mass)':
            pick = min(options, key=lambda x: (x['wastage_mass'], x['plates'], x['order_weight']))
        elif goal == 'Min order weight':
            pick = min(options, key=lambda x: (x['order_weight'], x['plates'], x['wastage_mass']))
        else:  # Min plates
            pick = min(options, key=lambda x: (x['plates'], x['wastage_mass'], x['order_weight']))

        recommended.append({
            'Plate Thickness (mm)': thk,
            'Plate Width (mm)': pick['plate_w'],
            'Recommended Plate Length (mm)': pick['plate_l'],
            'Plate weight (kg)': round(pick['plate_weight'], 1),
            'Plates Needed': pick['plates'],
            'Order weight (kg)': round(pick['order_weight'], 1),
            'Total Qty': pick['total_qty'],
            'Total Weight (kg)': round(pick['total_weight'], 3),
            'Wastage (mass %)': pick['wastage_mass'],
            'Wastage (area %)': pick['wastage_area'],
            '# Splices': pick['splices_created']
        })

    return pd.DataFrame(compare_rows), pd.DataFrame(recommended)

# -----------------------------
# Sanity checks for input webs
# -----------------------------
def sanity_check_webs(df_calc):
    problems = []
    for i, r in df_calc.iterrows():
        wt = float(r['web_thk'])
        wh = float(r['web_height'])
        if wh <= 0:
            problems.append(f"Row {i+1}: web_height <= 0 (PROFILE={r['PROFILE']})")
        if abs(wh - wt) < 1e-6:
            problems.append(f"Row {i+1}: web_height equals web_thk (H - 2*ft == wt) (PROFILE={r['PROFILE']})")
        if wh < 5 * wt:
            problems.append(f"Row {i+1}: web_height unusually small vs thickness "
                            f"(web_height={wh}, web_thk={wt}) (PROFILE={r['PROFILE']})")
    return problems

# ===================
#         UI
# ===================
st.title("BH Profile Plate Optimizer")

@st.cache_data(show_spinner=False)
def read_excel(file):
    return pd.read_excel(file)

uploaded = st.file_uploader("Upload Excel (columns: PROFILE, LENGTH(mm), QTY.)", type=["xlsx", "xls"])

if uploaded:
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

    try:
        df_calc, cuts = compute_augmented(df)
    except Exception as e:
        st.error(f"Problem while parsing/deriving beam parts: {e}")
        st.stop()

    # Sanity guard
    issues = sanity_check_webs(df_calc)
    if issues:
        st.warning("Sanity checks flagged potential issues:\n- " + "\n- ".join(issues))

    # KPI quick glance
    total_parts = sum(int(p.get('qty', 0)) for p in cuts)
    total_cut_weight = sum(float(p.get('weight_each', 0.0)) * int(p.get('qty', 0)) for p in cuts)
    k1, k2 = st.columns(2)
    k1.metric("Total rectangles (parts)", f"{total_parts}")
    k2.metric("Total part weight", _fmt_kg(total_cut_weight))

    # ---------------- Inputs ----------------
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        widths_input = st.text_input("Plate widths to test (mm, comma-separated)", "2000,2500")
        widths = []
        for token in widths_input.split(","):
            token = token.strip()
            if token:
                try:
                    widths.append(int(float(token)))
                except:
                    st.warning(f"Ignoring bad width: {token}")
        widths = sorted(set([w for w in widths if w > 0]))
    with c2:
        lengths_input = st.text_input("Plate lengths to test (mm, comma-separated)",
                                      "10000,10500,11000,11500,12000")
        lengths = []
        for token in lengths_input.split(","):
            token = token.strip()
            if token:
                try:
                    lengths.append(int(float(token)))
                except:
                    st.warning(f"Ignoring bad length: {token}")
        lengths = sorted(set([L for L in lengths if L > 0]))
    with c3:
        kerf = st.number_input("Kerf between parts (mm)", min_value=0, value=6)
    with c4:
        rotation = st.checkbox("Allow rotation during packing", value=True,
                               help="Uncheck if rolling/grain direction must be preserved")

    with st.expander("Splice settings", expanded=False):
        allow_splice = st.checkbox("Allow splicing for lengths over max plate length", value=True)
        max_plate_len = st.number_input("Max plate length available (mm)", min_value=1000, value=12000, step=100)
        splice_overlap = st.number_input("Overlap allowance per joint (mm)", min_value=0, value=50, step=5,
                                         help="Extra length per joint (lap/prep). Set 0 for butt joints.")
        include_kerf_in_area = st.checkbox("(Advanced) Include kerf in area wastage (slower)", value=False,
                                           help="Area wastage computed using inflated dims. Off = faster.")

    goal = st.selectbox("Optimization goal", ["Min wastage (mass)", "Min order weight", "Min plates"], index=0)

    with st.expander("Cuts generated", expanded=False):
        st.dataframe(pd.DataFrame(cuts), use_container_width=True)
    
        

    # --------------- Run sweep ---------------
    with st.spinner("Running sweep and selecting best per thickness..."):
        cmp_df, rec_df = sweep_and_pick(
            cuts, widths, lengths, kerf, goal,
            allow_splice, max_plate_len, splice_overlap,
            include_kerf_in_area, rotation
        )

    with st.expander("Comparison table (all width × length)", expanded=False):
        cmp_sorted = cmp_df.sort_values(
            ['Plate Thickness (mm)', 'Plate Width (mm)', 'Plate Length (mm)']
        )
        st.dataframe(cmp_sorted, use_container_width=True, height=420)
        st.download_button(
           "⬇️ Download Comparison CSV",
            cmp_sorted.to_csv(index=False),
            "width_length_comparison.csv"
        )

    st.subheader("Recommended mixed order (best size per thickness)")
    if not rec_df.empty:
        show_cols = [
            'Plate Thickness (mm)', 'Plate Width (mm)', 'Recommended Plate Length (mm)',
            'Plate weight (kg)', 'Plates Needed', 'Order weight (kg)',
            'Total Qty', 'Total Weight (kg)',
            'Wastage (mass %)', 'Wastage (area %)', '# Splices'
        ]
        st.dataframe(rec_df[show_cols].sort_values('Plate Thickness (mm)'), use_container_width=True)
        st.download_button("⬇️ Download Recommended Mix CSV",
                           rec_df[show_cols].to_csv(index=False), "recommended_mix.csv")

        # KPIs from recommended mix
        total_order_wt = rec_df['Order weight (kg)'].sum()
        total_plates   = rec_df['Plates Needed'].sum()
        s1, s2 = st.columns(2)
        s1.metric("Total plates (all thicknesses)", f"{int(total_plates)}")
        s2.metric("Total order weight", _fmt_kg(total_order_wt))
    else:
        st.info("No recommendations could be compiled (check inputs).")

else:
    st.info("Upload Excel with columns: PROFILE, LENGTH(mm), QTY.")



# ============================
# DROP-IN: Marking Planner (viewer for Recommended Mix)
# ============================
import plotly.graph_objects as go
from rectpack import newPacker

def _pack_with_positions(parts, plate_w, plate_l, kerf=6, rotation=True,
                         allow_splice=True, max_plate_len=12000, splice_overlap=50):
    """
    Packs and returns placements with exact X,Y. Uses your existing splicing logic.
    placements: [ {plate_no,x,y,w_true,h_true,w_infl,h_infl,rid,type,thickness}, ... ]
    """
    # Reuse your make_rects + split_long_rects to build rect list
    rects, _ = make_rects(parts)

    if allow_splice:
        rects, _splice_counts = split_long_rects(rects, max_plate_len, splice_overlap)

    # Inflate for layout
    add = max(0.0, float(kerf))
    inflated = []
    rid_map = {}
    rid_counter = 0
    for (w_true, h_true, _oldrid) in rects:
        W = max(1, int(round(w_true + add)))
        H = max(1, int(round(h_true + add)))
        inflated.append((W, H, rid_counter))
        rid_map[rid_counter] = dict(w_true=w_true, h_true=h_true)  # keep true sizes
        rid_counter += 1

    if not inflated:
        return [], 0

    # Pack into many bins once (fast heuristic)
    bin_guess = max(50, len(inflated)//2)
    packer = newPacker(rotation=rotation)
    for W, H, rid in inflated:
        packer.add_rect(W, H, rid)
    for _ in range(bin_guess):
        packer.add_bin(int(plate_w), int(plate_l))
    packer.pack()

    packed = packer.rect_list()  # (bin_id, x, y, w, h, rid)
    if not packed:
        return [], 0

    # Assign human plate numbers
    order = []
    for b, *_ in packed:
        if b not in order:
            order.append(b)
    bin_to_plate = {b: i+1 for i, b in enumerate(order)}

    placements = []
    for (b, x, y, w_infl, h_infl, rid) in packed:
        meta = rid_map[rid]
        placements.append({
            'plate_no': bin_to_plate[b],
            'x': int(x), 'y': int(y),
            'w_true': float(meta['w_true']), 'h_true': float(meta['h_true']),
            'w_infl': int(w_infl), 'h_infl': int(h_infl),
            'rid': int(rid),
            'type': 'part',            # generic; we don’t track web/flange at this stage
            'thickness': None,         # optional
        })
    placements.sort(key=lambda d: (d['plate_no'], d['y'], d['x']))
    plate_count = len(set(p['plate_no'] for p in placements))
    return placements, plate_count

def _build_cutlist_dataframe_for_plate(placements, plate_no, kerf):
    rows = []
    for p in placements:
        if p['plate_no'] != plate_no:
            continue
        rows.append({
            'Plate No': plate_no,
            'X (mm)': int(p['x']),
            'Y (mm)': int(p['y']),
            'Cut Width (mm)': round(p['w_true'], 1),
            'Cut Length (mm)': round(p['h_true'], 1),
            'Kerf (mm)': kerf,
            'RID': p['rid']
        })
    return pd.DataFrame(rows).sort_values(['Y (mm)', 'X (mm)'])

def _draw_plate_plotly(plate_no, placements, plate_w, plate_l, kerf,
                       show_ids=True, show_kerf_envelope=True,
                       min_label_area_mm2=60000, label_font_size=9):
    import plotly.graph_objects as go

    fig = go.Figure()
    # Plate boundary
    fig.add_shape(type="rect", x0=0, y0=0, x1=plate_w, y1=plate_l, line=dict(width=2))
    for p in placements:
        if p['plate_no'] != plate_no:
            continue
        x, y = p['x'], p['y']
        W_env, H_env = p['w_infl'], p['h_infl']
        W_true, H_true = p['w_true'], p['h_true']
        W_draw, H_draw = (W_env, H_env) if show_kerf_envelope else (W_true, H_true)
        # Outline
        fig.add_shape(type="rect", x0=x, y0=y, x1=x + W_draw, y1=y + H_draw, line=dict(width=1))
        # Hover
        hover = (f"True: {int(W_true)} × {int(H_true)} mm<br>"
                 f"Start: ({int(x)}, {int(y)}) mm<br>"
                 f"Kerf: {kerf} mm<br>"
                 f"RID: {p['rid']}")
        fig.add_trace(go.Scatter(
            x=[x, x+W_draw, x+W_draw, x, x],
            y=[y, y, y+H_draw, y+H_draw, y],
            mode="lines",
            line=dict(width=0),
            fill="toself",
            fillcolor="rgba(0,0,0,0)",
            hovertemplate=hover,
            showlegend=False
        ))
        # RID label (annotation supports rotation)
        if show_ids:
            area = W_draw * H_draw
            if area >= min_label_area_mm2:
                angle = 0 if W_draw >= H_draw else 90
                cx = x + W_draw / 2
                cy = y + H_draw / 2
                fig.add_annotation(
                    x=cx, y=cy, text=str(p['rid']),
                    showarrow=False, font=dict(size=label_font_size),
                    textangle=angle, xanchor="center", yanchor="middle"
                )
    # Clamp axes; fixed X ticks @ 500 mm; Y ticks @ 1000 mm
    PAD = max(int(kerf), 20)
    fig.update_xaxes(range=[0, plate_w + PAD], showgrid=True, gridwidth=1,
                     dtick=1000, tick0=0, ticks="outside", mirror=True, zeroline=False)
    fig.update_yaxes(range=[0, plate_l + PAD], showgrid=True, gridwidth=1,
                     dtick=1000, tick0=0, ticks="outside", mirror=True, zeroline=False,
                     scaleanchor="x", scaleratio=1)
    fig.update_shapes(xref="x", yref="y")
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20),
                      title=f"Plate {plate_no} — {plate_w}×{plate_l} mm (kerf {kerf} mm)",
                      dragmode="pan", uirevision="planner")
    return fig


# ---- UI block (appears under your Recommended mix) ----
with st.expander("Marking Planner (for the Recommended Mix)", expanded=False):
    if rec_df.empty:
        st.info("Run the sweep to get a recommended mix first.")
    else:
        thk_options = list(rec_df['Plate Thickness (mm)'].astype(float).unique())
        thk_pick = st.selectbox("Select thickness for marking plan", thk_options, index=0)

        pick_row = rec_df[rec_df['Plate Thickness (mm)'] == thk_pick].iloc[0]
        plate_w_rec = int(pick_row['Plate Width (mm)'])
        plate_l_rec = int(pick_row['Recommended Plate Length (mm)'])
        st.write(f"Using recommended plate size: **{plate_w_rec} × {plate_l_rec} mm** for **{thk_pick} mm**")

        # Filter parts for this thickness
        parts_this_thk = [c for c in cuts if abs(float(c['thickness']) - float(thk_pick)) < 1e-6]
        if not parts_this_thk:
            st.warning("No parts found for the selected thickness.")
        else:
            placements, plate_count = _pack_with_positions(
                parts_this_thk, plate_w_rec, plate_l_rec, kerf=kerf, rotation=rotation,
                allow_splice=allow_splice, max_plate_len=max_plate_len, splice_overlap=splice_overlap
            )
            if plate_count == 0:
                st.error("Packing failed for this selection. Try enabling rotation, reducing kerf, or larger plate.")
            else:
                # Controls
                cA, cB, cC, cD = st.columns(4)
                with cA:
                    plate_to_view = st.number_input("View Plate No.", min_value=1, max_value=plate_count, value=1, step=1)
                with cB:
                    show_ids = st.checkbox("Show RID", value=True)
                with cC:
                    show_env = st.checkbox("Show kerf envelope", value=True)
                with cD:
                    min_label_area = st.number_input("Min area to show RID (mm²)", min_value=0, value=60000, step=5000)

                # Side-by-side: chart (wider) and table+download
                left, right = st.columns([2, 1], gap="medium")

                with left:
                    fig = _draw_plate_plotly(
                        plate_to_view, placements, plate_w_rec, plate_l_rec, kerf,
                        show_ids=show_ids, show_kerf_envelope=show_env,
                        min_label_area_mm2=min_label_area, label_font_size=9
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

                with right:
                    plate_df = _build_cutlist_dataframe_for_plate(placements, plate_to_view, kerf)
                    st.dataframe(plate_df, use_container_width=True, height=420)
                    st.download_button(
                        f"⬇️ Download Plate {plate_to_view} Cut List (CSV)",
                        plate_df.to_csv(index=False),
                        file_name=f"plate_{plate_to_view}_{plate_w_rec}x{plate_l_rec}_thk{int(thk_pick)}.csv"
                    )

# ============================
# DROP-IN: ZIP export (DXF + CSVs) for current Marking Planner selection
# ============================
import io, zipfile
import ezdxf

def _export_plate_dxf(plate_no, placements, plate_w, plate_l) -> bytes:
    """
    Build a DXF (as bytes) for a single plate:
      - Plate boundary on layer 'PLATE'
      - True-size rectangles on layer 'PARTS'
    """
    text_buffer = io.StringIO()
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
        W, H = float(p['w_true']), float(p['h_true'])
        msp.add_lwpolyline(
            [(x, y), (x + W, y), (x + W, y + H), (x, y + H), (x, y)],
            dxfattribs={"layer": "PARTS", "closed": True}
        )

    # Write text DXF → bytes
    doc.write(text_buffer)
    return text_buffer.getvalue().encode("utf-8")

# ============================
# DROP-IN: Global ZIP Export for Recommended Mix
# ============================
import io, zipfile
import ezdxf

def _export_plate_dxf_global(plate_no, placements, plate_w, plate_l):
    """DXF (as bytes) for one plate."""
    buf_txt = io.StringIO()
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

    # Parts
    for p in placements:
        if p['plate_no'] != plate_no:
            continue
        x, y = p['x'], p['y']
        W, H = float(p['w_true']), float(p['h_true'])
        msp.add_lwpolyline(
            [(x, y), (x + W, y), (x + W, y + H), (x, y + H), (x, y)],
            dxfattribs={"layer": "PARTS", "closed": True}
        )

    doc.write(buf_txt)
    return buf_txt.getvalue().encode("utf-8")

def _export_recommended_mix_zip(rec_df, cuts, kerf, rotation,
                                allow_splice, max_plate_len, splice_overlap):
    """
    Bundle all recommended thicknesses into one ZIP:
      - DXF + CSV per plate
      - index.csv (per thickness summary)
      - all_plates.csv (all cuts combined)
    """
    buf = io.BytesIO()
    all_plate_rows = []
    index_rows = []

    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        for _, rec in rec_df.iterrows():
            thk = float(rec['Plate Thickness (mm)'])
            W   = int(rec['Plate Width (mm)'])
            L   = int(rec['Recommended Plate Length (mm)'])

            # Filter parts for this thickness
            parts_this_thk = [c for c in cuts if abs(float(c['thickness']) - thk) < 1e-6]
            if not parts_this_thk:
                continue

            placements, plate_count = _pack_with_positions(
                parts_this_thk, W, L, kerf=kerf, rotation=rotation,
                allow_splice=allow_splice, max_plate_len=max_plate_len, splice_overlap=splice_overlap
            )

            for n in range(1, plate_count + 1):
                # DXF
                dxf_bytes = _export_plate_dxf_global(n, placements, W, L)
                dxf_name = f"plate_{n}_{W}x{L}_thk{int(thk)}.dxf"
                z.writestr(dxf_name, dxf_bytes)

                # CSV (cut list)
                df_plate = _build_cutlist_dataframe_for_plate(placements, n, kerf)
                csv_name = f"plate_{n}_{W}x{L}_thk{int(thk)}.csv"
                z.writestr(csv_name, df_plate.to_csv(index=False))

                # Collect for index + combined
                index_rows.append({
                    "Plate No": n,
                    "Thickness (mm)": thk,
                    "Plate Size (mm)": f"{W}x{L}",
                    "Items": len(df_plate),
                })
                df_plate = df_plate.copy()
                df_plate["Thickness (mm)"] = thk
                all_plate_rows.append(df_plate)

        # Write index.csv
        if index_rows:
            index_df = pd.DataFrame(index_rows).sort_values(["Thickness (mm)", "Plate No"])
            z.writestr("index.csv", index_df.to_csv(index=False))

        # Write all_plates.csv
        if all_plate_rows:
            all_df = pd.concat(all_plate_rows, ignore_index=True)
            z.writestr("all_plates.csv", all_df.to_csv(index=False))

    buf.seek(0)
    return buf.read()

# ---- UI: Global ZIP Download ----
if uploaded and not rec_df.empty:
    zip_bytes = _export_recommended_mix_zip(
        rec_df, cuts, kerf, rotation,
        allow_splice, max_plate_len, splice_overlap
    )
    st.subheader("📦 Download Complete Cutting Package")
    st.download_button(
        "⬇️ Download ALL Plates (DXF + CSVs + Index) as ZIP",
        data=zip_bytes,
        file_name="cutting_plan_recommended_mix.zip",
        mime="application/zip"
    )
