import streamlit as st
import pandas as pd
import re
from math import ceil
from rectpack import newPacker

st.set_page_config(layout="wide", page_title="BH Plate Size Optimizer")

DENSITY_STEEL = 7850.0  # kg/m^3

# ---------- Utilities

def _fmt_kg(x):
    try:
        x = float(x)
    except:
        return x
    if x >= 1000:
        return f"{x/1000:.2f} t"
    return f"{x:.0f} kg"

def _ensure_pos(*vals):
    return all(float(v) > 0 for v in vals)

# Regex:
# Optional prefix BH/NPB/WPB, then four numeric fields separated by X/x, allowing spaces.
# Captures as: (t_web, t_flange, H, W) but we reorder to match your original tuple.
PROFILE_RE = re.compile(
    r"^\s*(?:BH|NPB|WPB)?\s*"
    r"(\d+(?:\.\d+)?)\s*[Xx]\s*"
    r"(\d+(?:\.\d+)?)\s*[Xx]\s*"
    r"(\d+(?:\.\d+)?)\s*[Xx]\s*"
    r"(\d+(?:\.\d+)?)\s*$"
)

def extract_dims(profile: str):
    """Return (web_thk, flange_thk, H, W) in mm from a PROFILE string."""
    s = str(profile).strip()
    m = PROFILE_RE.match(s)
    if not m:
        raise ValueError(f"Bad PROFILE format: {profile!r}. Expected like 'BH8x20x500x200'.")
    a, b, c, d = map(float, m.groups())
    # Your original assumption was: return (wt, ft, H, W) == (parts[2], parts[3], parts[0], parts[1])
    # Here we keep SAME ordering as your downstream usage: (wt, ft, H, W)
    wt, ft, H, W = c, d, a, b
    if not _ensure_pos(wt, ft, H, W):
        raise ValueError(f"Non-positive dimensions in PROFILE: {profile}")
    return wt, ft, H, W

def calc_row_weights(row, density=DENSITY_STEEL):
    wt, ft, H, W = extract_dims(row['PROFILE'])
    try:
        Lm = float(row['LENGTH(mm)']) / 1000.0  # m
    except Exception:
        raise ValueError(f"Bad LENGTH(mm) value: {row['LENGTH(mm)']}")
    qty = int(row['QTY.'])
    if qty < 0:
        raise ValueError("QTY. must be >= 0")

    # Web height (clear between flanges)
    web_h = max(0.0, H - 2.0 * ft)

    # Convert mm^2 to m^2 by /1e6; then multiply by length (m) to get m^3; then * density => kg
    web_area_mm2   = (web_h * wt)
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

def build_cuts(df):
    cuts = []
    for _, r in df.iterrows():
        qty = int(r['QTY.'])
        L = float(r['LENGTH(mm)'])
        if qty <= 0 or L <= 0:
            continue

        # web strip: width = web_thk
        if r['web_thk'] > 0 and r['web_height'] > 0:
            total_web_wt = float(r['web_weight'])
            web_each = total_web_wt / qty if qty else 0.0
            cuts.append({
                'thickness': float(r['web_thk']),
                'type': 'web',
                'width': float(r['web_thk']),   # strip width
                'length': float(L),             # along member length
                'qty': qty,
                'weight_each': web_each
            })

        # flange plates: two per beam
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
    rects = []
    id_to_weight = {}
    rid = 0
    for p in parts:
        w = float(p['width'])
        l = float(p['length'])
        q = int(p['qty'])
        we = float(p['weight_each'])
        if not _ensure_pos(w, l) or q <= 0:
            continue
        for _ in range(q):
            rects.append((w, l, rid))
            id_to_weight[rid] = we
            rid += 1
    return rects, id_to_weight

def pack_thickness_rects(rects, plate_w, plate_l, kerf=6, rotation=True):
    """Fast single-shot packing: inflate once, add a generous fixed number of bins, pack once."""
    if not rects:
        return 0, 0.0

    add = max(0.0, float(kerf))
    # Inflate by kerf once (as full gap); keep area as *true* rect area for wastage calc
    inflated = [(max(1, int(round(w + add))), max(1, int(round(h + add))), rid, w, h)
                for (w, h, rid) in rects]
    true_area = sum(w * h for (w, h, _) in rects)

    # Heuristic: bins ~= max(n/2, 50) — like your original idea, with a floor
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


def eval_for_size(parts, thk, plate_w, plate_l, kerf, include_kerf_in_area=False):
    rects, id_weights = make_rects(parts)
    total_weight = round(sum(id_weights.values()), 3)
    total_qty = len(rects)

    plate_wt = (plate_w/1000.0)*(plate_l/1000.0)*(thk/1000.0)*DENSITY_STEEL
    min_by_weight = ceil(total_weight / max(plate_wt, 1e-9)) if total_weight > 0 else 0

    # Single fast pack
    plates_used, true_area = pack_thickness_rects(
        rects, plate_w, plate_l, kerf=kerf, rotation=True
    )

    # Respect weight lower bound
    if plates_used < min_by_weight:
        plates_used = min_by_weight

    total_plate_area = plates_used * (plate_w * plate_l) if plates_used else 0.0

    # Area wastage: use true part area (fast) unless you really need kerf included
    area_used_for_wastage = true_area if not include_kerf_in_area else true_area  # keep fast
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
        'wastage_area': wastage_area
    }


def sweep_and_pick(cuts, widths, lengths, kerf, goal='Min wastage (mass)', include_kerf_in_area=True):
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
                opt = eval_for_size(parts, thk, W, L, kerf, include_kerf_in_area=include_kerf_in_area)
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
                    'Wastage (area %)': opt['wastage_area']
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
            'Wastage (area %)': pick['wastage_area']
        })

    return pd.DataFrame(compare_rows), pd.DataFrame(recommended)

# ---------- UI

st.title("Plate Size Optimizer (Width × Length Sweep)")

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

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        widths_input = st.text_input("Plate widths to test (comma-separated mm)", "2000,2500")
        widths = []
        for token in widths_input.split(","):
            token = token.strip()
            if token:
                try:
                    widths.append(int(float(token)))
                except:
                    st.warning(f"Ignoring bad width entry: {token}")
        widths = sorted(set(w for w in widths if w > 0))
    with c2:
        lengths_input = st.text_input("Plate lengths to test (comma-separated mm)", "10000,10500,11000,11500,12000")
        lengths = []
        for token in lengths_input.split(","):
            token = token.strip()
            if token:
                try:
                    lengths.append(int(float(token)))
                except:
                    st.warning(f"Ignoring bad length entry: {token}")
        lengths = sorted(set(L for L in lengths if L > 0))
    with c3:
        kerf = st.number_input("Kerf between parts (mm)", min_value=0, value=6)
    with c4:
        include_kerf_in_area = st.checkbox("Include kerf in area wastage %", value=True, help="If on, area wastage accounts for cut gaps.")

    goal = st.selectbox("Optimization goal", ["Min wastage (mass)", "Min order weight", "Min plates"], index=0)

    # KPIs
    total_parts = sum(int(p.get('qty', 0)) for p in cuts)
    total_cut_weight = sum(float(p.get('weight_each', 0.0)) * int(p.get('qty', 0)) for p in cuts)
    k1, k2 = st.columns(2)
    k1.metric("Total parts (rectangles)", f"{total_parts}")
    k2.metric("Total part weight", _fmt_kg(total_cut_weight))

    st.subheader("Cuts generated")
    st.dataframe(pd.DataFrame(cuts), use_container_width=True)

    with st.spinner("Running sweep and selecting best per thickness..."):
        cmp_df, rec_df = sweep_and_pick(cuts, widths, lengths, kerf, goal, include_kerf_in_area=include_kerf_in_area)

    st.subheader("Comparison table (all width × length)")
    cmp_sorted = cmp_df.sort_values(['Plate Thickness (mm)', 'Plate Width (mm)', 'Plate Length (mm)'])
    st.dataframe(cmp_sorted, use_container_width=True, height=420)
    st.download_button("⬇️ Download Comparison CSV", cmp_sorted.to_csv(index=False), "width_length_comparison.csv")

    st.subheader("Recommended mixed order (best size per thickness)")
    if not rec_df.empty:
        show_cols = ['Plate Thickness (mm)', 'Plate Width (mm)', 'Recommended Plate Length (mm)',
                     'Plate weight (kg)', 'Plates Needed', 'Order weight (kg)',
                     'Total Weight (kg)', 'Wastage (mass %)', 'Wastage (area %)']
        st.dataframe(rec_df[show_cols].sort_values('Plate Thickness (mm)'), use_container_width=True)
        st.download_button("⬇️ Download Recommended Mix CSV", rec_df[show_cols].to_csv(index=False), "recommended_mix.csv")

        # Tiny summary
        total_order_wt = rec_df['Order weight (kg)'].sum()
        total_plates   = rec_df['Plates Needed'].sum()
        s1, s2 = st.columns(2)
        s1.metric("Total plates (all thicknesses)", f"{int(total_plates)}")
        s2.metric("Total order weight", _fmt_kg(total_order_wt))
    else:
        st.info("No recommendations could be compiled (check inputs).")

else:
    st.info("Upload Excel with columns: PROFILE, LENGTH(mm), QTY.")
