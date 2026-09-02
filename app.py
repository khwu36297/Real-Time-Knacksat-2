import streamlit as st
import pandas as pd
import numpy as np
import calendar
import math
from datetime import datetime, timezone, timedelta
from skyfield.api import load, wgs84, EarthSatellite
from skyfield.framelib import itrs
import requests
import traceback
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Satellite Command Delay Calculator | CDH Team",
    page_icon="🛰️",
    layout="wide"
)

SPEED_OF_LIGHT_KM_S = 299792.458
ts = load.timescale()

ICT_TZ = timezone(timedelta(hours=7))

# TLE ของ LEO จะเริ่มคลาดเคลื่อนมากหลังจากนี้ (วัน) เพราะแรงต้านบรรยากาศ (drag)
TLE_MAX_AGE_DAYS_WARN = 7
TLE_MAX_AGE_DAYS_DANGER = 21

GS_PRESETS = {
    "GISTDA ศรีราชา (Chonburi)": {"lat": 13.1014, "lon": 100.9234, "alt": 50.0},
    "สถานีเชียงใหม่ (Chiang Mai GS)": {"lat": 18.8524, "lon": 98.9642, "alt": 300.0},
    "Custom Location": {"lat": 13.7563, "lon": 100.5018, "alt": 10.0},
}

TARGET_PRESETS = {
    "ปารีส, ฝรั่งเศส (Paris, France)": {"lat": 48.8566, "lon": 2.3522},
    "โตเกียว, ญี่ปุ่น (Tokyo, Japan)": {"lat": 35.6762, "lon": 139.6503},
    "นิวยอร์ก, สหรัฐฯ (New York, USA)": {"lat": 40.7128, "lon": -74.0060},
    "Custom Pin บนแผนที่": {"lat": 48.8566, "lon": 2.3522},
}

THAI_MONTHS = [
    "มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
    "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม",
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
@st.cache_data(ttl=3600)
def fetch_tle_candidates(query: str):
    """ดึงค่า TLE จาก CelesTrak API — คืนเป็น list เพราะการค้นหาบางชื่ออาจได้หลายดวง"""
    url = f"https://celestrak.org/NORAD/elements/gp.php?NAME={query}&FORMAT=tle"
    try:
        response = requests.get(url, timeout=10)
    except Exception as e:
        return [], f"เชื่อมต่อ CelesTrak ไม่สำเร็จ: {e}"

    if response.status_code != 200:
        return [], f"CelesTrak ตอบกลับสถานะ HTTP {response.status_code}"

    lines = [l.strip() for l in response.text.strip().splitlines() if l.strip()]
    if len(lines) < 3:
        return [], "ไม่พบข้อมูลดาวเทียมที่ตรงกับคำค้นหา"

    candidates = []
    for i in range(0, len(lines) - 2, 3):
        name, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
        if l1.startswith("1 ") and l2.startswith("2 "):
            candidates.append((name, l1, l2))

    if not candidates:
        return [], "รูปแบบข้อมูลที่ได้รับไม่ถูกต้อง (ไม่พบบรรทัด TLE ที่สมบูรณ์)"

    return candidates, None


def validate_tle(line1: str, line2: str):
    """ตรวจสอบรูปแบบ TLE คร่าว ๆ ก่อนสร้าง EarthSatellite"""
    errors = []
    if not line1.strip().startswith("1 "):
        errors.append("Line 1 ต้องขึ้นต้นด้วย '1 '")
    if not line2.strip().startswith("2 "):
        errors.append("Line 2 ต้องขึ้นต้นด้วย '2 '")
    if len(line1.strip()) < 60:
        errors.append("Line 1 สั้นเกินไป (ควรยาวประมาณ 69 ตัวอักษร)")
    if len(line2.strip()) < 60:
        errors.append("Line 2 สั้นเกินไป (ควรยาวประมาณ 69 ตัวอักษร)")
    return errors


def check_tle_freshness(satellite: EarthSatellite, reference_dt: datetime):
    """
    เช็คว่า TLE เก่าไปหรือยัง เทียบกับวันที่ที่จะใช้คำนวณ (ไม่ใช่แค่ 'วันนี้')
    เพราะ TLE ของ LEO แม่นยำจริงแค่ ~1-2 สัปดาห์รอบ epoch เนื่องจากแรงต้านบรรยากาศ
    คืนค่า (age_days: int, level: "ok" | "warn" | "danger")
    """
    epoch_dt = satellite.epoch.utc_datetime()
    age_days = abs((reference_dt - epoch_dt).days)
    if age_days > TLE_MAX_AGE_DAYS_DANGER:
        level = "danger"
    elif age_days > TLE_MAX_AGE_DAYS_WARN:
        level = "warn"
    else:
        level = "ok"
    return age_days, level, epoch_dt


def find_all_passes(satellite, lat, lon, alt_m, start_dt, duration_hours=48, min_el=10.0):
    """
    ค้นหารอบบินผ่านทั้งหมด (Multi-Pass Search) ตาม Threshold ที่ผู้ใช้กำหนด
    หมายเหตุ: คืนค่าเวลาเป็น python datetime (UTC) ธรรมดา ไม่ใช่ skyfield Time
    เพื่อให้ผลลัพธ์ serializable และใช้กับ st.cache_data ได้ (ดู find_passes_cached ด้านล่าง)
    """
    t0 = ts.from_datetime(start_dt)
    t1 = ts.from_datetime(start_dt + timedelta(hours=duration_hours))
    location = wgs84.latlon(lat, lon, elevation_m=alt_m)

    times, events = satellite.find_events(location, t0, t1, altitude_degrees=min_el)

    passes = []
    current_pass = {}

    for t, event in zip(times, events):
        if event == 0:  # Rise
            current_pass = {"rise": t.utc_datetime()}
        elif event == 1:  # Culmination (Max Elevation)
            current_pass["max"] = t.utc_datetime()
            topocentric = (satellite - location).at(t)
            alt, az, distance = topocentric.altaz()
            current_pass["max_el"] = alt.degrees
            current_pass["distance_km"] = distance.km
        elif event == 2:  # Set
            current_pass["set"] = t.utc_datetime()
            if "max" in current_pass:
                passes.append(current_pass)
            current_pass = {}

    return passes


@st.cache_data(ttl=3600, show_spinner=False)
def find_passes_cached(tle_line1, tle_line2, tle_name, lat, lon, alt_m, start_iso, duration_hours, min_el):
    """
    Wrapper ของ find_all_passes ที่ cache ผลลัพธ์ตาม TLE + ตำแหน่ง + ช่วงเวลา
    ทำให้เวลาผู้ใช้สลับดูเดือนที่เคยค้นหาไปแล้ว (เช่น กลับไปดูเดือนก่อนหน้า) ไม่ต้องคำนวณ SGP4 ซ้ำ
    รับเฉพาะ primitive types เป็น argument เพื่อให้ st.cache_data hash ได้ (EarthSatellite hash ไม่ได้)
    """
    satellite = EarthSatellite(tle_line1, tle_line2, tle_name, ts)
    start_dt = datetime.fromisoformat(start_iso)
    return find_all_passes(satellite, lat, lon, alt_m, start_dt, duration_hours, min_el)


def passes_to_dataframe(passes, time_col_label):
    rows = []
    for i, p in enumerate(passes):
        max_dt = p["max"]
        rows.append({
            "Pass #": i + 1,
            "Date": max_dt.strftime("%Y-%m-%d"),
            f"{time_col_label} (UTC)": max_dt.strftime("%H:%M:%S"),
            f"{time_col_label} (ICT)": max_dt.astimezone(ICT_TZ).strftime("%H:%M:%S"),
            "Max Elevation (°)": round(p["max_el"], 1),
            "Distance (km)": round(p["distance_km"], 2),
        })
    return pd.DataFrame(rows)


def format_hms(total_seconds: float) -> str:
    sign = "-" if total_seconds < 0 else ""
    total_seconds = abs(total_seconds)
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{sign}{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def month_end_utc(year: int, month: int) -> datetime:
    """คืนวันที่ 00:00 UTC ของวันแรกของเดือนถัดไป (ใช้หาขอบเขตสิ้นเดือน)"""
    if month == 12:
        return datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    return datetime(year, month + 1, 1, tzinfo=timezone.utc)


# =============================================================================
# REAL-TIME ORBIT TRACKER HELPERS
# (พอร์ตแนวคิดจากสคริปต์ MATLAB satelliteScenario มาเป็น Skyfield + Streamlit)
# =============================================================================
def get_orbital_period_minutes(satellite: EarthSatellite) -> float:
    """คาบการโคจรโดยประมาณ (นาที) จากค่า mean motion (no_kozai) ใน TLE"""
    return 2 * math.pi / satellite.model.no_kozai


def get_current_state(satellite: EarthSatellite, at_dt: datetime):
    """คืนตำแหน่งปัจจุบันของดาวเทียม ณ เวลาที่กำหนด: (lat_deg, lon_deg, alt_km, speed_km_s)"""
    t = ts.from_datetime(at_dt)
    geocentric = satellite.at(t)
    subpoint = wgs84.subpoint(geocentric)
    vx, vy, vz = geocentric.velocity.km_per_s
    speed_km_s = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    return subpoint.latitude.degrees, subpoint.longitude.degrees, subpoint.elevation.km, speed_km_s


def compute_ground_track(satellite: EarthSatellite, start_dt: datetime, duration_min: float, step_sec: int = 30):
    """สุ่มตัวอย่างตำแหน่ง subpoint (lat, lon) ล่วงหน้า duration_min นาที ทุก step_sec วินาที
    (เทียบเท่า groundTrack(..., 'LeadTime', ...) ในสคริปต์ MATLAB)"""
    n_steps = max(2, int((duration_min * 60) // step_sec) + 1)
    points = []
    for i in range(n_steps):
        dt = start_dt + timedelta(seconds=i * step_sec)
        t = ts.from_datetime(dt)
        subpoint = wgs84.subpoint(satellite.at(t))
        points.append((subpoint.latitude.degrees, subpoint.longitude.degrees))
    return points


def split_track_on_dateline(points):
    """ตัดเส้นทางเป็นช่วง ๆ ตรงที่ลองจิจูดกระโดดข้าม ±180° เพื่อไม่ให้ folium ลากเส้นพาดผ่านแผนที่ผิดฝั่ง"""
    if not points:
        return []
    segments = [[points[0]]]
    for prev, curr in zip(points, points[1:]):
        if abs(curr[1] - prev[1]) > 180:
            segments.append([curr])
        else:
            segments[-1].append(curr)
    return segments


def latlon_list_to_ecef(points_latlon, radius_km):
    """แปลงพิกัด lat/lon (องศา) เป็นพิกัด ECEF บนผิวทรงกลมรัศมี radius_km (ใช้เพื่อวาดกราฟ 3 มิติเท่านั้น)"""
    result = []
    for lat_deg, lon_deg in points_latlon:
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        x = radius_km * math.cos(lat) * math.cos(lon)
        y = radius_km * math.cos(lat) * math.sin(lon)
        z = radius_km * math.sin(lat)
        result.append((x, y, z))
    return result


def build_globe_figure(satellite: EarthSatellite, at_dt: datetime, track_points_latlon, sat_name: str,
                        earth_radius_km: float = 6371.0):
    """สร้างกราฟ 3 มิติ (Plotly) ของโลกทรงกลม + ตำแหน่งดาวเทียม + เส้นทางวงโคจร ในกรอบอ้างอิง ECEF
    (หมุนไปกับโลก คล้ายมุมมองใน satelliteScenarioViewer ของ MATLAB — ผู้ใช้หมุน/ซูมด้วยเมาส์ได้)"""
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 24)
    xs = earth_radius_km * np.outer(np.cos(u), np.sin(v))
    ys = earth_radius_km * np.outer(np.sin(u), np.sin(v))
    zs = earth_radius_km * np.outer(np.ones_like(u), np.cos(v))

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=zs,
        colorscale=[[0, "#0a3d62"], [1, "#0a3d62"]],
        showscale=False, opacity=0.85, hoverinfo="skip", name="Earth",
    ))

    track_xyz = latlon_list_to_ecef(track_points_latlon, earth_radius_km)
    if track_xyz:
        fig.add_trace(go.Scatter3d(
            x=[p[0] for p in track_xyz], y=[p[1] for p in track_xyz], z=[p[2] for p in track_xyz],
            mode="lines", line=dict(color="#00e5ff", width=4), name="Ground Track",
        ))

    t = ts.from_datetime(at_dt)
    sat_xyz = satellite.at(t).frame_xyz(itrs).km
    fig.add_trace(go.Scatter3d(
        x=[sat_xyz[0]], y=[sat_xyz[1]], z=[sat_xyz[2]],
        mode="markers", marker=dict(size=6, color="#ff3333"), name=sat_name,
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        height=480,
        paper_bgcolor="rgba(0,0,0,0)",
        uirevision="orbit_globe",  # กันมุมกล้อง/ซูมที่ผู้ใช้หมุนไว้ไม่ให้รีเซ็ตทุกครั้งที่ rerun
    )
    return fig


def build_ground_track_figure(lat_now, lon_now, track_points_latlon, sat_name, hover_text):
    """สร้างแผนที่ 2 มิติ (Ground Track) ด้วย Plotly Scattergeo แทน folium
    เหตุผล: folium/st_folium ฝังแผนที่เป็น iframe แล้ว 'วาดใหม่ทั้งอัน' ทุกครั้งที่ rerun
    จึงกระพริบเสมอต่อให้ rerun แค่บางส่วนของหน้าด้วย st.fragment ก็ตาม
    ส่วน Plotly นั้น Streamlit อัปเดตข้อมูลกราฟแบบ in-place (Plotly.react) เมื่อ key เดิม
    โครงสร้างกราฟเดิม จึงได้ตำแหน่ง/เส้นทางที่ขยับแบบเรียลไทม์โดยไม่กระพริบ"""
    fig = go.Figure()

    for seg in split_track_on_dateline(track_points_latlon):
        fig.add_trace(go.Scattergeo(
            lat=[p[0] for p in seg], lon=[p[1] for p in seg],
            mode="lines", line=dict(color="#00e5ff", width=2),
            hoverinfo="skip", showlegend=False,
        ))

    fig.add_trace(go.Scattergeo(
        lat=[lat_now], lon=[lon_now], mode="markers",
        marker=dict(size=11, color="#ff3333", line=dict(color="white", width=1)),
        text=[hover_text], hoverinfo="text", name=sat_name, showlegend=False,
    ))

    fig.update_geos(
        projection_type="natural earth",
        showland=True, landcolor="#16213e",
        showocean=True, oceancolor="#0b1120",
        showcountries=True, countrycolor="#2b3a55",
        showcoastlines=True, coastlinecolor="#2b3a55",
        showframe=False,
        bgcolor="rgba(0,0,0,0)",
        lataxis=dict(showgrid=True, gridcolor="#22314f", gridwidth=0.5),
        lonaxis=dict(showgrid=True, gridcolor="#22314f", gridwidth=0.5),
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        uirevision="orbit_track_map",  # กันไม่ให้ zoom/pan ของผู้ใช้ถูกรีเซ็ตทุกครั้งที่ rerun
    )
    return fig


# =============================================================================
# HEADER
# =============================================================================
st.title("🛰️ Satellite Command Delay Calculator")
st.caption("ระบบคำนวณเวลาดีเลย์การสั่งงานดาวเทียมสำหรับทีม CDH (Command & Data Handling)")
st.markdown("---")

# =============================================================================
# SIDEBAR: TLE SELECTION
# =============================================================================
st.sidebar.header("⚙️ Configuration Settings")
st.sidebar.subheader("1. 🛰️ ข้อมูลวงโคจรดาวเทียม (TLE)")

tle_source = st.sidebar.radio("แหล่งข้อมูล TLE:", ["CelesTrak API (สด)", "ป้อน TLE เอง (Custom)"])

tle_name, tle_line1, tle_line2 = "", "", ""
tle_ready = False

if tle_source == "CelesTrak API (สด)":
    sat_search = st.sidebar.text_input("ค้นหาชื่อดาวเทียม (เช่น ISS, NOAA 19, THAICOM 8):", value="ISS")
    with st.sidebar:
        with st.spinner("กำลังดึงข้อมูล TLE..."):
            candidates, err = fetch_tle_candidates(sat_search) if sat_search.strip() else ([], "กรุณาระบุชื่อดาวเทียม")

        if err:
            st.error(f"❌ {err}")
        elif len(candidates) == 1:
            tle_name, tle_line1, tle_line2 = candidates[0]
            st.success(f"พบข้อมูล TLE ล่าสุด: {tle_name}")
            tle_ready = True
        else:
            st.info(f"พบ {len(candidates)} ดวงที่ตรงกับคำค้นหา กรุณาเลือก:")
            options = [c[0] for c in candidates]
            chosen = st.selectbox("เลือกดาวเทียม:", options)
            tle_name, tle_line1, tle_line2 = next(c for c in candidates if c[0] == chosen)
            tle_ready = True
else:
    tle_name = st.sidebar.text_input("Satellite Name:", "CUSTOM-SAT")
    tle_line1 = st.sidebar.text_input(
        "TLE Line 1:",
        "1 25544U 98067A   24065.51782569  .00014790  00000+0  26388-3 0  9999",
    )
    tle_line2 = st.sidebar.text_input(
        "TLE Line 2:",
        "2 25544  51.6416 288.4237 0004868 200.5986 270.8164 15.49544464442542",
    )
    tle_errors = validate_tle(tle_line1, tle_line2)
    if tle_errors:
        for e in tle_errors:
            st.sidebar.error(f"⚠️ {e}")
        tle_ready = False
    else:
        st.sidebar.success("รูปแบบ TLE ถูกต้อง")
        tle_ready = True

st.sidebar.markdown("---")

# =============================================================================
# SIDEBAR: SEARCH PARAMETERS
# =============================================================================
st.sidebar.subheader("2. 🎛️ ตั้งค่าเงื่อนไขการค้นหา (Parameters)")

search_mode = st.sidebar.radio(
    "ช่วงเวลาค้นหา Ground Station Passes:",
    ["ระบุจำนวนชั่วโมงล่วงหน้า", "เลือกหลายเดือน (รายเดือน)"],
)

now_utc_for_ui = datetime.now(timezone.utc)

# month_ranges: list ของช่วงเวลาที่จะค้นหา แต่ละช่วงจะถูกคำนวณและเก็บผลแยกกัน
# ทำให้เลือกได้หลายเดือน/ทั้งปีพร้อมกันในการค้นหาครั้งเดียว แล้วสลับดูทีหลังได้โดยไม่ต้องกดค้นหาใหม่
month_ranges = []

if search_mode == "ระบุจำนวนชั่วโมงล่วงหน้า":
    duration_hours_gs = st.sidebar.slider(
        "ระยะเวลาค้นหาล่วงหน้า (ชั่วโมง):", min_value=12, max_value=120, value=48, step=12
    )
    month_ranges.append({
        "label": f"ล่วงหน้า {duration_hours_gs} ชม.",
        "start": now_utc_for_ui,
        "duration": float(duration_hours_gs),
        "year": None,
        "month": None,
    })
else:
    col_y, col_all = st.sidebar.columns([1, 1])
    with col_y:
        sel_year = st.selectbox("ปี (ค.ศ.):", [now_utc_for_ui.year, now_utc_for_ui.year + 1], index=0)
    with col_all:
        st.write("")
        st.write("")
        select_all_year = st.checkbox("เลือกทั้งปี (12 เดือน)", value=False, key="select_all_months")

    if select_all_year:
        sel_months = list(range(1, 13))
        st.sidebar.caption("เลือกทั้ง 12 เดือนของปีนี้ — การค้นหาอาจใช้เวลาสักครู่")
    else:
        default_month = [now_utc_for_ui.month] if sel_year == now_utc_for_ui.year else [1]
        sel_months = st.sidebar.multiselect(
            "เลือกเดือน (เลือกได้หลายเดือน):",
            list(range(1, 13)),
            default=default_month,
            format_func=lambda m: THAI_MONTHS[m - 1],
            key=f"sel_months_widget_{sel_year}",
        )

    if not sel_months:
        st.sidebar.warning("กรุณาเลือกอย่างน้อย 1 เดือน")

    for m in sorted(sel_months):
        start = datetime(sel_year, m, 1, tzinfo=timezone.utc)
        # ถ้าเลือกเดือนปัจจุบัน ให้เริ่มค้นจาก "ตอนนี้" แทนต้นเดือน จะได้ไม่เห็นรอบที่ผ่านไปแล้ว
        if sel_year == now_utc_for_ui.year and m == now_utc_for_ui.month:
            start = now_utc_for_ui
        end = month_end_utc(sel_year, m)
        duration = (end - start).total_seconds() / 3600.0
        month_ranges.append({
            "label": f"{THAI_MONTHS[m - 1]} {sel_year}",
            "start": start,
            "duration": duration,
            "year": sel_year,
            "month": m,
        })

duration_hours_target = st.sidebar.slider(
    "ค้นหารอบเป้าหมายล่วงหน้าจากเวลา Uplink (ชั่วโมง):", min_value=12, max_value=120, value=48, step=12
)
min_el_gs = st.sidebar.slider("มุม Elevation ขั้นต่ำเหนือ GS (°):", min_value=0.0, max_value=45.0, value=10.0, step=1.0)
min_el_target = st.sidebar.slider("มุม Elevation ขั้นต่ำเหนือเป้าหมาย (°):", min_value=0.0, max_value=45.0, value=10.0, step=1.0)

# =============================================================================
# MAIN: LOCATION SELECTION
# =============================================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📡 1. สถานีส่งคำสั่ง (Uplink Location)")
    gs_choice = st.selectbox("เลือกสถานีภาคพื้นดิน:", list(GS_PRESETS.keys()))

    if gs_choice == "Custom Location":
        gs_lat = st.number_input("Ground Station Lat:", value=13.1014, format="%.4f", key="gs_lat_input")
        gs_lon = st.number_input("Ground Station Lon:", value=100.9234, format="%.4f", key="gs_lon_input")
        gs_alt = st.number_input("Ground Station Altitude (m):", value=50.0, key="gs_alt_input")
    else:
        gs_lat = GS_PRESETS[gs_choice]["lat"]
        gs_lon = GS_PRESETS[gs_choice]["lon"]
        gs_alt = GS_PRESETS[gs_choice]["alt"]
        st.info(f"📍 Lat: `{gs_lat}` | Lon: `{gs_lon}` | Alt: `{gs_alt}m`")

with col2:
    st.subheader("🎯 2. พิกัดเป้าหมาย (Target Location)")
    target_choice = st.selectbox("เลือกเป้าหมายถ่ายภาพ:", list(TARGET_PRESETS.keys()))

    if "t_lat_input" not in st.session_state:
        st.session_state["t_lat_input"] = TARGET_PRESETS[target_choice]["lat"]
        st.session_state["t_lon_input"] = TARGET_PRESETS[target_choice]["lon"]
        st.session_state["_prev_target_choice"] = target_choice

    if target_choice != "Custom Pin บนแผนที่" and st.session_state.get("_prev_target_choice") != target_choice:
        st.session_state["t_lat_input"] = TARGET_PRESETS[target_choice]["lat"]
        st.session_state["t_lon_input"] = TARGET_PRESETS[target_choice]["lon"]
    st.session_state["_prev_target_choice"] = target_choice

    st.caption("👇 คลิกบนแผนที่เพื่อปักหมุดเลือกพิกัดเป้าหมายถ่ายภาพ (Interactive Map)")
    m = folium.Map(
        location=[st.session_state["t_lat_input"], st.session_state["t_lon_input"]],
        zoom_start=3,
    )
    folium.Marker(
        [st.session_state["t_lat_input"], st.session_state["t_lon_input"]],
        popup="Target Location",
        icon=folium.Icon(color="red", icon="camera"),
    ).add_to(m)

    map_data = st_folium(m, height=220, use_container_width=True, key="target_map")

    if map_data and map_data.get("last_clicked"):
        st.session_state["t_lat_input"] = map_data["last_clicked"]["lat"]
        st.session_state["t_lon_input"] = map_data["last_clicked"]["lng"]

    target_lat = st.number_input("Target Lat:", format="%.4f", key="t_lat_input")
    target_lon = st.number_input("Target Lon:", format="%.4f", key="t_lon_input")

st.markdown("---")

# =============================================================================
# PASS SEARCH & SELECTION
# =============================================================================
st.subheader("📅 3. ค้นหาและเลือกรอบวงโคจร (Satellite Pass Selection)")

if search_mode != "ระบุจำนวนชั่วโมงล่วงหน้า" and len(month_ranges) > 1:
    st.caption(
        f"จะค้นหาทั้งหมด {len(month_ranges)} เดือน "
        f"({', '.join(r['label'] for r in month_ranges)}) ในการกดค้นหาครั้งเดียว "
        "แล้วสลับดูแต่ละเดือนได้จากแท็บด้านล่างโดยไม่ต้องกดค้นหาใหม่"
    )

search_clicked = st.button(
    "🔍 ค้นหารอบการบินผ่านทั้งหมด (Search Passes)",
    type="secondary",
    use_container_width=True,
    disabled=(len(month_ranges) == 0),
)

if search_clicked and month_ranges:
    st.session_state["gs_search_results"] = []
    progress = st.progress(0.0, text="กำลังค้นหา...")
    for idx, rng in enumerate(month_ranges):
        progress.progress(
            (idx) / len(month_ranges),
            text=f"กำลังค้นหา {rng['label']} ({idx + 1}/{len(month_ranges)})...",
        )
        passes = find_passes_cached(
            tle_line1, tle_line2, tle_name,
            gs_lat, gs_lon, gs_alt,
            rng["start"].isoformat(), rng["duration"], min_el_gs,
        )
        st.session_state["gs_search_results"].append({**rng, "passes": passes})
    progress.progress(1.0, text="ค้นหาเสร็จสิ้น")
    progress.empty()
    st.session_state["search_executed"] = True

if not tle_ready:
    st.error("กรุณาตรวจสอบข้อมูล TLE ดาวเทียมใน Sidebar ก่อนเริ่มค้นหา")
elif not st.session_state.get("search_executed"):
    st.info("กด '🔍 ค้นหารอบการบินผ่านทั้งหมด' เพื่อเริ่มค้นหารอบวงโคจรตามเงื่อนไขที่ตั้งค่าไว้")
else:
    try:
        satellite = EarthSatellite(tle_line1, tle_line2, tle_name, ts)
        results = st.session_state.get("gs_search_results", [])

        # ---- เช็คความสดของ TLE ก่อนแสดงผล ----
        # เทียบกับ "วันที่ปลายสุดของทุกช่วงที่ค้นหา" ไม่ใช่แค่วันนี้ เพราะถ้าเลือกหลายเดือน/ทั้งปี
        # ต้องมั่นใจว่า TLE ยังแม่นยำพอตลอดช่วงที่ไกลที่สุดที่จะทำนาย
        if results:
            reference_check_dt = max(r["start"] + timedelta(hours=r["duration"]) for r in results)
        else:
            reference_check_dt = now_utc_for_ui
        age_days, freshness_level, epoch_dt = check_tle_freshness(satellite, reference_check_dt)

        if freshness_level == "danger":
            st.error(
                f"🚫 TLE นี้มี epoch วันที่ {epoch_dt.strftime('%Y-%m-%d')} ห่างจากช่วงที่จะคำนวณ "
                f"ประมาณ {age_days} วัน — เกินขีดจำกัดที่ TLE ของดาวเทียมวงโคจรต่ำ (LEO) จะแม่นยำได้ "
                f"(ปกติแม่นยำจริงแค่ ~1-2 สัปดาห์รอบ epoch เพราะแรงต้านบรรยากาศ) "
                f"**ผลลัพธ์ที่ได้จะไม่น่าเชื่อถือ** กรุณาดึง TLE ใหม่จาก CelesTrak ก่อนใช้งาน"
            )
        elif freshness_level == "warn":
            st.warning(
                f"⚠️ TLE นี้มี epoch วันที่ {epoch_dt.strftime('%Y-%m-%d')} ห่างจากช่วงที่จะคำนวณ "
                f"ประมาณ {age_days} วัน ความแม่นยำอาจเริ่มลดลง แนะนำให้อัปเดต TLE ให้ใหม่ที่สุดก่อนใช้งานจริง"
            )
        else:
            st.success(f"✅ TLE สดพอสำหรับการคำนวณ (epoch {epoch_dt.strftime('%Y-%m-%d')}, ห่าง {age_days} วัน)")

        total_passes = sum(len(r["passes"]) for r in results)

        if total_passes == 0:
            st.warning(
                f"❌ ไม่พบรอบที่ดาวเทียมบินผ่าน Ground Station "
                f"(Elevation ≥ {min_el_gs}°) ในทุกช่วงที่ค้นหา "
                f"ลองปรับลด Elevation Threshold หรือเลือกเดือน/ช่วงเวลาอื่น"
            )
        else:
            combined_for_select = []  # [(range_dict, pass_index, pass_dict), ...]

            if len(results) == 1 and results[0].get("month") is None:
                # โหมด "ระบุจำนวนชั่วโมงล่วงหน้า" — แสดงตารางเดียวเหมือนเดิม
                r = results[0]
                st.markdown(
                    f"##### 📡 รายการรอบที่ผ่านสถานีภาคพื้นดิน (Ground Station Passes) — พบทั้งหมด {len(r['passes'])} รอบ"
                )
                if r["passes"]:
                    gs_df = passes_to_dataframe(r["passes"], "Max Time")
                    st.dataframe(gs_df, use_container_width=True, hide_index=True)
                    for i, p in enumerate(r["passes"]):
                        combined_for_select.append((r, i, p))
                else:
                    st.info("ไม่พบรอบในช่วงเวลาที่เลือก")
            else:
                # โหมดหลายเดือน — แสดงเป็นแท็บแยกตามเดือน ดูของแต่ละเดือนได้พร้อมกันโดยไม่ต้องค้นหาใหม่
                st.markdown(
                    f"##### 📡 รายการรอบที่ผ่านสถานีภาคพื้นดิน (Ground Station Passes) — "
                    f"พบทั้งหมด {total_passes} รอบ ใน {len(results)} เดือน"
                )
                tab_labels = [f"{r['label']} ({len(r['passes'])})" for r in results]
                tabs = st.tabs(tab_labels)
                for tab, r in zip(tabs, results):
                    with tab:
                        if not r["passes"]:
                            st.info(f"ไม่พบรอบในเดือน {r['label']}")
                        else:
                            df = passes_to_dataframe(r["passes"], "Max Time")
                            st.dataframe(
                                df, use_container_width=True, hide_index=True,
                                height=min(430, 40 + 35 * len(df)),
                            )
                    for i, p in enumerate(r["passes"]):
                        combined_for_select.append((r, i, p))

            st.markdown("---")

            # ---- เลือกรอบ Uplink จากรายการรวมทุกช่วง/ทุกเดือนที่ค้นหา ----
            combined_for_select.sort(key=lambda x: x[2]["max"])
            pass_labels = [
                f"[{r['label']}] Pass #{i + 1}: {p['max'].strftime('%Y-%m-%d %H:%M:%S UTC')} | Max El: {p['max_el']:.1f}°"
                for r, i, p in combined_for_select
            ]
            selected_combo_idx = st.selectbox(
                "👉 เลือกรอบที่ต้องการ Uplink สัญญาณ (จากทุกช่วง/เดือนที่ค้นหา):",
                range(len(pass_labels)),
                format_func=lambda i: pass_labels[i],
            )
            selected_gs_pass = combined_for_select[selected_combo_idx][2]
            gs_max_dt = selected_gs_pass["max"]

            st.markdown("---")

            # ---- 2. ค้นหาทุก Target Pass ที่เกิดขึ้นหลังจากเวลา Uplink ที่เลือก ----
            target_passes = find_all_passes(
                satellite, target_lat, target_lon, 0.0, gs_max_dt,
                duration_hours=duration_hours_target, min_el=min_el_target,
            )

            if not target_passes:
                st.warning(
                    f"❌ ไม่พบรอบที่ดาวเทียมบินผ่านพิกัดเป้าหมาย (Elevation ≥ {min_el_target}°) "
                    f"หลังจากเวลา Uplink ที่เลือกไว้"
                )
            else:
                st.markdown("##### 🎯 รายการรอบที่ผ่านเป้าหมายหลังเวลา Uplink (Target Passes)")
                st.caption(
                    "คำนวณจาก SGP4 จริง (ไม่ใช่ข้อมูลจำลอง) แต่ความแม่นยำขึ้นกับความสดของ TLE ด้านบน — "
                    "ดูสถานะ TLE ก่อนเชื่อผลลัพธ์นี้ 100%"
                )
                st.dataframe(
                    passes_to_dataframe(target_passes, "Target Time"),
                    use_container_width=True, hide_index=True,
                )

                target_labels = [
                    f"Pass #{i+1}: {p['max'].strftime('%Y-%m-%d %H:%M:%S UTC')} | Max El: {p['max_el']:.1f}°"
                    for i, p in enumerate(target_passes)
                ]
                selected_target_idx = st.selectbox(
                    "👉 เลือกรอบเป้าหมายที่ต้องการปฏิบัติงานถ่ายภาพ:",
                    range(len(target_labels)),
                    format_func=lambda i: target_labels[i],
                )
                selected_target_pass = target_passes[selected_target_idx]
                target_max_dt = selected_target_pass["max"]

                st.markdown("---")

                # =============================================================
                # CALCULATION & OUTPUT
                # =============================================================
                if st.button("🚀 คำนวณค่า Command Delay สำหรับรอบที่เลือก", type="primary", use_container_width=True):
                    distance_km = selected_gs_pass["distance_km"]
                    propagation_delay_sec = distance_km / SPEED_OF_LIGHT_KM_S
                    delta_t_sec = (target_max_dt - gs_max_dt).total_seconds()
                    total_command_delay_sec = delta_t_sec - propagation_delay_sec
                    hms_str = format_hms(total_command_delay_sec)
                    target_ict_dt = target_max_dt.astimezone(ICT_TZ)

                    if total_command_delay_sec < 0:
                        st.error(
                            "⚠️ คำนวณได้ค่า Delay ติดลบ — แปลว่ารอบเป้าหมายที่เลือกเกิดขึ้น "
                            "เร็วกว่าเวลาที่สัญญาณเดินทางไปถึงดาวเทียม กรุณาเลือกรอบเป้าหมายอื่น หรือรอบ Uplink อื่น"
                        )

                    st.markdown("### 📊 ผลลัพธ์การคำนวณ (Command Delay Results)")
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric(label="🎯 COMMAND DELAY (วินาที)", value=f"{total_command_delay_sec:.3f} s")
                    with m2:
                        st.metric(label="⏱️ FORMATTED DELAY (HH:MM:SS)", value=hms_str)

                    st.markdown("#### 📑 สรุปรายละเอียดภารกิจ (Mission Execution Summary)")
                    res_col1, res_col2 = st.columns(2)

                    with res_col1:
                        st.markdown(f"""
                        * **สถานีภาคพื้นดิน:** `{gs_choice}`
                        * **เวลาเริ่ม Uplink (UTC):** `{gs_max_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}`
                        * **เวลาเริ่ม Uplink (ICT):** `{gs_max_dt.astimezone(ICT_TZ).strftime('%Y-%m-%d %H:%M:%S ICT')}`
                        * **ระยะทาง Uplink:** `{distance_km:.2f} km` (Propagation Delay: `{propagation_delay_sec:.6f} s`)
                        """)

                    with res_col2:
                        st.markdown(f"""
                        * **พิกัดเป้าหมาย:** Lat `{target_lat:.4f}`, Lon `{target_lon:.4f}`
                        * **เวลาผ่านเป้าหมายจริง (UTC):** `{target_max_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}`
                        * **เวลาผ่านเป้าหมายจริง (ICT):** `{target_ict_dt.strftime('%Y-%m-%d %H:%M:%S ICT')}`
                        * **มุม Elevation เหนือเป้าหมาย:** `{selected_target_pass['max_el']:.2f}°`
                        """)

                    if selected_target_pass["max_el"] < 15.0:
                        st.warning(
                            f"⚠️ **ข้อควรระวัง:** รอบเป้าหมายที่เลือกมีมุม Elevation ต่ำ "
                            f"({selected_target_pass['max_el']:.2f}°) ภาพถ่ายอาจเกิด distortion "
                            f"จากชั้นบรรยากาศหรือถูกสิ่งกีดขวางบัง"
                        )
                    else:
                        st.success("✅ มุม Elevation อยู่ในเกณฑ์ดีเยี่ยมสำหรับการปฏิบัติภารกิจถ่ายภาพ")

                    if freshness_level != "ok":
                        st.info(
                            "ℹ️ อย่าลืม: ผลลัพธ์นี้อ้างอิงจาก TLE ที่มีสถานะ "
                            f"'{ 'ควรอัปเดต' if freshness_level=='warn' else 'เก่าเกินไป' }' ตามที่แจ้งด้านบน"
                        )

    except Exception:
        st.error("❌ เกิดข้อผิดพลาดในการคำนวณ กรุณาตรวจสอบข้อมูล TLE และพิกัดที่กรอก")
        with st.expander("🛠️ รายละเอียดข้อผิดพลาดสำหรับ Developer (Traceback)"):
            st.code(traceback.format_exc())

# =============================================================================
# 4. REAL-TIME ORBIT TRACKER (พอร์ตมาจากสคริปต์ MATLAB satelliteScenario)
# =============================================================================
st.markdown("---")
st.subheader("🌍 4. ติดตามตำแหน่งดาวเทียมแบบเรียลไทม์ (Real-Time Orbit Tracker)")
st.caption(
    "คำนวณตำแหน่งปัจจุบันและเส้นทางวงโคจร (Ground Track) จาก TLE เดียวกับที่ตั้งค่าไว้ใน Sidebar ด้วย SGP4 จริง — "
    "แนวคิดเทียบเท่าสคริปต์ MATLAB ที่ใช้ satelliteScenario + satelliteScenarioViewer แต่ทำงานใน Python/Streamlit ล้วนๆ"
)

if not tle_ready:
    st.info("กรุณาตั้งค่า TLE ให้พร้อมใน Sidebar (หัวข้อ 1) ก่อน จึงจะติดตามตำแหน่งแบบเรียลไทม์ได้")
else:
    track_col1, track_col2, track_col3 = st.columns([2, 1, 1])
    with track_col1:
        auto_refresh_on = st.checkbox("🔄 อัปเดตอัตโนมัติ (Auto-refresh)", value=False, key="orbit_autorefresh_toggle")
    with track_col2:
        refresh_sec = st.number_input(
            "ทุกกี่วินาที:", min_value=1, max_value=60, value=5, key="orbit_refresh_sec"
        )
    with track_col3:
        st.write("")
        st.write("")
        st.button("↻ รีเฟรชตำแหน่งตอนนี้", key="orbit_manual_refresh", use_container_width=True)

    # หมายเหตุ: เดิมใช้ streamlit_autorefresh ซึ่งสั่ง rerun ทั้งหน้า (ทั้ง sidebar, ส่วนคำนวณ
    # Command Delay ด้านบน ฯลฯ) ทุกครั้ง ทำให้ทั้งหน้าจอกระพริบ/รีเซ็ต scroll position
    # เปลี่ยนมาใช้ @st.fragment(run_every=...) แทน — Streamlit จะ rerun เฉพาะฟังก์ชันนี้
    # (metric + แผนที่) เท่านั้น ส่วนอื่นของหน้าไม่ถูกวาดซ้ำ จึงไม่กระพริบ
    # ต้องใช้ Streamlit >= 1.37 (ถ้าเวอร์ชันเก่ากว่านี้ ให้ pip install -U streamlit)
    @st.fragment(run_every=refresh_sec if auto_refresh_on else None)
    def render_orbit_tracker():
        try:
            track_satellite = EarthSatellite(tle_line1, tle_line2, tle_name, ts)
            now_dt = datetime.now(timezone.utc)
            lat_now, lon_now, alt_now_km, speed_now_kms = get_current_state(track_satellite, now_dt)
            period_min = get_orbital_period_minutes(track_satellite)

            m1c, m2c, m3c, m4c = st.columns(4)
            m1c.metric("🕒 เวลา (UTC)", now_dt.strftime("%H:%M:%S"))
            m2c.metric("📍 Lat / Lon", f"{lat_now:.2f}°, {lon_now:.2f}°")
            m3c.metric("📏 ความสูง (Alt)", f"{alt_now_km:,.1f} km")
            m4c.metric("🚀 ความเร็ว", f"{speed_now_kms:.2f} km/s")
            st.caption(
                f"ดาวเทียม: `{tle_name}` | คาบวงโคจรโดยประมาณ: {period_min:.1f} นาที "
                f"— เส้นทางด้านล่างแสดงล่วงหน้า ~1 รอบวงโคจรนับจากตอนนี้"
            )

            track_points = compute_ground_track(track_satellite, now_dt, period_min)

            map_tab, globe_tab = st.tabs(["🗺️ แผนที่ 2 มิติ (Ground Track)", "🌐 มุมมอง 3 มิติ (Globe View)"])

            with map_tab:
                map_fig = build_ground_track_figure(
                    lat_now, lon_now, track_points, tle_name,
                    hover_text=(
                        f"{tle_name}<br>{now_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}<br>"
                        f"Alt: {alt_now_km:.1f} km"
                    ),
                )
                st.plotly_chart(map_fig, use_container_width=True, key="orbit_track_map")

            with globe_tab:
                fig = build_globe_figure(track_satellite, now_dt, track_points, tle_name)
                st.plotly_chart(fig, use_container_width=True, key="orbit_globe_fig")
                st.caption("หมุน/ซูมได้ด้วยเมาส์ — จุดสีแดงคือตำแหน่งดาวเทียมปัจจุบัน เส้นสีฟ้าคือเส้นทางวงโคจรล่วงหน้า")

        except Exception:
            st.error("❌ เกิดข้อผิดพลาดในการคำนวณตำแหน่งเรียลไทม์ของดาวเทียม")
            with st.expander("🛠️ รายละเอียดข้อผิดพลาดสำหรับ Developer (Traceback)"):
                st.code(traceback.format_exc())

    render_orbit_tracker()

#python -m streamlit run "c:/Users/ACER/OneDrive - kmutnb.ac.th/Desktop/Command Delay Calculator/app.py