import streamlit as st
import ee
import folium
from streamlit_folium import st_folium
import geemap.foliumap as gmapimport geopandas as gpd
import tempfile
import os
import math
import zipfile
import json
import google.auth
from google.oauth2 import service_account

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Parcel Site Characterization", layout="wide")
st.title("🗺️ Parcel Site Characterization Tool")
st.write("Upload a zipped shapefile to generate a site report and map.")

# ── GEE Auth ──────────────────────────────────────────────────────────────────
@st.cache_resource
def init_gee():
    key = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
    credentials = service_account.Credentials.from_service_account_info(
        key, scopes=["https://www.googleapis.com/auth/earthengine"]
    )
    ee.Initialize(credentials)

init_gee()

# ── File Upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload zipped shapefile (.zip containing .shp .dbf .shx .prj)",
    type="zip"
)

if uploaded:
    with tempfile.TemporaryDirectory() as tmpdir:

        # Unzip
        zip_path = os.path.join(tmpdir, "parcel.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded.read())
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmpdir)

        # Find .shp
        shp_files = [f for f in os.listdir(tmpdir) if f.endswith(".shp")]
        if not shp_files:
            st.error("No .shp file found in the zip.")
            st.stop()

        shp_path = os.path.join(tmpdir, shp_files[0])

        with st.spinner("Loading shapefile and running analysis — 30–60 seconds..."):

            # Load shapefile with pyogrio
            gdf = gpd.read_file(shp_path, engine="pyogrio")
            if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
            geojson = json.loads(gdf.to_json())
            parcel   = ee.FeatureCollection(geojson)
            geometry = parcel.geometry()

            # ── Centroid ──────────────────────────────────────────────────────
            coords   = geometry.centroid(maxError=1).coordinates().getInfo()
            lon, lat = coords[0], coords[1]

            # ── Elevation & Terrain ───────────────────────────────────────────
            dem       = ee.Image("USGS/SRTMGL1_003").clipToCollection(parcel)
            slope     = ee.Terrain.slope(dem)
            aspect    = ee.Terrain.aspect(dem)
            hillshade = ee.Terrain.hillshade(dem, 315, 45)
            slope_rad = slope.multiply(math.pi / 180)
            twi       = dem.focalMean(radius=3, kernelType="square") \
                          .divide(slope_rad.tan().max(0.001)).log().rename("twi")

            elev_stats = dem.reduceRegion(
                reducer=ee.Reducer.minMax().combine(ee.Reducer.mean(), sharedInputs=True),
                geometry=geometry, scale=30, maxPixels=1e9
            ).getInfo()

            slope_stats = slope.reduceRegion(
                reducer=ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True),
                geometry=geometry, scale=30, maxPixels=1e9
            ).getInfo()

            twi_stats = twi.reduceRegion(
                reducer=ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True),
                geometry=geometry, scale=30, maxPixels=1e9
            ).getInfo()

            aspect_mean = round(
                aspect.reduceRegion(ee.Reducer.mean(), geometry, 30, maxPixels=1e9)
                .getInfo().get("aspect", 0), 1)

            elev_min   = round(elev_stats["elevation_min"], 1)
            elev_max   = round(elev_stats["elevation_max"], 1)
            elev_mean  = round(elev_stats["elevation_mean"], 1)
            elev_range = round(elev_max - elev_min, 1)
            slope_mean = round(slope_stats["slope_mean"], 1)
            slope_max  = round(slope_stats["slope_max"], 1)
            twi_mean   = round(twi_stats.get("twi_mean", 0), 2)
            twi_max    = round(twi_stats.get("twi_max", 0), 2)

            dirs       = ["N","NE","E","SE","S","SW","W","NW","N"]
            aspect_dir = dirs[round(aspect_mean / 45) % 8]

            slope_total = slope.reduceRegion(
                ee.Reducer.count(), geometry, 30, maxPixels=1e9
            ).getInfo().get("slope", 0)

            def slope_pct(low, high):
                if slope_total == 0: return 0.0
                mask  = slope.gt(low) if high is None else slope.gt(low).And(slope.lte(high))
                count = mask.reduceRegion(ee.Reducer.sum(), geometry, 30, maxPixels=1e9).getInfo().get("slope", 0)
                return round((count / slope_total) * 100, 1)

            pct_flat       = slope_pct(0, 5)
            pct_gentle     = slope_pct(5, 15)
            pct_moderate   = slope_pct(15, 30)
            pct_steep      = slope_pct(30, 45)
            pct_very_steep = slope_pct(45, None)

            # ── Canopy Height (Meta) ──────────────────────────────────────────
            band = "cover_code"
            ch   = ee.ImageCollection("projects/sat-io/open-datasets/facebook/meta-canopy-height") \
                     .filterBounds(geometry).mosaic().clip(geometry)

            ch_stats = ch.reduceRegion(
                reducer=ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True)
                    .combine(ee.Reducer.percentile([25, 50, 75]), sharedInputs=True),
                geometry=geometry, scale=10, maxPixels=1e9
            ).getInfo()

            ch_total = ch.reduceRegion(
                ee.Reducer.count(), geometry, 10, maxPixels=1e9
            ).getInfo().get(band, 0)

            def ch_pct(low, high):
                if ch_total == 0: return 0.0
                if low == 0 and high == 0: mask = ch.eq(0)
                elif high is None:         mask = ch.gt(low)
                else:                      mask = ch.gt(low).And(ch.lte(high))
                count = mask.reduceRegion(ee.Reducer.sum(), geometry, 10, maxPixels=1e9).getInfo().get(band, 0)
                return round((count / ch_total) * 100, 1)

            ch_mean   = round(ch_stats.get(f"{band}_mean", 0), 1)
            ch_max    = round(ch_stats.get(f"{band}_max", 0), 1)
            ch_p25    = round(ch_stats.get(f"{band}_p25", 0), 1)
            ch_p50    = round(ch_stats.get(f"{band}_p50", 0), 1)
            ch_p75    = round(ch_stats.get(f"{band}_p75", 0), 1)
            pct_bare  = ch_pct(0, 0)
            pct_low   = ch_pct(0, 5)
            pct_mid   = ch_pct(5, 15)
            pct_tall  = ch_pct(15, 30)
            pct_vtall = ch_pct(30, None)

            # ── Dynamic World ─────────────────────────────────────────────────
            label    = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1") \
                         .filterBounds(geometry).filterDate("2024-05-01","2025-05-01") \
                         .median().select("label")
            dw_total = label.reduceRegion(
                ee.Reducer.count(), geometry, 10, maxPixels=1e9
            ).getInfo().get("label", 0)

            def dw_pct(c):
                if dw_total == 0: return 0.0
                count = label.eq(c).reduceRegion(
                    ee.Reducer.sum(), geometry, 10, maxPixels=1e9
                ).getInfo().get("label", 0)
                return round((count / dw_total) * 100, 1)

            pct_trees   = dw_pct(1)
            pct_grass   = dw_pct(2)
            pct_shrub   = dw_pct(5)
            pct_water   = dw_pct(0)
            pct_dw_bare = dw_pct(7)

            # ── ESA WorldCover (built) ─────────────────────────────────────────
            wc         = ee.ImageCollection("ESA/WorldCover/v200").first().clip(geometry)
            built_mask = wc.eq(50)
            wc_total   = wc.reduceRegion(
                ee.Reducer.count(), geometry, 10, maxPixels=1e9
            ).getInfo().get("Map", 0)
            wc_built   = built_mask.reduceRegion(
                ee.Reducer.sum(), geometry, 10, maxPixels=1e9
            ).getInfo().get("Map", 0)
            pct_built  = round((wc_built / wc_total) * 100, 1) if wc_total > 0 else 0.0

            # ── Soil ──────────────────────────────────────────────────────────
            def soil_mean(img, b):
                return round(img.reduceRegion(
                    ee.Reducer.mean(), geometry, 250, maxPixels=1e9
                ).getInfo()[b], 1)

            clay_surf = soil_mean(ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select("b0"),  "b0")
            clay_sub  = soil_mean(ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select("b30"), "b30")
            sand_surf = soil_mean(ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02").select("b0"),  "b0")
            sand_sub  = soil_mean(ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02").select("b30"), "b30")

            # ── Climate ───────────────────────────────────────────────────────
            wc_stats      = ee.Image("WORLDCLIM/V1/BIO").reduceRegion(
                ee.Reducer.mean(), geometry, 1000, maxPixels=1e9
            ).getInfo()
            mean_temp     = round((wc_stats.get("bio01", 0) or 0) / 10, 1)
            annual_precip = int(wc_stats.get("bio12", 0) or 0)
            precip_seas   = round(wc_stats.get("bio15", 0) or 0, 1)

            # ── Wildfire ──────────────────────────────────────────────────────
            burned      = ee.ImageCollection("MODIS/061/MCD64A1") \
                            .filterBounds(geometry).filterDate("2000-01-01","2025-01-01") \
                            .select("BurnDate")
            ever_burned = burned.max().gt(0).clip(geometry)
            pct_burned  = round(
                (ever_burned.reduceRegion(
                    ee.Reducer.mean(), geometry, 500, maxPixels=1e9
                ).getInfo().get("BurnDate", 0) or 0) * 100, 1)

            # ── Distance to water ─────────────────────────────────────────────
            dist_water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater") \
                           .select("occurrence").gt(10) \
                           .fastDistanceTransform().sqrt() \
                           .multiply(ee.Image.pixelArea().sqrt()).clip(geometry)
            min_dist   = round(
                dist_water.reduceRegion(
                    ee.Reducer.min(), geometry, 30, maxPixels=1e9
                ).getInfo().get("occurrence", 0) or 0, 0)

        # ── Map ───────────────────────────────────────────────────────────────
        st.subheader("Map")
        layer_choice = st.selectbox("Select layer to display", [
            "Canopy Height (Meta)",
            "Slope",
            "Wetness (TWI)",
            "Elevation",
            "Land Cover (Dynamic World)",
            "Built (ESA WorldCover)",
            "Ever Burned (MODIS)"
        ])

        Map = gmap.Map(center=[lat, lon], zoom=15)
        Map.add_basemap("SATELLITE")TE")

        parcel_vis = {"color": "000000", "fillColor": "00000000", "width": 1}

        if layer_choice == "Canopy Height (Meta)":
            Map.addLayer(ch, {"min":0,"max":30,"palette":["ffffff","a8dda8","2d6a2d"]}, "Canopy Height")
        elif layer_choice == "Slope":
            Map.addLayer(slope.clip(geometry), {"min":0,"max":45,"palette":["00ff00","ffff00","ff0000"]}, "Slope")
        elif layer_choice == "Wetness (TWI)":
            Map.addLayer(twi.clip(geometry), {"min":0,"max":20,"palette":["ffffff","0066cc"]}, "TWI")
        elif layer_choice == "Elevation":
            Map.addLayer(dem, {"min":elev_min,"max":elev_max,"palette":["006633","E5FFCC","662A00","D8D8D8","F5F5F5"]}, "Elevation")
        elif layer_choice == "Land Cover (Dynamic World)":
            Map.addLayer(label.clip(geometry), {"min":0,"max":8,"palette":["419BDF","397D49","88B053","7A87C6","E49635","DFC35A","C4281B","A59B8F","B39FE1"]}, "Land Cover")
        elif layer_choice == "Built (ESA WorldCover)":
            Map.addLayer(built_mask.selfMask(), {"palette":["C4281B"]}, "Built")
        elif layer_choice == "Ever Burned (MODIS)":
            Map.addLayer(ever_burned, {"min":0,"max":1,"palette":["ffffff","ff4400"]}, "Ever Burned")

        Map.addLayer(parcel.style(**parcel_vis), {}, "Parcel")
        st_folium(Map, height=500, width=None)

        # ── Report ────────────────────────────────────────────────────────────
        st.subheader("Site Characterization Report")
        st.code(f"""
{'='*52}
         SITE CHARACTERIZATION REPORT
{'='*52}

📍 ELEVATION & TERRAIN
  Low / High:   {elev_min} m → {elev_max} m  (relief: {elev_range} m)
  Mean elev:    {elev_mean} m
  Slope:        mean {slope_mean}°   max {slope_max}°
  Aspect:       {aspect_mean}° ({aspect_dir}-facing)

  Slope breakdown:
    Flat (0–5°):        {pct_flat}%
    Gentle (5–15°):     {pct_gentle}%
    Moderate (15–30°):  {pct_moderate}%  ← erosion-prone
    Steep (30–45°):     {pct_steep}%   ← landslide risk
    Very steep (>45°):  {pct_very_steep}%

  Topographic Wetness Index:
    Mean: {twi_mean}   Max: {twi_max}  (>15 = high drainage accumulation)

🌳 CANOPY HEIGHT  (Meta 2024, 1 m res)
  Mean: {ch_mean} m   Max: {ch_max} m
  Percentiles:  25th={ch_p25} m   50th={ch_p50} m   75th={ch_p75} m

  Height classes:
    Bare (0 m):         {pct_bare}%
    Low (0–5 m):        {pct_low}%   shrubs / regeneration
    Mid (5–15 m):       {pct_mid}%   young / mid-successional
    Tall (15–30 m):     {pct_tall}%   mature canopy
    Very tall (>30 m):  {pct_vtall}%   old growth / emergent

🌿 LAND COVER  (Dynamic World 2024–25)
  Trees:   {pct_trees}%
  Grass:   {pct_grass}%
  Shrub:   {pct_shrub}%
  Water:   {pct_water}%
  Bare:    {pct_dw_bare}%

🏗️  BUILT ENVIRONMENT  (ESA WorldCover 2021)
  Impervious cover:  {pct_built}%  (roads, structures, hardscape)

🪨 SOIL  (SoilGrids 250 m)
  Clay:  surface {clay_surf}%   subsoil {clay_sub}%
  Sand:  surface {sand_surf}%   subsoil {sand_sub}%

🌡️  CLIMATE  (WorldClim 1 km normals)
  Mean annual temp:    {mean_temp} °C
  Annual precip:       {annual_precip} mm
  Precip seasonality:  {precip_seas}

💧 WATER PROXIMITY  (JRC Global Surface Water)
  Nearest permanent/seasonal water:  {int(min_dist)} m

🔥 WILDFIRE  (MODIS 2000–2025, 500 m res)
  Ever burned:  {pct_burned}%

⚠️  VERIFY LOCALLY
  Bedrock:     geoscan.nrcan.gc.ca
  BC geology:  DataBC Geoscience BC
  BC floods:   maps.gov.bc.ca/ess/hm/imap4m
  BC slides:   governmentofbc.maps.arcgis.com
  Seismic:     earthquakescanada.nrcan.gc.ca

{'='*52}
All metrics are reference-level only — verify in field.
{'='*52}
""", language=None)
