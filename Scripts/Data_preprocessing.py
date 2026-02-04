#==============================================================================
# IMPORT LIBRARIES
#==============================================================================
import pandas as pd
import geopandas as gpd
from shapely import wkt, Point
import numpy as np
import matplotlib.pyplot as plt

#==============================================================================
# FILE PATHS
#==============================================================================
print(">>> Loading input files...")
GMM_PATH = "../Data/GMM/GMM4/out_RoadRC_LinkLoadsToD.csv"
LINKS_PATH = "../Data/GMM/GMM4/OUT_ROADRC_LINKS.csv"
CATEGORIES_PATH = "../Data/GMM/GMM4/TEMP_sys_RoadRC_Categories.csv"
REGIONS_PATH = "../Data/GMM/Region_onehot.csv"

#==============================================================================
# GMM TRAFFIC
#==============================================================================
print(">>> Processing GMM traffic data...")
GMM = pd.read_csv(GMM_PATH)
Categories = pd.read_csv(CATEGORIES_PATH)

# --- Category to TrafTypeID map ---
Categories.loc[Categories['TrafTypeID'] < 4, 'TrafTypeID'] = 1
Categories.loc[Categories['TrafTypeID'] == 4, 'TrafTypeID'] = 3
Categories = Categories[['ID', 'TrafTypeID']].rename(columns={'ID': 'CategoryID'})

# --- Merge to get TrafTypeID ---
GMM = GMM.merge(Categories, on='CategoryID', how='left')
GMM['AvgSpeed'] *= GMM['TotalTraf']

# --- Calculate weighted averages of speed ---
print("    Aggregating weighted averages...")
GMM = (
    GMM.groupby(['LinkID', 'TrafTypeID'])
        .agg(AvgSpeed=('AvgSpeed', 'sum'), TotalTraf=('TotalTraf', 'sum'))
        .reset_index()
)
GMM['AvgSpeed'] = GMM['AvgSpeed'] / GMM['TotalTraf']

# --- Pivot ---
print("    Pivoting traffic data (Car/Van vs Truck)...")
def name_map(typ):
    return {1: 'Car_Van', 3: 'Truck'}[typ]

GMM = GMM.pivot_table(index='LinkID', columns='TrafTypeID', values=['AvgSpeed', 'TotalTraf'])
GMM.columns = [f"{val}_{name_map(typ)}" for val, typ in GMM.columns]
GMM = GMM.rename(columns={'TotalTraf_Car_Van': 'Traf_Car_Van', 'TotalTraf_Truck': 'Traf_Truck'}).reset_index()
print(f"    GMM traffic table shape: {GMM.shape}")

#==============================================================================
# GMM LINK INFORMATION
#==============================================================================
print(">>> Loading and processing GMM link geometries...")
LINKS = pd.read_csv(LINKS_PATH)
LINKS = LINKS[['ID', 'LinkTypeID', 'RoadClassID', 'Draw', 'FreeSpeed', 
               'LanesFor', 'LanesBack', 'LaneHCFor', 'LaneHCBack', 
               'Length', 'URBAN', 'Domestic', 'WKT']]

LINKS = LINKS[LINKS['Domestic'] == 'DK'] # Only dainish links
LINKS = LINKS[LINKS['LinkTypeID'] < 90] # Assumption: no traffic incidents occur on ferry links
LINKS['WKT'] = LINKS['WKT'].apply(wkt.loads)
LINKS = gpd.GeoDataFrame(LINKS, geometry='WKT', crs='EPSG:25832').rename(columns={'ID': 'LinkID'})

print(f"    Number of Danish links: {len(LINKS)}")

LINKS_SHAPE = LINKS[['LinkID', 'Draw', 'WKT']] # used later

# Compute lane/geometry info
LINKS_INFO = gpd.GeoDataFrame(LINKS[['LinkID', 'LinkTypeID', 'RoadClassID', 'Draw', 
                                     'FreeSpeed', 'Length', 'URBAN', 'WKT']], 
                              geometry='WKT', crs='EPSG:25832')

LINKS_INFO['Lanes'] = LINKS['LanesFor'] + LINKS['LanesBack']
LINKS_INFO['LaneHC'] = LINKS['LaneHCFor'] + LINKS['LaneHCBack']
LINKS_INFO['Oneway'] = ((LINKS['LanesFor'] > 0) ^ (LINKS['LanesBack'] > 0)).astype(int)


#==============================================================================
# ACCIDENT DATA (UHELDS DATA)
#==============================================================================
print(">>> Loading accident (uheld) data...")
uheld = pd.read_excel('../Data/Uheld/Ulykker 2019-2023.xlsx', sheet_name='Ulykker til GIS')
print(f"    Raw accidents loaded: {len(uheld)}")

# Filter
uheld = uheld[uheld['KRYDS_UHELD'] == 'Nej'] # remove acidents in intersections
years = [2022, 2023] # data period
uheld = uheld[uheld['AAR'].isin(years)]

print("    Filtering out weekend accidents...")
uheld['datetime'] = pd.to_datetime(dict(year=uheld['AAR'], month=uheld['måned'], 
                                        day=uheld['DAG'], hour=uheld['TIME']))
uheld['weekday'] = uheld['datetime'].dt.weekday
uheld = uheld[uheld['weekday'] < 5].copy()
print(f"    Accidents after weekday filter: {len(uheld)}")

# Keep relevant columns
uheld = uheld[['AAR', 'måned', 'DAG', 'TIME', 
               'UH_UHID_UHANTLETTILS', 'UH_UHID_UHANTALVTILS',
               'ANTAL_DRAEBTE', 'ANTAL_TILSKADEKOMNE', 'VEJARBEJDE',
               'X_KOORDINAT', 'Y_KOORDINAT']]

# Convert to GeoDataFrame
uheld = gpd.GeoDataFrame(uheld,
                         geometry=gpd.points_from_xy(uheld['X_KOORDINAT'], uheld['Y_KOORDINAT']),
                         crs="EPSG:25832")
print(f"    Accident GeoDataFrame created with {uheld.shape[0]} rows.")

# Type conversion and damage flags
uheld = uheld.astype({
    'ANTAL_TILSKADEKOMNE': int,
    'UH_UHID_UHANTLETTILS': int,
    'UH_UHID_UHANTALVTILS': int,
    'ANTAL_DRAEBTE': int
})

uheld['Personskade'] = (uheld['ANTAL_DRAEBTE'] + uheld['ANTAL_TILSKADEKOMNE'] > 0)
uheld['Materielskade'] = ~uheld['Personskade']
uheld = uheld.rename(columns={'ANTAL_DRAEBTE': 'fatal',
                              'UH_UHID_UHANTLETTILS': 'slight',
                              'UH_UHID_UHANTALVTILS': 'serious'})

invalid = (~uheld.geometry.is_valid).sum()
print(f"    Invalid geometries: {invalid}")
uheld = uheld[uheld.geometry.is_valid]

# Spatial join
print("    Matching accidents to nearest road links (≤15m)...")
uheld = gpd.sjoin_nearest(uheld, LINKS_SHAPE, how='inner', distance_col='dist_to_road')

thresh = 15
before = len(uheld)
uheld = uheld[uheld['dist_to_road'] <= thresh]
print(f"    Accidents matched within {thresh}m: {len(uheld)} / {before}")

# Map VEJARBEJDE to numeric
uheld['VEJARBEJDE'] = uheld['VEJARBEJDE'].map({'Ja': 1, 'Nej': 0})

# Aggregate
print("    Aggregating by LinkID and year...")
uheld = uheld.groupby(['LinkID', 'AAR']).agg({
    'Personskade': 'sum',
    'Materielskade': 'sum',
    'slight': 'sum',
    'serious': 'sum',
    'fatal': 'sum',
    'VEJARBEJDE': 'mean'
}).reset_index()

#==============================================================================
# MERGE ALL DATASETS
#==============================================================================
print(">>> Merging accidents with GMM and link data...")
temps = []
for year in years:
    temp = uheld.loc[uheld['AAR'] == year].copy()
    temp = pd.merge(LINKS_INFO, temp, on='LinkID', how='left')
    temp = pd.merge(GMM, temp, how='right', on='LinkID')
    temp['AAR'] = year
    temp = temp.fillna(0)
    temps.append(temp)
uheld = pd.concat(temps, ignore_index=True)

# Link grouping
uheld['Link_group_0'] = uheld['LinkTypeID'].isin([1]).astype(int)
uheld['Link_group_1'] = uheld['LinkTypeID'].isin([3, 4, 5, 7, 9]).astype(int)
uheld['Link_group_2'] = uheld['LinkTypeID'].isin([6, 8, 10]).astype(int)
uheld['Link_group_3'] = uheld['LinkTypeID'].isin([2, 11]).astype(int)

uheld['Merging'] *= uheld['Link_group_0']
uheld['Diverging'] *= uheld['Link_group_0']
uheld.loc[uheld['Link_group_0'], 'Oneway'] = 0 # disregard motorways as oneway

# Region merge
REGIONS = pd.read_csv(REGIONS_PATH)
uheld = pd.merge(REGIONS, uheld, how='right', on='LinkID')
print(f"    Final merged dataset shape: {uheld.shape}")

#==============================================================================
# SAVE OUTPUTS
#==============================================================================
print(">>> Saving outputs...")
uheld = uheld[['LinkID', 'AAR', 
               'Personskade', 'Materielskade', 'slight', 'serious', 'fatal', 
               'LinkTypeID', 'Link_group_0', 'Link_group_1','Link_group_2','Link_group_3', 
               'Length', 'Lanes', 'LaneHC', 'FreeSpeed',  'Oneway',
               'AvgSpeed_Car_Van', 'AvgSpeed_Truck', 'Traf_Car_Van', 'Traf_Truck', 
               'URBAN','Region Hovedstaden', 'Region Midtjylland', 'Region Nordjylland', 'Region Sjælland','Region Syddanmark', 
               'Merging', 'Diverging', 'VEJARBEJDE', 'WKT']] 
GDF = gpd.GeoDataFrame(uheld, geometry='WKT', crs='EPSG:25832') 

GDF.to_file('../Data/Model_Input/Uheld_LINKS.gpkg', driver='GPKG') 
uheld = uheld[['LinkID', 'AAR', 
               'Personskade', 'Materielskade', 'slight', 'serious', 'fatal', 
               'LinkTypeID', 'Link_group_0', 'Link_group_1','Link_group_2','Link_group_3', 
               'Length', 'Lanes', 'LaneHC', 'FreeSpeed', 
               'AvgSpeed_Car_Van', 'AvgSpeed_Truck', 'Traf_Car_Van', 'Traf_Truck', 'Oneway',
               'URBAN','Region Hovedstaden', 'Region Midtjylland', 'Region Nordjylland', 'Region Sjælland','Region Syddanmark', 
               'Merging', 'Diverging', 'VEJARBEJDE', 'WKT']] 

uheld.to_csv('../Data/Model_Input/Uheld_LINKS.csv', index=False)
print("✅ Done! Files saved to '../Data/Model_Input/'.")
