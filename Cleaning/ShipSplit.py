import os
import dask
import dask.dataframe as dd
from dask import delayed
from distributed import Client, LocalCluster
import logging
from main import DTYPES

# Configure logging to suppress shuffle warnings
logging.getLogger('distributed.shuffle._scheduler_plugin').setLevel(logging.ERROR)

# Dictionary to map ship types to folder names
SHIP_TYPE_FOLDERS = {
    'Cargo': 'Cargo_Ships',
    'Fishing': 'Fishing_Ships',
    'Pleasure craft': 'Pleasure_Ships',
    'Pleasure Craft': 'Pleasure_Ships',
    'Class B': 'Pleasure_Ships',
    'Helicopter': 'Helicopters',
    'Passenger': 'Passenger_Ships',
    'Tanker': 'Tanker_Ships',
    'High speed': 'High_Speed_Vessels',
    'Tug': 'Tugs',
    'Military': 'Military_Ships',
    'Law enforcement': 'Law_Enforcement',
    'Medical': 'Medical_Transport',
    'Pilot': 'Pilot_Vessels',
    'Search and rescue': 'SAR_Vessels',
    'Port tender': 'Port_Service',
    'Anti pollution': 'Special_Purpose',
    'WIG': 'WIG_Vessels',              
    'Dredging': 'Dredging_Vessels',
    'Diving': 'Diving_Operations',
    'Towing': 'Towing_Vessels',
    'Sailing': 'Sailing_Vessels',      
    'Research': 'Research_Vessels',
    'Survey': 'Survey_Vessels',
    'Reserved': 'Reserved_Vessels',
    'Supply': 'Supply_Vessels',        
    'HSC': 'High_Speed_Vessels',       
    'SAR': 'SAR_Vessels',              
    'Wing in ground': 'WIG_Vessels'
}

def get_ship_type_folder(ship_type):
    """Determine the appropriate folder based on ship type"""
    for key, folder in SHIP_TYPE_FOLDERS.items():
        if key.lower() in str(ship_type).lower():
            return folder
    return "Other_Ships"

def save_mmsi_to_csv(df, mmsi, output_folder):
    """Save data for a specific MMSI to a CSV file in the appropriate folder"""
    mmsi_df = df[df['MMSI'] == mmsi].sort_values(by='# Timestamp')
    
    ship_type = "Unknown"
    
    if 'Ship type' in mmsi_df.columns and not mmsi_df.empty:
        non_null = mmsi_df[mmsi_df['Ship type'].notna()]
        
        if not non_null.empty:
            defined_ships = non_null[~non_null['Ship type'].str.lower().isin(["undefined", "unknown", ""])]
            if not defined_ships.empty:
                ship_type = defined_ships['Ship type'].iloc[0]
            else:
                ship_type = non_null['Ship type'].iloc[0]
    
    folder = get_ship_type_folder(ship_type)
    folder_path = os.path.join(output_folder, folder)
    os.makedirs(folder_path, exist_ok=True)
    mmsi_output_path = os.path.join(folder_path, f"{mmsi}.csv")
    mmsi_df.to_csv(mmsi_output_path, index=False)


def split_ships(removed_data=None):
    input_folder = '/ceph/project/P6-data-cleaning/sprint1/Ships_data2'
    output_folder = '/ceph/project/P6-data-cleaning/sprint1/Ships_data'
    os.makedirs(output_folder, exist_ok=True)
    csv_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {input_folder}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    all_tasks = []
    for input_file in csv_files:
        print(f"Reading data from {input_file}...")
        df = dd.read_csv(input_file, dtype=DTYPES, assume_missing=True)    
        partitions = df.npartitions
        print(f"Processing {partitions} partitions for {input_file}")
        for i in range(partitions):
            print(f"  Processing partition {i+1}/{partitions}")
            partition = df.get_partition(i).compute()
            unique_mmsi_in_partition = partition['MMSI'].unique()
            print(f"  Found {len(unique_mmsi_in_partition)} unique ships in partition {i+1}")
            for mmsi in unique_mmsi_in_partition:
                mmsi_df = partition[partition['MMSI'] == mmsi]
                if not mmsi_df.empty:
                    save_mmsi_to_csv(mmsi_df, mmsi, output_folder)

    total_files = 0
    for root, dirs, files in os.walk(output_folder):
        if files:
            folder_name = os.path.basename(root)
            file_count = len(files)
            print(f"  {folder_name}: {file_count} files")
            total_files += file_count
    
    print(f"Total ship files created: {total_files}")


def analyze_ship_types(input_folder, sample_limit=10):
    """Analyze ship types in the data to identify missing categories"""
    csv_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) 
                if f.endswith('.csv')][:sample_limit]
    
    ship_types = set()
    
    for input_file in csv_files:
        print(f"Analyzing ship types in {input_file}...")
        df = dd.read_csv(input_file, assume_missing=True)
        if 'Ship type' in df.columns:
            types = df['Ship type'].dropna().unique().compute().tolist()
            ship_types.update(types)
    
    print("Found ship types:")
    for st in sorted(ship_types):
        print(f"  - {st}")
    return ship_types

if __name__ == "__main__":
    cluster = LocalCluster(n_workers=8, threads_per_worker=15, memory_limit="300GB")
    client = Client(cluster)
    found_types = analyze_ship_types('/ceph/project/P6-data-cleaning/sprint1/Ships_data2', 20)
    for ship_type in found_types:
        if ship_type and ship_type.lower() not in [k.lower() for k in SHIP_TYPE_FOLDERS.keys()]:
            print(f"Adding new ship type to mapping: {ship_type}")
            SHIP_TYPE_FOLDERS[ship_type] = f"{ship_type.replace(' ', '_')}_Vessels"
    
    split_ships()
    client.close()