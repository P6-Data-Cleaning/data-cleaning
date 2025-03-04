import dask.dataframe as dd

def cargo_filter(df):
    # Filter ships with cargo
    classA = df[df['Type of mobile'] == "Class A"]

    #list
    shipTypes = ['Cargo', 'Tanker', 'Passenger', 'Reserved', 'Undefined']

    selectedShipTypes = classA[classA['Ship type'].isin(shipTypes)]
    
    # Reset index to ensure MMSI is only in columns, not in index
    selectedShipTypes = selectedShipTypes.reset_index(drop=True)

    # Get counts per MMSI and ship type combination
    mmsi_type_counts = (selectedShipTypes
                       .groupby('MMSI')['Ship type']
                       .agg(list, meta=('Ship type', 'object'))
                       .reset_index())

    def get_replacement_type(types):
        if 'Undefined' in types and len(types) > 1:
            replacement = next((t for t in types if t != 'Undefined'), None)
            return replacement
        return None

    
    mmsi_type_counts['replacement_type'] = mmsi_type_counts['Ship type'].map(
        get_replacement_type, 
        meta=('replacement_type', 'object')
    )
    
    # Keep only the MMSI and needs_cleaning columns for the merge
    cleaning_flags = mmsi_type_counts[['MMSI', 'replacement_type']]
    
    # Merge the cleaning flags back to the original data
    merged_data = selectedShipTypes.merge(cleaning_flags, on='MMSI', how='left')

    def update_ship_type(row):
        if row['Ship type'] == 'Undefined' and row['replacement_type'] is not None:
            return row['replacement_type']
        return row['Ship type']
    
    merged_data['Ship type'] = merged_data.apply(
        update_ship_type,
        axis=1,
        meta=('Ship type', 'object')
    )

    # Filter out rows where needs_cleaning is True AND Ship type is Undefined
    cleaned_data = merged_data[merged_data['Ship type'] != 'Undefined']
    
    # Drop the temporary column
    cleaned_data = cleaned_data.drop('replacement_type', axis=1)

    # Return the cleaned data
    return cleaned_data