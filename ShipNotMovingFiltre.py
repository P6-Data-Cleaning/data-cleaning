def filter_moving_ships(cleaned_data):
    moving_ships = cleaned_data[(cleaned_data["SOG"] > 0) & 
                                (cleaned_data["Longitude"] != 0) & 
                                (cleaned_data["Latitude"] != 0)]
    return moving_ships