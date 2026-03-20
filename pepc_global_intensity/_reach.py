def get_intensity_reach_dict(basin):

    basins  = {
        'AS': 60.0,
        'BoB': 60.0,
        'WNP': 60.0,
        'ENP': 60.0,
        'NA':  20.0,
        'SI':  60.0,
        'SP':  65.0
    }
    return basins[basin]
