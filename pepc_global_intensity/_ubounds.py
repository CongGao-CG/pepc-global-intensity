def get_intensity_ubounds_dict(basin):

    basins  = {
        'AS': 155.0,
        'BoB': 170.0,
        'WNP': 185.0,
        'ENP': 170.0,
        'NA':  175.0,
        'SI':  170.0,
        'SP':  170.0
    }
    return basins[basin]
