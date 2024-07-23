import datetime


def printt(message: str):
    """
    Small function to track the different step of the program with the time
    """
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}")


def int_or_none(value):
    if value.lower() == "none":
        return None
    return int(value)


def is_in_europe(lon, lat):
    """
    Check if the given longitude and latitude are within the bounds of Europe.
    """
    # Define Europe boundaries (these are approximate)
    lon_min, lon_max = -31.266, 39.869  # Longitude boundaries
    lat_min, lat_max = 27.636, 81.008  # Latitude boundaries

    # Check if the point is within the defined boundaries
    in_europe = (
        (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
    )
    return in_europe
