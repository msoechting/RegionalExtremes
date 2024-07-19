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
