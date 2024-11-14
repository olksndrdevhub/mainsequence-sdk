


def string_freq_to_time_delta(frequency):
    import datetime
    if "min" in frequency:
        kwargs={"minutes":int(frequency.replace("min",""))}
    elif "minutes" in frequency:
        kwargs={"minutes":int(frequency.replace("minutes",""))}
    elif "minute" in frequency:
        kwargs={"minutes":int(frequency.replace("minute",""))}
    elif "days" in frequency:
        kwargs = {"days": int(frequency.replace("days", ""))}
    elif "day" in frequency:
        kwargs = {"days": int(frequency.replace("day", ""))}
    else:
        raise NotImplementedError

    time_delta=datetime.timedelta(**kwargs)
    return time_delta

def string_frequency_to_minutes(frequency):
    if "min" in frequency:
        minutes= int(frequency.replace("min",""))
    elif "minutes" in frequency:
        minutes= int(frequency.replace("minutes",""))
    elif "minute" in frequency:
        minutes= int(frequency.replace("minute",""))
    elif "days" in frequency:
        minutes=int(frequency.replace("days",""))*24*60
    elif "day" in frequency:
        minutes = int(frequency.replace("day", "")) * 24 * 60
    else:
        raise NotImplementedError

    return minutes