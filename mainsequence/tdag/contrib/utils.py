


def transform_frequency_to_seconds(frequency_id:str)->int:
    if frequency_id in ["1min"]:
        return 60
    elif frequency_id in ["5min"]:
        return 60 * 5
    elif frequency_id in ["15min"]:
        return 60 * 15
    else:
        raise NotImplementedError
