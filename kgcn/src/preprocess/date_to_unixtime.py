
def datetime_to_unixtime(datetime_array):
    return datetime_array.astype('datetime64[s]').astype('int64')
