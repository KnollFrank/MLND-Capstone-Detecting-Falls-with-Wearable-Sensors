def time_window(df, window_center_index, half_window_size):
    lower_bound = window_center_index - half_window_size
    upper_bound = window_center_index + half_window_size + 1
    if not (0 <= lower_bound < len(df) and 0 <= upper_bound <= len(df)):
        raise IndexError()
    return df[lower_bound:upper_bound].reset_index(drop=True)
