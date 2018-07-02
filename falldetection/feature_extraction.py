def time_window(df, midpoint_index, window_size):
    half_window_size = int(window_size / 2)
    lower_bound = midpoint_index - half_window_size
    upper_bound = midpoint_index + half_window_size + 1
    return df[lower_bound:upper_bound].reset_index(drop=True)
