def time_window(df, midpoint_index, window_size):
    half_window_size = int(window_size / 2)
    lower_boud = midpoint_index - half_window_size
    uppder_bound = midpoint_index + half_window_size + 1
    return df[lower_boud:uppder_bound].reset_index(drop=True)
