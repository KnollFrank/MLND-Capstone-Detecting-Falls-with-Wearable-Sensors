def time_window(df, window_center_index, half_window_size):
    lower_bound_inclusive = window_center_index - half_window_size
    upper_bound_exclusive = window_center_index + half_window_size + 1
    if not (0 <= lower_bound_inclusive < len(df) and 0 <= upper_bound_exclusive <= len(df)):
        raise IndexError()
    return df[lower_bound_inclusive:upper_bound_exclusive].reset_index(drop=True)


def get_index_of_maximum_total_acceleration(df):
    squared_total_acceleration = df['Acc_X'] ** 2 + df['Acc_Y'] ** 2 + df['Acc_Z'] ** 2
    return squared_total_acceleration.idxmax()
