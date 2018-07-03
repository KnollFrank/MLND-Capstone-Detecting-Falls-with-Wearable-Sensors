from falldetection.slicer import slice_with_window


def get_index_of_maximum_total_acceleration(df):
    squared_total_acceleration = df['Acc_X'] ** 2 + df['Acc_Y'] ** 2 + df['Acc_Z'] ** 2
    return squared_total_acceleration.idxmax()


def get_window_around_maximum_total_acceleration(df, half_window_size, index_error_msg=""):
    return slice_with_window(
        df,
        window_center_index=get_index_of_maximum_total_acceleration(df),
        half_window_size=half_window_size,
        index_error_msg=index_error_msg)
