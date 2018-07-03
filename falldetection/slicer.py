def slice_with_window(df, window_center_index, half_window_size, index_error_msg=""):
    lower_bound_inclusive = window_center_index - half_window_size
    upper_bound_exclusive = window_center_index + half_window_size + 1
    if not (0 <= lower_bound_inclusive < len(df) and 0 <= upper_bound_exclusive <= len(df)):
        raise IndexError(
            "{}: not (0 <= {} < {} and 0 <= {} <= {})".format(index_error_msg, lower_bound_inclusive, len(df),
                                                              upper_bound_exclusive, len(df)))
    return df.iloc[lower_bound_inclusive:upper_bound_exclusive]
