import numba


@numba.njit(cache=True)
def query_prefix_sum(arr_prefix_sum, start, stop):
    return (
        arr_prefix_sum[stop - 1] - arr_prefix_sum[start - 1]
        if start > 0
        else arr_prefix_sum[stop - 1]
    )
