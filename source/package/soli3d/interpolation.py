def interpolate_temperature(
    t_seconds: float, days: list[float], temps: list[float]
) -> float:
    seconds_per_day = 86400
    t_days = t_seconds / seconds_per_day

    # print("t_days: ", t_days)

    # handel the values out of input data boundaries
    if t_days <= days[0]:
        return temps[0]
    if t_days >= days[-1]:
        return temps[-1]

    # linear interpolation
    for i in range(len(days) - 1):
        if days[i] <= t_days <= days[i + 1]:
            temp_lower = temps[i]
            temp_upper = temps[i + 1]
            frac = (t_days - days[i]) / (days[i + 1] - days[i])
            return temp_lower + frac * (temp_upper - temp_lower)

    # If something went wrong
    raise RuntimeError("Interpolation failed: no valid time interval found.")
