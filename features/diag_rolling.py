import numpy as np

highs = np.array([10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0,
                   14.0, 13.5, 12.0, 11.0, 10.5, 9.0, 8.5, 7.0, 8.0, 9.0,
                   10.0, 11.0, 10.5, 9.5, 8.0, 7.5, 6.0, 5.5, 6.0, 7.0])
size = 5

print("=== My _rolling_high/low (Pine-consistent) ===")
for i in range(size, len(highs)):
    ref_i = i - size
    start = ref_i + 1
    end = i + 1
    window = highs[start:end]
    h_max = float(np.max(window))
    l_min = float(np.min(highs[start:end]))
    val = highs[ref_i]
    is_high = val > h_max
    is_low = val < l_min
    print(f"i={i:2d} ref={ref_i} val={val:.1f} window={window} h_max={h_max:.1f} is_high={is_high} is_low={is_low}")

print()
print("=== What Pine says: high[size] vs ta.highest(size) ===")
print("In Pine: high[size] is the bar at index=size (ref_i), ta.highest(size) is max of last 'size' bars ending at bar i")
print("At i=size: high[size] vs ta.highest(5) where highest covers bars [size-4, size]=[1,5]")
print("That means window is arr[ref_i-size+1 : i+1] = arr[ref_i-4 : ref_i+1]")
print()
print("=== Corrected rolling ===")
for i in range(size, len(highs)):
    ref_i = i - size
    s = max(0, ref_i - size + 1)
    e = ref_i + 1
    window = highs[s:e]
    h_max = float(np.max(window))
    l_min = float(np.min(window))
    val = highs[ref_i]
    is_high = val > h_max
    is_low = val < l_min
    print(f"i={i:2d} ref={ref_i} val={val:.1f} window_idx=[{s}:{e}] window={window} h_max={h_max:.1f} is_high={is_high} is_low={is_low}")