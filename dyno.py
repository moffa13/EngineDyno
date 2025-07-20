import tkinter
from tkinter import ttk as ttkb
import sv_ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import messagebox, filedialog
from idlelib.tooltip import Hovertip
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.optimize import root_scalar
import tempfile
import os
import platform
import csv
from PIL import Image, ImageDraw, ImageFont
import re
import math
import pywinstyles, sys
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

log_file_path = None
update_job = None
last_changed = None  # Will be 'temp' or 'density'
canvas_widget = None
all_valid_runs = None
runs_to_compare = []
combo_runs_compare_contents = []
re_submit_needed = False

def moving_average(y, window_size):
    y = np.asarray(y)
    N = len(y)
    result = np.empty(N, dtype=float)
    half = window_size // 2
    
    for i in range(N):
        start = max(0, i - half)
        end = min(N, i + half + 1)
        result[i] = y[start:end].mean()
    
    return result

def temp_to_density(temp_c, humidity):
    T = temp_c + 273.15
    es = 6.1078 * 10 ** ((7.5 * temp_c) / (237.3 + temp_c))
    e = es * (humidity / 100.0)
    p = float(entry_col_air_pressure.get())
    pd = p - e
    Rd = 287.05
    Rv = 461.495
    rho = ((pd * 100) / (Rd * T)) + ((e * 100) / (Rv * T))
    return round(rho, 4)

def f_to_c(temp_f):
    return (temp_f - 32) / 1.8

def c_to_f(temp_c):
    return (temp_c * 1.8) + 32

def kW_to_HP(power):
    return power * 1.34102

def kW_to_PS(power):
    return power * 1.35962

def Nm_to_lbft(torque):
    return torque * 0.73756

def convert_kW_to_imperial(hps):
    return [kW_to_HP(hp) for hp in hps]

def convert_kW_to_metric(hps):
    return [kW_to_PS(hp) for hp in hps]

def convert_torque_to_imperial(torques):
    return [Nm_to_lbft(torque) for torque in torques]

def density_to_temp(density, humidity):
    def f(temp_c):
        return temp_to_density(temp_c, humidity) - density
    sol = root_scalar(f, bracket=(-50, 200), method='brentq')
    if sol.converged:
        return sol.root
    return None

def schedule_update(source):
    global update_job, last_changed
    last_changed = source
    if update_job is not None:
        root.after_cancel(update_job)
    update_job = root.after(500, apply_update)


# This function will make the density / humidity / temp_C / temp_F consistent
# By updating based on the modified one (last_changed)
def apply_update():
    try:
        humidity = float(entry_humidity.get())
    except:
        return

    if last_changed == 'temp_C':
        try:
            temp_c = entry_temp_C_var.get()
            entry_temp_F_var.set(round(c_to_f(temp_c), 1))
            rho = temp_to_density(temp_c, humidity)
            entry_density_var.set(rho)
        except:
            pass
    elif last_changed == 'temp_F':
        try:
            temp_f = entry_temp_F_var.get()
            temp_c = f_to_c(temp_f)
            entry_temp_C_var.set(round(temp_c, 1))
            rho = temp_to_density(temp_c, humidity)
            entry_density_var.set(rho)
        except:
            pass
    elif last_changed == 'density':
        try:
            rho = float(entry_air_density.get())
            temp_c_calc = density_to_temp(rho, humidity)
            if temp_c_calc is not None:
                entry_temp_C_var.set(round(temp_c_calc, 1))
                temp_f = c_to_f(temp_c_calc)
                entry_temp_F_var.set(round(temp_f, 1))
        except:
            pass
    
    root.after_cancel(update_job)

# This function returns whether or not all the values in the range are in range
def validate_rows(run, min_run_size=8, min_run_time=2, min_rpm_range=500, min_rpm_allowed=2500, max_rpm=20000, max_speed=500, check_speed=True):

    min_rpm = min(run['rpms'])
    max_rpm = max(run['rpms'])
    total_time = run['times'][-1] - run['times'][0]

    if max_rpm - min_rpm < min_rpm_range: # too small rpm range
        return False

        
    if max_rpm < min_rpm_allowed: # max rpm in the run is too low
        return False
    
    if len(run['times']) < min_run_size: # not enough data points in the run
        return False
    
    if total_time < min_run_time: # run not long enough in time
        return False


    if any(time < 0 for time in run['times']):
        return False
    
    if check_speed and any(speed < 0 or speed > max_speed for speed in run['speeds']):
        return False
    
    if any(rpm < 0 or rpm > max_rpm for rpm in run['rpms']):
        return False
    
    return True


def on_compare_runs():
    if len(runs_to_compare) < 2:
        messagebox.showerror("Error", "2 Runs of more are needed for comparison")
        return
    
    toggle_params('hide')
    print_graph_compare(runs_to_compare, graph_frame, 5)

# converts the raw run in csv mode in a dict with times, rpms and speeds keys
# speed is converted to m/s
def runs_to_dict(runs, important_cols):
    col_time_i = entry_col_time_var.get()
    col_speed_i = entry_col_speed_var.get()
    col_rpm_i = entry_col_rpm_var.get()

    runs_dict = []

    for run in runs:

        run = sanitize_run(run, important_cols)

        if len(run) == 0:
            continue

        speeds = []
        times = []
        rpms = []
        for elem in run:
            
            
            times.append(float(elem[col_time_i])) 

            # rpm is in the logs, retrieve it
            if not deduce_rpm_from_speed_var.get():
                rpm = int(float(elem[col_rpm_i]))

            if deduce_speed_from_rpm_var.get():
                speed = get_speed_from_rpm(rpm) / 3.6
            else:
                conversion_factor = 2.237 if speed_log_mph_var.get() else 3.6
                speed = float(elem[col_speed_i]) / conversion_factor

            
            if deduce_rpm_from_speed_var.get():
                rpm = get_rpm_from_speed(speed)
                

            rpms.append(rpm)
            speeds.append(speed)

        sampling_rate = 1 / np.mean(np.diff(times))

        runs_dict.append({
            'sampling_rate': sampling_rate,
            'times': np.array(times),
            'rpms': np.array(rpms),
            'speeds': np.array(speeds)
        })
    return runs_dict

'''def run_prefilter(run, target_rate=10, sg_window_sec=1.0, sg_poly=3):
    """
    Preprocess run data by interpolating and smoothing speed & RPM.
    
    Parameters:
        run (dict): {'times': [...], 'speeds': [...], 'rpms': [...]}
        target_rate (float): Resampling rate in Hz (e.g., 10 for 100ms steps)
        sg_window_sec (float): Savitzky-Golay window in seconds
        sg_poly (int): Polynomial order for Savitzky-Golay filter
        
    Returns:
        dict: {'times': ..., 'speeds': ..., 'rpms': ...} (filtered & resampled)
    """
    
    # Extract arrays
    times = np.array(run['times'], dtype=float)
    speeds = np.array(run['speeds'], dtype=float)
    rpms = np.array(run['rpms'], dtype=float)

    # --- 1. Validate data ---
    if len(times) < 5:
        raise ValueError("Not enough data points for filtering.")
    
    # --- 2. Create uniform time base ---
    t_start, t_end = times[0], times[-1]
    dt = 1.0 / target_rate
    t_uniform = np.arange(t_start, t_end, dt)
    
    # --- 3. Linear interpolate speeds & rpms ---
    speeds_interp = np.interp(t_uniform, times, speeds)
    rpms_interp = np.interp(t_uniform, times, rpms)
    
    # --- 4. Apply Savitzky-Golay smoothing ---
    # Convert sg_window_sec to samples (must be odd and >= polyorder+2)
    sg_window = int(sg_window_sec * target_rate)
    if sg_window % 2 == 0:  # must be odd
        sg_window += 1
    sg_window = max(sg_window, sg_poly + 2 | 1)  # ensure valid size
    
    speeds_filt = savgol_filter(speeds_interp, sg_window, sg_poly)
    rpms_filt = savgol_filter(rpms_interp, sg_window, sg_poly)
    
    return t_uniform, rpms_filt, speeds_filt'''

'''def run_prefilter(run):
    times = np.array(run['times'])
    speeds = np.array(run['speeds'])  # m/s
    rpms = np.array(run['rpms'])
        
    # 2. Assess log quality
    avg_dt = np.mean(np.diff(times))
    std_dt = np.std(np.diff(times))
    jitter_ratio = std_dt / avg_dt
    
    # Decide resampling step
    if avg_dt <= 0.02:      # ~50 Hz
        target_dt = 0.05    # 20 Hz
    elif avg_dt <= 0.1:     # 10 Hz
        target_dt = 0.1     # 10 Hz
    else:                   # Low rate (GPS-like)
        target_dt = avg_dt  # Don't fake detail
    
    new_times = np.arange(times[0], times[-1], target_dt)
    
    # 3. Interpolate
    interp_speed = interp1d(times, speeds, kind='linear', fill_value="extrapolate")
    interp_rpm = interp1d(times, rpms, kind='linear', fill_value="extrapolate")
    
    speeds_u = interp_speed(new_times)
    rpms_u = interp_rpm(new_times)
    
    # 4. Smoothing based on jitter
    if jitter_ratio > 0.1 or target_dt >= 0.1:
        window = max(5, int(0.5 / target_dt))  # ~0.5s window
        if window % 2 == 0: window += 1        # make it odd
        speeds_u = savgol_filter(speeds_u, window, polyorder=3)
        rpms_u = savgol_filter(rpms_u, window, polyorder=3)
    
    return new_times, rpms_u, speeds_u'''

def run_postfilter(data, sg_window_rpm=300, sg_poly=3):
    data = sorted(data, key=lambda x: x[0])
    rpms = np.array([x[0] for x in data], dtype=float)
    hp = np.array([x[1] for x in data], dtype=float)
    tq = np.array([x[2] for x in data], dtype=float)

    rpm_span = rpms[-1] - rpms[0]
    n_points = len(rpms)
    avg_step = rpm_span / max(n_points - 1, 1)

    sg_window_pts = int(sg_window_rpm / avg_step)
    if sg_window_pts % 2 == 0:
        sg_window_pts += 1
    sg_window_pts = max(sg_window_pts, sg_poly + 2 | 1)
    sg_window_pts = min(sg_window_pts, n_points if n_points % 2 == 1 else n_points - 1)

    hp_smoothed = savgol_filter(hp, sg_window_pts, sg_poly)
    tq_smoothed = savgol_filter(tq, sg_window_pts, sg_poly)

    return list(zip(rpms, hp_smoothed, tq_smoothed))

# This is the data pre filter
def run_prefilter(run):

    ok_time_sr = 10 # sr will be minimum 1 / that
    avg_window_time_ms = 1000

    rpms = run['rpms']
    times = run['times']
    speeds = run['speeds']


    sampling_rate = run['sampling_rate']

    
    time_step = 1 / max(sampling_rate, ok_time_sr)

    t_uniform = np.arange(times.min(), times.max(), time_step)
    interp_speed = interp1d(times, speeds, kind='linear')    
    interp_rpm = interp1d(times, rpms, kind='linear')

    speed_uniform = interp_speed(t_uniform)
    rpm_uniform = interp_rpm(t_uniform)

    smoothed_speeds = savgol_filter(speed_uniform, 5, 3)
    smoothed_rpms = savgol_filter(rpm_uniform, 5, 3)

    window_size = max(3, int((avg_window_time_ms / 1000) / time_step))
    if window_size % 2 == 0:  # Ensure window size is odd
        window_size += 1
    
    avg_times = []
    avg_speeds = []
    avg_rpms = []

    for i in range(len(t_uniform)):
        # Compute safe slice indices with padding at edges
        start_prev = max(0, i - window_size)
        end_prev = i
        start_next = i
        end_next = min(len(t_uniform), i + window_size)

        # Previous and next slices
        prev_speeds = smoothed_speeds[start_prev:end_prev]
        prev_times = t_uniform[start_prev:end_prev]

        next_speeds = smoothed_speeds[start_next:end_next]
        next_times = t_uniform[start_next:end_next]

        # Compute averages (handle empty slices at edges)
        avg_prev_speed = np.mean(prev_speeds) if len(prev_speeds) > 0 else smoothed_speeds[i]
        avg_prev_time = np.mean(prev_times) if len(prev_times) > 0 else t_uniform[i]

        avg_next_speed = np.mean(next_speeds) if len(next_speeds) > 0 else smoothed_speeds[i]
        avg_next_time = np.mean(next_times) if len(next_times) > 0 else t_uniform[i]

        avg_rpm = int(np.mean(smoothed_rpms[start_prev:end_next]))

        # Centered values
        avg_times.append((avg_prev_time + avg_next_time) / 2)
        avg_speeds.append((avg_prev_speed + avg_next_speed) / 2)
        avg_rpms.append(avg_rpm)
        

    # if by any chance there is a duplicated rpm value
    # add a small residual value to not make the plotting crash with
    # expect x to not have duplicates
    _, idx = np.unique(avg_rpms, return_index=True)
    if len(idx) != len(avg_rpms): 
        avg_rpms += np.arange(len(avg_rpms)) * 1e-6

    return avg_times, avg_rpms, avg_speeds

def on_compare_add():
    global runs_to_compare
    index = run_selector.current()
    if index >= 0:

        run_calculated = run_postfilter(analyse_run(*run_prefilter(all_valid_runs[index])))

        if run_calculated in runs_to_compare:
            messagebox.showerror("Error", "The same run has already been added to the compare list")
            return

        # add the hp version of everything since the parameters might change between two logs to compare
        runs_to_compare.append(run_calculated)
        combo_runs_compare_contents.append(run_selector['values'][index])
        combo_runs_compare['values'] = combo_runs_compare_contents

        # Select the freshly added run, to show the user something happened
        combo_runs_compare.set(combo_runs_compare_contents[-1])

def on_compare_remove():
    global runs_to_compare
    global combo_runs_compare_contents
    index = combo_runs_compare.current()
    if index >= 0:
        del combo_runs_compare_contents[index]
        del runs_to_compare[index]
        combo_runs_compare['values'] = combo_runs_compare_contents
        # Select the last run, to show the user something happened
        if len(runs_to_compare) > 0:
            combo_runs_compare.set(combo_runs_compare_contents[-1])
        else:
            on_compare_clear()

def on_compare_clear():
    global runs_to_compare
    global combo_runs_compare_contents
    runs_to_compare = []
    combo_runs_compare_contents = []
    combo_runs_compare['values'] = []
    combo_runs_compare.set('Empty')

# Called by UI when a run is selected in the combo box or reloaded
# If the file was not accepted it will be submitted again (e.g if the columns parameters were wrong)
def on_select_run(e=None):
    if re_submit_needed:
        submit()
        return
    index = run_selector.current()
    if index >= 0:
        select_run(index)

# Selects a run based on the index, calculates power & torque figures
# And plot in the graph
# All useless data in the UI is then hidden
def select_run(index):
    run = all_valid_runs[index]

    try:
        hp_torque = run_postfilter(analyse_run(*run_prefilter(run))) # Will calculate rpm, hp & torque
    except ValueError as e:
        messagebox.showerror("Error", "Please enter valid numeric values.")
        print(e)

    print_graph(hp_torque, graph_frame, int(window_size_var.get())) # Will plot        
    toggle_params('hide')

# Returns the best supposed run
# Used when loading a file and automatically select the most meaningful run
# RPM range is the most important, second is the length of the run
def find_best_run(runs):
    if len(runs) == 0: return None
    def score(run):
        return (max(run['rpms']) - min(run['rpms'])) * (len(run) ** 0.5)
    
    return max(range(len(runs)), key=lambda i: score(runs[i]))

def is_valid_numeric(value):
    try:
        value = float(value)
        return value > 0  # Or >= 0 depending on what you want
    except (ValueError, TypeError):
        return False

def sanitize_run(run, important_cols):
    run_sanitized = []
    for row in run:
        rowOk = True
        for imp in important_cols:
            if len(row) <= imp or not is_valid_numeric(row[imp]):
                rowOk = False
                break
        if rowOk:
            run_sanitized.append(row)
    return run_sanitized

def detect_columns_from_rows(rows):
    """
    Detect timestamp, RPM, and speed columns from CSV rows.

    Args:
        rows (list[list[str]]): CSV rows as list of lists.

    Returns:
        dict with keys: found, timestamp_idx, rpm_idx, speed_idx
    """

    # Keywords for header detection
    keywords = {
        "timestamp": [r"\btime\b", r"stamp", r"timestamp"],
        "rpm": [r"engine\s*speed", r"^rpm$", r"/min"],
        "speed": [r"\bspeed\b", r"vehicle\s*speed", r"^km/h$", r"^mph$"]
    }

    exclusions = {
        "speed": [r"engine\s*speed"]
    }
    
    def is_header_row(row):
        for i in keywords:
            for j in keywords[i]:
                for c in row:
                    if re.match(j, c.lower()):
                        return True
        return False
    
    # Search function
    def find_index(headers, patterns, exclude=None):
        for i, col in enumerate(headers):
            col_lower = col.strip().lower()
            if any(re.search(p, col_lower) for p in patterns):
                if exclude and any(re.search(e, col_lower) for e in exclude):
                    continue
                return i
        return None


    header_rows = []

    # Find headers
    for row in rows:
        if is_header_row(row):
            header_rows.append(row)

    if len(header_rows) == 0:
        return {"found": False}
    
    timestamp_idx = None
    rpm_idx = None
    speed_idx = None

    for row in header_rows:
        if timestamp_idx is None:
            timestamp_idx = find_index(row, keywords["timestamp"])
        if rpm_idx is None:
            rpm_idx = find_index(row, keywords["rpm"])
        if speed_idx is None:
            speed_idx = find_index(row, keywords["speed"], exclude=exclusions['speed'])

    return {
        "found": True,
        "timestamp_idx": timestamp_idx,
        "rpm_idx": rpm_idx,
        "speed_idx": speed_idx
    }

# This function retrieves the column infos using the csv headers and sets the Spinbox accordingly
def auto_set_columns_infos(rows):
    col_infos = detect_columns_from_rows(rows)

    if col_infos['found']:
        timestamp_idx = col_infos['timestamp_idx']
        rpm_idx = col_infos['rpm_idx']
        speed_idx = col_infos['speed_idx']

        if timestamp_idx is not None:
            entry_col_time_var.set(timestamp_idx)
        if rpm_idx is not None:
            entry_col_rpm_var.set(rpm_idx)
            deduce_rpm_from_speed_var.set(False)
            deduce_speed_from_rpm_var.set(True)
        else:
            deduce_rpm_from_speed_var.set(True)
            deduce_speed_from_rpm_var.set(False)
            messagebox.showwarning("Submission Result",
                        "Vehicle RPM not found in log, enabling auto deduce from speed.\n"
                        "Please fill in correct tire infos and gear infos."
            )

        if speed_idx is not None:
            deduce_speed_from_rpm_var.set(False)
            entry_col_speed_var.set(speed_idx)
        else:
            deduce_speed_from_rpm_var.set(True)
            deduce_rpm_from_speed_var.set(False)
            messagebox.showwarning("Submission Result",
                                    "Vehicle speed not found in log, enabling auto deduce from RPM.\n"
                                    "Please fill in correct tire infos and gear infos."
                                    )
        toggle_deduce_fields()
    else:
        messagebox.showwarning("Submission Result", f"Column infos not found.\n Be sure to manually set it properly")

def submit(auto_load_col_fields=False):

    print('submit!')

    global all_valid_runs
    global re_submit_needed

    re_submit_needed = True

    try:
        rows = []
        if log_file_path:
            try:
                with open(log_file_path, newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    rows = list(reader)
                    if auto_load_col_fields:
                        auto_set_columns_infos(rows)

            except Exception as e:
                messagebox.showinfo("Submission Result", f"Error reading log file: {e}")
                return
        else:
            messagebox.showinfo("Submission Result", "No log file loaded.")
            # If no file was selected, reload should not trigger that function again
            re_submit_needed = False
            return

        runs = []
        deduce_disabled = not deduce_speed_from_rpm_var.get() and not deduce_rpm_from_speed_var.get()
        deduce_speed = deduce_speed_from_rpm_var.get()
        filter_col_idx = entry_col_rpm_var.get() if deduce_speed or deduce_disabled else entry_col_speed_var.get()

        important_cols = [entry_col_time_var.get()]
        if deduce_disabled:
            important_cols.append(entry_col_rpm_var.get())
            important_cols.append(entry_col_speed_var.get())
        elif deduce_speed_from_rpm_var.get():
            important_cols.append(entry_col_rpm_var.get())
        elif deduce_rpm_from_speed_var.get():
            important_cols.append(entry_col_speed_var.get())
        
        if rows:
            runs = find_probable_runs(rows, filter_col_idx=filter_col_idx) # Will get the run range
            runs = runs_to_dict(runs, important_cols)
        else:
            # If file is empty, reload should not trigger that function again
            re_submit_needed = False
            messagebox.showinfo("Submission Result", "File is empty.")
            return

        all_valid_runs = list(filter(lambda x: validate_rows(
            x, check_speed=not deduce_speed_from_rpm_var.get()
        ), runs))

        if not all_valid_runs:
            if deduce_speed_from_rpm_var.get():
                messagebox.showwarning("Submission Result",
                                        "No valid run was found.\nBe sure you entered correct column info values.\n"
                                        "Click reload to try again."
                                        )
            else:
                messagebox.showwarning("Submission Result",
                                        "No valid run was found.\nCheck that the column info values are correct.\n"
                                        "Deduce speed from RPM is OFF, be sure to enable it if vehicle speed is not in the loaded log.\n"
                                        "Click reload to try again.")                      
        best_index = find_best_run(all_valid_runs)

        run_summaries = []
        for i, run in enumerate(all_valid_runs):
            run_time = (max(run['times']) - min(run['times']))
            rpm_range = f"{int(min(run['rpms']))}-{int(max(run['rpms']))}"
            run_len = len(run['times'])
            text = ""
            text += f"Run {i+1}: {run_len} points, "
            text += f"RPM {rpm_range}, "
            text += f"{run_time:.1f} s, "
            text += f"Data acquisition freq: {(run_len / run_time):.1f} Hz"
            run_summaries.append(text)

        run_selector['values'] = run_summaries
        run_selector_var.set(run_summaries[best_index] if run_summaries else "No valid runs")

        if best_index is not None:
            select_run(best_index)
            re_submit_needed = False
        
        
    
    except Exception as e:
        messagebox.showerror("Error", "Unknown error")
        print(e)

# will calculate power and torque based on many parameters
# Output is in kW and Nm
def analyse_run(times, rpms, speeds):
    car_weight = int(entry_mass.get())
    air_density = float(entry_air_density.get())
    air_temp = float(entry_temp_C.get())
    air_pressure_mbar = float(entry_col_air_pressure.get())
    gravity = float(entry_gravity.get())
    rolling_coeff = float(entry_crr.get())
    scx = float(entry_scx.get())
    drivetrain_loss = float(entry_gearbox_loss.get())

    if drivetrain_loss == 100:
        drivetrain_loss = 99.9

    final_hp_torque_curve = []
    for i in range(len(times)):
        if i == 0: continue

        # All the cool calculated values in order to extract two values, rpm and hp => torque
        delta_time = float(times[i]) - float(times[i-1])
        rpm = float(rpms[i])

        prev_speed_ms = speeds[i-1]
        speed_ms = speeds[i]

        if delta_time == 0:
            continue
        delta_speed = speed_ms - prev_speed_ms
        acceleration = delta_speed / delta_time
        force = acceleration * car_weight
        power_kw = force * speed_ms * 0.001
        air_loss_kw = 0.5 * air_density * scx * speed_ms ** 3 * 0.001
        rolling_loss_kw = car_weight * rolling_coeff * gravity * speed_ms * 0.001
        power_with_losses  = power_kw + air_loss_kw + rolling_loss_kw
        crank_power = power_with_losses / ((100 - drivetrain_loss) / 100)
        crank_torque_Nm = (crank_power * 9549.29) / rpm
        crank_torque = crank_torque_Nm

        # Apply DIN 70020 if needed
        if din_var.get():
            crank_power, crank_torque = apply_din_correction(
                crank_power, crank_torque, air_temp, air_pressure_mbar
            )

        final_hp_torque_curve.append((rpm, crank_power, crank_torque))
    return final_hp_torque_curve

def print_graph_to_printer():
    if not canvas_widget or not param_frame:
        return
    
    # Step 1: Save graph as PNG
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_graph:
        canvas_widget.figure.savefig(tmp_graph.name, bbox_inches='tight')
        graph_path = tmp_graph.name
    
    # Step 2: Generate text image from param_frame contents
    # Extract text from param_frame widgets (labels/entries)
    param_text = ""
    for child in param_frame.winfo_children():
        if hasattr(child, 'cget'):
            try:
                text = child.cget('text')
            except:
                try:
                    text = child.cget('textvariable')
                except:
                    continue
            if not text.startswith('PY_') and text != '':
                param_text += f'{text}: '
            if 'DIN' in text:
                param_text += 'yes' if din_var.get() else 'no'
                param_text += "\n"

        if hasattr(child, 'get'):
            try:
                value = child.get()
                param_text += str(value) + "\n"
            except:
                pass

    # Create image for text
    font = ImageFont.load_default()
    lines = param_text.strip().split('\n')
    
    # Get text bounding box to calculate width and height
    width = 0
    line_height = 0
    for line in lines:
        bbox = font.getbbox(line)
        line_width = bbox[2] - bbox[0]
        width = max(width, line_width)
        line_height = max(line_height, bbox[3] - bbox[1])
    
    height = line_height * len(lines) + 20
    text_img = Image.new('RGB', (width + 20, height), color='white')
    draw = ImageDraw.Draw(text_img)
    
    y = 10
    for line in lines:
        draw.text((10, y), line, font=font, fill='black')
        y += line_height
    
    # Step 3: Combine text image and graph image vertically
    graph_img = Image.open(graph_path)
    total_width = max(text_img.width, graph_img.width)
    total_height = text_img.height + graph_img.height
    combined_img = Image.new('RGB', (total_width, total_height), color='white')
    
    combined_img.paste(text_img, (0, 0))
    combined_img.paste(graph_img, (0, text_img.height))
    
    # Save combined image
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_combined:
        combined_img.save(tmp_combined.name)
        combined_path = tmp_combined.name

    # Step 4: Print
    if platform.system() == 'Windows':
        os.startfile(combined_path, 'print')
    elif platform.system() == 'Darwin':
        os.system(f'lpr "{combined_path}"')
    elif platform.system() == 'Linux':
        os.system(f'lpr "{combined_path}"')
    else:
        print("Unsupported OS for direct printing")

def parse_tire_size(tire_str):
    """
    Parses a tire size string (e.g. '205/45 R16') and returns (width_mm, aspect_ratio, diameter_inch)
    
    :param tire_str: str, e.g. '205/45 R16' or '195/65R15'
    :return: tuple (width_mm, aspect_ratio, diameter_inch)
    :raises: ValueError if format is invalid
    """
    # Normalize the string
    tire_str = tire_str.strip().upper()
    
    # Regex to match formats like 205/45R16 or 205/45 R16
    pattern = r"^(\d{2,3})/(\d{1,2})\s*R(\d{1,2})$"
    match = re.match(pattern, tire_str)
    
    if not match:
        raise ValueError(f"Invalid tire size format: '{tire_str}'. Expected format '205/45 R16'")
    
    width = int(match.group(1))
    aspect = int(match.group(2))
    diameter = int(match.group(3))
    
    return width, aspect, diameter

def wheel_perimeter_cm(width_mm, aspect_ratio, rim_diameter_inch):
    sidewall = width_mm * (aspect_ratio / 100)
    rim_dia_mm = rim_diameter_inch * 25.4
    total_dia_mm = rim_dia_mm + 2 * sidewall
    perimeter_mm = math.pi * total_dia_mm
    return perimeter_mm / 10  # mm to cm

def get_rpm_from_speed(speed):
    width, aspect, diameter = parse_tire_size(entry_tire.get())
    gear_ratio = float(entry_gearbox_ratio.get())
    diff_ratio = float(entry_diff_ratio.get())

    perim = wheel_perimeter_cm(width, aspect, diameter)

    final_ratio = gear_ratio * diff_ratio

    if final_ratio == 0:
        final_ratio = 0.01

    rpm = speed * final_ratio * 60 / perim * 100

    return rpm

def get_speed_from_rpm(rpm):

    width, aspect, diameter = parse_tire_size(entry_tire.get())
    gear_ratio = float(entry_gearbox_ratio.get())
    diff_ratio = float(entry_diff_ratio.get())

    perim = wheel_perimeter_cm(width, aspect, diameter)

    final_ratio = gear_ratio * diff_ratio

    if final_ratio == 0:
        final_ratio = 0.01

    speed = rpm / final_ratio * perim * 60 /100000

    return speed

def apply_din_correction(hp_measured, torque_measured, temp_c, pressure_mbar):
    """
    Apply DIN 70020 correction to measured HP and torque.

    :param hp_measured: measured horsepower
    :param torque_measured: measured torque
    :param temp_c: measured temperature (°C)
    :param pressure_mbar: measured atmospheric pressure (mbar)
    :return: tuple (hp_corrected, torque_corrected)
    """
    T0 = 293  # K (20°C)
    p0 = 1013  # mbar

    T = temp_c + 273.15  # Convert to K

    correction_factor = (p0 / pressure_mbar) * (T / T0) ** 0.5

    hp_corrected = hp_measured * correction_factor
    torque_corrected = torque_measured * correction_factor

    return hp_corrected, torque_corrected

def apply_theme_to_titlebar(root):
    version = sys.getwindowsversion()

    if version.major == 10 and version.build >= 22000:
        # Set the title bar color to the background color on Windows 11 for better appearance
        pywinstyles.change_header_color(root, "#1c1c1c" if sv_ttk.get_theme() == "dark" else "#fafafa")
    elif version.major == 10:
        pywinstyles.apply_style(root, "dark" if sv_ttk.get_theme() == "dark" else "normal")

        # A hacky way to update the title bar's color on Windows 10 (it doesn't update instantly like on Windows 11)
        root.wm_attributes("-alpha", 0.99)
        root.wm_attributes("-alpha", 1)

def print_graph_compare(rpm_hp_torque_list, graph_frame, smoothing_window_size=5):
    global canvas_widget

    """
    Plot HP and torque vs RPM on two Y axes in a Tkinter canvas frame,
    with smoothing, peak labels, and interactive pointer.

    Args:
        rpm_hp_torque (list of tuple): List of (RPM, HP, Torque) values.
        graph_frame (tk.Frame): The Tkinter frame to hold the canvas.
        smoothing_window_size (int): The moving_average smoothing window size.
    """

    # Extract data per run and make a flat out rpm list to get the min and max
    rpm_per_run = [np.array([elem[0] for elem in run]) for run in rpm_hp_torque_list]
    hps_per_run = [np.array([elem[1] for elem in run]) for run in rpm_hp_torque_list]
    torques_per_run = [np.array([elem[2] for elem in run]) for run in rpm_hp_torque_list]
    rpm_ranges = [(min(run), max(run)) for run in rpm_per_run]
    rpm_min = max(range[0] for range in rpm_ranges)
    rpm_max = min(range[1] for range in rpm_ranges)
    overlap = rpm_max - rpm_min



    overlap_percentages = [
        overlap / (max_rpm - min_rpm) * 100
        for min_rpm, max_rpm in rpm_ranges
    ]

    warning_shown = False
    for v in overlap_percentages:
        if v <= 0:
            messagebox.showerror(
                "Error",
                "At least one of the graphs does not overlap at all, comparison is not possible.\nRemove that run and try again"
            )
            return
        elif v < 25:
            messagebox.showerror(
                "Error",
                "The rpm overlap of a graph in the compare list is too low, comparison is not possible\nRemove that run and try again"
            )
            return
        elif v < 50 and not warning_shown:
            messagebox.showwarning(
                "Error",
                "The rpm overlap of a graph in the compare list is quite low, comparison is not optimal"
            )
            warning_shown = True

    if interpolate_var.get():
        # I want for ex 100 points per 1000 rpm
        npoints = int((int(rpm_max) - int(rpm_min)) / 10)
    else:
        mean_size = int(sum(len(run) for run in rpm_per_run) / len(rpm_per_run))
        npoints = mean_size

    # The common rpm range for all the figures to compare
    rpm_smooth = np.linspace(rpm_min, rpm_max, npoints)
    
    # Clear the frame
    for widget in graph_frame.winfo_children():
        widget.destroy()


    # Plot
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_hp = 'tab:red'
    color_torque = 'tab:blue'

    ax1.set_xlabel('RPM')
    ax1.set_ylabel('Power', color=color_hp)
    
    ax1.tick_params(axis='y', labelcolor=color_hp)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Torque', color=color_torque)
    
    ax2.tick_params(axis='y', labelcolor=color_torque)
    
    # Disable top spines to better see the peak figures
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    for i in range(len(rpm_hp_torque_list)):

        rpm_i = rpm_per_run[i]
        hp_i = hps_per_run[i]
        tq_i = torques_per_run[i]

        # if imperial, convert kW to HP and Nm to lb-ft
        if use_imperial.get():
            hp_i = convert_kW_to_imperial(hp_i)
            tq_i = convert_torque_to_imperial(tq_i)
        else: # if not, just convert kW to PS
            hp_i = convert_kW_to_metric(hp_i)


        # Smoothing only if user allowed it
        if smooth_var.get():
            hp_i = moving_average(hp_i, smoothing_window_size)
            tq_i = moving_average(tq_i, smoothing_window_size)


        hp_spline = make_interp_spline(rpm_i, hp_i, k=3)(rpm_smooth)
        torque_spline = make_interp_spline(rpm_i, tq_i, k=3)(rpm_smooth)

        ax1.plot(rpm_smooth, hp_spline, color=color_hp, label='Power')
        ax2.plot(rpm_smooth, torque_spline, color=color_torque, label='Torque')

    fig.tight_layout()

    if canvas_widget:
        canvas_widget.get_tk_widget().destroy()

    canvas_widget = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas_widget.draw()
    canvas_widget.get_tk_widget().pack(fill="both", expand=True)

    plt.close(fig)
    print_button.grid()

def print_graph(rpm_hp_torque, graph_frame, smoothing_window_size=5):

    global canvas_widget

    """
    Plot HP and torque vs RPM on two Y axes in a Tkinter canvas frame,
    with smoothing, peak labels, and interactive pointer.

    Args:
        rpm_hp_torque (list of tuple): List of (RPM, HP, Torque) values.
        graph_frame (tk.Frame): The Tkinter frame to hold the canvas.
        smoothing_window_size (int): The moving_average smoothing window size.
    """
    # Clear the frame
    for widget in graph_frame.winfo_children():
        widget.destroy()

    # Extract data
    rpm = np.array([point[0] for point in rpm_hp_torque])
    hp = np.array([point[1] for point in rpm_hp_torque])
    torque = np.array([point[2] for point in rpm_hp_torque])


    # if imperial, convert kW to HP and Nm to lb-ft
    if use_imperial.get():
        hp = convert_kW_to_imperial(hp)
        torque = convert_torque_to_imperial(torque)
    else: # if not, just convert kW to PS
        hp = convert_kW_to_metric(hp)


    # I want for ex 100 points per 1000 rpm
    npoints = int((int(rpm.max()) - int(rpm.min())) / 10)

    def moving_average(y, window_size):
        if len(y) < window_size:
            return y
        return np.convolve(y, np.ones(window_size)/window_size, mode='same')
    
    # Smoothing only if user allowed it
    if smooth_var.get():
        hp = moving_average(hp, smoothing_window_size)
        torque = moving_average(torque, smoothing_window_size)

    # Iterpolate if more than 4 data points and the interpolation is allowed
    if len(rpm) >= 4 and interpolate_var.get():
        rpm_smooth = np.linspace(rpm.min(), rpm.max(), npoints)
        hp_spline = make_interp_spline(rpm, hp, k=3)(rpm_smooth)
        torque_spline = make_interp_spline(rpm, torque, k=3)(rpm_smooth)
    else:
        rpm_smooth = rpm
        hp_spline = hp
        torque_spline = torque


    

    # Plot
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_hp = 'tab:red'
    color_torque = 'tab:blue'

    ax1.set_xlabel('RPM')
    ax1.set_ylabel('Power', color=color_hp)
    ax1.plot(rpm_smooth, hp_spline, color=color_hp, label='Power')
    ax1.tick_params(axis='y', labelcolor=color_hp)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Torque', color=color_torque)
    ax2.plot(rpm_smooth, torque_spline, color=color_torque, label='Torque')
    ax2.tick_params(axis='y', labelcolor=color_torque)

    # Peak labels
    hp_peak_idx = np.argmax(hp_spline)
    torque_peak_idx = np.argmax(torque_spline)

    power_unit = 'HP' if use_imperial.get() else 'PS'
    torque_unit = 'lb-ft' if use_imperial.get() else 'Nm'

    annot_peak_hp = ax1.annotate(f'Peak power: {hp_spline[hp_peak_idx]:.1f} {power_unit} @ {rpm_smooth[hp_peak_idx]:.0f} RPM', 
                 textcoords="offset points",
                 xy=(rpm_smooth[hp_peak_idx], hp_spline[hp_peak_idx]),
                 xytext=(5, 10),
                 arrowprops=dict(facecolor=color_hp, arrowstyle="->"),
                 color=color_hp)

    

    annot_peak_torque = ax2.annotate(f'Peak Torque: {torque_spline[torque_peak_idx]:.1f} {torque_unit} @ {rpm_smooth[torque_peak_idx]:.0f} RPM',
                 textcoords="offset points",
                 xy=(rpm_smooth[torque_peak_idx], torque_spline[torque_peak_idx]),
                 xytext=(5, 10),
                 arrowprops=dict(facecolor=color_torque, arrowstyle="->"),
                 color=color_torque)
    
    # Disable top spines to better see the peak figures
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # Annot per axis
    annot_hp = ax1.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"), color='black')
    annot_hp.set_visible(False)

    annot_torque = ax2.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                                bbox=dict(boxstyle="round", fc="w"),
                                arrowprops=dict(arrowstyle="->"), color='black')
    annot_torque.set_visible(False)

    
    # Draw dynamic annotations on top of everything
    if annot_hp not in fig.artists:
        fig.artists.append(annot_hp)
    if annot_torque not in fig.artists:
        fig.artists.append(annot_torque)
    

    def on_mouse_move(event):
        if event.inaxes not in [ax1, ax2]:
            annot_hp.set_visible(False)
            annot_torque.set_visible(False)
            annot_peak_hp.set_visible(True)
            annot_peak_torque.set_visible(True)
            fig.canvas.draw_idle()
            return

        x = event.xdata
        if x is None:
            return
        
        annot_peak_hp.set_visible(False)
        annot_peak_torque.set_visible(False)
        
        # Find closest index
        idx = np.abs(rpm_smooth - x).argmin()

        # Update HP annot
        annot_hp.xy = (rpm_smooth[idx], hp_spline[idx])
        annot_hp.set_text(f"Power: {hp_spline[idx]:.1f} {power_unit} @ {rpm_smooth[idx]:.0f} RPM")
        annot_hp.set_visible(True)

        # Update Torque annot
        annot_torque.xy = (rpm_smooth[idx], torque_spline[idx])
        annot_torque.set_text(f"Torque: {torque_spline[idx]:.1f} {torque_unit} @ {rpm_smooth[idx]:.0f} RPM")
        annot_torque.set_visible(True)

        annot_torque.set_position((20, 20))
        
        renderer = fig.canvas.get_renderer()
        bbox1 = annot_hp.get_window_extent(renderer)
        bbox2 = annot_torque.get_window_extent(renderer)

        if bbox1.overlaps(bbox2):
            annot_torque.set_position((-200, 20))

        
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)

    def check_peak_overlap():
        annot_peak_torque.set_position((5, 10))
        renderer = canvas_widget.get_renderer()
        bbox1 = annot_peak_hp.get_window_extent(renderer)
        bbox2 = annot_peak_torque.get_window_extent(renderer)
        # if peaks overlap, only one offset is enough to fix the problem
        if bbox1.overlaps(bbox2):
            annot_peak_torque.set_position((-200, 10))
            fig.canvas.draw_idle()

    if canvas_widget:
        canvas_widget.get_tk_widget().destroy()

    canvas_widget = FigureCanvasTkAgg(fig, master=graph_frame)

    fig.tight_layout()
    canvas_widget.draw()
    canvas_widget.get_tk_widget().pack(fill="both", expand=True)
   
    
    # This is needed to check overlap, the figure needs to be fully rendered in tkinter
    canvas_widget.get_tk_widget().after_idle(check_peak_overlap)

    # Re-check for peak annot collisions when resizing
    graph_frame.bind('<Configure>', lambda e: check_peak_overlap())

    
    plt.close(fig)
    print_button.grid()

def find_probable_runs(rows, filter_col_idx=2):
    runs = []
    current_start = None
    prev_rpm = None

    def save_run(start, end):
        raw_run = rows[start:end]
        runs.append(raw_run)

    for i, row in enumerate(rows):
        # Skip completely invalid rows
        if not row or len(row) <= filter_col_idx or row[filter_col_idx] == '' or not is_valid_numeric(row[filter_col_idx]):
            if current_start is not None:
                save_run(current_start, i)
                current_start = None
            prev_rpm = None
            continue

        curr_rpm_or_speed = float(row[filter_col_idx])

        if prev_rpm is None:
            current_start = i
        elif curr_rpm_or_speed < prev_rpm:
            save_run(current_start, i)
            current_start = i

        prev_rpm = curr_rpm_or_speed

    # Save any remaining run
    if current_start is not None:
        save_run(current_start, len(rows))

    return runs

def load_log_file():
    global log_file_path
    file_path = filedialog.askopenfilename(title="Select log file", filetypes=[("CSV or log files", "*.csv *.log *.txt"), ("All files", "*.*")])
    if file_path:
        log_file_path = file_path
        log_label.config(text=f"Loaded: {file_path}")
        submit(auto_load_col_fields=True)

def toggle_params(action=None):
    if action == "hide" or param_frame.winfo_ismapped():
        param_frame.grid_remove()
        actions_frame.grid_remove()
        toggle_btn.config(text="Show all")
    else:
        param_frame.grid()
        actions_frame.grid()
        toggle_btn.config(text="Hide all")

def toggle_deduce_fields(element=None):
    global re_submit_needed
    
    # This makes sure that runs are reloaded
    re_submit_needed = True


    if element == deduce_rpm_from_speed_var and deduce_rpm_from_speed_var.get():
        deduce_speed_from_rpm_var.set(False)
    elif element == deduce_speed_from_rpm_var and deduce_speed_from_rpm_var.get():
        deduce_rpm_from_speed_var.set(False)

    one_is_checked = deduce_speed_from_rpm_var.get() or deduce_rpm_from_speed_var.get()

    state = "normal" if one_is_checked else "disabled"
    entry_tire.config(state=state)
    entry_gearbox_ratio.config(state=state)
    entry_diff_ratio.config(state=state)
    entry_col_speed.config(state="disabled" if deduce_speed_from_rpm_var.get() else "normal")
    entry_col_rpm.config(state="disabled" if deduce_rpm_from_speed_var.get() else "normal")
    speed_mph_checkbox.config(state="disabled" if deduce_speed_from_rpm_var.get() else "normal")

def critical_value_changed():
    global re_submit_needed

    # This makes sure that runs are reloaded
    re_submit_needed = True

def toggle_window_size_param():
    state = "normal" if smooth_var.get() else "disabled"
    window_size_spinbox.config(state=state)

root = tkinter.Tk()
sv_ttk.use_dark_theme()
root.title("VCDS Log Dyno")
root.geometry("1000x850")

param_frame = ttkb.LabelFrame(root, text="Parameters")
param_frame.pack(padx=10, pady=10)
param_frame.grid(row=0, column=0, columnspan=4, sticky="we", padx=10, pady=10)

for i in range(4):
    param_frame.columnconfigure(i, weight=1)

# FIRST COLUMN
label_mass = ttkb.Label(param_frame, text="Car mass (kg)")
label_mass.grid(row=0, column=0, sticky="e", padx=5, pady=5)
entry_mass = ttkb.Spinbox(param_frame, from_=500, to=5000, increment=1)
entry_mass.insert(0, "1240")
entry_mass.grid(row=0, column=1, sticky="we", padx=5, pady=5)

label_air_density = ttkb.Label(param_frame, text="Air density (kg/m³)")
label_air_density.grid(row=1, column=0, sticky="e", padx=5, pady=5)
entry_density_var = tkinter.DoubleVar()
entry_air_density = ttkb.Spinbox(param_frame, from_=0.9, to=1.4, increment=0.01, textvariable=entry_density_var)
entry_density_var.set("1.1755")
entry_air_density.grid(row=1, column=1, sticky="we", padx=5, pady=5)

label_outside_temp = ttkb.Label(param_frame, text="Temperature °C / ° F")
label_outside_temp.grid(row=2, column=0, sticky="e", padx=5, pady=5)


temp_frame = ttkb.Frame(param_frame)
temp_frame.grid(row=2, column=1, sticky="ew")
temp_frame.grid_columnconfigure(0, weight=1)
temp_frame.grid_columnconfigure(1, weight=1)

entry_temp_C_var = tkinter.DoubleVar()
entry_temp_C = ttkb.Spinbox(temp_frame, from_=-50, to=60, increment=1, textvariable=entry_temp_C_var, width=3)
entry_temp_C_var.set("25")
entry_temp_C.grid(row=0, column=0, padx=5, pady=0, sticky="ew")

entry_temp_F_var = tkinter.DoubleVar()
entry_temp_F = ttkb.Spinbox(temp_frame, from_=-58, to=140, increment=1, textvariable=entry_temp_F_var, width=3)
entry_temp_F_var.set("77")
entry_temp_F.grid(row=0, column=1, padx=5, pady=0, sticky="ew")

label_outside_humidity = ttkb.Label(param_frame, text="Humidity (%)")
label_outside_humidity.grid(row=3, column=0, sticky="e", padx=5, pady=5)
entry_humidity_var = tkinter.IntVar()
entry_humidity = ttkb.Spinbox(param_frame, from_=0, to=100, increment=1, textvariable=entry_humidity_var)
entry_humidity_var.set("60")
entry_humidity.grid(row=3, column=1, sticky="we", padx=5, pady=5)

label_scx = ttkb.Label(param_frame, text="SCx")
label_scx.grid(row=4, column=0, sticky="e", padx=5, pady=5)
entry_scx = ttkb.Spinbox(param_frame, from_=0.2, to=1.2, increment=0.01)
entry_scx.insert(0, "0.653")
entry_scx.grid(row=4, column=1, sticky="we", padx=5, pady=5)

label_gearbox_loss = ttkb.Label(param_frame, text="Gearbox loss %")
label_gearbox_loss.grid(row=5, column=0, sticky="e", padx=5, pady=5)
entry_gearbox_loss = ttkb.Spinbox(param_frame, from_=0, to=30, increment=1)
entry_gearbox_loss.insert(0, "13")
entry_gearbox_loss.grid(row=5, column=1, sticky="we", padx=5, pady=5)

# --- Tire info field ---
label_tire_size = ttkb.Label(param_frame, text="Tire size (e.g. 205/45 R16)")
label_tire_size.grid(row=6, column=0, sticky="e", padx=5, pady=5)
entry_tire = ttkb.Entry(param_frame)
entry_tire.insert(0, "205/45 R16")
entry_tire.grid(row=6, column=1, sticky="we", padx=5, pady=5)

# --- Gearbox ratio field ---
label_gearbox_ratio = ttkb.Label(param_frame, text="Gearbox ratio")
label_gearbox_ratio.grid(row=7, column=0, sticky="e", padx=5, pady=5)
entry_gearbox_ratio = ttkb.Spinbox(param_frame, from_=0.1, to=5, increment=0.001)
entry_gearbox_ratio.insert(0, "1.324")
entry_gearbox_ratio.grid(row=7, column=1, sticky="we", padx=5, pady=5)

# --- Differential ratio field ---
label_diff_ratio = ttkb.Label(param_frame, text="Differential ratio")
label_diff_ratio.grid(row=8, column=0, sticky="e", padx=5, pady=5)
entry_diff_ratio = ttkb.Spinbox(param_frame, from_=0.1, to=5, increment=0.001)
entry_diff_ratio.insert(0, "3.238")
entry_diff_ratio.grid(row=8, column=1, sticky="we", padx=5, pady=5)


use_imperial = tkinter.BooleanVar(value=False)

unit_checkbox = ttkb.Checkbutton(
    param_frame,
    text="Use imperial units",
    variable=use_imperial
)
unit_checkbox.grid(row=9, column=1, sticky="we", padx=10, pady=10)

speed_log_mph_var = tkinter.BooleanVar(value=False)

speed_mph_checkbox = ttkb.Checkbutton(
    param_frame,
    text="Speed in logs is in mph",
    variable=speed_log_mph_var,
    command=critical_value_changed
)
speed_mph_checkbox.grid(row=9, column=2, sticky="we", padx=10, pady=10)

# Apply DIN correction checkbox
din_var = tkinter.BooleanVar()
din_var.set(True)
din_checkbox = ttkb.Checkbutton(param_frame, text="Apply DIN correction", variable=din_var)
din_checkbox.grid(row=9, column=3, columnspan=2, sticky="w", padx=5, pady=5)

# Apply smoothing checkbox
smooth_var = tkinter.BooleanVar()
smooth_var.set(True)
smooth_checkbox = ttkb.Checkbutton(param_frame, text="Apply graph smoothing", variable=smooth_var, command=toggle_window_size_param)
smooth_checkbox.grid(row=7, column=3, columnspan=2, sticky="w", padx=5, pady=5)

label_window_size = ttkb.Label(param_frame, text="Smoothing window size")
label_window_size.grid(row=8, column=3, sticky="w", padx=5, pady=5)
window_size_var = tkinter.IntVar(value=5)
window_size_spinbox = ttkb.Spinbox(param_frame, from_=2, to=10, textvariable=window_size_var, width=5)
window_size_spinbox.grid(row=8, column=3, sticky="e", padx=5, pady=5)

# Apply point interpolation checkbox
interpolate_var = tkinter.BooleanVar()
interpolate_var.set(True)
interpolate_checkbox = ttkb.Checkbutton(param_frame, text="Apply point interp.", variable=interpolate_var)
interpolate_checkbox.grid(row=6, column=3, columnspan=2, sticky="w", padx=5, pady=5)

deduce_speed_from_rpm_var = tkinter.BooleanVar(value=False)
deduce_speed_from_rpm_checkbox = ttkb.Checkbutton(
    param_frame, text="Deduce speed from RPM", variable=deduce_speed_from_rpm_var, command=lambda: toggle_deduce_fields(deduce_speed_from_rpm_var)
)
deduce_speed_from_rpm_checkbox.grid(row=6, column=2, columnspan=2, sticky="w", padx=5, pady=5)

deduce_rpm_from_speed_var = tkinter.BooleanVar(value=False)
deduce_rpm_from_speed_checkbox = ttkb.Checkbutton(
    param_frame, text="Deduce RPM from speed", variable=deduce_rpm_from_speed_var, command=lambda: toggle_deduce_fields(deduce_rpm_from_speed_var)
)
deduce_rpm_from_speed_checkbox.grid(row=7, column=2, columnspan=2, sticky="w", padx=5, pady=5)

# SECOND COLUMN
label_crr = ttkb.Label(param_frame, text="Crr")
label_crr.grid(row=0, column=2, sticky="e", padx=5, pady=5)
entry_crr = ttkb.Spinbox(param_frame, from_=0, to=0.04, increment=0.001)
entry_crr.insert(0, "0.019")
entry_crr.grid(row=0, column=3, sticky="we", padx=5, pady=5)

label_gravity = ttkb.Label(param_frame, text="Gravity (m/s²)")
label_gravity.grid(row=1, column=2, sticky="e", padx=5, pady=5)
entry_gravity = ttkb.Spinbox(param_frame, from_=9, to=10, increment=0.01)
entry_gravity.insert(0, "9.81")
entry_gravity.grid(row=1, column=3, sticky="we", padx=5, pady=5)

entry_col_time_var = tkinter.IntVar()
entry_col_time_var.set(1)
label_col_time = ttkb.Label(param_frame, text="Time stamp column")
label_col_time.grid(row=2, column=2, sticky="e", padx=5, pady=5)
entry_col_time = ttkb.Spinbox(param_frame, from_=0, to=10, increment=1, textvariable=entry_col_time_var, command=critical_value_changed)
entry_col_time.grid(row=2, column=3, sticky="we", padx=5, pady=5)

entry_col_speed_var = tkinter.IntVar()
entry_col_speed_var.set(3)
label_col_speed = ttkb.Label(param_frame, text="Vehicle speed column")
label_col_speed.grid(row=3, column=2, sticky="e", padx=5, pady=5)
entry_col_speed = ttkb.Spinbox(param_frame, from_=0, to=10, increment=1, textvariable=entry_col_speed_var, command=critical_value_changed)
entry_col_speed.grid(row=3, column=3, sticky="we", padx=5, pady=5)

entry_col_rpm_var = tkinter.IntVar()
entry_col_rpm_var.set(2)
label_col_rpm = ttkb.Label(param_frame, text="RPM column")
label_col_rpm.grid(row=4, column=2, sticky="e", padx=5, pady=5)
entry_col_rpm = ttkb.Spinbox(param_frame, from_=0, to=10, increment=1, textvariable=entry_col_rpm_var, command=critical_value_changed)
entry_col_rpm.grid(row=4, column=3, sticky="we", padx=5, pady=5)


label_col_air_pressure = ttkb.Label(param_frame, text="Air pressure (hPa)")
label_col_air_pressure.grid(row=5, column=2, sticky="e", padx=5, pady=5)
entry_col_air_pressure = ttkb.Spinbox(param_frame, from_=700, to=1400, increment=1)
entry_col_air_pressure.insert(0, "1013.25")
entry_col_air_pressure.grid(row=5, column=3, sticky="we", padx=5, pady=5)




actions_frame = ttkb.LabelFrame(root, text="Actions")
actions_frame.grid(row=1, column=0, columnspan=4, sticky="we", padx=10, pady=10)
for i in range(4):
    actions_frame.columnconfigure(i, weight=1)

# LOAD LOG FILE
ttkb.Button(actions_frame, text="Load log file", command=load_log_file).grid(row=0, column=0, columnspan=4, padx=20, pady=10, sticky="ew")
log_label = ttkb.Label(actions_frame, text="No log file loaded", wraplength=700, justify="left")
log_label.grid(row=1, column=0, columnspan=4)

graph_frame = ttkb.Frame(root, relief="solid", borderwidth=2)
graph_frame.grid(row=7, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)

run_selector_var = tkinter.StringVar()
run_selector = ttkb.Combobox(root, textvariable=run_selector_var, state="readonly")
run_selector.grid(row=3, column=0, columnspan=2, sticky="we", padx=10, pady=5)
run_selector.set('No file selected')

root.rowconfigure(7, weight=1)
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)



# RELOAD + TOGGLE BUTTON
ttkb.Button(actions_frame, text="Reload", command=on_select_run).grid(row=3, column=0, padx=20, pady=10, sticky="ew")
toggle_btn = ttkb.Button(root, text="Hide all", command=toggle_params)
toggle_btn.grid(row=2, column=0, columnspan=4, padx=20, pady=20, sticky="ew")

ttkb.Button(actions_frame, text="Add to compare", command=on_compare_add).grid(row=4, column=0, padx=20, pady=10, sticky="ew")
ttkb.Button(actions_frame, text="Compare runs", command=on_compare_runs).grid(row=4, column=1, padx=20, pady=10, sticky="ew")
ttkb.Button(actions_frame, text="Clear compare list", command=on_compare_clear).grid(row=4, column=2, padx=20, pady=10, sticky="ew")
ttkb.Button(actions_frame, text="Remove selected compare run", command=on_compare_remove).grid(row=4, column=3, padx=20, pady=10, sticky="ew")


ttkb.Label(actions_frame, text="Loaded runs to compare", wraplength=700, justify="left").grid(row=5, column=0)
combo_runs_compare = ttkb.Combobox(actions_frame, state="readonly")

combo_runs_compare.set('Empty')

combo_runs_compare.grid(row=5, column=1, columnspan=3, sticky="ew", padx=20, pady=10)

print_button = ttkb.Button(root, text="Print Graph", command=print_graph_to_printer)
print_button.grid(row=15, column=1, padx=50, pady=20, sticky="ew")
print_button.grid_remove()  # Start hidden

entry_tire.config(state="disabled")
entry_gearbox_ratio.config(state="disabled")
entry_diff_ratio.config(state="disabled")

# TOOLTIPS
Hovertip(label_mass, "The total weight of your vehicle during the test run")
Hovertip(label_air_density, "The outside air density")
Hovertip(label_outside_temp, "The outside air temperature")
Hovertip(label_outside_humidity, "The outside relative humidity. Set it to 0 for dry air")
Hovertip(label_scx, "Drag Area SCx (Drag Coefficient × Frontal Area)")
Hovertip(label_gearbox_loss, "Percentage of power loss from the engine to the wheels")
Hovertip(label_tire_size, "Informations about your tire. Write it just as marked on the tire")
Hovertip(
    label_gearbox_ratio,
    "The internal gear ratio for the selected gear.\n"
    "This value changes based on what gear you're in."
)
Hovertip(
    label_diff_ratio,
    "The differential ratio.\n"
    "That value does not depend on what gear is selected."
)
Hovertip(unit_checkbox, "Check this if you want the dyno results in imperial units (HP and lb-ft)")
Hovertip(
    deduce_speed_from_rpm_checkbox,
    "Check this if the vehicle speed is not in the log values.\n"
    "It will then be calculted based on tire infos and final gear ratio."
)
Hovertip(speed_mph_checkbox, "Check this if the speed in the logs is in mph")
Hovertip(label_crr, "The rolling resistance coefficient.")
Hovertip(
    label_gravity,
    "Gravitational acceleration.\n"
    "Used to calculate rolling resistance loss\n"
    "This value changes based on where on earth you are\n"
    "Can very well be left unchanged to the standard value."
)
Hovertip(label_col_time, "The index where to find the time stamp in the logs (starts at 0)")
Hovertip(label_col_speed, "The index where to find the vehicle speed in the logs (starts at 0)")
Hovertip(label_col_rpm, "The index where to find the vehicle RPM in the logs (starts at 0)")
Hovertip(label_col_air_pressure, "The outside air pressure")
Hovertip(
    interpolate_checkbox,
    "Check this if you want to create more points than the log has.\n"
    "This helps eliminate sharp edges or jagged transitions, "
    "making the graph easier to read and more representative of real-world behavior"
)
Hovertip(
    smooth_checkbox,
    "Check this if you want to smooth the graph.\n"
    "This makes the overall curve look smoother by adjusting values based on the neighbors"
)
Hovertip(
    label_window_size,
    "The size of the window used for graphg smoothing.\n"
    "This value represents the surrounding neighbors used to smooth the current value."
)
Hovertip(din_checkbox, "Check this to apply the DIN correction to the power figures.")

entry_temp_C_var.trace_add("write", lambda *args: schedule_update('temp_C'))
entry_temp_F_var.trace_add("write", lambda *args: schedule_update('temp_F'))
entry_density_var.trace_add("write", lambda *args: schedule_update('density'))
entry_humidity_var.trace_add("write", lambda *args: schedule_update(last_changed if last_changed else 'temp_C'))
entry_col_air_pressure.bind("<KeyRelease>", lambda e: schedule_update(last_changed if last_changed else 'temp_C'))
run_selector.bind("<<ComboboxSelected>>", on_select_run)

plt.style.use('dark_background')

# Example usage (replace `root` with the reference to your main/Toplevel window)
apply_theme_to_titlebar(root)

root.mainloop()