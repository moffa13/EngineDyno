import tkinter as tk
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import messagebox, filedialog
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

log_file_path = None
update_job = None
last_changed = None  # Will be 'temp' or 'density'
canvas_widget = None

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

def apply_update():
    try:
        humidity = float(entry_humidity.get())
    except:
        return

    if last_changed == 'temp':
        try:
            temp_c = float(entry_temp.get())
            rho = temp_to_density(temp_c, humidity)
            entry_density_var.set(str(rho))
        except:
            pass
    elif last_changed == 'density':
        try:
            rho = float(entry_air_density.get())
            temp_c_calc = density_to_temp(rho, humidity)
            if temp_c_calc is not None:
                entry_temp_var.set(str(temp_c_calc))
        except:
            pass


def validate_rows(rows, time_col_idx=1, rpm_col_idx=2, speed_col_idx=3, max_rpm=20000, max_speed=500, check_speed=True):

    for row in rows:
        try:
            time = float(row[time_col_idx])
            rpm = float(row[rpm_col_idx])
            if check_speed:
                speed = float(row[speed_col_idx])
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid row data or format: {row} - {e}")

        if time < 0:
            raise ValueError(f"Negative time found: {time} in row {row}")
        if rpm < 0:
            raise ValueError(f"Negative RPM found: {rpm} in row {row}")
        if check_speed and speed < 0:
            raise ValueError(f"Negative speed found: {rpm} in row {row}")
        if rpm > max_rpm:
            raise ValueError(f"Unrealistically high RPM found: {rpm} in row {row}")
        if check_speed and speed > max_speed:
            raise ValueError(f"Unrealistically high speed found: {rpm} in row {row}")

def submit():
    try:
        rows = []
        if log_file_path:
            try:
                with open(log_file_path, newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    rows = list(reader)
            except Exception as e:
                messagebox.showinfo("Submission Result", f"\n\nError reading log file: {e}")
                return
        else:
            messagebox.showinfo("Submission Result", "\n\nNo log file loaded.")
            return

        run = []
        if rows:
            run = run_remove_duplicates(extract_data(rows), int(entry_col_rpm.get()))
        else:
            messagebox.showinfo("Submission Result", f"\n\nNo data found")
            return
        
        try:
            if run:
                validate_rows(run, int(entry_col_time.get()), int(entry_col_rpm.get()), int(entry_col_speed.get()), check_speed=not deduce_var.get())
            else:
                messagebox.showinfo("Submission Result", f"\n\nNo VCDS log data found")
                return
        except ValueError as e:
            messagebox.showerror("Submission Result", e)
            return

        try:
            if run: 
                hp_torque = analyse_run(run) # Will calculate rpm, hp & torque
            else:
                messagebox.showinfo("Submission Result", f"\n\nNo VCDS log data found")
        except ValueError as e:
            messagebox.showinfo("Submission Result", e)
                

        if hp_torque:
            print_graph(hp_torque, graph_frame) # Will plot
        else:
            messagebox.showinfo("Submission Result", f"\n\nNo viable data found, check column parameters")
        param_frame.grid_remove()
        toggle_btn.config(text="Show parameters")
    
    except ValueError as e:
        messagebox.showerror("Error", "Please enter valid numeric values.")
        print(e)

# this function will find when the real data actually starts
def extract_data(rows):
    wordlist = ['Marker', 'STAMP', '/min', '°BTDC', '°KW']
    for i in range(len(rows)):
        row = rows[i]
        if any(word in row for word in wordlist):
            run = find_run_probable_range(rows[i+1:], int(entry_col_rpm.get())) # Will get the run range
            return run
    return None

def analyse_run(rows):
    car_weight = int(entry_mass.get())
    col_time_i = int(entry_col_time.get())
    col_rpm_i = int(entry_col_rpm.get())
    col_speed_i = int(entry_col_speed.get())
    air_density = float(entry_air_density.get())
    air_temp = float(entry_temp.get())
    air_pressure_mbar = float(entry_col_air_pressure.get())
    gravity = float(entry_gravity.get())
    rolling_coeff = float(entry_crr.get())
    scx = float(entry_scx.get())
    drivetrain_loss = float(entry_gearbox_loss.get())
    final_hp_torque_curve = []
    for i in range(len(rows)):
        prev_row = None if i == 0 else rows[i-1]
        if prev_row is None: continue
        row = rows[i]

        # All the cool calculated values in order to extract two values, rpm and hp => torque
        delta_time = float(row[col_time_i]) - float(prev_row[col_time_i])
        prev_rpm = int(float(prev_row[col_rpm_i]))
        rpm = int(float(row[col_rpm_i]))
        if deduce_var.get():
            prev_speed_ms = get_speed_from_rpm(prev_rpm) / 3.6
            speed_ms = get_speed_from_rpm(rpm) / 3.6
        else:
            prev_speed_ms = float(prev_row[col_speed_i]) / 3.6
            speed_ms = float(row[col_speed_i]) / 3.6
        speed_mean = (speed_ms + prev_speed_ms) / 2
        delta_speed = speed_ms - prev_speed_ms
        acceleration = delta_speed / delta_time
        force = acceleration * car_weight
        power_kw = force * speed_mean * 0.001
        whp = power_kw * 1.341
        air_loss_hp = 0.5 * air_density * scx * speed_ms ** 3 * 0.001 * 1.341
        rolling_loss_hp = car_weight * rolling_coeff * gravity * speed_ms * 0.001 * 1.341
        whp_with_losses  = whp + air_loss_hp + rolling_loss_hp
        crank_hp = whp_with_losses / drivetrain_loss
        crank_torque = (crank_hp * 7022) / rpm

        # Apply DIN 70020 if needed
        if din_var.get():
            crank_hp, crank_torque = apply_din_correction(
                crank_hp, crank_torque, air_temp, air_pressure_mbar
            )

        final_hp_torque_curve.append((rpm, crank_hp, crank_torque))
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
            text = child.cget('text')
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
        print(param_text)

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

def get_speed_from_rpm(rpm):

    width, aspect, diameter = parse_tire_size(entry_tire.get())
    gear_ratio = float(entry_gearbox_ratio.get())
    diff_ratio = float(entry_diff_ratio.get())

    perim = wheel_perimeter_cm(width, aspect, diameter)

    final_ratio = gear_ratio * diff_ratio

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

def print_graph(rpm_hp_torque, graph_frame):

    global canvas_widget

    """
    Plot HP and torque vs RPM on two Y axes in a Tkinter canvas frame,
    with smoothing, peak labels, and interactive pointer.

    Args:
        rpm_hp_torque (list of tuple): List of (RPM, HP, Torque) values.
        graph_frame (tk.Frame): The Tkinter frame to hold the canvas.
    """
    # Clear the frame
    for widget in graph_frame.winfo_children():
        widget.destroy()

    # Extract data
    rpm = np.array([point[0] for point in rpm_hp_torque])
    hp = np.array([point[1] for point in rpm_hp_torque])
    torque = np.array([point[2] for point in rpm_hp_torque])


    # I want for ex 100 points per 1000 rpm
    npoints = int((int(rpm.max()) - int(rpm.min())) / 10)

    def moving_average(y, window_size=5):
        return np.convolve(y, np.ones(window_size)/window_size, mode='same')
    
    if smooth_var.get():
        hp = moving_average(hp)
        torque = moving_average(torque)

    # Smoothing
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
    ax1.set_ylabel('HP', color=color_hp)
    ax1.plot(rpm_smooth, hp_spline, color=color_hp, label='HP')
    ax1.tick_params(axis='y', labelcolor=color_hp)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Torque', color=color_torque)
    ax2.plot(rpm_smooth, torque_spline, color=color_torque, label='Torque')
    ax2.tick_params(axis='y', labelcolor=color_torque)

    # Peak labels
    hp_peak_idx = np.argmax(hp_spline)
    torque_peak_idx = np.argmax(torque_spline)
    ax1.annotate(f'Peak HP: {hp_spline[hp_peak_idx]:.1f}', 
                 xy=(rpm_smooth[hp_peak_idx], hp_spline[hp_peak_idx]),
                 xytext=(rpm_smooth[hp_peak_idx], hp_spline[hp_peak_idx] + 10),
                 arrowprops=dict(facecolor=color_hp, arrowstyle="->"),
                 color=color_hp)

    ax2.annotate(f'Peak Torque: {torque_spline[torque_peak_idx]:.1f}',
                 xy=(rpm_smooth[torque_peak_idx], torque_spline[torque_peak_idx]),
                 xytext=(rpm_smooth[torque_peak_idx], torque_spline[torque_peak_idx] + 10),
                 arrowprops=dict(facecolor=color_torque, arrowstyle="->"),
                 color=color_torque)

    # Annot per axis
    annot_hp = ax1.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
    annot_hp.set_visible(False)

    annot_torque = ax2.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                                bbox=dict(boxstyle="round", fc="w"),
                                arrowprops=dict(arrowstyle="->"))
    annot_torque.set_visible(False)


    def on_mouse_move(event):
        if event.inaxes not in [ax1, ax2]:
            annot_hp.set_visible(False)
            annot_torque.set_visible(False)
            fig.canvas.draw_idle()
            return

        x = event.xdata
        if x is None:
            return

        # Find closest index
        idx = np.abs(rpm_smooth - x).argmin()

        # Update HP annot
        annot_hp.xy = (rpm_smooth[idx], hp_spline[idx])
        annot_hp.set_text(f"HP: {hp_spline[idx]:.1f} @ {rpm_smooth[idx]:.0f} RPM")
        annot_hp.set_visible(True)

        # Update Torque annot
        annot_torque.xy = (rpm_smooth[idx], torque_spline[idx])
        annot_torque.set_text(f"Torque: {torque_spline[idx]:.1f} @ {rpm_smooth[idx]:.0f} RPM")
        annot_torque.set_visible(True)

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)

    fig.tight_layout()

    if canvas_widget:
        canvas_widget.get_tk_widget().destroy()

    canvas_widget = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas_widget.draw()
    canvas_widget.get_tk_widget().pack(fill="both", expand=True)

    plt.close(fig)
    print_button.grid()

def run_remove_duplicates(rows, rpm_col_idx=2):

    if not rows: return None

    # To remove duplicates and keep the first occurrence of each RPM
    unique_data = []
    seen_rpms = set()

    for row in rows:
        rpm = row[rpm_col_idx]
        if rpm not in seen_rpms:
            unique_data.append(row)
            seen_rpms.add(rpm)
    return unique_data

def find_run_probable_range(rows, rpm_col_idx=2):
    """
    Find the longest sequence of rows where RPMs continuously increase or stay the same,
    skipping empty rows or rows with empty RPM.

    Args:
        rows (list of list): The data rows (each row is a list of values).
        rpm_col_idx (int): The index of the RPM column.

    Returns:
        list of list: The subset of rows representing the probable run.
    """
    best_start = 0
    best_end = 0
    best_length = 0
    best_diff = 0
    current_start = None
    prev_rpm = None
    i = 0
    min_rpm = 0

    def is_run_better():
        is_rpm_diff_acceptable = rpm_diff > 800
        is_rpm_diff_bigger_than_best = rpm_diff > best_diff
        is_run_larger_than_best = current_length > best_length
        min_correct_run_size = max(15, rpm_diff / 200) 
        return is_rpm_diff_acceptable and (
            (is_rpm_diff_bigger_than_best and is_run_larger_than_best) or 
            (is_rpm_diff_bigger_than_best and current_length >= min_correct_run_size)
        )

    for i, row in enumerate(rows):
        
        # Skip rows that are empty or with missing RPM
        if not row or len(row) <= rpm_col_idx or row[rpm_col_idx] == '':
            break

        try:
            curr_rpm = float(row[rpm_col_idx])
        except ValueError:
            continue

        if prev_rpm is None:
            # First valid RPM we find
            current_start = i
            min_rpm = curr_rpm
        elif curr_rpm >= prev_rpm:
            # RPM continues to rise or stays flat -> keep going
            pass
        else:
            # RPM dropped -> check if this is the best run so far
            current_length = i - current_start
            rpm_diff = curr_rpm - min_rpm
            if is_run_better():
                best_length = current_length
                best_diff = rpm_diff
                best_start = current_start
                best_end = i
            # Start a new run
            current_start = i

        prev_rpm = curr_rpm

    # Final check after the loop in case the longest run is at the end
    current_length = i - current_start
    if current_start is not None and is_run_better():
        best_start = current_start
        best_end = i

    # That code is mostly useless, I don't see why there could be an invalid row in between data
    # Return only rows that are non-empty and have valid RPM in the range
    result = []
    for row in rows[best_start:best_end]:
        if row and len(row) > rpm_col_idx and row[rpm_col_idx] != '':
            result.append(row)

    return result

def load_log_file():
    global log_file_path
    file_path = filedialog.askopenfilename(title="Select log file", filetypes=[("CSV or log files", "*.csv *.log *.txt"), ("All files", "*.*")])
    if file_path:
        log_file_path = file_path
        log_label.config(text=f"Loaded: {file_path}", bootstyle="success")
    else:
        log_file_path = None
        log_label.config(text="No log file loaded", bootstyle='warning')

def toggle_params():
    if param_frame.winfo_ismapped():
        param_frame.grid_remove()
        toggle_btn.config(text="Show parameters")
    else:
        param_frame.grid()
        toggle_btn.config(text="Hide parameters")

def toggle_params():
    if param_frame.winfo_ismapped():
        param_frame.grid_remove()
        toggle_btn.config(text="Show parameters")
    else:
        param_frame.grid()
        toggle_btn.config(text="Hide parameters")

root = ttkb.Window(themename="superhero")
root.title("VCDS Log Dyno")
root.geometry("1000x850")

param_frame = ttkb.LabelFrame(root, text="Parameters")
param_frame.pack(padx=10, pady=10)
param_frame.grid(row=0, column=0, columnspan=2, sticky="we", padx=10, pady=10)

# FIRST COLUMN
ttkb.Label(param_frame, text="Car mass (kg)").grid(row=0, column=0, sticky="e", padx=5, pady=5)
entry_mass = ttkb.Entry(param_frame)
entry_mass.insert(0, "1240")
entry_mass.grid(row=0, column=1, sticky="we", padx=5, pady=5)

ttkb.Label(param_frame, text="Air density (kg/m³)").grid(row=1, column=0, sticky="e", padx=5, pady=5)
entry_density_var = ttkb.StringVar()
entry_air_density = ttkb.Entry(param_frame, textvariable=entry_density_var)
entry_density_var.set("1.1755")
entry_air_density.grid(row=1, column=1, sticky="we", padx=5, pady=5)

ttkb.Label(param_frame, text="Temperature (°C)").grid(row=2, column=0, sticky="e", padx=5, pady=5)
entry_temp_var = ttkb.StringVar()
entry_temp = ttkb.Entry(param_frame, textvariable=entry_temp_var)
entry_temp_var.set("25")
entry_temp.grid(row=2, column=1, sticky="we", padx=5, pady=5)

ttkb.Label(param_frame, text="Humidity (%)").grid(row=3, column=0, sticky="e", padx=5, pady=5)
entry_humidity_var = ttkb.StringVar()
entry_humidity = ttkb.Entry(param_frame, textvariable=entry_humidity_var)
entry_humidity_var.set("60")
entry_humidity.grid(row=3, column=1, sticky="we", padx=5, pady=5)

ttkb.Label(param_frame, text="SCx").grid(row=4, column=0, sticky="e", padx=5, pady=5)
entry_scx = ttkb.Entry(param_frame)
entry_scx.insert(0, "0.65")
entry_scx.grid(row=4, column=1, sticky="we", padx=5, pady=5)

ttkb.Label(param_frame, text="Gearbox loss").grid(row=5, column=0, sticky="e", padx=5, pady=5)
entry_gearbox_loss = ttkb.Entry(param_frame)
entry_gearbox_loss.insert(0, "0.87")
entry_gearbox_loss.grid(row=5, column=1, sticky="we", padx=5, pady=5)

def toggle_deduce_fields():
    state = "normal" if deduce_var.get() else "disabled"
    entry_tire.config(state=state)
    entry_gearbox_ratio.config(state=state)
    entry_diff_ratio.config(state=state)
    entry_col_speed.config(state="disabled" if deduce_var.get() else "normal")


# --- Tire info field ---
ttkb.Label(param_frame, text="Tire size (e.g. 205/45 R16)").grid(row=6, column=0, sticky="e", padx=5, pady=5)
entry_tire = ttkb.Entry(param_frame)
entry_tire.insert(0, "205/45 R16")
entry_tire.grid(row=6, column=1, sticky="we", padx=5, pady=5)

# --- Gearbox ratio field ---
ttkb.Label(param_frame, text="Gearbox ratio").grid(row=7, column=0, sticky="e", padx=5, pady=5)
entry_gearbox_ratio = ttkb.Entry(param_frame)
entry_gearbox_ratio.insert(0, "1.324")
entry_gearbox_ratio.grid(row=7, column=1, sticky="we", padx=5, pady=5)

# --- Differential ratio field ---
ttkb.Label(param_frame, text="Differential ratio").grid(row=8, column=0, sticky="e", padx=5, pady=5)
entry_diff_ratio = ttkb.Entry(param_frame)
entry_diff_ratio.insert(0, "3.238")
entry_diff_ratio.grid(row=8, column=1, sticky="we", padx=5, pady=5)

# Apply DIN correction checkbox
din_var = ttkb.BooleanVar()
din_var.set(True)
din_checkbox = ttkb.Checkbutton(param_frame, text="Apply DIN correction", variable=din_var)
din_checkbox.grid(row=8, column=3, columnspan=2, sticky="w", padx=5, pady=5)

# Apply smoothing checkbox
smooth_var = ttkb.BooleanVar()
smooth_var.set(True)
smooth_checkbox = ttkb.Checkbutton(param_frame, text="Apply graph smoothing", variable=smooth_var)
smooth_checkbox.grid(row=7, column=3, columnspan=2, sticky="w", padx=5, pady=5)

# Apply point interpolation checkbox
interpolate_var = ttkb.BooleanVar()
interpolate_var.set(True)
interpolate_checkbox = ttkb.Checkbutton(param_frame, text="Apply point interp.", variable=interpolate_var)
interpolate_checkbox.grid(row=6, column=3, columnspan=2, sticky="w", padx=5, pady=5)

deduce_var = ttkb.BooleanVar(value=False)
deduce_checkbox = ttkb.Checkbutton(
    param_frame, text="Deduce speed from RPM", variable=deduce_var, command=toggle_deduce_fields
)
deduce_checkbox.grid(row=6, column=2, columnspan=2, sticky="w", padx=5, pady=5)

# SECOND COLUMN
ttkb.Label(param_frame, text="Crr").grid(row=0, column=2, sticky="e", padx=5, pady=5)
entry_crr = ttkb.Entry(param_frame)
entry_crr.insert(0, "0.019")
entry_crr.grid(row=0, column=3, sticky="we", padx=5, pady=5)

ttkb.Label(param_frame, text="Gravity (m/s²)").grid(row=1, column=2, sticky="e", padx=5, pady=5)
entry_gravity = ttkb.Entry(param_frame)
entry_gravity.insert(0, "9.81")
entry_gravity.grid(row=1, column=3, sticky="we", padx=5, pady=5)

ttkb.Label(param_frame, text="Time stamp column").grid(row=2, column=2, sticky="e", padx=5, pady=5)
entry_col_time = ttkb.Entry(param_frame)
entry_col_time.insert(0, "1")
entry_col_time.grid(row=2, column=3, sticky="we", padx=5, pady=5)

ttkb.Label(param_frame, text="Vehicle speed column").grid(row=3, column=2, sticky="e", padx=5, pady=5)
entry_col_speed = ttkb.Entry(param_frame)
entry_col_speed.insert(0, "3")
entry_col_speed.grid(row=3, column=3, sticky="we", padx=5, pady=5)

ttkb.Label(param_frame, text="RPM column").grid(row=4, column=2, sticky="e", padx=5, pady=5)
entry_col_rpm = ttkb.Entry(param_frame)
entry_col_rpm.insert(0, "2")
entry_col_rpm.grid(row=4, column=3, sticky="we", padx=5, pady=5)


ttkb.Label(param_frame, text="Air pressure (hPa)").grid(row=5, column=2, sticky="e", padx=5, pady=5)
entry_col_air_pressure = ttkb.Entry(param_frame)
entry_col_air_pressure.insert(0, "1013.25")
entry_col_air_pressure.grid(row=5, column=3, sticky="we", padx=5, pady=5)

for i in range(4):
    param_frame.columnconfigure(i, weight=1)

# LOAD LOG FILE
ttkb.Button(root, text="Load log file", command=load_log_file).grid(row=1, column=0, columnspan=2, pady=10)
log_label = ttkb.Label(root, text="No log file loaded", bootstyle="warning", wraplength=700, justify="left")
log_label.grid(row=2, column=0, columnspan=2, padx=10)

graph_frame = ttkb.Frame(root, relief="solid", borderwidth=2, bootstyle="default")
graph_frame.grid(row=14, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
root.rowconfigure(14, weight=1)  # pour que la ligne 14 s'étire
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)



# SUBMIT + TOGGLE BUTTON
ttkb.Button(root, text="Submit", command=submit).grid(row=5, column=0, pady=10)
toggle_btn = ttkb.Button(root, text="Hide parameters", command=toggle_params)
toggle_btn.grid(row=5, column=1, pady=10)

print_button = ttkb.Button(root, text="Print Graph", command=print_graph_to_printer)
print_button.grid(row=15, column=1, pady=20, sticky="w")
print_button.grid_remove()  # Start hidden

def on_graph_resize(event):
    if canvas_widget:
        canvas_widget.get_tk_widget().config(width=event.width, height=event.height)
        canvas_widget.draw()

entry_tire.config(state="disabled")
entry_gearbox_ratio.config(state="disabled")
entry_diff_ratio.config(state="disabled")

graph_frame.bind('<Configure>', on_graph_resize)
entry_temp.bind("<KeyRelease>", lambda e: schedule_update('temp'))
entry_air_density.bind("<KeyRelease>", lambda e: schedule_update('density'))
entry_humidity.bind("<KeyRelease>", lambda e: schedule_update(last_changed if last_changed else 'temp'))
entry_col_air_pressure.bind("<KeyRelease>", lambda e: schedule_update(last_changed if last_changed else 'temp'))

root.mainloop()