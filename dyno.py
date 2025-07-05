import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import messagebox, filedialog
import numpy as np
from scipy.interpolate import make_interp_spline
import tempfile
import os
import platform
import csv

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

# That's weird I agree
def density_to_temp(density, humidity):
    for temp_c in np.arange(-273.15, 1000, 0.1):
        try:
            if abs(temp_to_density(temp_c, humidity) - density) < 0.001:
                return temp_c
        except OverflowError as e:
            print(e)
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

def submit():
    try:
        mass = float(entry_mass.get())
        air_density = float(entry_air_density.get())
        scx = float(entry_scx.get())
        gearbox_loss = float(entry_gearbox_loss.get())
        crr = float(entry_crr.get())
        gravity = float(entry_gravity.get())
        col_time = entry_col_time.get()
        col_speed = entry_col_speed.get()
        col_rpm = entry_col_rpm.get()

        info_text = (f"Mass = {mass} kg\n"
                     f"Air density = {air_density} kg/m³\n"
                     f"SCx = {scx}\n"
                     f"Gearbox loss = {gearbox_loss}\n"
                     f"Crr = {crr}\n"
                     f"Gravity = {gravity} m/s²\n"
                     f"Time stamp column: {col_time}\n"
                     f"Vehicle speed column: {col_speed}\n"
                     f"RPM column: {col_rpm}")

        if log_file_path:
            try:
                with open(log_file_path, newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    rows = list(reader)
                    if rows:
                        extract_data(rows)
                    else:
                        messagebox.showinfo("Submission Result", f"\n\nLog file is empty")
            except Exception as e:
                messagebox.showinfo("Submission Result", f"\n\nError reading log file: {e}")
        else:
            messagebox.showinfo("Submission Result", "\n\nNo log file loaded.")

    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric values.")

# this function will find when the real data actually starts
def extract_data(rows):
    wordlist = ['Marker', 'STAMP', '/min', '°BTDC', '°KW']
    for i in range(len(rows)):
        row = rows[i]
        if any(word in row for word in wordlist):
            run = find_run_probable_range(rows[i+1:], int(entry_col_rpm.get()))
            hp_torque = analyse_run(run)
            print_graph(hp_torque, graph_frame)
            param_frame.grid_remove()
            toggle_btn.config(text="Show parameters")
            break

def analyse_run(rows):
    car_weight = int(entry_mass.get())
    col_time_i = int(entry_col_time.get())
    col_rpm_i = int(entry_col_rpm.get())
    col_speed_i = int(entry_col_speed.get())
    air_density = float(entry_air_density.get())
    gravity = float(entry_gravity.get())
    rolling_coeff = float(entry_crr.get())
    scx = float(entry_scx.get())
    drivetrain_loss = float(entry_gearbox_loss.get())
    final_hp_torque_curve = []
    for i in range(len(rows)):
        prev_row = None if i == 0 else rows[i-1]
        if prev_row is None: continue
        row = rows[i]
        delta_time = float(row[col_time_i]) - float(prev_row[col_time_i])
        prev_speed_ms = float(prev_row[col_speed_i]) / 3.6
        speed_ms = float(row[col_speed_i]) / 3.6
        speed_mean = (speed_ms + prev_speed_ms) / 2
        rpm = int(row[col_rpm_i])
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
        final_hp_torque_curve.append((rpm, crank_hp, crank_torque))
    return final_hp_torque_curve

def print_graph_to_printer():
    if not canvas_widget:
        return
    
    # Save graph to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
        canvas_widget.figure.savefig(tmpfile.name)
        tmpfile_path = tmpfile.name

    # Send to printer based on OS
    if platform.system() == 'Windows':
        # On Windows use start command with /print
        os.startfile(tmpfile_path, 'print')
    elif platform.system() == 'Darwin':
        # macOS
        os.system(f'lpr "{tmpfile_path}"')
    elif platform.system() == 'Linux':
        # Linux
        os.system(f'lpr "{tmpfile_path}"')
    else:
        print("Unsupported OS for direct printing")

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

    # Smoothing
    if len(rpm) >= 4:
        rpm_smooth = np.linspace(rpm.min(), rpm.max(), 300)
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
    hp_line, = ax1.plot(rpm_smooth, hp_spline, color=color_hp, label='HP')
    ax1.tick_params(axis='y', labelcolor=color_hp)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Torque', color=color_torque)
    torque_line, = ax2.plot(rpm_smooth, torque_spline, color=color_torque, label='Torque')
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

    current_start = None
    prev_rpm = None

    for i, row in enumerate(rows):
        # Skip rows that are empty or with missing RPM
        if not row or len(row) <= rpm_col_idx or row[rpm_col_idx] == '':
            continue

        try:
            curr_rpm = float(row[rpm_col_idx])
        except ValueError:
            continue

        if prev_rpm is None:
            # First valid RPM we find
            current_start = i
        elif curr_rpm >= prev_rpm:
            # RPM continues to rise or stays flat -> keep going
            pass
        else:
            # RPM dropped -> check if this is the best run so far
            current_length = i - current_start
            if current_length > best_length:
                best_length = current_length
                best_start = current_start
                best_end = i
            # Start a new run
            current_start = i

        prev_rpm = curr_rpm

    # Final check after the loop in case the longest run is at the end
    if current_start is not None:
        current_length = len(rows) - current_start
        if current_length > best_length:
            best_start = current_start
            best_end = len(rows)

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
        log_label.config(text=f"Loaded: {file_path}")
    else:
        log_file_path = None
        log_label.config(text="No log file loaded")

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

root = tk.Tk()
root.title("VCDS Log Dyno")
root.geometry("750x700")

param_frame = tk.LabelFrame(root, text="Parameters", padx=10, pady=10)
param_frame.grid(row=0, column=0, columnspan=2, sticky="we", padx=10, pady=10)

# FIRST COLUMN
tk.Label(param_frame, text="Car mass (kg)").grid(row=0, column=0, sticky="e", padx=5, pady=5)
entry_mass = tk.Entry(param_frame)
entry_mass.insert(0, "1240")
entry_mass.grid(row=0, column=1, sticky="we", padx=5, pady=5)

tk.Label(param_frame, text="Air density (kg/m³)").grid(row=1, column=0, sticky="e", padx=5, pady=5)
entry_density_var = tk.StringVar()
entry_air_density = tk.Entry(param_frame, textvariable=entry_density_var)
entry_density_var.set("1.1755")
entry_air_density.grid(row=1, column=1, sticky="we", padx=5, pady=5)

tk.Label(param_frame, text="Temperature (°C)").grid(row=2, column=0, sticky="e", padx=5, pady=5)
entry_temp_var = tk.StringVar()
entry_temp = tk.Entry(param_frame, textvariable=entry_temp_var)
entry_temp_var.set("25")
entry_temp.grid(row=2, column=1, sticky="we", padx=5, pady=5)

tk.Label(param_frame, text="Humidity (%)").grid(row=3, column=0, sticky="e", padx=5, pady=5)
entry_humidity_var = tk.StringVar()
entry_humidity = tk.Entry(param_frame, textvariable=entry_humidity_var)
entry_humidity_var.set("60")
entry_humidity.grid(row=3, column=1, sticky="we", padx=5, pady=5)

tk.Label(param_frame, text="SCx").grid(row=4, column=0, sticky="e", padx=5, pady=5)
entry_scx = tk.Entry(param_frame)
entry_scx.insert(0, "0.65")
entry_scx.grid(row=4, column=1, sticky="we", padx=5, pady=5)

tk.Label(param_frame, text="Gearbox loss").grid(row=5, column=0, sticky="e", padx=5, pady=5)
entry_gearbox_loss = tk.Entry(param_frame)
entry_gearbox_loss.insert(0, "0.87")
entry_gearbox_loss.grid(row=5, column=1, sticky="we", padx=5, pady=5)

# SECOND COLUMN
tk.Label(param_frame, text="Crr").grid(row=0, column=2, sticky="e", padx=5, pady=5)
entry_crr = tk.Entry(param_frame)
entry_crr.insert(0, "0.012")
entry_crr.grid(row=0, column=3, sticky="we", padx=5, pady=5)

tk.Label(param_frame, text="Gravity (m/s²)").grid(row=1, column=2, sticky="e", padx=5, pady=5)
entry_gravity = tk.Entry(param_frame)
entry_gravity.insert(0, "9.81")
entry_gravity.grid(row=1, column=3, sticky="we", padx=5, pady=5)

tk.Label(param_frame, text="Time stamp column").grid(row=2, column=2, sticky="e", padx=5, pady=5)
entry_col_time = tk.Entry(param_frame)
entry_col_time.insert(0, "1")
entry_col_time.grid(row=2, column=3, sticky="we", padx=5, pady=5)

tk.Label(param_frame, text="Vehicle speed column").grid(row=3, column=2, sticky="e", padx=5, pady=5)
entry_col_speed = tk.Entry(param_frame)
entry_col_speed.insert(0, "3")
entry_col_speed.grid(row=3, column=3, sticky="we", padx=5, pady=5)

tk.Label(param_frame, text="RPM column").grid(row=4, column=2, sticky="e", padx=5, pady=5)
entry_col_rpm = tk.Entry(param_frame)
entry_col_rpm.insert(0, "2")
entry_col_rpm.grid(row=4, column=3, sticky="we", padx=5, pady=5)


tk.Label(param_frame, text="Air pressure (hPa)").grid(row=5, column=2, sticky="e", padx=5, pady=5)
entry_col_air_pressure = tk.Entry(param_frame)
entry_col_air_pressure.insert(0, "1013.25")
entry_col_air_pressure.grid(row=5, column=3, sticky="we", padx=5, pady=5)

for i in range(4):
    param_frame.columnconfigure(i, weight=1)

# LOAD LOG FILE
tk.Button(root, text="Load log file", command=load_log_file).grid(row=1, column=0, columnspan=2, pady=10)
log_label = tk.Label(root, text="No log file loaded", fg="gray", wraplength=700, justify="left")
log_label.grid(row=2, column=0, columnspan=2, padx=10)

graph_frame = tk.Frame(root, relief="sunken", borderwidth=2, bg="white")
graph_frame.grid(row=14, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
root.rowconfigure(14, weight=1)  # pour que la ligne 14 s'étire
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)



# SUBMIT + TOGGLE BUTTON
tk.Button(root, text="Submit", command=submit).grid(row=5, column=0, pady=10)
toggle_btn = tk.Button(root, text="Hide parameters", command=toggle_params)
toggle_btn.grid(row=5, column=1, pady=10)

print_button = tk.Button(root, text="Print Graph", command=print_graph_to_printer)
print_button.grid(row=15, column=1, pady=20, sticky="w")
print_button.grid_remove()  # Start hidden

def on_graph_resize(event):
    if canvas_widget:
        canvas_widget.get_tk_widget().config(width=event.width, height=event.height)
        canvas_widget.draw()


graph_frame.bind('<Configure>', on_graph_resize)
entry_temp.bind("<KeyRelease>", lambda e: schedule_update('temp'))
entry_air_density.bind("<KeyRelease>", lambda e: schedule_update('density'))
entry_humidity.bind("<KeyRelease>", lambda e: schedule_update(last_changed if last_changed else 'temp'))
entry_col_air_pressure.bind("<KeyRelease>", lambda e: schedule_update(last_changed if last_changed else 'temp'))

root.mainloop()