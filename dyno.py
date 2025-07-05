import tkinter as tk
from tkinter import messagebox, filedialog
import csv

log_file_path = None
update_job = None
last_changed = None  # Will be 'temp' or 'density'

def temp_to_density(temp_c, humidity):
    T = temp_c + 273.15
    es = 6.1078 * 10 ** ((7.5 * temp_c) / (237.3 + temp_c))
    e = es * (humidity / 100.0)
    p = 1013.25
    pd = p - e
    Rd = 287.05
    Rv = 461.495
    rho = ((pd * 100) / (Rd * T)) + ((e * 100) / (Rv * T))
    return round(rho, 4)

def density_to_temp(density, humidity):
    for temp_c in range(-40, 50):
        if abs(temp_to_density(temp_c, humidity) - density) < 0.01:
            return temp_c
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
            density_var.set(str(rho))
        except:
            pass
    elif last_changed == 'density':
        try:
            rho = float(entry_air_density.get())
            temp_c_calc = density_to_temp(rho, humidity)
            if temp_c_calc is not None:
                temp_var.set(str(temp_c_calc))
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
            print(rows[i+1:10])
    run = find_run_probable_range(rows, int(entry_col_rpm.get()))
    analyse_run(run)

def analyse_run(a):
    for arr in a:
        line = ""
        for num in arr:
            line += str(num)+" "
        print(line)

def find_run_probable_range(rows, rpm_col_idx=2):
    """
    Find the longest sequence of rows where RPMs continuously increase or stay the same.

    Args:
        rows (list of list): The data rows (each row is a list of values).
        rpm_col_idx (int): The index of the RPM column.

    Returns:
        list of list: The subset of rows representing the probable run.
    """
    best_start = 0
    best_end = 0
    best_length = 0

    current_start = 0

    for i in range(1, len(rows)):
        try:
            prev_rpm = float(rows[i - 1][rpm_col_idx])
            curr_rpm = float(rows[i][rpm_col_idx])
        except (ValueError, IndexError):
            # Skip rows with invalid or missing RPM
            continue

        if curr_rpm >= prev_rpm:
            # continue the current increasing sequence
            continue
        else:
            # end of current increasing sequence
            current_length = i - current_start
            if current_length > best_length:
                best_length = current_length
                best_start = current_start
                best_end = i
            # start a new sequence
            current_start = i

    # Check the last sequence
    current_length = len(rows) - current_start
    if current_length > best_length:
        best_start = current_start
        best_end = len(rows)

    return rows[best_start:best_end]

def load_log_file():
    global log_file_path
    file_path = filedialog.askopenfilename(title="Select log file", filetypes=[("CSV or log files", "*.csv *.log *.txt"), ("All files", "*.*")])
    if file_path:
        log_file_path = file_path
        log_label.config(text=f"Loaded: {file_path}")
    else:
        log_file_path = None
        log_label.config(text="No log file loaded")

root = tk.Tk()
root.title("Car Parameters Input")
root.geometry("750x780")

tk.Label(root, text="Car mass (kg)").grid(row=0, column=0, sticky="e", padx=5, pady=5)
entry_mass = tk.Entry(root)
entry_mass.insert(0, "1240")
entry_mass.grid(row=0, column=1, sticky="we", padx=5, pady=5)

tk.Label(root, text="Air density (kg/m³)").grid(row=1, column=0, sticky="e", padx=5, pady=5)
density_var = tk.StringVar()
entry_air_density = tk.Entry(root, textvariable=density_var)
entry_air_density.insert(0, "1.22")
entry_air_density.grid(row=1, column=1, sticky="we", padx=5, pady=5)

tk.Label(root, text="Temperature (°C)").grid(row=2, column=0, sticky="e", padx=5, pady=5)
temp_var = tk.StringVar()
entry_temp = tk.Entry(root, textvariable=temp_var)
entry_temp.insert(0, "15")
entry_temp.grid(row=2, column=1, sticky="we", padx=5, pady=5)

tk.Label(root, text="Humidity (%)").grid(row=3, column=0, sticky="e", padx=5, pady=5)
entry_humidity = tk.Entry(root)
entry_humidity.insert(0, "50")
entry_humidity.grid(row=3, column=1, sticky="we", padx=5, pady=5)

tk.Label(root, text="SCx").grid(row=4, column=0, sticky="e", padx=5, pady=5)
entry_scx = tk.Entry(root)
entry_scx.insert(0, "0.65")
entry_scx.grid(row=4, column=1, sticky="we", padx=5, pady=5)

tk.Label(root, text="Gearbox loss").grid(row=5, column=0, sticky="e", padx=5, pady=5)
entry_gearbox_loss = tk.Entry(root)
entry_gearbox_loss.insert(0, "0.9")
entry_gearbox_loss.grid(row=5, column=1, sticky="we", padx=5, pady=5)

tk.Label(root, text="Crr").grid(row=6, column=0, sticky="e", padx=5, pady=5)
entry_crr = tk.Entry(root)
entry_crr.insert(0, "0.012")
entry_crr.grid(row=6, column=1, sticky="we", padx=5, pady=5)

tk.Label(root, text="Gravity (m/s²)").grid(row=7, column=0, sticky="e", padx=5, pady=5)
entry_gravity = tk.Entry(root)
entry_gravity.insert(0, "9.81")
entry_gravity.grid(row=7, column=1, sticky="we", padx=5, pady=5)

tk.Label(root, text="Time stamp column").grid(row=8, column=0, sticky="e", padx=5, pady=5)
entry_col_time = tk.Entry(root)
entry_col_time.insert(0, "1")
entry_col_time.grid(row=8, column=1, sticky="we", padx=5, pady=5)

tk.Label(root, text="Vehicle speed column").grid(row=9, column=0, sticky="e", padx=5, pady=5)
entry_col_speed = tk.Entry(root)
entry_col_speed.insert(0, "3")
entry_col_speed.grid(row=9, column=1, sticky="we", padx=5, pady=5)

tk.Label(root, text="RPM column").grid(row=10, column=0, sticky="e", padx=5, pady=5)
entry_col_rpm = tk.Entry(root)
entry_col_rpm.insert(0, "2")
entry_col_rpm.grid(row=10, column=1, sticky="we", padx=5, pady=5)

tk.Button(root, text="Load log file", command=load_log_file).grid(row=11, column=0, columnspan=2, pady=10)
log_label = tk.Label(root, text="No log file loaded", fg="gray", wraplength=700, justify="left")
log_label.grid(row=12, column=0, columnspan=2, padx=10)

tk.Label(root, text="Graph placeholder:").grid(row=13, column=0, columnspan=2)
graph_canvas = tk.Canvas(root, width=700, height=250, bg="white", relief="sunken", borderwidth=2)
graph_canvas.grid(row=14, column=0, columnspan=2, padx=10, pady=10)

tk.Button(root, text="Submit", command=submit).grid(row=15, column=0, columnspan=2, pady=20)

root.columnconfigure(1, weight=1)

# Bind with info on source
entry_temp.bind("<KeyRelease>", lambda e: schedule_update('temp'))
entry_air_density.bind("<KeyRelease>", lambda e: schedule_update('density'))
entry_humidity.bind("<KeyRelease>", lambda e: schedule_update(last_changed if last_changed else 'temp'))

root.mainloop()
