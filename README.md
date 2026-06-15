# EngineDyno

**Dyno measurement based on logs and real data**

This project aims to compute dyno data based mainly on timestamp, vehicle speed, and RPM.

The calculated power/torque will depend on:
- vehicle mass
- air density / pressure
- outdoor temperature / humidity
- SCx
- drivetrain loss
- rolling resistance

The program is designed to retrieve data based on **CSV logs**, such as VCDS, Dragy, ...

At the end, you will see a dyno graph including **torque** and **power**.

There is also a feature to compare runs between each other

---

## Requirements

For this to work, the logs should contain at least a time stamp and vehicule RPM or speed.

On EDC15 ecu, you need to modify your ECU map to create a custom log channel that includes both **RPM** and **vehicle speed** if you don't have gear ratios.

📖 A tutorial for EDC15 can be found here:  
[https://www.ecuconnections.com/forum/viewtopic.php?f=6&t=23324](https://www.ecuconnections.com/forum/viewtopic.php?f=6&t=23324)

Log data on a **flat road**, record:
- temperature
- air pressure
- humidity  

…and make a full-throttle log from **low to high RPM**.  
Run the program and carefully set all parameters to accurate values.

---


## Infos for VCDS Column Mapping

The time stamp, vehicle speed, and RPM columns refer to the VCDS log columns:

- **Group A**: time stamp (1), vehicle speed (2), RPM (3) → starts at 1  
- **Group B**: time stamp (6), vehicle speed (7), RPM (8) → starts at 6  
- **Group C**: time stamp (11), vehicle speed (12), RPM (13) → starts at 11  

⚠️ **Be careful to have all data in the same group** — otherwise, data retrieval could lead to inconsistencies (e.g. not reading RPM at the same time as vehicle speed).

## Gearbox Loss File

The program supports the use of a **gearbox loss file** to apply a speed-dependent drivetrain loss model instead of a fixed drivetrain loss percentage.

By default, drivetrain losses are calculated using the value entered in the **Drivetrain Loss (%)** field. While this method is simple and works reasonably well, real drivetrain losses are not constant and generally increase with vehicle speed.

A gearbox loss file is generated from a coast-down log. During the dyno calculation, the software interpolates between the recorded points and automatically applies the corresponding drivetrain loss at each speed.

### Required Data

The source log can be any supported format and may contain additional columns.

It only needs to contain at least:

* Timestamp
* Vehicle speed

The column names do not matter, as the software allows you to manually select which columns contain the required data.

### How to Create a Gearbox Loss File

1. Find a flat and safe road.
2. Accelerate the vehicle to a sufficiently high speed.
3. Shift into neutral or fully disengage the clutch.
4. Let the vehicle coast down naturally without braking.
5. Record a log containing timestamp and vehicle speed.
6. Import the log into EngineDyno and use the **Generate Gearbox Loss File** function.

The software will calculate the drivetrain losses from the deceleration rate and generate a gearbox loss file that can later be reused for dyno calculations.

### VCDS Example

With VCDS, a simple way to record vehicle speed is by logging the ABS controller, which provides a reliable vehicle speed signal independent of engine RPM.

### Notes

* For best results, perform the test on a flat road with minimal wind.
* Multiple runs in opposite directions can help reduce the influence of road slope and wind.
* The generated file can usually be reused as long as the vehicle configuration remains unchanged (gearbox, differential, tires, wheel bearings, etc.).
* When a gearbox loss file is loaded, it overrides the fixed drivetrain loss percentage.


## Auto deduce speed from RPM

If you do not have the speed data recorded into your log, you can check the "Deduce speed from RPM" option
to automatically retrieve the vehicle speed based on RPM, tire info along with gear and differential ratios.

## Auto deduce RPM from speed

If you do not have the rpm data recorded into your log which happens in logs that are generally recorded with devices not connected to the engine such as Dragy, you can check the "Deduce RPM from speed" option
to automatically retrieve the RPM based on speed, tire info along with gear and differential ratios.

## Decrypt logs from Dragy

Long click on the dragy run (Found in Me -> History -> 0-100, 1/4 mile, ...) will create an encrypted **json** file which can be decrypted using the following tool:

- [https://dragy-decryptor.d3vl.com/](https://dragy-decryptor.d3vl.com/)
---

## Definitions

- **SCx**: Drag surface (Cx × S)  
  → See: [https://en.wikipedia.org/wiki/Drag_coefficient](https://en.wikipedia.org/wiki/Drag_coefficient)

- **Crr**: Rolling resistance coefficient  
  → See: [https://en.wikipedia.org/wiki/Rolling_resistance](https://en.wikipedia.org/wiki/Rolling_resistance)

---

## How to Run the Program

To get started with the EngineDyno program, follow these steps:

1. **Install Dependencies**  
   Before running the program, you need to install the required dependencies. In the project folder, you should find a `requirements.txt` file. To install the necessary packages, run the following command in your terminal:

   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Program**
   After installing the dependencies, you can execute the program. Make sure to adjust the parameters (such as temperature, air pressure, and rolling resistance) to your actual conditions for accurate results.

Enjoy.  

*Moffa13*