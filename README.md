# EngineDyno
Dyno measure based on logs and real data

This project aims at measuring dyno data mainly based on timestamp, vehicle speed and rpm

All the values will depend on Vehicle mass, Air Density, Outdoor temperature/humidity, SCx, drivetrain loss and rolling resistance


The program is made to retrieve data based on VCDS logs but can be adjusted to read other CSV files

At the end you will see a Dyno graph including torque and power

For it to work, you need to change your ecu map to create a custom log channel including both RPM and vehicle speed.

A tutorial can be found here: https://www.ecuconnections.com/forum/viewtopic.php?f=6&t=23324

Do the log on a plane road, record temperature and humidity and make a log from low to high rpm full throttle
Run the program and carefully set all the parameters to good values.

Enjoy.

Moffa13