# Previously built firmware of MicroPython with ULAB support for selected ports <br>

## MicroPython Version V1.23, ULab version: V6.5 <br>
**Note: Flashing guide For ESP32 Generic port** <br>
1. Make sure that esptool is already installed in your system (both Unix and Windows) <br>
2. Run **./flash.sh** in Unix/Linux machine <br>
3. For Windows users, Install Python (if not installed). Then run: ***python pip install esptool*** from command prompt (if not installed previously) <br>
4. Then run the following command from your shell or command prompt:<br>

```
python esptool.py -p /dev/ttyUSB0 -b 460800 --before default_reset --after hard_reset --chip esp32  write_flash --flash_mode dio --flash_size detect --flash_freq 40m 0x1000 bootloader/bootloader.bin 0x8000 partition_table/partition-table.bin 0x10000 firmware.bin
```

**Note:** For Windows user replace serial port /dev/ttyUSB0 by COM port number such as COM5 etc... <br>
          In Windows, check COM port number from the device manager, it will show something like CP2102 USB to serial interface. <br>
For windows machines, use the following command to install esptool: <br>
```
python -m pip install esptool
```
<br>While flashing the script, make sure that your command prompt/shell should point the current directory i.e. ESP32_GENERIC_ULAB_Micropython
           
