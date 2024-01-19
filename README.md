microSWIFT Expendable Wave Buoy

Repository for operational code that runs on the microSWIFT

https://apl.washington.edu/SWIFT


To run:
  1. Enable UART on rPi instead of USB: page 9 of cdn-learn.adafruit.com/downloads/pdf/adafruit-ultimate-gps-on-the-raspberry-pi.pdf
  2. `sudo pip install pynmea2`
  3. "cd" to the microSWIFT folder and switch branches:
```bash 
    cd /home/pi/microSWIFT
    git checkout witt_interference
```
  4. Add the following text to rc.local to run on boot (`sudo nano /etc/rc.local`)
```bash 
    # Run microswift
    cd /home/pi/microSWIFT
    bash run_microswift.sh &
```
  5. reboot the rPi
  6. Add scripts to the main .sh file as necessary
