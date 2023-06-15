microSWIFT Expendable Wave Buoy

Repository for operational code that runs on the microSWIFT

https://apl.washington.edu/SWIFT


To run:
  1. Enable UART on rPi instead of USB: cdn-learn.adafruit.com/downloads/pdf/adafruit-ultimate-gps-on-the-raspberry-pi.pdf
  1. `sudo pip install pynmea2`
  2. Add the following text to rc.local to run on boot (`sudo nano /etc/rc.local`)
```bash 
    # Run microswift
    cd /home/pi/microSWIFT
    bash run_microswift.sh &
```
  3. reboot the rPi
  4. Add scripts to the main .sh file as necessary
