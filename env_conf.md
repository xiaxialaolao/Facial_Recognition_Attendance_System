
# Linux Development Environment and Python Virtual Environments

## Part 1: Burn the OS to the SIM Card

1. Install the Imager from official website:  
   https://www.raspberrypi.com/software/

2. Prepare a SIM card and a SIM card reader.

3. Run the software and insert the card reader with the SIM card into the computer.

4. Select the **Raspberry Pi 5** for the devices.

5. Select **64-bit OS version**.

6. Select the **storage** for burning OS.

7. Edit the **OS Customisation** and save.

8. Wait for the OS to be burned.

9. After finish, take out the SIM card, put it into the card slot, and run Raspberry Pi.

---

## Part 2: Environment Configuration

```bash
sudo passwd root
sudo raspi-config  # Select VNC or another development-friendly option
su root
sudo nano /boot/firmware/config.txt  # Add at the end of the file
```

```bash
[all]
dtoverlay=over5647
dtoverlay=over5647,cam1
```

```bash
sudo apt install libcamera-apps python3-libcamera libcap-dev cmake -y
```

---

## Part 3: Python Virtual Environment

```bash
python3 -m venv FRAS_env
cd FRAS_env
source bin/activate  # Use 'deactivate' to close
```

Install requi#C42C48 packages:

```bash
pip install opencv-python
pip install opencv-contrib-python==4.7.0.68
pip install picamera2
pip install numpy
pip install face_recognition
pip install scipy
pip install mysql.connector
pip install RPi.GPIO
pip install psutil
```

Fix libcamera import failure:

```bash
echo 'export PYTHONPATH="/usr/lib/python3/dist-packages:$PYTHONPATH"' >> ~/FRAS_env/bin/activate
```

---

## Part 4: Apache and SQL

```bash
sudo apt install apache2 phpmyadmin php mariadb-server -y
sudo nano /etc/apache2/apache2.conf  # Add at the end
```

```bash
#Include & Configure FRAS_env soft link and permissions:
Include /etc/phpmyadmin/apache.conf

<Directory /var/www/html/FRAS_env>
   Options +FollowSymLinks
    AllowOverride All
   Require all granted
</Directory>
```

Restart and enable Apache:

```bash
sudo systemctl restart apache2
sudo systemctl enable apache2
sudo systemctl start apache2
```

Set up MySQL user:

```bash
mysql -u root

create user 'xiaxialaolao'@'localhost' identified by 'xiaxialaolao';
GRANT ALL PRIVILEGES ON *.* TO 'xiaxialaolao'@'localhost' WITH GRANT OPTION;
FLUSH PRIVILEGES;
```

Set up symbolic link:

```bash
cd /var/www/html/
ln -s /home/xiaxialaolao/FRAS_env/ FRAS_env
```

Create directories and set permissions:

```bash
cd /home/xiaxialaolao/FRAS/
mkdir -p Profile_Pictures/
mkdir -p Image_DataSet/
mkdir -p Encoding_DataSet/

sudo chmod o+x /home/
sudo chmod o+x /home/xiaxialaolao/
sudo chmod o+x -R /home/xiaxialaolao/FRAS_env/

sudo chmod 777 /home/xiaxialaolao/FRAS_env/Profile_Pictures/
sudo chmod 777 /home/xiaxialaolao/FRAS_env/Image_DataSet/
sudo chmod 777 /home/xiaxialaolao/FRAS_env/Encoding_DataSet/
```

---

## Part 5: Firewall Configuration

```bash
sudo apt install ufw -y
su root

ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 5000/tcp
ufw allow 5900/tcp
ufw reload

sudo systemctl restart ufw
sudo systemctl enable ufw
sudo systemctl start ufw
sudo systemctl status ufw
```

---

## Part 6: Permission Settings

```bash
cd /home/xiaxialaolao/FRAS/
mkdir -p Profile_Pictures/
mkdir -p Image_DataSet/
mkdir -p Encoding_DataSet/

sudo chmod o+x /home/
sudo chmod o+x /home/xiaxialaolao/
sudo chmod o+x -R /home/xiaxialaolao/FRAS_env/

sudo chmod 777 /home/xiaxialaolao/FRAS_env/Profile_Pictures/
sudo chmod 777 /home/xiaxialaolao/FRAS_env/Image_DataSet/
sudo chmod 777 /home/xiaxialaolao/FRAS_env/Encoding_DataSet/
```

# Linux Development Environment and Python Virtual Environments

## Part 1: Burn the OS to the SIM Card

1. Install the Imager from official website:  
   https://www.raspberrypi.com/software/

2. Prepare a SIM card and a SIM card reader.

3. Run the software and insert the card reader with the SIM card into the computer.

4. Select the **Raspberry Pi 5** for the devices.

5. Select **64-bit OS version**.

6. Select the **storage** for burning OS.

7. Edit the **OS Customisation** and save.

8. Wait for the OS to be burned.

9. After finish, take out the SIM card, put it into the card slot, and run Raspberry Pi.

---

## Part 2: Environment Configuration

```bash
sudo passwd root
sudo raspi-config  # Select VNC or another development-friendly option
su root
sudo nano /boot/firmware/config.txt  # Add at the end of the file
```

```bash
[all]
dtoverlay=over5647
dtoverlay=over5647,cam1
```

```bash
sudo apt install libcamera-apps python3-libcamera libcap-dev cmake -y
```

---

## Part 3: Python Virtual Environment

```bash
python3 -m venv FRAS_env
cd FRAS_env
source bin/activate  # Use 'deactivate' to close
```

Install requi#C42C48 packages:

```bash
pip install opencv-python
pip install opencv-contrib-python==4.7.0.68
pip install picamera2
pip install numpy
pip install face_recognition
pip install scipy
pip install mysql.connector
pip install RPi.GPIO
pip install psutil
```

Fix libcamera import failure:

```bash
echo 'export PYTHONPATH="/usr/lib/python3/dist-packages:$PYTHONPATH"' >> ~/FRAS_env/bin/activate
```

---

## Part 4: Apache and SQL

```bash
sudo apt install apache2 phpmyadmin php mariadb-server -y
sudo nano /etc/apache2/apache2.conf  # Add at the end
```

```bash
#Include & Configure FRAS_env soft link and permissions:
Include /etc/phpmyadmin/apache.conf

<Directory /var/www/html/FRAS_env>
   Options +FollowSymLinks
    AllowOverride All
   Require all granted
</Directory>
```

Restart and enable Apache:

```bash
sudo systemctl restart apache2
sudo systemctl enable apache2
sudo systemctl start apache2
```

Set up MySQL user:

```bash
mysql -u root

create user 'xiaxialaolao'@'localhost' identified by 'xiaxialaolao';
GRANT ALL PRIVILEGES ON *.* TO 'xiaxialaolao'@'localhost' WITH GRANT OPTION;
FLUSH PRIVILEGES;
```

Set up symbolic link:

```bash
cd /var/www/html/
ln -s /home/xiaxialaolao/FRAS_env/ FRAS_env
```

Create directories and set permissions:

```bash
cd /home/xiaxialaolao/FRAS/
mkdir -p Profile_Pictures/
mkdir -p Image_DataSet/
mkdir -p Encoding_DataSet/

sudo chmod o+x /home/
sudo chmod o+x /home/xiaxialaolao/
sudo chmod o+x -R /home/xiaxialaolao/FRAS_env/

sudo chmod 777 /home/xiaxialaolao/FRAS_env/Profile_Pictures/
sudo chmod 777 /home/xiaxialaolao/FRAS_env/Image_DataSet/
sudo chmod 777 /home/xiaxialaolao/FRAS_env/Encoding_DataSet/
```

---

## Part 5: Firewall Configuration

```bash
sudo apt install ufw -y
su root

ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 5000/tcp
ufw allow 5900/tcp
ufw reload

sudo systemctl restart ufw
sudo systemctl enable ufw
sudo systemctl start ufw
sudo systemctl status ufw
```

---

## Part 6: Permission Settings

```bash
cd /home/xiaxialaolao/FRAS/
mkdir -p Profile_Pictures/
mkdir -p Image_DataSet/
mkdir -p Encoding_DataSet/

sudo chmod o+x /home/
sudo chmod o+x /home/xiaxialaolao/
sudo chmod o+x -R /home/xiaxialaolao/FRAS_env/

sudo chmod 777 /home/xiaxialaolao/FRAS_env/Profile_Pictures/
sudo chmod 777 /home/xiaxialaolao/FRAS_env/Image_DataSet/
sudo chmod 777 /home/xiaxialaolao/FRAS_env/Encoding_DataSet/
```
