
#For CPU monitoring:
pip3 install bpytop --upgrade

For GPU monitorin:
sudo apt-get update

#Install gpu drivers, see glances_ins.sh gist or : https://help.ubuntu.com/community/BinaryDriverHowto/Nvidia

# For Ubuntu 19:
# sudo apt install nvtop

#Older Ubuntu versions:
git clone https://github.com/Syllo/nvtop.git
mkdir -p nvtop/build && cd nvtop/build
cmake ..
make

# Install globally on the system
sudo make install

# Alternatively, install without privileges at a location of your choosing
# make DESTDIR="/your/install/path" install
