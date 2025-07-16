#!/bin/bash

apt-get update && apt-get install -y sudo

sudo apt-get update && \
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common && \
sudo add-apt-repository -y universe && \
sudo apt-get install -y qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools build-essential xvfb freeglut3-dev
sudo apt-get install -y python3 python3-pip python3-numpy python3-pandas python3-skimage python3-munkres
sudo pip3 install --break-system-packages skan torch

ln -s /usr/lib/x86_64-linux-gnu/libglut.so.3.12 /usr/lib/x86_64-linux-gnu/libglut.so.3

# To remove/kill all Xvfb or X display instances for all users (requires sudo/root):
sudo pkill Xvfb || true
sudo pkill X || true

# Kill all Xvfb or X display instances except the current user's shell/script
# This kills all Xvfb/X for the current user, but not the current shell:
pkill -u "$USER" Xvfb || true
pkill -u "$USER" X || true

# Initialize virtual display for vlab
# Remove 2>/dev/null to see error messages if Xvfb fails
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
# Initialize the runtime directory for XDG
export XDG_RUNTIME_DIR=/tmp/runtime-root

# Start browser and source the environment setup script
./bin/govlab.sh
# Run environment setup script
. ./bin/sourceme.sh
cd oofs/ext/bushell/nn
