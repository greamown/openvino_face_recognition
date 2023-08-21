#!/bin/bash
# ---------------------------------------------------------
# color ANIS
RED='\033[1;31m'
BLUE='\033[1;34m'
YELLOW='\033[1;33m'
GREEN='\033[1;32m'
NC='\033[0m'

function printstr(){
    echo -e "${BLUE}"
    echo $1
    echo -e "${NC}"
}

# Initial
printstr "$(date +"%T") Initialize ... "
apt-get update -qqy

ROOT=`pwd`
echo "Workspace is ${ROOT}"

# OpenCV
printstr "$(date +"%T") Install OpenCV " 
apt-get install -qqy ffmpeg libsm6 libxext6 #> /dev/null 2>&1

# Colorlog
printstr "$(date +"%T") Pip install colorlog " 
pip3 install colorlog

printstr "Done"
