#!/bin/bash
# ---------------------------------------------------------
# color ANIS
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

# ---------------------------------------------------------
function word(){
echo -e "${YELLOW}"
echo '
                             
 _____     _ _   _ _         
| __  |_ _|_| |_| |_|___ ___ 
| __ -| | | | | . | |   | . |
|_____|___|_|_|___|_|_|_|_  |
                        |___|
'
echo -e "${NC}"
}

function face(){
echo -e "${YELLOW}"
echo "
 ____  __    ___  ____ 
(  __)/ _\  / __)(  __)
 ) _)/    \( (__  ) _) 
(__) \_/\_/ \___)(____)
"
echo -e "${NC}"
}

word
face

ROOT=$(dirname `realpath $0`)
cd $ROOT

echo "Build the docker image."
docker build -t openvino_face_cv .