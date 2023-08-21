#!/bin/bash
# ---------------------------------------------------------
# color ANIS
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

function printstr(){
    echo -e "${BLUE}"
    echo $1
    echo -e "${NC}"
}

# ---------------------------------------------------------
printstr "---Install Sqlite3---"

apt update
apt install sqlite3 
sqlite3 database/face.db ".databases" ".quit"

printstr "Done"