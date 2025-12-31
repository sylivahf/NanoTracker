#!/bin/bash

IP="172.16.192.238"
REMOTE_DIR="/root/nanotrack"
LOCAL_DIR="/home/itds/NanoTracker/build"
LOCAL_DIR_MODEL="/home/itds/NanoTracker/models"
# Kill existing processes
ssh root@$IP "pkill gdbserver; fuser -k $REMOTE_DIR/nanotrack"

# Copy files
scp $LOCAL_DIR/nanotrack root@$IP:$REMOTE_DIR/
scp -r $LOCAL_DIR_MODEL root@$IP:$REMOTE_DIR/
ssh root@$IP "
    cd $REMOTE_DIR && chmod +x nanotrack && ./nanotrack girl_dance.mp4
"
