#!/bin/bash

IP="172.17.192.217"
REMOTE_DIR="/root/nanotrack"
LOCAL_DIR="/home/NanoTracker/build"
LOCAL_DIR_MODEL="/home/NanoTracker/models"
# Kill existing processes
ssh root@$IP "pkill gdbserver; fuser -k $REMOTE_DIR/nanotrack"

# Copy files
scp $LOCAL_DIR/nanotrack root@$IP:$REMOTE_DIR/
scp -r $LOCAL_DIR_MODEL root@$IP:$REMOTE_DIR/

# Run
ssh root@$IP "
    cd $REMOTE_DIR && chmod +x nanotrack && ./nanotrack girl_dance.mp4
"
