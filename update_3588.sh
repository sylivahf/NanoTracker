#!/bin/bash

IP="172.16.192.238"
REMOTE_DIR="/root/nanotrack"
LOCAL_DIR="/home/itds/NanoTracker/build"
LOCAL_DIR_MODEL="/home/itds/NanoTracker/models"
# Kill existing processes
ssh root@$IP "pkill gdbserver; fuser -k $REMOTE_DIR/nanotrack"

# Copy files
scp $LOCAL_DIR/nanotrack root@$IP:$REMOTE_DIR/
# scp -r $LOCAL_DIR_MODEL root@$IP:$REMOTE_DIR/

ssh root@$IP "
    # 检查当前目录是否存在 output/det_txt 文件夹
    if [ -d "/root/nanotrack/output/" ]; then
        echo "发现 output 文件夹，清空文件夹内容..."
        rm -rf /root/nanotrack/output/*
    else
        echo "未发现 output 文件夹..."
    fi

    if [ -d "/root/nanotrack/output_txt/" ]; then
        echo "发现 output_txt 文件夹，清空文件夹内容..."
        rm -rf /root/nanotrack/output_txt/*
    else
        echo "未发现 output_txt 文件夹..."
    fi

    # cd $REMOTE_DIR && chmod +x nanotrack && ./nanotrack /root/gdbserver_test/data/20251114pcie_output.h264
    cd $REMOTE_DIR && chmod +x nanotrack && ./nanotrack VTOL-Fixed-Wing-Model-Flight-down.mp4
    echo "Done"

"
