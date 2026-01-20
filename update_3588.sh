#!/bin/bash

IP="175.17.192.200"
REMOTE_DIR="/root/nanotrack"
LOCAL_DIR="/home/NanoTracker/build"
LOCAL_DIR_MODEL="/home/NanoTracker/models"
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

    cd $REMOTE_DIR && chmod +x nanotrack && ./nanotrack girl_dance.mp4
    echo "Done"

"
