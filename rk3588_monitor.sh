#!/bin/bash
# RK3588 综合监控脚本 - CPU / GPU / RGA / NPU / DDR
interval=${1:-1}  # 默认刷新间隔 1 秒

RED='\033[0;31m'
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

print_color() { echo -e "$1$2${NC}"; }

monitor_cpu() {
    echo "=== CPU Status ==="
    for i in {0..7}; do
        cur=$(cat /sys/devices/system/cpu/cpu$i/cpufreq/cpuinfo_cur_freq 2>/dev/null)
        gov=$(cat /sys/devices/system/cpu/cpufreq/policy$i/scaling_governor 2>/dev/null)
        if [ -n "$cur" ]; then
            freq_mhz=$((cur/1000))
            color=$GREEN
            [ "$freq_mhz" -gt 2000 ] && color=$RED
            [ "$freq_mhz" -gt 1500 ] && color=$YELLOW
            printf "CPU$i: "
            print_color $color "${freq_mhz} MHz"
            [ -n "$gov" ] && echo " (gov: $gov)"
        fi
    done
    if [ -f /proc/loadavg ]; then
        load=$(awk '{print $1,$2,$3}' /proc/loadavg)
        echo "Load avg: $load"
    fi
    echo ""
}

monitor_gpu() {
    echo "=== GPU Status ==="
    dev="/sys/class/devfreq/fb000000.gpu"
    [ ! -d "$dev" ] && dev="/sys/class/misc/mali0/device"
    # 频率
    freq_file="$dev/cur_freq"
    if [ -f "$freq_file" ]; then
        freq=$(cat "$freq_file" 2>/dev/null)
        if [ -n "$freq" ] && [ "$freq" -gt 0 ] 2>/dev/null; then
            freq_mhz=$((freq/1000000))
            color=$GREEN
            [ "$freq_mhz" -gt 800 ] && color=$RED
            [ "$freq_mhz" -gt 500 ] && color=$YELLOW
            print_color $color "GPU freq: ${freq_mhz} MHz"
            echo ""
        fi
    fi
    # 负载
    load_file="$dev/load"
    if [ -f "$load_file" ]; then
        load=$(cat "$load_file" 2>/dev/null)
        [ -n "$load" ] && echo "GPU load: ${load}%"
    fi
    # 可用频率
    avail_file="$dev/available_frequencies"
    if [ -f "$avail_file" ]; then
        freqs=$(cat "$avail_file" 2>/dev/null)
        [ -n "$freqs" ] && echo -n "GPU available freqs: " && for f in $freqs; do echo -n "$((f/1000000))MHz "; done && echo ""
    fi
    # governor
    gov_file="$dev/governor"
    [ -f "$gov_file" ] && echo "GPU governor: $(cat $gov_file 2>/dev/null)"
    echo ""
}

monitor_rga() {
    echo "=== RGA Status ==="
    dev="/sys/devices/platform/fdb80000.rga"
    if [ -d "$dev" ]; then
        # 频率
        freq_file="$dev/devfreq/fdb80000.rga/cur_freq"
        if [ -f "$freq_file" ]; then
            freq=$(cat "$freq_file" 2>/dev/null)
            [ -n "$freq" ] && [ "$freq" -gt 0 ] 2>/dev/null && echo "RGA freq: $((freq/1000000)) MHz"
        fi
        # 负载
        load_file="/sys/kernel/debug/rkrga/load"
        [ -f "$load_file" ] && echo -n "RGA load: " && cat "$load_file"
    fi
    echo ""
}

monitor_npu() {
    echo "=== NPU Status ==="
    freq_file="/sys/kernel/debug/rknpu/freq"
    load_file="/sys/kernel/debug/rknpu/load"
    [ -f "$freq_file" ] && echo -n "NPU freq: " && cat "$freq_file"
    [ -f "$load_file" ] && echo -n "NPU load: " && cat "$load_file"
    echo ""
}

monitor_ddr() {
    echo "=== DDR Status ==="
    ddr_file="/sys/class/devfreq/dmc/cur_freq"
    if [ -f "$ddr_file" ]; then
        ddr=$(cat "$ddr_file" 2>/dev/null)
        [ -n "$ddr" ] && [ "$ddr" -gt 0 ] 2>/dev/null && echo "DDR freq: $((ddr/1000)) MHz"
    else
        ddr=$(grep dmc /sys/kernel/debug/clk/clk_summary 2>/dev/null | awk '{print $3}' | head -n1)
        [ -n "$ddr" ] && [ "$ddr" -gt 0 ] 2>/dev/null && echo "DDR freq: $((ddr/1000000)) MHz"
    fi
    echo ""
}

trap 'echo "Monitor stopped"; exit 0' INT

while true; do
    clear
    echo "=== RK3588 Monitor - $(date "+%Y-%m-%d %H:%M:%S") ==="
    monitor_cpu
    monitor_gpu
    monitor_rga
    monitor_npu
    monitor_ddr
    echo "Refresh interval: ${interval}s | Ctrl+C to exit"
    sleep $interval
done