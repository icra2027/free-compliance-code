#!/usr/bin/env bash
# Fix device node permissions for the Orbbec Femto Bolt (scene camera) in containers with no
# udev daemon, where new nodes come up root-only and neither access path can open them:
#   - orbbec_camera (the ROS driver, raw USB via libuvc): "usbEnumerator openUsbDevice failed!
#     status:113" -- fixed by chmod'ing /dev/bus/usb/<bus>/<dev>.
#   - data_recorder's use_scene_cv2_capture fallback (cv2.VideoCapture / V4L2):
#     opens but read()s fail, or open() itself is denied -- fixed by chmod'ing the matching
#     /dev/video* node(s); these are usually group `video`, mode 660, and the recording user is
#     not necessarily in that group in this container.
# The Femto Bolt has been observed to drop off the
# USB bus and re-enumerate to a different node repeatedly within a single session, so this looks
# it up by vendor ID (Orbbec = 2bc5) each run rather than assuming a fixed bus/device number or
# video index.
#
# Usage: sudo ./fix_femto_bolt_usb_perms.sh
# Then either (re)launch the ROS driver:
#   ros2 launch orbbec_camera femto_bolt.launch.py enable_depth:=false enable_ir:=false
# or use the cv2 fallback in data_recorder:
#   ros2 run data_recorder record_lerobot --ros-args -p use_scene_cv2_capture:=true \
#     -p scene_cv2_device_path:=/dev/videoN   # see below for how to pick N
#
# If the device keeps re-enumerating every few seconds even right after this script runs, that is
# NOT a permissions problem -- it is the separate, harder USB3 connection-stability issue observed
# earlier (try a different port, a shorter/rated-USB3 cable, or a powered hub).
set -euo pipefail

ORBBEC_VENDOR_ID="2bc5"

device_path=""
for candidate in /sys/bus/usb/devices/*/idVendor; do
    if [ -f "${candidate}" ] && [ "$(cat "${candidate}")" = "${ORBBEC_VENDOR_ID}" ]; then
        device_path="$(dirname "${candidate}")"
        break
    fi
done

if [ -z "${device_path}" ]; then
    echo "No USB device with vendor ID ${ORBBEC_VENDOR_ID} (Orbbec) found." >&2
    echo "Is the Femto Bolt plugged in? Check with: find /sys/bus/usb/devices -name idVendor" >&2
    exit 1
fi

busnum="$(cat "${device_path}/busnum")"
devnum="$(cat "${device_path}/devnum")"
usb_node="/dev/bus/usb/$(printf '%03d' "${busnum}")/$(printf '%03d' "${devnum}")"

chmod 666 "${usb_node}"
echo "Fixed permissions on ${usb_node} ($(cat "${device_path}/product" 2>/dev/null || echo 'Femto Bolt'))"

# Any /dev/videoN whose sysfs device symlink resolves underneath this same USB device belongs to
# the Femto Bolt (it exposes several: color's alternate UVC formats plus depth/IR).
device_path_real="$(readlink -f "${device_path}")"
found_video_node=0
for v4l_node in /dev/video*; do
    [ -e "${v4l_node}" ] || continue
    name="$(basename "${v4l_node}")"
    sys_device="/sys/class/video4linux/${name}/device"
    [ -e "${sys_device}" ] || continue
    resolved="$(readlink -f "${sys_device}")"
    case "${resolved}" in
        "${device_path_real}"*)
            chmod 666 "${v4l_node}"
            echo "Fixed permissions on ${v4l_node} ($(cat "${sys_device}/../name" 2>/dev/null || cat "/sys/class/video4linux/${name}/name" 2>/dev/null))"
            found_video_node=1
            ;;
    esac
done

if [ "${found_video_node}" -eq 0 ]; then
    echo "No /dev/video* nodes found for this device (only relevant if you use" >&2
    echo "use_scene_cv2_capture -- the ROS driver path doesn't need them)." >&2
fi
