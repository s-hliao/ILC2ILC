#!/bin/bash
# dds_env.sh: CycloneDDS for running the car remotely (as LLA-MPC-f1tenth llampc/utils/cyclonedds{car,laptop}.xml):
# unicast discovery between the car and the laptop over the lab WiFi (multicast off), ROS 2 on rmw_cyclonedds_cpp.
#   source scripts/dds_env.sh car                       # on the car     (config/cyclonedds/cyclonedds_car.xml)
#   source scripts/dds_env.sh laptop                    # on the laptop  (config/cyclonedds/cyclonedds_laptop.xml)
#   source scripts/dds_env.sh local                     # desk tests, one machine (loopback)
#   source scripts/dds_env.sh car --car-ip A --laptop-ip B [--peer C ...]   # other addresses: writes ~/.ros/cyclonedds_<role>.xml
# Both machines must use the same ROS_DOMAIN_ID (default 0; set it before sourcing to change it).
_dds_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../config/cyclonedds" && pwd)
_role=${1:-}; shift || true
_car_ip=""; _lap_ip=""; _extra=()
while [ $# -gt 0 ]; do
  case $1 in
    --car-ip) _car_ip=$2; shift 2;;
    --laptop-ip) _lap_ip=$2; shift 2;;
    --peer) _extra+=("$2"); shift 2;;
    *) echo "dds_env.sh: unknown argument $1"; return 1 2>/dev/null || exit 1;;
  esac
done
case $_role in
  car|laptop|local) ;;
  *) echo "usage: source dds_env.sh car|laptop|local [--car-ip A --laptop-ip B] [--peer C]"; return 1 2>/dev/null || exit 1;;
esac
_xml=$_dds_dir/cyclonedds_$_role.xml
if [ -n "$_car_ip$_lap_ip" ]; then
  [ -n "$_car_ip" ] && [ -n "$_lap_ip" ] || { echo "dds_env.sh: give both --car-ip and --laptop-ip"; return 1 2>/dev/null || exit 1; }
  _self=$_car_ip; [ "$_role" = laptop ] && _self=$_lap_ip
  _peers=""; for _ip in "$_car_ip" "$_lap_ip" "${_extra[@]}"; do _peers+="        <Peer Address=\"$_ip\"/>\n"; done
  mkdir -p ~/.ros; _xml=~/.ros/cyclonedds_$_role.xml
  sed -e "s|@SELF_IP@|$_self|" -e "s|@PEERS@|${_peers%\\n}|" "$_dds_dir/template.xml.in" | sed 's/\\n/\n/g' > "$_xml"
fi
_distro=${ROS_DISTRO:-$(ls /opt/ros 2>/dev/null | head -1)}
if [ ! -e /opt/ros/$_distro/lib/librmw_cyclonedds_cpp.so ]; then
  echo "dds_env.sh: rmw_cyclonedds_cpp is not installed: sudo apt install ros-$_distro-rmw-cyclonedds-cpp"
  return 1 2>/dev/null || exit 1
fi
_self_ip=$(sed -n 's|.*<NetworkInterfaceAddress>\(.*\)</NetworkInterfaceAddress>.*|\1|p' "$_xml")
if command -v ip >/dev/null && ! ip -o addr show 2>/dev/null | grep -q " $_self_ip/"; then
  echo "dds_env.sh: WARNING: $_self_ip (this machine in $_xml) is not on any local interface -- wrong role or address?"
fi
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file://$_xml
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}
unset ROS_LOCALHOST_ONLY
echo "DDS: $RMW_IMPLEMENTATION, $CYCLONEDDS_URI (self $_self_ip), ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
ros2 daemon stop >/dev/null 2>&1 || true        # a running daemon keeps the old middleware / discovery settings
unset _distro _dds_dir _role _car_ip _lap_ip _extra _xml _self _peers _ip _self_ip
