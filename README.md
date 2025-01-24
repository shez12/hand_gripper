#  Hand_Gripper
## Use IM948 in Ubuntu 20.04
### Check bluetoothd version
```bash
bluetoothd --version
```
### Find IM948 MAC address

```bash
# something like 1f:5f:6e:e7:74:89
bluetoothctl
```
### install python env (recommend py 3.10)
```bash
pip install gatt
pip install socket
```

