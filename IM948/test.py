import gatt

'''export DBUS_SYSTEM_BUS_ADDRESS=unix:path=/var/run/dbus/system_bus_socket'''


manager = gatt.DeviceManager(adapter_name='hci0')

class AnyDevice(gatt.Device):
    def services_resolved(self):
        super().services_resolved()

        device_information_service = next(
            s for s in self.services
            if s.uuid == '0000180a-0000-1000-8000-00805f9b34fb')

        firmware_version_characteristic = next(
            c for c in device_information_service.characteristics
            if c.uuid == '00002a26-0000-1000-8000-00805f9b34fb')

        firmware_version_characteristic.read_value()

    def characteristic_value_updated(self, characteristic, value):
        print("Firmware version:", value.decode("utf-8"))


device = AnyDevice(mac_address='AA:BB:CC:DD:EE:FF', manager=manager)
device.connect()
manager.run()