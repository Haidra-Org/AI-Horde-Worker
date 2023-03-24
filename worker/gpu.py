import contextlib

from pynvml.smi import nvidia_smi


class GPUInfo:
    def __init__(self):
        pass

    # Return a value from the given dictionary supporting dot notation
    def get(self, data, key, default=""):
        # Handle nested structures
        path = key.split(".")

        if len(path) == 1:
            # Simple case
            return data[key] if key in data else default
        # Nested case
        walkdata = data
        for element in path:
            if element in walkdata:
                walkdata = walkdata[element]
            else:
                walkdata = ""
                break
        return walkdata

    def _get_gpu_data(self):
        with contextlib.suppress(Exception):
            nvsmi = nvidia_smi.getInstance()
            data = nvsmi.DeviceQuery()
            return data.get("gpu", [None])[0]

    def _mem(self, raw):
        unit = "GB"
        mem = raw / 1024
        if mem < 1:
            unit = "MB"
            raw *= 1024
        return f"{int(mem)} {unit}"

    def get_info(self):
        data = self._get_gpu_data()
        if not data:
            return
        return {
            "product": self.get(data, "product_name", "unknown"),
            "pci_gen": self.get(data, "pci.pci_gpu_link_info.pcie_gen.current_link_gen", "?"),
            "pci_width": self.get(data, "pci.pci_gpu_link_info.link_widths.current_link_width", "?"),
            "fan_speed": f"{self.get(data, 'fan_speed')}{self.get(data, 'fan_speed_unit')}",
            "vram_total": f"{self._mem(self.get(data, 'fb_memory_usage.total', '0'))}",
            "vram_used": f"{self._mem(self.get(data, 'fb_memory_usage.used', '0'))}",
            "vram_free": f"{self._mem(self.get(data, 'fb_memory_usage.free', '0'))}",
            "load": f"{self.get(data, 'utilization.gpu_util')}{self.get(data, 'utilization.unit')}",
            "temp": f"{self.get(data, 'temperature.gpu_temp')}{self.get(data, 'temperature.unit')}",
            "power": f"{int(self.get(data, 'power_readings.power_draw', 0))}{self.get(data, 'power_readings.unit')}",
        }


if __name__ == "__main__":
    gpu = GPUInfo()
    print(gpu.get_info())
