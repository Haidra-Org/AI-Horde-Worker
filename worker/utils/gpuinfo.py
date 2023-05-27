import contextlib
import os

from pynvml.smi import nvidia_smi


class GPUInfo:
    def __init__(self):
        self.avg_load = []
        self.avg_temp = []
        self.avg_power = []
        # Average period in samples, default 10 samples per second, period 5 minutes
        self.samples_per_second = 10
        # Look out for device env var hack
        self.device = int(os.getenv("CUDA_VISIBLE_DEVICES", 0))

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
            return data.get("gpu", [None])[self.device]

    def _mem(self, raw):
        unit = "GB"
        mem = raw / 1024
        if mem < 1:
            unit = "MB"
            raw *= 1024
        return f"{round(mem)} {unit}"

    def get_total_vram_mb(self):
        """Get total VRAM in MB as an integer, or 0"""
        value = 0
        data = self._get_gpu_data()
        if data:
            value = self.get(data, "fb_memory_usage.total", "0")
            try:
                value = int(value)
            except ValueError:
                value = 0
        return value

    def get_free_vram_mb(self):
        """Get free VRAM in MB as an integer, or 0"""
        value = 0
        data = self._get_gpu_data()
        if data:
            value = self.get(data, "fb_memory_usage.free", "0")
            try:
                value = int(value)
            except ValueError:
                value = 0
        return value

    def get_info(self):
        data = self._get_gpu_data()
        if not data:
            return None

        # Calculate averages
        try:
            gpu_util = int(self.get(data, "utilization.gpu_util", 0))
        except ValueError:
            gpu_util = 0

        try:
            gpu_temp = int(self.get(data, "temperature.gpu_temp", 0))
        except ValueError:
            gpu_temp = 0

        try:
            gpu_power = int(self.get(data, "power_readings.power_draw", 0))
        except ValueError:
            gpu_power = 0

        self.avg_load.append(gpu_util)
        self.avg_temp.append(gpu_temp)
        self.avg_power.append(gpu_power)
        self.avg_load = self.avg_load[-(self.samples_per_second * 60 * 5) :]
        self.avg_power = self.avg_power[-(self.samples_per_second * 60 * 5) :]
        self.avg_temp = self.avg_temp[-(self.samples_per_second * 60 * 5) :]
        avg_load = int(sum(self.avg_load) / len(self.avg_load))
        avg_power = int(sum(self.avg_power) / len(self.avg_power))
        avg_temp = int(sum(self.avg_temp) / len(self.avg_temp))

        return {
            "product": self.get(data, "product_name", "unknown"),
            "pci_gen": self.get(data, "pci.pci_gpu_link_info.pcie_gen.current_link_gen", "?"),
            "pci_width": self.get(data, "pci.pci_gpu_link_info.link_widths.current_link_width", "?"),
            "fan_speed": f"{self.get(data, 'fan_speed')}{self.get(data, 'fan_speed_unit')}",
            "vram_total": f"{self._mem(self.get(data, 'fb_memory_usage.total', '0'))}",
            "vram_used": f"{self._mem(self.get(data, 'fb_memory_usage.used', '0'))}",
            "vram_free": f"{self._mem(self.get(data, 'fb_memory_usage.free', '0'))}",
            "load": f"{gpu_util}{self.get(data, 'utilization.unit')}",
            "temp": f"{gpu_temp}{self.get(data, 'temperature.unit')}",
            "power": f"{gpu_power}{self.get(data, 'power_readings.unit')}",
            "avg_load": f"{avg_load}{self.get(data, 'utilization.unit')}",
            "avg_temp": f"{avg_temp}{self.get(data, 'temperature.unit')}",
            "avg_power": f"{avg_power}{self.get(data, 'power_readings.unit')}",
        }
