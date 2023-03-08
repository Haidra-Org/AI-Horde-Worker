"""Bridge Stats Tracker"""
import json
import time


class BridgeStats:
    """Convenience functions for the stats"""

    stats = {}  # Deliberately on class level

    def __init__(self):
        self.startup_time = time.monotonic()

    def update_inference_stats(self, model_name, kudos):
        """Updates the stats for a model inference"""
        if "inference" not in self.stats:
            self.stats["inference"] = {}
        if model_name not in self.stats["inference"]:
            self.stats["inference"][model_name] = {"kudos": 0, "count": 0}
        self.stats["inference"][model_name]["count"] += 1
        self.stats["inference"][model_name]["kudos"] = round(self.stats["inference"][model_name]["kudos"] + kudos)
        stats_for_model = self.stats["inference"][model_name]
        self.stats["inference"][model_name]["avg_kpr"] = round(stats_for_model["kudos"] / stats_for_model["count"], 2)

        # Update overall kudos stats
        if "total_kudos" in self.stats:
            self.stats["total_kudos"] += kudos
        else:
            self.stats["total_kudos"] = kudos
        self.stats["total_uptime"] = time.monotonic() - self.startup_time
        self.stats["kudos_per_hour"] = int((self.stats["total_kudos"] / self.stats["total_uptime"]) * 3600)

    def get_pretty_stats(self):
        """Returns a pretty string of the stats"""
        return json.dumps(self.stats, indent=4)


bridge_stats = BridgeStats()
