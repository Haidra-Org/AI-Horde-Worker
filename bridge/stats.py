import json


class BridgeStats:
    
    def __init__(self):
        self.stats = {}

    def update_inference_stats(self, model_name, kudos):
        if "inference" not in self.stats:
            self.stats["inference"] = {}
        if model_name not in self.stats["inference"]:
            self.stats["inference"][model_name] = { "kudos": 0, "count": 0 }
        self.stats["inference"][model_name]["count"] += 1
        self.stats["inference"][model_name]["kudos"] = round(self.stats["inference"][model_name]["kudos"] + kudos)
        stats_for_model = self.stats["inference"][model_name]
        self.stats["inference"][model_name]["avg_kpr"] = round(stats_for_model["kudos"] / stats_for_model["count"], 2)

    def get_pretty_stats(self):
        return json.dumps(self.stats, indent=4)
