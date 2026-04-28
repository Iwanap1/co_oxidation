from pymongo import MongoClient
from dotenv import load_dotenv
import os

class DB:
    def __init__(self, uri=None, db_name="hybrid_model"):
        if uri is None:
            load_dotenv()
            uri = os.getenv("MONGO")
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collections = {
            "materials": self.db["materials"],
            "reactions": self.db["reactions"],
            "h2_tpr_peaks": self.db["h2_tpr_peaks"],
            "osc": self.db["osc"],
            "o2_tpd_peaks": self.db["o2_tpd_peaks"],
            "co2_tpd_peaks": self.db["co2_tpd_peaks"],
        }

    def close(self):
        self.client.close()
        