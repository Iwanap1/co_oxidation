import json, hashlib
from pymongo import ReturnDocument
from bson import ObjectId
from utils.data_utils import load_mongo_db

MIGRATION_FILE = "migrations/o2_tpd_mats.json"

def normalize(v):
    if isinstance(v, float):
        return round(v, 12)
    if isinstance(v, list):
        return [normalize(x) for x in v]
    if isinstance(v, dict):
        return {k: normalize(v[k]) for k in sorted(v)}
    return v

def fingerprint(doc):
    blob = json.dumps(normalize(doc), sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()

def create_material(entry, mat_coll):
    mat = entry["material"]
    mat["fingerprint"] = fingerprint(mat)
    mat_row = mat_coll.find_one_and_update(
        {"fingerprint": mat["fingerprint"]},
        {"$setOnInsert": mat},
        upsert=True,
        return_document=ReturnDocument.AFTER
    )
    return mat_row["_id"]

def create_o2_tpd(entry, mat_id, o2_tpd_coll):
    tpd = entry["o2_tpd"]
    tpd["material_id"] = mat_id
    res = o2_tpd_coll.insert_one(tpd)
    return res.inserted_id is not None

def migrate_o2_tpd(docs):
    db = load_mongo_db()
    mat_coll = db["materials"]
    o2_tpd_coll = db["o2_tpd"]

    updates = 0
    for entry in docs:
        mat_id = create_material(entry, mat_coll)
        created = create_o2_tpd(entry, mat_id, o2_tpd_coll)
        if created:
            updates += 1
    return updates

if __name__ == "__main__":
    with open(MIGRATION_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    num_updates = migrate_o2_tpd(data["create"])
    print(f"Migrated {num_updates} O2-TPD entries.")