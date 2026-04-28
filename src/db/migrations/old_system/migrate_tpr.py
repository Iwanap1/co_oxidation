from utils.data_utils import load_mongo_db
import json, hashlib
from pymongo import ReturnDocument
from bson import ObjectId

MIGRATION_FILE = "migrations/tpr_tpd_retry.json"

REACTION_KEYS = {
    "temps","conversion","flow_mL_h_g","flow_normalisation_note",
    "gas_co_content","gas_o2_content","gas_co2_content","gas_h2o_content","gas_air_content"
}

def fingerprint(doc):
    blob = json.dumps(normalize(doc), sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()

def normalize(v):
    if isinstance(v, float):
        return round(v, 12)
    if isinstance(v, list):
        return [normalize(x) for x in v]
    if isinstance(v, dict):
        return {k: normalize(v[k]) for k in sorted(v)}
    return v


def new_material(mat_entry, mat_coll):
    mat = {k: mat_entry[k] for k in mat_entry if k not in REACTION_KEYS and k != "id"}
    mat["fingerprint"] = fingerprint(mat)
    mat_row = mat_coll.find_one_and_update(
        {"fingerprint": mat["fingerprint"]},
        {"$setOnInsert": mat},
        upsert=True,
        return_document=ReturnDocument.AFTER
    )
    return mat_row["_id"]


def new_lightoff(entry, reactions_coll, mat_id):
    rxn = {k: entry.get(k, None) for k in REACTION_KEYS if k in entry}
    rxn["doi"] = entry.get("doi", None)
    rxn["material_id"] = mat_id
    res = reactions_coll.insert_one(rxn)
    return res.inserted_id is not None

def update_material(update_entry, mat_coll):
    mat_id = update_entry["id"]
    res = mat_coll.update_one(
        {"_id": ObjectId(mat_id)},
        {"$set": {k: update_entry[k] for k in update_entry if k not in REACTION_KEYS and k != "id" and k != "material_id"}}
    )
    return res.modified_count > 0


def find_material_or_create_new(tp_entry, mat_coll, reactions_coll):
    if "material" in tp_entry.keys():
        mat_entry = tp_entry.pop("material")
        mat_id = new_material(mat_entry, mat_coll)
        if "reactions" in tp_entry:
            r_entry = tp_entry.pop("reactions")
            r = new_lightoff(r_entry, reactions_coll, mat_id)
            if r:
                print("Created new material and associated reaction for material id:", mat_id)
        return mat_id
    else:
        id = tp_entry.get("material_id", "")
        mat = mat_coll.find_one({"_id": ObjectId(id)})
        if mat:
            return mat["_id"]
    return None
    

def create_tp_entries(tp_entries, mat_coll, reactions_coll, tp_coll):
    created_count = 0
    for entry in tp_entries:
        mat_id = find_material_or_create_new(entry, mat_coll, reactions_coll)
        if mat_id is None:
            print("Failed to find or create material for entry:", entry)
            continue
        entry["material_id"] = mat_id

        tp_coll.insert_one(entry)
    return created_count


def run_migration():
    db = load_mongo_db()

    co_tpr_coll = db["co_tpr"]
    co_tpr_coll.delete_many({})

    o2_tpr_coll = db["o2_tpd"]
    o2_tpr_coll.delete_many({})

    co_tpd_coll = db["co_tpd"]
    co_tpd_coll.delete_many({})

    materials_coll = db["materials"]

    reactions_coll = db["reactions"]

    with open(MIGRATION_FILE, "r", encoding="utf-8") as f:
        docs = json.load(f)

    update_count = 0
    update_fail_count = 0
    update_entries = docs["update"]
    for e in update_entries:
        updated = update_material(e, materials_coll)
        if updated:
            update_count += 1
        else:
            print("Failed to update material with id:", e.get("id"))
            update_fail_count += 1
    
    for coll_name in ["co_tpr", "o2_tpd", "co_tpd"]:
        entries = docs[f"create_{coll_name}"]
        tp_coll = db[coll_name]
        created = create_tp_entries(entries, materials_coll, reactions_coll, tp_coll)
        print(f"Created {created} new entries in collection {coll_name}.")


if __name__ == "__main__":
    run_migration()