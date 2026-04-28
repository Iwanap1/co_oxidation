from utils.data_utils import load_mongo_db
import json, hashlib
from pymongo import ReturnDocument
from bson import ObjectId

MIGRATION_FILE = "migrations/19-01-2026.json"

REACTION_KEYS = {
    "temps","conversion","flow_mL_h_g","flow_normalisation_note",
    "gas_co_content","gas_o2_content","gas_co2_content","gas_h2o_content","gas_air_content"
}

def normalize(v):
    if isinstance(v, float):
        return round(v, 12)
    if isinstance(v, list):
        return [normalize(x) for x in v]
    if isinstance(v, dict):
        return {k: normalize(v[k]) for k in sorted(v)}
    return v

def material_doc(entry):
    return {k: entry[k] for k in entry if k not in REACTION_KEYS and k != "id"}

def fingerprint(doc):
    blob = json.dumps(normalize(doc), sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def create(mat_coll, rxn_coll, docs):
    updates = 0
    for entry in docs:
        mat = material_doc(entry)
        mat["fingerprint"] = fingerprint(mat)
        if mat_coll.find_one({"fingerprint": mat["fingerprint"]}):
            print(f"Material with fingerprint {mat['fingerprint']} already exists. Skipping creation.")
            continue
        mat_row = mat_coll.find_one_and_update(
            {"fingerprint": mat["fingerprint"]},
            {"$setOnInsert": mat},
            upsert=True,
            return_document=ReturnDocument.AFTER
        )
        mat_id = mat_row["_id"]
        rxn = {k: entry[k] for k in REACTION_KEYS if k in entry}
        rxn["material_id"] = mat_id
        res = rxn_coll.insert_one(rxn)
        if res.inserted_id:
            updates += 1
    return updates

def update(mat_coll, rxn_coll, docs):
    updates = 0
    for entry in docs:
        if "reaction_id" in entry and "id" not in entry:
            rxn_id = entry["reaction_id"]
            res = rxn_coll.update_one(
                {"_id": ObjectId(rxn_id)},
                {"$set": {k: entry[k] for k in REACTION_KEYS if k in entry}}
            )
            if res.modified_count == 1:
                updates += 1
        elif "id" in entry:
            reaction_keys = {k: entry[k] for k in REACTION_KEYS if k in entry}
            if reaction_keys:
                reactions = rxn_coll.find({"material_id": ObjectId(entry["id"])})
                rxns = list(reactions)
                if len(rxns) > 1:
                    print(f"Warning: multiple reactions found for material_id {entry['id']}. Skipping update.")
                elif len(rxns) == 1:
                    res = rxn_coll.update_one(
                        {"_id": rxns[0]["_id"]},
                        {"$set": reaction_keys}
                    )
                    if res.modified_count == 1:
                        updates += 1
    return updates




if __name__ == "__main__":
    db = load_mongo_db()
    mats_coll = db["materials"]
    rxns_coll = db["reactions"]
    with open(MIGRATION_FILE, "r") as f:
        migrations = json.load(f)   
    print(f"{len(migrations['create'])} entries to create.")
    print(f"{len(migrations['update'])} entries to update.")
    creations = create(mats_coll, rxns_coll, migrations["create"])
    print(f"created {creations} new entries.")
    updates = update(mats_coll, rxns_coll, migrations["update"])
    print(f"updated {updates} existing entries.")
    print("Migration complete.")
