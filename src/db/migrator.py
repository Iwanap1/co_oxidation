from .database import DB
from typing import List, Dict, Tuple
import json, os, hashlib, shutil
from pathlib import Path
from bson import ObjectId

class Migrator:
    def __init__(self, db: DB, microscopy_dir="db/microscopy_images", staging_dir="db/migrations/staged_microscopy", migration_failure_dir="db/migrations/failures"):
        self.db = DB
        self.microscopy_dir = Path(microscopy_dir)
        self.staging_dir = Path(staging_dir)
        self.staged_images = set(os.listdir(self.staging_dir))
        self.materials_coll = db.collections["materials"]
        self.reactions_coll = db.collections["reactions"]
        self.failure_dir = Path(migration_failure_dir)
        self.failure_dir.mkdir(exist_ok=True)
        return
    
    def migrate_file(self, file_path: str):
        data = self.load_file(file_path)
        root_name = Path(file_path).stem
        create = data.get("create", [])
        update = data.get("update", [])
        create_from_existing = data.get("create_from_existing", [])
        self.check_uniques(create, check_ids=False)
        self.check_uniques(update, check_ids=True)
        created_materials, created_reactions, created_h2_tpr, created_co2_tpd, created_o2_tpd, created_osc, failures, subcreation_failures = self.run_creations(create)


        print(f"Created {created_materials} new materials")
        n_material_failures = len(failures)
        print(f"Failed to insert {n_material_failures} new materials into the DB.")
        if n_material_failures > 0:
            failure_path = root_name + "_material_failures.json"
            print(f"Logging Failed Materials To {str(self.failure_dir)}/{failure_path}")
            self.save_failures(failures, failure_path)

        print(f"Created {created_reactions} new reactions")
        print(f"Created {created_h2_tpr} new H2 TPRs")
        print(f"Created {created_co2_tpd} new CO2 TPDs")
        print(f"Created {created_o2_tpd} new O2 TPDs")
        print(f"Created {created_osc} new OSC entries")



        successes, failed_updates = self.update(update)
        print(f"Successfully updates {successes} documents")
        if len(failed_updates) > 0:
            failure_path = root_name + "_update_failures.json"
            self.save_failures(failed_updates, failure_path)

        failed_creations, failed_subcreations2, h2_tpr_successes, o2_tpd_successes, co2_tpd_successes, osc_successes = self.create_from_existing(create_from_existing)
        subcreation_failures.extend(failed_subcreations2)
        print(f"Created {h2_tpr_successes} H2 TPRs from existing materials")
        print(f"Created {co2_tpd_successes} CO2 TPDs from existing materials")
        print(f"Created {o2_tpd_successes} O2 TPDs from existing materials")
        print(f"Created {osc_successes} OSC entries from existing materials")        

        n_rxn_failures = len(subcreation_failures)
        print(f"Failed to insert {n_rxn_failures} new reactions or characterisations into the DB.")
        if n_rxn_failures > 0:
            failure_path = root_name + "_subcreation_failures.json"
            print(f"Logging Failed Subcreations To {str(self.failure_dir)}/{failure_path}")
            self.save_failures(subcreation_failures, failure_path)


        update_all_successes, failed_update_by_filter = self.update_all_by_filter(data.get("update_by_filter", []))
        print(f"Successfully updated {update_all_successes} documents by filter")
        if len(failed_update_by_filter) > 0:
            failure_path = root_name + "_update_by_filter_failures.json"
            self.save_failures(failed_update_by_filter, failure_path)
        return
    
    
    def load_file(self, file_path) -> Dict[str, List]:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    

    def check_uniques(self, data: List[Dict], check_ids=True):
        image_paths = [
            d.get("image_path")
            for d in data
            if d.get("image_path") not in (None, "")
        ]
        if len(image_paths) != len(set(image_paths)):
            seen, dupes = set(), set()
            for x in image_paths:
                if x in seen:
                    dupes.add(x)
                else:
                    seen.add(x)
            raise ValueError(f"Duplicate Image Paths Found: {dupes}")

        if check_ids:
            all_ids = []
            for entry in data:
                _id = entry.get("_id", entry.get("id"))
                if _id is not None:
                    all_ids.append(_id)
            if len(all_ids) != len(set(all_ids)):
                seen, dupes = set(), set()
                for _id in all_ids:
                    if _id in seen:
                        dupes.add(_id)
                    else:
                        seen.add(_id)
                raise ValueError(f"Repeat IDs found: {dupes}")
            

    def run_creations(self, creations: List[Dict]):
        failures = []
        subcreation_failures = []
        created_materials = 0
        created_reactions = 0
        created_h2_tpr = 0
        created_o2_tpd = 0
        created_co2_tpd = 0
        created_osc = 0
        for entry in creations:
            image_path = entry.get("image_path", "")
            if not (image_path == "" or image_path is None) and image_path not in self.staged_images:
                failures.append((entry, "Invalid Image Path"))
                continue
            
            reactions = entry.pop("reactions", [])
            h2_tpr = entry.pop("h2_tpr_peaks", {})
            o2_tpd = entry.pop("o2_tpd_peaks", {})
            co2_tpd = entry.pop("co2_tpd_peaks", {})
            oscs = entry.pop("osc_entries", [])

            entry["fingerprint"] = self.fingerprint(entry)
            if self.materials_coll.find_one({"fingerprint": entry["fingerprint"]}):
                failures.append((entry, "Fingerprint already in DB"))
                continue
            
            insert = self.materials_coll.insert_one(entry)
            _id = insert.inserted_id
            if _id:
                created_materials += 1
                try:
                    new_image_path = self.move_image(image_path, _id)
                    self.materials_coll.find_one_and_update({"_id": _id}, {"$set": {"image_path": new_image_path}})
                except Exception as e:
                    print(f"Could not move image for new entry: {_id} because of error: {e}")
            else:
                failures.append((entry, "failed to add to DB"))
                continue
            
            # Create Reactions
            for r in reactions:
                r["material_id"] = _id
                r["doi"] = entry["doi"]
                res = self.reactions_coll.insert_one(r)
                if res.inserted_id:
                    created_reactions += 1
                else:
                    subcreation_failures.append((entry, "Could not insert reaction"))

            # H2 TPR
            if h2_tpr != {} and h2_tpr.get("temps", False):
                res, fail_reason = self.upload_characterisation(h2_tpr, "h2_tpr_peaks", entry["doi"], _id)
                if res:
                    created_h2_tpr += 1
                else:
                    subcreation_failures.append((h2_tpr, fail_reason))

            if o2_tpd != {} and o2_tpd.get("temps", False):
                res, fail_reason = self.upload_characterisation(o2_tpd, "o2_tpd_peaks", entry["doi"], _id)
                if res:
                    created_o2_tpd += 1
                else:
                    subcreation_failures.append((o2_tpd, fail_reason))
            
            if co2_tpd != {} and co2_tpd.get("temps", False):
                res, fail_reason = self.upload_characterisation(co2_tpd, "co2_tpd_peaks", entry["doi"], _id)
                if res:
                    created_co2_tpd += 1
                else:
                    subcreation_failures.append((co2_tpd, fail_reason))

            n_inserted, osc_failures = self.upload_oscs(oscs, _id, entry["doi"])
            subcreation_failures.extend(osc_failures)
            created_osc += n_inserted

        return created_materials, created_reactions, created_h2_tpr, created_co2_tpd, created_o2_tpd, created_osc, failures, subcreation_failures
    

    def fingerprint(self, doc):
        blob = json.dumps(self.normalize(doc), sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()
    

    def normalize(self, v):
        if isinstance(v, float):
            return round(v, 12)
        if isinstance(v, list):
            return [self.normalize(x) for x in v]
        if isinstance(v, dict):
            return {k: self.normalize(v[k]) for k in sorted(v)}
        return v


    def move_image(self, old_name: str, _id: ObjectId):
        if old_name in (None, ""):
            return old_name

        _, resolution, micro_type = old_name.split("_")
        new_name = f"{micro_type.replace('.png', '')}_{resolution}_{_id}.png"
        shutil.move(self.staging_dir / old_name, self.microscopy_dir / new_name)
        return new_name


    def save_failures(self, failures: List[Tuple[Dict, str]], file_name):
        to_save = []
        for doc, reason in failures:
            row = dict(doc)
            row["failed_because"] = str(reason)
            to_save.append(row)

        with open(self.failure_dir / file_name, "w") as f:
            json.dump(to_save, f, indent=2, default=str)


    def update(self, updates: List[Dict]): 
        failed_updates = []
        successes = 0
        for u in updates:
            try:
                coll_name = u.pop("collection")
                coll = self.database.collections[coll_name]
                _id = ObjectId(u.pop("_id"))
                res = coll.find_one_and_update({"_id": _id}, {"$set": u})
            except Exception as e:
                failed_updates.append((u, e))
                continue
    
            if res:
                successes += 1
            else:
                failed_updates.append((u, "ID not in DB"))

        return successes, failed_updates
    

    def create_from_existing(self, creations: List[Dict]):
        failed_creations = []
        failed_subcreations = []
        h2_tpr_successes = 0
        o2_tpd_successes = 0
        co2_tpd_successes = 0
        osc_successes = 0

        for entry in creations:
            try:
                material_id = ObjectId(entry["material_id"])
                doi = entry["doi"]
                material_doc = self.materials_coll.find_one({"_id": material_id})
                if material_doc["doi"] != doi:
                    raise ValueError("DOI does not match existing material")
                
                h2_tpr = entry.get("h2_tpr_peaks", {})
                if h2_tpr != {} and h2_tpr.get("temps", False):
                    res, failure_reason = self.upload_characterisation(h2_tpr, "h2_tpr_peaks", doi, material_id)
                    if res:
                        h2_tpr_successes += 1
                    else:
                        failed_subcreations.append((entry, failure_reason))

                o2_tpd = entry.get("o2_tpd_peaks", {})
                if o2_tpd != {} and o2_tpd.get("temps", False):
                    res, failure_reason = self.upload_characterisation(o2_tpd, "o2_tpd_peaks", doi, material_id)
                    if res:
                        o2_tpd_successes += 1
                    else:
                        failed_subcreations.append((entry, failure_reason))


                co2_tpd = entry.get("co2_tpd_peaks", {})
                if co2_tpd != {} and co2_tpd.get("temps", False):
                    res, failure_reason = self.upload_characterisation(co2_tpd, "co2_tpd_peaks", doi, material_id)
                    if res:
                        co2_tpd_successes += 1
                    else:
                        failed_subcreations.append((entry, failure_reason))

                osc = entry.get("osc_entries", [])
                n_inserted, failures = self.upload_oscs(osc, material_id, doi)
                osc_successes += n_inserted
                failed_subcreations.extend(failures)

            except:
                failed_creations.append(entry)
                continue

        return failed_creations, failed_subcreations, h2_tpr_successes, o2_tpd_successes, co2_tpd_successes, osc_successes
        
        
    def upload_characterisation(self, entry: Dict, collection_name: str, doi: str, material_id: ObjectId):
        # upload O2_TPD, H2_TPR, CO2_TPD or OSC
        try:
            coll = self.database.collections[collection_name]
            entry.update({"doi": doi, "material_id": material_id})
            res = coll.insert_one(entry)
            if res.inserted_id:
                return True, ""
            else:
                return False, "insertion error"

        except Exception as e: 
            return False, "upload_characterisation function failure"
        

    def upload_oscs(self, osc_entries: List[Dict], material_id: ObjectId, doi: str):
        try:
            coll = self.database.collections["osc"]
            successes = 0
            failures = []
            for osc_entry in osc_entries:
                osc_entry.update({"doi": doi, "material_id": material_id})
                res = coll.insert_one(osc_entry)
                if res.inserted_id:
                    successes += 1
                else:
                    failures.append((osc_entry, "insertion error"))
            return successes, failures
                
        except Exception as e:
            return 0, osc_entries
        
    def update_all_by_filter(self, update_by_filter_entries: List[Dict]):
        failed_updates = []
        successes = 0
        for entry in update_by_filter_entries:
            try:
                coll_name = entry.pop("collection")
                coll = self.database.collections[coll_name]
                filter_ = entry.pop("filter")
                update = entry.pop("update")
                res = coll.update_many(filter_, update)
            except Exception as e:
                failed_updates.append((entry, e))
                continue
    
            if res.modified_count > 0:
                successes += res.modified_count
            else:
                failed_updates.append((entry, "No documents matched the filter"))

        return successes, failed_updates
