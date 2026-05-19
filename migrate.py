from src.db import Migrator, DB

file_path = "src/db/migrations/18-5-26.json"

db = DB()
migrator = Migrator(
    db,
    microscopy_dir="/Users/iwanpavord/desktop/HybridModel/db/microscopy_images",
    staging_dir="src/db/migrations/staged_microscopy",
    migration_failure_dir="src/db/migrations/failed_migrations",
)
migrator.migrate_file(file_path)
# migrator.backfill_subcreations_by_fingerprint(file_path)
