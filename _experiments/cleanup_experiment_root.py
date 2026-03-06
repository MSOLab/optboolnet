import argparse
import shutil
from pathlib import Path


def resolve_root(root_arg: str) -> Path:
    root = Path(root_arg)
    if root.is_absolute():
        return root

    # Prefer caller cwd; fallback to _experiments-relative.
    if root.exists():
        return root.resolve()
    return (Path(__file__).resolve().parent / root).resolve()


def cleanup(root: Path, dry_run: bool = False):
    preserved_files = []
    removed_files = []
    removed_dirs = []

    # Remove all files except alg_config.json.
    for file_path in sorted(p for p in root.rglob("*") if p.is_file()):
        if file_path.name == "alg_config.json":
            preserved_files.append(file_path)
            continue
        removed_files.append(file_path)
        if not dry_run:
            file_path.unlink()

    # Remove directories that become empty (bottom-up), but keep root itself.
    all_dirs = sorted(
        (p for p in root.rglob("*") if p.is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    )
    for dir_path in all_dirs:
        if any(dir_path.iterdir()):
            continue
        removed_dirs.append(dir_path)
        if not dry_run:
            dir_path.rmdir()

    return preserved_files, removed_files, removed_dirs


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Clean an experiments root by removing everything under it except files "
            "named 'alg_config.json'."
        )
    )
    parser.add_argument(
        "root_folder",
        help=(
            "Root experiments folder to clean. Can be absolute or relative. "
            "Examples: '_experiments/260306_full', '260306_full'."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without deleting anything.",
    )
    args = parser.parse_args()

    root = resolve_root(args.root_folder)
    if not root.exists() or not root.is_dir():
        raise NotADirectoryError(f"Root folder not found: {root}")

    preserved_files, removed_files, removed_dirs = cleanup(root, dry_run=args.dry_run)
    mode = "DRY RUN" if args.dry_run else "CLEANUP"
    print(f"[{mode}] root={root}")
    print(f"Preserved alg_config.json files: {len(preserved_files)}")
    for path in preserved_files:
        print(f"  = {path}")
    print(f"Removed files: {len(removed_files)}")
    for path in removed_files:
        print(f"  - {path}")
    print(f"Removed directories: {len(removed_dirs)}")
    for path in removed_dirs:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
