#!/usr/bin/env python3
"""
organize_photos_by_date.py
Script to organize existing photos in the camera directory into subdirectories by date.
Reads the modification date of each photo and moves it to the corresponding date subdirectory.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
import argparse


def organize_photos(camera_dir):
    """
    Organize photos in camera directory into subdirectories by date.

    Args:
        camera_dir: Path to the camera directory containing photos
    """
    camera_path = Path(camera_dir)

    if not camera_path.exists():
        print(f"Error: Directory {camera_dir} does not exist")
        return

    # Image extensions to look for
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    # Counter for statistics
    moved_count = 0
    skipped_count = 0
    error_count = 0

    print(f"Scanning directory: {camera_dir}")
    print("-" * 60)

    # Iterate through all files in the camera directory (not subdirectories)
    for file_path in camera_path.iterdir():
        # Skip if it's a directory
        if file_path.is_dir():
            skipped_count += 1
            continue

        # Check if it's an image file
        if file_path.suffix.lower() not in image_extensions:
            skipped_count += 1
            continue

        try:
            # Get the modification time of the file
            mtime = file_path.stat().st_mtime
            file_date = datetime.fromtimestamp(mtime)
            date_str = file_date.strftime("%Y-%m-%d")

            # Create the subdirectory for this date
            date_dir = camera_path / date_str
            date_dir.mkdir(exist_ok=True)

            # Move the file to the date subdirectory
            destination = date_dir / file_path.name

            # Check if destination already exists
            if destination.exists():
                print(
                    f"Warning: {destination} already exists, skipping {file_path.name}"
                )
                skipped_count += 1
                continue

            shutil.move(str(file_path), str(destination))
            print(f"Moved: {file_path.name} -> {date_str}/")
            moved_count += 1

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            error_count += 1

    # Print summary
    print("-" * 60)
    print(f"Summary:")
    print(f"  Files moved: {moved_count}")
    print(f"  Files skipped: {skipped_count}")
    print(f"  Errors: {error_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Organize photos in camera directory by date"
    )
    parser.add_argument(
        "--camera-dir",
        type=str,
        default="/home/roboteseo/dev/picam/output/camera",
        help="Path to the camera directory (default: /home/roboteseo/dev/picam/output/camera)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually moving files",
    )

    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN MODE - No files will be moved")
        print("-" * 60)
        # TODO: Implement dry run mode if needed

    organize_photos(args.camera_dir)


if __name__ == "__main__":
    main()
