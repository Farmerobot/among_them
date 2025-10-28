#!/usr/bin/env python3
"""Script to update llm_model_name in game state JSON files."""

import argparse
import json
from pathlib import Path
from typing import Any


def update_model_name(data: Any, new_model_name: str) -> tuple[Any, int]:
    """Recursively update all llm_model_name fields in the data structure.
    
    Args:
        data: JSON data structure to update
        new_model_name: New model name to set
        
    Returns:
        Tuple of (updated data, count of changes made)
    """
    changes = 0
    
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "llm_model_name":
                if data[key] != new_model_name:
                    data[key] = new_model_name
                    changes += 1
            else:
                _, sub_changes = update_model_name(value, new_model_name)
                changes += sub_changes
    elif isinstance(data, list):
        for item in data:
            _, sub_changes = update_model_name(item, new_model_name)
            changes += sub_changes
    
    return data, changes


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Update llm_model_name in game state JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Update default data/game_state.json with default model
            python update_model_name.py
            
            # Update specific file in data/ directory with default model
            python update_model_name.py --file old_games/20250402_064841.json
            
            # Update with custom model name
            python update_model_name.py --model "deepseek-r1:14b"
            
            # Update specific file with custom model
            python update_model_name.py --file my_game.json --model "gpt-4"
        """
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default="game_state.json",
        help="Path to the JSON file to update (default: data/game_state.json)"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="deepseek-r1:1.5b",
        help="New model name to set (default: deepseek-r1:1.5b)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying the file"
    )
    
    args = parser.parse_args()
    
    # Get project root (parent of scripts directory)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / "data"
    
    # Resolve file path relative to data directory if not absolute
    file_path = Path(args.file)
    if not file_path.is_absolute():
        file_path = data_dir / file_path
    
    # Check if file exists
    if not file_path.exists():
        print(f"Error: File '{file_path}' does not exist")
        return
    
    # Read JSON file
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file: {e}")
        return
    except Exception as e:
        print(f"Error: Failed to read file: {e}")
        return
    
    # Update model names
    updated_data, changes = update_model_name(data, args.model)
    
    # Report results
    if changes == 0:
        print(f"No changes needed. All llm_model_name fields already set to '{args.model}'")
        return
    
    print(f"Found {changes} instance(s) of 'llm_model_name' to update")
    
    if args.dry_run:
        print(f"[DRY RUN] Would update '{file_path}' to set llm_model_name = '{args.model}'")
        return
    
    # Write back to file
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(updated_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Successfully updated '{file_path}'")
        print(f"  Set {changes} instance(s) of llm_model_name to '{args.model}'")
    except Exception as e:
        print(f"Error: Failed to write file: {e}")


if __name__ == "__main__":
    main()
