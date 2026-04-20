#!/usr/bin/env python3
"""Script to rename gpt4_score to judge_score and gpt4_reasoning to judge_reasoning in all CSV files."""

import csv
import os
from pathlib import Path

def rename_columns_in_csv(file_path):
    """Rename columns in a CSV file."""
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        # Check if columns exist
        if 'gpt4_score' not in fieldnames and 'gpt4_reasoning' not in fieldnames:
            print(f"Skipping {file_path}: columns not found")
            return False
        
        # Create new fieldnames with renamed columns
        new_fieldnames = []
        for field in fieldnames:
            if field == 'gpt4_score':
                new_fieldnames.append('judge_score')
            elif field == 'gpt4_reasoning':
                new_fieldnames.append('judge_reasoning')
            else:
                new_fieldnames.append(field)
        
        # Read all rows
        rows = list(reader)
    
    # Write the file with new column names
    with open(file_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        
        # Rename keys in each row
        for row in rows:
            new_row = {}
            for key, value in row.items():
                if key == 'gpt4_score':
                    new_row['judge_score'] = value
                elif key == 'gpt4_reasoning':
                    new_row['judge_reasoning'] = value
                else:
                    new_row[key] = value
            writer.writerow(new_row)
    
    print(f"Updated {file_path}")
    return True

def main():
    """Process all CSV files in automate_scores directories."""
    base_dir = Path(__file__).parent / 'results' / 'automate_scores'
    
    csv_files = list(base_dir.rglob('*.csv'))
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    updated_count = 0
    for csv_file in csv_files:
        if rename_columns_in_csv(csv_file):
            updated_count += 1
    
    print(f"\nSuccessfully updated {updated_count} files")

if __name__ == '__main__':
    main()

