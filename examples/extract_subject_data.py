"""
Script to extract subject data from zip files.
Useful when you have limited disk space - extracts subjects one at a time.
"""

import os
import zipfile
import argparse
import shutil


def get_available_subjects(data_dir):
    """Get list of available zip files."""
    zip_files = [f for f in os.listdir(data_dir) if f.endswith('.zip')]
    zip_files.sort()
    return zip_files


def extract_subject(subject_zip, data_dir, overwrite=False):
    """
    Extract a single subject's data.
    
    Args:
        subject_zip (str): Name of the zip file
        data_dir (str): Path to data directory
        overwrite (bool): Whether to overwrite existing extraction
    """
    zip_path = os.path.join(data_dir, subject_zip)
    subject_name = subject_zip.replace('.zip', '')
    subject_dir = os.path.join(data_dir, subject_name)
    
    # Check if already extracted
    if os.path.exists(subject_dir) and not overwrite:
        print(f"✓ {subject_name} already extracted (use --overwrite to re-extract)")
        return True
    
    # Extract
    try:
        print(f"Extracting {subject_name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print(f"✓ {subject_name} extracted successfully")
        return True
    except Exception as e:
        print(f"✗ Error extracting {subject_name}: {str(e)}")
        return False


def cleanup_subject(subject_name, data_dir):
    """
    Remove extracted subject data to free up space.
    
    Args:
        subject_name (str): Name of subject folder
        data_dir (str): Path to data directory
    """
    subject_dir = os.path.join(data_dir, subject_name)
    
    if os.path.exists(subject_dir):
        try:
            shutil.rmtree(subject_dir)
            print(f"✓ Removed {subject_name} directory")
            return True
        except Exception as e:
            print(f"✗ Error removing {subject_name}: {str(e)}")
            return False
    else:
        print(f"✗ {subject_name} directory not found")
        return False


def list_extracted_subjects(data_dir):
    """List all extracted subject folders."""
    subjects = [d for d in os.listdir(data_dir) 
                if os.path.isdir(os.path.join(data_dir, d)) 
                and d.startswith('subject_')]
    subjects.sort()
    return subjects


def main():
    parser = argparse.ArgumentParser(description='Manage subject data extraction')
    parser.add_argument('action', choices=['list', 'extract', 'extract-all', 'cleanup', 'status'],
                       help='Action to perform')
    parser.add_argument('--subject', type=str, help='Subject name or number (e.g., subject_1 or 1)')
    parser.add_argument('--data-dir', type=str, default='../data/sEMG_data',
                       help='Path to data directory')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing extracted data')
    
    args = parser.parse_args()
    
    # Resolve data directory path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, args.data_dir))
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    # Execute action
    if args.action == 'list':
        zip_files = get_available_subjects(data_dir)
        print(f"\nAvailable subjects ({len(zip_files)} zip files):")
        for zf in zip_files:
            print(f"  - {zf}")
    
    elif args.action == 'status':
        zip_files = get_available_subjects(data_dir)
        extracted = list_extracted_subjects(data_dir)
        
        print(f"\nData Status:")
        print(f"  Total subjects (zip files): {len(zip_files)}")
        print(f"  Extracted subjects: {len(extracted)}")
        
        if extracted:
            print(f"\nExtracted:")
            for subject in extracted:
                print(f"  ✓ {subject}")
        
        not_extracted = [zf.replace('.zip', '') for zf in zip_files 
                        if zf.replace('.zip', '') not in extracted]
        if not_extracted:
            print(f"\nNot extracted:")
            for subject in not_extracted:
                print(f"  - {subject}")
    
    elif args.action == 'extract':
        if not args.subject:
            print("Error: --subject is required for extract action")
            return
        
        # Handle both formats: "subject_1" or "1"
        if not args.subject.startswith('subject_'):
            subject_name = f'subject_{args.subject}'
        else:
            subject_name = args.subject
        
        subject_zip = f'{subject_name}.zip'
        zip_path = os.path.join(data_dir, subject_zip)
        
        if not os.path.exists(zip_path):
            print(f"Error: {subject_zip} not found in {data_dir}")
            return
        
        extract_subject(subject_zip, data_dir, args.overwrite)
    
    elif args.action == 'extract-all':
        zip_files = get_available_subjects(data_dir)
        print(f"\nExtracting all {len(zip_files)} subjects...")
        
        success_count = 0
        for zip_file in zip_files:
            if extract_subject(zip_file, data_dir, args.overwrite):
                success_count += 1
        
        print(f"\n✓ Successfully extracted {success_count}/{len(zip_files)} subjects")
    
    elif args.action == 'cleanup':
        if not args.subject:
            print("Error: --subject is required for cleanup action")
            return
        
        # Handle both formats
        if not args.subject.startswith('subject_'):
            subject_name = f'subject_{args.subject}'
        else:
            subject_name = args.subject
        
        cleanup_subject(subject_name, data_dir)


if __name__ == '__main__':
    main()
