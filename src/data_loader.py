"""
Data loader module for Zenodo EMG datasets.
Provides utilities to download and load EMG data from Zenodo repositories.
"""

import os
import urllib.request
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json


class ZenodoDataLoader:
    """
    Loader for EMG datasets from Zenodo.
    Supports downloading and loading data from Zenodo record 14182446 and similar EMG datasets.
    """
    
    def __init__(self, zenodo_record_id: str = "14182446"):
        """
        Initialize the Zenodo data loader.
        
        Args:
            zenodo_record_id (str): Zenodo record ID (default: 14182446)
        """
        self.zenodo_record_id = zenodo_record_id
        self.base_url = f"https://zenodo.org/records/{zenodo_record_id}"
        self.api_url = f"https://zenodo.org/api/records/{zenodo_record_id}"
        self.download_dir = "data/zenodo"
        
    def get_record_info(self) -> Dict:
        """
        Fetch metadata about the Zenodo record.
        
        Returns:
            dict: Record metadata including title, description, and file list
        """
        try:
            with urllib.request.urlopen(self.api_url) as response:
                data = json.loads(response.read())
                return {
                    'title': data.get('metadata', {}).get('title'),
                    'description': data.get('metadata', {}).get('description'),
                    'files': data.get('files', [])
                }
        except Exception as e:
            print(f"Error fetching record info: {e}")
            print(f"Please visit {self.base_url} to download data manually")
            return {}
    
    def list_files(self) -> List[Dict]:
        """
        List all files available in the Zenodo record.
        
        Returns:
            list: List of file information dictionaries
        """
        info = self.get_record_info()
        return info.get('files', [])
    
    def download_file(self, filename: str, force: bool = False) -> str:
        """
        Download a specific file from the Zenodo record.
        
        Args:
            filename (str): Name of the file to download
            force (bool): Force re-download even if file exists
            
        Returns:
            str: Path to the downloaded file
        """
        os.makedirs(self.download_dir, exist_ok=True)
        filepath = os.path.join(self.download_dir, filename)
        
        if os.path.exists(filepath) and not force:
            print(f"File {filename} already exists. Use force=True to re-download.")
            return filepath
        
        try:
            # Get file list and find download URL
            files = self.list_files()
            file_info = next((f for f in files if f.get('key') == filename), None)
            
            if not file_info:
                raise ValueError(f"File {filename} not found in record")
            
            download_url = file_info.get('links', {}).get('self')
            if not download_url:
                raise ValueError(f"Download URL not found for {filename}")
            
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(download_url, filepath)
            print(f"Downloaded to {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            print(f"\nPlease download manually from: {self.base_url}")
            print(f"Save the file to: {filepath}")
            raise
    
    def download_all(self, force: bool = False) -> List[str]:
        """
        Download all files from the Zenodo record.
        
        Args:
            force (bool): Force re-download even if files exist
            
        Returns:
            list: Paths to all downloaded files
        """
        files = self.list_files()
        downloaded = []
        
        for file_info in files:
            filename = file_info.get('key')
            try:
                filepath = self.download_file(filename, force=force)
                downloaded.append(filepath)
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
        
        return downloaded
    
    def load_emg_csv(self, filepath: str, 
                     signal_column: str = 'emg',
                     label_column: str = 'fatigue_label') -> Tuple[np.ndarray, np.ndarray]:
        """
        Load EMG data from a CSV file.
        
        Args:
            filepath (str): Path to CSV file
            signal_column (str): Name of the column containing EMG signal
            label_column (str): Name of the column containing fatigue labels
            
        Returns:
            tuple: (signals, labels) as numpy arrays
        """
        try:
            df = pd.read_csv(filepath)
            
            # Extract signals
            if signal_column in df.columns:
                signals = df[signal_column].values
            else:
                # If column not found, try to auto-detect numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    signals = df[numeric_cols[0]].values
                else:
                    raise ValueError(f"Could not find signal column '{signal_column}'")
            
            # Extract labels if available
            labels = None
            if label_column in df.columns:
                labels = df[label_column].values
            
            return signals, labels
            
        except Exception as e:
            raise ValueError(f"Error loading EMG data from {filepath}: {e}")
    
    def load_dataset(self, 
                    data_dir: Optional[str] = None,
                    file_pattern: str = "*.csv") -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Load entire EMG dataset from a directory.
        
        Args:
            data_dir (str): Directory containing data files (default: self.download_dir)
            file_pattern (str): Pattern to match files (default: *.csv)
            
        Returns:
            dict: Dictionary mapping filenames to (signals, labels) tuples
        """
        if data_dir is None:
            data_dir = self.download_dir
        
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory {data_dir} does not exist")
        
        import glob
        dataset = {}
        files = glob.glob(os.path.join(data_dir, file_pattern))
        
        for filepath in files:
            filename = os.path.basename(filepath)
            try:
                signals, labels = self.load_emg_csv(filepath)
                dataset[filename] = (signals, labels)
                print(f"Loaded {filename}: {len(signals)} samples")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        return dataset


def download_zenodo_dataset(record_id: str = "14182446", 
                            output_dir: str = "data/zenodo") -> str:
    """
    Convenience function to download a Zenodo dataset.
    
    Args:
        record_id (str): Zenodo record ID
        output_dir (str): Directory to save downloaded files
        
    Returns:
        str: Path to the download directory
    """
    loader = ZenodoDataLoader(record_id)
    loader.download_dir = output_dir
    
    print(f"Downloading Zenodo record {record_id}...")
    print(f"URL: https://zenodo.org/records/{record_id}")
    
    try:
        files = loader.download_all()
        print(f"\nSuccessfully downloaded {len(files)} files to {output_dir}")
        return output_dir
    except Exception as e:
        print(f"\nAutomatic download failed: {e}")
        print(f"\nPlease download manually:")
        print(f"1. Visit: https://zenodo.org/records/{record_id}")
        print(f"2. Download all files")
        print(f"3. Save them to: {output_dir}")
        return output_dir


def load_zenodo_emg_data(data_dir: str = "data/zenodo") -> Dict:
    """
    Load EMG data from Zenodo dataset directory.
    
    Args:
        data_dir (str): Directory containing Zenodo data files
        
    Returns:
        dict: Loaded dataset with signals and labels
    """
    loader = ZenodoDataLoader()
    return loader.load_dataset(data_dir)


if __name__ == "__main__":
    # Example usage
    print("Zenodo EMG Data Loader")
    print("=" * 60)
    
    # Initialize loader
    loader = ZenodoDataLoader("14182446")
    
    # Get record information
    print("\nFetching record information...")
    info = loader.get_record_info()
    
    if info:
        print(f"\nTitle: {info.get('title', 'N/A')}")
        print(f"\nFiles available:")
        for file_info in info.get('files', []):
            filename = file_info.get('key')
            size_mb = file_info.get('size', 0) / (1024 * 1024)
            print(f"  - {filename} ({size_mb:.2f} MB)")
    else:
        print("\nCould not fetch record information automatically.")
        print(f"Please visit: https://zenodo.org/records/14182446")
        print("And download the files manually to data/zenodo/")
