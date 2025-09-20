#!/usr/bin/env python3
"""
BatGentic Automated Data Ingestion Pipeline
Processes new Titley Chorus data dumps into standardized structure
"""

import os
import shutil
import re
from pathlib import Path
from datetime import datetime
import pandas as pd
import hashlib
import json

class TitleyChorusIngestor:
    def __init__(self, project_root="."):
        self.project_root = Path(project_root)
        self.raw_data_root = self.project_root / "data" / "raw"
        self.log_file = self.project_root / "data" / "ingestion_log.json"
        
        # Ensure target directories exist
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directory structure"""
        directories = [
            self.raw_data_root / "acoustic",
            self.raw_data_root / "logs", 
            self.raw_data_root / "metadata",
            self.project_root / "data" / "interim" / "ingestion_staging"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def log_ingestion(self, batch_info):
        """Log ingestion details for audit trail"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "batch_id": batch_info["batch_id"],
            "source_path": str(batch_info["source_path"]),
            "files_processed": batch_info["files_processed"],
            "duplicates_found": batch_info["duplicates_found"],
            "errors": batch_info["errors"]
        }
        
        # Load existing log or create new
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                ingestion_log = json.load(f)
        else:
            ingestion_log = []
            
        ingestion_log.append(log_entry)
        
        with open(self.log_file, 'w') as f:
            json.dump(ingestion_log, f, indent=2)
            
    def calculate_file_hash(self, file_path):
        """Calculate MD5 hash for duplicate detection"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
        
    def parse_audio_filename(self, filename):
        """Extract metadata from audio filename"""
        # Pattern: S4U12318_YYYYMMDD_HHMMSS.w4v
        pattern = r'(S4U\d+)_(\d{8})_(\d{6})\.w4v'
        match = re.match(pattern, filename)
        
        if match:
            device_id, date_str, time_str = match.groups()
            year = date_str[:4]
            month = date_str[4:6] 
            day = date_str[6:8]
            
            return {
                "device_id": device_id,
                "date": f"{year}-{month}-{day}",
                "time": f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}",
                "year": year,
                "month": month,
                "day": day,
                "filename": filename
            }
        return None
        
    def ingest_audio_files(self, source_data_dir, batch_id):
        """Ingest .w4v files from Titley Chorus Data/ directory"""
        source_path = Path(source_data_dir) / "Data"
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source data directory not found: {source_path}")
            
        processed_files = []
        duplicates = []
        errors = []
        
        for audio_file in source_path.glob("*.w4v"):
            try:
                # Parse filename metadata
                metadata = self.parse_audio_filename(audio_file.name)
                if not metadata:
                    errors.append(f"Invalid filename format: {audio_file.name}")
                    continue
                    
                # Calculate target path with date organization
                target_dir = (self.raw_data_root / "acoustic" / 
                             metadata["year"] / metadata["month"] / metadata["day"])
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = target_dir / audio_file.name
                
                # Check for duplicates using file hash
                if target_path.exists():
                    source_hash = self.calculate_file_hash(audio_file)
                    target_hash = self.calculate_file_hash(target_path)
                    
                    if source_hash == target_hash:
                        duplicates.append(audio_file.name)
                        continue
                    else:
                        # Same name, different content - create versioned name
                        timestamp = datetime.now().strftime("%H%M%S")
                        versioned_name = f"{audio_file.stem}_v{timestamp}{audio_file.suffix}"
                        target_path = target_dir / versioned_name
                        
                # Copy file to target location
                shutil.copy2(audio_file, target_path)
                processed_files.append({
                    "source": str(audio_file),
                    "target": str(target_path),
                    "metadata": metadata,
                    "batch_id": batch_id,
                    "ingestion_time": datetime.now().isoformat()
                })
                
                print(f"✓ Ingested: {audio_file.name} -> {target_path.relative_to(self.project_root)}")
                
            except Exception as e:
                errors.append(f"Error processing {audio_file.name}: {str(e)}")
                print(f"✗ Error: {audio_file.name} - {e}")
                
        return processed_files, duplicates, errors
        
    def ingest_log_files(self, source_data_dir, batch_id):
        """Ingest CSV log files from Titley Chorus logs/ directory"""
        source_path = Path(source_data_dir) / "logs"
        
        if not source_path.exists():
            print(f"Warning: No logs directory found at {source_path}")
            return [], [], []
            
        processed_logs = []
        duplicates = []
        errors = []
        
        target_dir = self.raw_data_root / "logs"
        
        for log_file in source_path.glob("*.csv"):
            try:
                target_path = target_dir / f"{batch_id}_{log_file.name}"
                
                # Check if file already exists
                if target_path.exists():
                    duplicates.append(log_file.name)
                    continue
                    
                shutil.copy2(log_file, target_path)
                processed_logs.append({
                    "source": str(log_file),
                    "target": str(target_path),
                    "batch_id": batch_id,
                    "ingestion_time": datetime.now().isoformat()
                })
                
                print(f"✓ Ingested log: {log_file.name} -> {target_path.relative_to(self.project_root)}")
                
            except Exception as e:
                errors.append(f"Error processing {log_file.name}: {str(e)}")
                print(f"✗ Error: {log_file.name} - {e}")
                
        return processed_logs, duplicates, errors
        
    def ingest_metadata_files(self, source_data_dir, batch_id):
        """Ingest metadata files like summary reports"""
        source_path = Path(source_data_dir)
        
        processed_metadata = []
        duplicates = []
        errors = []
        
        target_dir = self.raw_data_root / "metadata"
        
        # Look for summary files and other metadata
        metadata_patterns = ["*Summary*.txt", "*.json", "*.xml"]
        
        for pattern in metadata_patterns:
            for metadata_file in source_path.glob(pattern):
                try:
                    target_path = target_dir / f"{batch_id}_{metadata_file.name}"
                    
                    if target_path.exists():
                        duplicates.append(metadata_file.name)
                        continue
                        
                    shutil.copy2(metadata_file, target_path)
                    processed_metadata.append({
                        "source": str(metadata_file),
                        "target": str(target_path),
                        "batch_id": batch_id,
                        "ingestion_time": datetime.now().isoformat()
                    })
                    
                    print(f"✓ Ingested metadata: {metadata_file.name} -> {target_path.relative_to(self.project_root)}")
                    
                except Exception as e:
                    errors.append(f"Error processing {metadata_file.name}: {str(e)}")
                    print(f"✗ Error: {metadata_file.name} - {e}")
                    
        return processed_metadata, duplicates, errors
        
    def generate_batch_report(self, batch_info):
        """Generate ingestion report for the batch"""
        report_dir = self.project_root / "outputs" / "ingestion_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = report_dir / f"ingestion_report_{batch_info['batch_id']}.json"
        
        with open(report_path, 'w') as f:
            json.dump(batch_info, f, indent=2)
            
        print(f"✓ Generated ingestion report: {report_path}")
        return report_path
        
    def ingest_batch(self, source_directory, batch_name=None):
        """Complete ingestion pipeline for a new Titley Chorus data dump"""
        
        source_path = Path(source_directory)
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory not found: {source_path}")
            
        # Generate batch ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_id = batch_name or f"chorus_{timestamp}"
        
        print(f"Starting ingestion batch: {batch_id}")
        print(f"Source: {source_path}")
        print("=" * 50)
        
        # Ingest each data type
        audio_files, audio_dupes, audio_errors = self.ingest_audio_files(source_path, batch_id)
        log_files, log_dupes, log_errors = self.ingest_log_files(source_path, batch_id)
        metadata_files, meta_dupes, meta_errors = self.ingest_metadata_files(source_path, batch_id)
        
        # Compile batch information
        batch_info = {
            "batch_id": batch_id,
            "source_path": str(source_path),
            "ingestion_timestamp": datetime.now().isoformat(),
            "files_processed": {
                "audio": len(audio_files),
                "logs": len(log_files),
                "metadata": len(metadata_files)
            },
            "duplicates_found": {
                "audio": len(audio_dupes),
                "logs": len(log_dupes), 
                "metadata": len(meta_dupes)
            },
            "errors": {
                "audio": audio_errors,
                "logs": log_errors,
                "metadata": meta_errors
            },
            "detailed_files": {
                "audio": audio_files,
                "logs": log_files,
                "metadata": metadata_files
            }
        }
        
        # Log the ingestion
        self.log_ingestion(batch_info)
        
        # Generate report
        self.generate_batch_report(batch_info)
        
        # Summary
        total_processed = sum(batch_info["files_processed"].values())
        total_duplicates = sum(batch_info["duplicates_found"].values())
        total_errors = len(audio_errors) + len(log_errors) + len(meta_errors)
        
        print("=" * 50)
        print(f"Ingestion Complete: {batch_id}")
        print(f"Files processed: {total_processed}")
        print(f"Duplicates skipped: {total_duplicates}")
        print(f"Errors encountered: {total_errors}")
        
        if total_errors > 0:
            print("\nErrors:")
            for error in audio_errors + log_errors + meta_errors:
                print(f"  - {error}")
                
        return batch_info

def main():
    """Command line interface for data ingestion"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest Titley Chorus data into BatGentic")
    parser.add_argument("source_dir", help="Path to Titley Chorus data directory")
    parser.add_argument("--batch-name", help="Custom batch name (default: auto-generated)")
    parser.add_argument("--project-root", default=".", help="BatGentic project root directory")
    
    args = parser.parse_args()
    
    ingestor = TitleyChorusIngestor(project_root=args.project_root)
    
    try:
        batch_info = ingestor.ingest_batch(args.source_dir, args.batch_name)
        print(f"\n✓ Ingestion successful! Batch ID: {batch_info['batch_id']}")
        
    except Exception as e:
        print(f"\n✗ Ingestion failed: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())
