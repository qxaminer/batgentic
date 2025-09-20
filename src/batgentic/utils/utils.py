"""
Utility Functions Module

Common utility functions for BatGentic package including file management,
validation, and directory operations.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_output_directory(base_path: Union[str, Path], 
                          subdirectory: str = "validated_calls",
                          clean_existing: bool = False) -> Path:
    """
    Create output directory structure for processed files.
    
    Args:
        base_path: Base path for output directory
        subdirectory: Name of the subdirectory to create
        clean_existing: Whether to clean existing directory contents
        
    Returns:
        Path: Path to the created output directory
    """
    base_path = Path(base_path)
    output_dir = base_path / subdirectory
    
    if clean_existing and output_dir.exists():
        logger.info(f"Cleaning existing directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for organization
    subdirs = ["original", "processed", "metadata", "exports"]
    for subdir in subdirs:
        (output_dir / subdir).mkdir(exist_ok=True)
    
    logger.info(f"Created output directory structure: {output_dir}")
    return output_dir


def validate_audio_file(file_path: Union[str, Path]) -> Dict[str, Union[bool, str]]:
    """
    Validate if a file is a valid audio file.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Dict: Validation result with status and message
    """
    file_path = Path(file_path)
    
    result = {
        'is_valid': False,
        'message': '',
        'file_path': str(file_path)
    }
    
    # Check if file exists
    if not file_path.exists():
        result['message'] = 'File does not exist'
        return result
    
    # Check file extension
    valid_extensions = ['.wav', '.WAV', '.mp3', '.MP3', '.flac', '.FLAC', '.aiff', '.AIFF']
    if file_path.suffix not in valid_extensions:
        result['message'] = f'Invalid file extension. Supported: {valid_extensions}'
        return result
    
    # Check file size (basic validation)
    file_size = file_path.stat().st_size
    if file_size == 0:
        result['message'] = 'File is empty'
        return result
    
    if file_size > 500 * 1024 * 1024:  # 500 MB limit
        result['message'] = 'File is too large (>500MB)'
        return result
    
    # Try to import soundfile to validate audio format
    try:
        import soundfile as sf
        info = sf.info(str(file_path))
        
        if info.duration <= 0:
            result['message'] = 'Invalid audio duration'
            return result
            
        result['is_valid'] = True
        result['message'] = f'Valid audio file: {info.duration:.2f}s, {info.samplerate}Hz'
        
    except ImportError:
        # Fallback validation without soundfile
        result['is_valid'] = True
        result['message'] = 'Basic validation passed (soundfile not available for detailed check)'
        
    except Exception as e:
        result['message'] = f'Audio validation failed: {str(e)}'
    
    return result


def organize_files_by_species(file_list: List[Tuple[str, Path]], 
                             output_dir: Path) -> Dict[str, List[Path]]:
    """
    Organize files into species-specific subdirectories.
    
    Args:
        file_list: List of (species, file_path) tuples
        output_dir: Base output directory
        
    Returns:
        Dict[str, List[Path]]: Dictionary mapping species to file paths
    """
    species_files = {}
    
    for species, file_path in file_list:
        if species not in species_files:
            species_files[species] = []
            # Create species directory
            species_dir = output_dir / "by_species" / species
            species_dir.mkdir(parents=True, exist_ok=True)
        
        species_files[species].append(file_path)
    
    logger.info(f"Organized {len(file_list)} files into {len(species_files)} species groups")
    return species_files


def get_file_hash(file_path: Union[str, Path]) -> str:
    """
    Generate MD5 hash of a file for integrity checking.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: MD5 hash of the file
    """
    import hashlib
    
    file_path = Path(file_path)
    hash_md5 = hashlib.md5()
    
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {e}")
        return ""


def save_processing_log(results: List[Dict], output_file: Union[str, Path]) -> None:
    """
    Save processing results to a CSV log file.
    
    Args:
        results: List of processing result dictionaries
        output_file: Path to output CSV file
    """
    output_file = Path(output_file)
    
    if not results:
        logger.warning("No results to save")
        return
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    logger.info(f"Saved processing log with {len(results)} entries to {output_file}")


def cleanup_temporary_files(directory: Union[str, Path], pattern: str = "temp_*") -> int:
    """
    Clean up temporary files in a directory.
    
    Args:
        directory: Directory to clean
        pattern: Glob pattern for temporary files
        
    Returns:
        int: Number of files cleaned up
    """
    directory = Path(directory)
    
    if not directory.exists():
        return 0
    
    temp_files = list(directory.glob(pattern))
    
    for file_path in temp_files:
        try:
            file_path.unlink()
            logger.debug(f"Removed temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to remove {file_path}: {e}")
    
    logger.info(f"Cleaned up {len(temp_files)} temporary files")
    return len(temp_files)


def estimate_processing_time(num_files: int, avg_duration: float = 5.0) -> Dict[str, float]:
    """
    Estimate processing time for batch operations.
    
    Args:
        num_files: Number of files to process
        avg_duration: Average file duration in seconds
        
    Returns:
        Dict[str, float]: Time estimates in different units
    """
    # Rough estimates based on typical processing performance
    time_per_file = 0.5 + (avg_duration * 0.1)  # Base overhead + duration factor
    total_seconds = num_files * time_per_file
    
    estimates = {
        'total_seconds': total_seconds,
        'total_minutes': total_seconds / 60,
        'total_hours': total_seconds / 3600,
        'files_per_minute': 60 / time_per_file if time_per_file > 0 else 0
    }
    
    return estimates


def check_disk_space(directory: Union[str, Path], required_mb: float = 100) -> Dict[str, Union[bool, float]]:
    """
    Check available disk space in a directory.
    
    Args:
        directory: Directory to check
        required_mb: Required space in MB
        
    Returns:
        Dict: Disk space information and availability status
    """
    directory = Path(directory)
    
    try:
        # Get disk usage statistics
        stat = shutil.disk_usage(directory)
        
        available_mb = stat.free / (1024 * 1024)
        total_mb = stat.total / (1024 * 1024)
        used_mb = (stat.total - stat.free) / (1024 * 1024)
        
        result = {
            'has_space': available_mb >= required_mb,
            'available_mb': available_mb,
            'total_mb': total_mb,
            'used_mb': used_mb,
            'required_mb': required_mb,
            'usage_percent': (used_mb / total_mb) * 100 if total_mb > 0 else 0
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error checking disk space for {directory}: {e}")
        return {
            'has_space': False,
            'error': str(e)
        }


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        str: Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_system_info() -> Dict[str, str]:
    """
    Get basic system information for debugging.
    
    Returns:
        Dict[str, str]: System information
    """
    import platform
    import sys
    
    info = {
        'platform': platform.platform(),
        'python_version': sys.version,
        'processor': platform.processor(),
        'architecture': platform.architecture()[0]
    }
    
    # Add package versions if available
    try:
        import librosa
        info['librosa_version'] = librosa.__version__
    except ImportError:
        info['librosa_version'] = 'Not installed'
    
    try:
        import streamlit
        info['streamlit_version'] = streamlit.__version__
    except ImportError:
        info['streamlit_version'] = 'Not installed'
    
    return info
