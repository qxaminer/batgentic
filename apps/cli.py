# BatGentic - CLI Version for BatDetect2 v1.3.0

import os
import subprocess
import sys
from pathlib import Path
import pandas as pd
import json

def setup_batgentic_structure():
    """Create BatGentic project structure"""
    dirs = [
        'batgentic/detection',
        'batgentic/processing', 
        'batgentic/generative',
        'batgentic/analysis',
        'data/raw',
        'data/candidates',
        'data/processed',
        'data/metadata',
        'data/nabat_exports',
        'templates',
        'species_lists'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created: {dir_path}")

def test_batdetect2_cli():
    """Test BatDetect2 CLI installation"""
    try:
        result = subprocess.run(['batdetect2', '--help'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ BatDetect2 CLI available")
            return True
        else:
            print(f"❌ BatDetect2 CLI error: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ BatDetect2 CLI not found in PATH")
        return False
    except subprocess.TimeoutExpired:
        print("❌ BatDetect2 CLI timeout")
        return False

def basic_detection_test_cli(audio_dir_path, output_dir):
    """Test BatDetect2 CLI on audio directory"""
    try:
        print(f"🔍 Testing CLI detection on: {audio_dir_path}")
        print(f"📁 Output will go to: {output_dir}")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Run BatDetect2 CLI on directory (limit to first few files for testing)
        cmd = [
            'batdetect2', 'detect', 
            str(audio_dir_path),
            '--output_dir', str(output_dir),
            '--detection_threshold', '0.3'
        ]
        
        print(f"🚀 Running command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ CLI detection completed successfully!")
            print(f"Output: {result.stdout}")
            
            # Check what files were created
            output_path = Path(output_dir)
            result_files = list(output_path.glob("**/*"))
            print(f"📊 Created {len(result_files)} output files:")
            for f in result_files[:5]:  # Show first 5
                print(f"  {f.name}")
            
            return output_path
        else:
            print(f"❌ CLI detection failed:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print("❌ Detection timeout (>60s)")
        return None
    except Exception as e:
        print(f"❌ Detection failed: {e}")
        return None

def scan_recordings_directory(recordings_path):
    """Scan for WAV files in recordings directory"""
    recordings_path = Path(recordings_path)
    
    if not recordings_path.exists():
        print(f"❌ Directory not found: {recordings_path}")
        return []
    
    wav_files = list(recordings_path.glob("**/*.wav"))
    wav_files.extend(list(recordings_path.glob("**/*.WAV")))
    
    print(f"📁 Found {len(wav_files)} WAV files in {recordings_path}")
    
    # Show first few files
    for i, wav_file in enumerate(wav_files[:5]):
        file_size = wav_file.stat().st_size / (1024*1024)  # MB
        print(f"  {i+1}. {wav_file.name} ({file_size:.1f} MB)")
    
    if len(wav_files) > 5:
        print(f"  ... and {len(wav_files) - 5} more files")
    
    return wav_files

def main():
    """Main setup and test function"""
    print("🦇 BatGentic - Generative Bioacoustic Analysis Pipeline")
    print("=" * 60)
    
    # 1. Create project structure
    print("\n📁 Setting up project structure...")
    setup_batgentic_structure()
    
    # 2. Test BatDetect2 CLI
    print("\n🔧 Testing BatDetect2 CLI...")
    if not test_batdetect2_cli():
        return
    
    # 3. Scan for recordings
    print("\n🔍 Scanning for recordings...")
    recordings_path = "~/Desktop/batRecordings/recordings"
    recordings_path_expanded = Path(recordings_path).expanduser()
    wav_files = scan_recordings_directory(recordings_path_expanded)
    
    if not wav_files:
        print("❌ No WAV files found. Please check the recordings path.")
        return
    
    # 4. Test detection on first subdirectory
    if wav_files:
        print(f"\n🦇 Testing CLI detection...")
        
        # Find first subdirectory with files
        first_file = wav_files[0]
        test_dir = first_file.parent
        output_dir = "./data/candidates/test_detection"
        
        print(f"Testing on directory: {test_dir}")
        results = basic_detection_test_cli(test_dir, output_dir)
        
        if results:
            print("✅ BatGentic CLI detection test successful!")
            print(f"\n📊 Ready to process {len(wav_files)} total files")
            print("\nNext steps:")
            print("1. Run batch detection on all directories")
            print("2. Parse detection results") 
            print("3. Build sonification pipeline")
            print(f"\n🎯 Test results in: {output_dir}")
        else:
            print("⚠️  Detection may need troubleshooting")

if __name__ == "__main__":
    main()