#!/usr/bin/env python3
"""
BatGentic Data Migration Script
Reorganizes existing directory structure to industry standards
"""

import os
import shutil
from pathlib import Path
import re
from datetime import datetime

class BatGenticMigrator:
    def __init__(self, base_path="."):
        self.base_path = Path(base_path)
        self.dry_run = True  # Set to False to actually move files
        
    def log(self, message):
        """Simple logging function"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
        
    def setup_directory_structure(self):
        """Create the standard directory structure"""
        
        directories = [
            "src/batgentic/analysis",
            "src/batgentic/processing", 
            "src/batgentic/generative",
            "src/batgentic/utils",
            "apps",
            "external_tools",
            "data/raw/acoustic",
            "data/raw/logs",
            "data/raw/metadata",
            "data/processed/validated",
            "data/processed/filtered", 
            "data/processed/spectrograms",
            "data/interim/candidates",
            "data/interim/temp",
            "data/exports/nabat",
            "data/exports/research",
            "models/trained",
            "models/checkpoints",
            "outputs/reports",
            "outputs/visualizations",
            "outputs/audio_samples",
            "tests/fixtures",
            "docs",
            "scripts",
            "config"
        ]
        
        for directory in directories:
            dir_path = self.base_path / directory
            if not self.dry_run:
                dir_path.mkdir(parents=True, exist_ok=True)
            self.log(f"{'[DRY RUN] ' if self.dry_run else ''}Created: {directory}")
            
            # Create __init__.py files for Python packages
            if "src/batgentic" in directory:
                init_file = dir_path / "__init__.py"
                if not self.dry_run and not init_file.exists():
                    init_file.touch()
                    
    def migrate_audio_files(self):
        """Migrate .w4v files to date-organized structure"""
        
        source_dirs = [
            self.base_path / "batRecordings" / "Data",
            self.base_path / "sonoRaw" / "Data"
        ]
        
        target_base = self.base_path / "data" / "raw" / "acoustic"
        
        # Pattern to extract date from filename: S4U12318_YYYYMMDD_HHMMSS.w4v
        date_pattern = r'S4U\d+_(\d{8})_\d{6}\.w4v'
        
        moved_files = set()  # Track files to avoid duplicates
        
        for source_dir in source_dirs:
            if source_dir.exists():
                self.log(f"Processing directory: {source_dir}")
                
                for file_path in source_dir.glob("*.w4v"):
                    filename = file_path.name
                    
                    # Skip if we've already processed this file
                    if filename in moved_files:
                        self.log(f"Skipping duplicate: {filename}")
                        continue
                        
                    match = re.match(date_pattern, filename)
                    if match:
                        date_str = match.group(1)  # YYYYMMDD
                        year = date_str[:4]
                        month = date_str[4:6]
                        day = date_str[6:8]
                        
                        target_dir = target_base / year / month / day
                        target_path = target_dir / filename
                        
                        if not self.dry_run:
                            target_dir.mkdir(parents=True, exist_ok=True)
                            
                            if not target_path.exists():
                                shutil.move(str(file_path), str(target_path))
                                moved_files.add(filename)
                                self.log(f"Moved: {filename} -> {target_dir}")
                            else:
                                self.log(f"Target exists: {filename}")
                        else:
                            self.log(f"[DRY RUN] Would move: {filename} -> {year}/{month}/{day}")
                            moved_files.add(filename)
                    else:
                        self.log(f"Invalid filename format: {filename}")
                        
    def migrate_logs(self):
        """Migrate log files to standardized location"""
        
        source_dirs = [
            self.base_path / "batRecordings" / "logs",
            self.base_path / "sonoRaw" / "logs"
        ]
        target_dir = self.base_path / "data" / "raw" / "logs"
        
        if not self.dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
        
        moved_files = set()
        
        for source_dir in source_dirs:
            if source_dir.exists():
                for file_path in source_dir.iterdir():
                    if file_path.is_file() and file_path.name not in moved_files:
                        target_path = target_dir / file_path.name
                        
                        if not self.dry_run:
                            if not target_path.exists():
                                shutil.move(str(file_path), str(target_path))
                                moved_files.add(file_path.name)
                                self.log(f"Moved log: {file_path.name}")
                            else:
                                self.log(f"Log exists: {file_path.name}")
                        else:
                            self.log(f"[DRY RUN] Would move log: {file_path.name}")
                            moved_files.add(file_path.name)
                            
    def migrate_metadata(self):
        """Migrate metadata files"""
        
        source_files = [
            self.base_path / "batRecordings" / "S4U12318_A_Summary.txt",
            self.base_path / "sonoRaw" / "S4U12318_A_Summary.txt"
        ]
        
        target_dir = self.base_path / "data" / "raw" / "metadata"
        
        if not self.dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
        
        for source_file in source_files:
            if source_file.exists():
                target_path = target_dir / source_file.name
                
                if not self.dry_run:
                    if not target_path.exists():
                        shutil.copy2(str(source_file), str(target_path))
                        self.log(f"Copied metadata: {source_file.name}")
                    else:
                        self.log(f"Metadata exists: {source_file.name}")
                else:
                    self.log(f"[DRY RUN] Would copy metadata: {source_file.name}")
                    
    def migrate_external_tools(self):
        """Move BattyBirdNET-Analyzer to external_tools"""
        
        source_dir = self.base_path / "batgentic" / "BattyBirdNET-Analyzer"
        target_dir = self.base_path / "external_tools" / "BattyBirdNET-Analyzer"
        
        if source_dir.exists():
            if not self.dry_run:
                target_dir.parent.mkdir(parents=True, exist_ok=True)
                if not target_dir.exists():
                    shutil.move(str(source_dir), str(target_dir))
                    self.log(f"Moved BattyBirdNET-Analyzer to external_tools/")
                else:
                    self.log(f"BattyBirdNET-Analyzer already in external_tools/")
            else:
                self.log(f"[DRY RUN] Would move BattyBirdNET-Analyzer to external_tools/")
                
    def migrate_existing_code(self):
        """Migrate existing batgentic modules to src/"""
        
        # List of Python files to migrate
        source_files = [
            ("batgentic/batgentic_cli_version.py", "apps/cli.py"),
            ("batgentic/batgentic_sonobat_parser.py", "src/batgentic/processing/sonobat_parser.py"),
            ("batgentic/batgentic_working.py", "src/batgentic/processing/audio_processor.py"),
        ]
        
        for source_rel, target_rel in source_files:
            source_path = self.base_path / source_rel
            target_path = self.base_path / target_rel
            
            if source_path.exists():
                if not self.dry_run:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    if not target_path.exists():
                        shutil.copy2(str(source_path), str(target_path))
                        self.log(f"Migrated: {source_rel} -> {target_rel}")
                    else:
                        self.log(f"Target exists: {target_rel}")
                else:
                    self.log(f"[DRY RUN] Would migrate: {source_rel} -> {target_rel}")
                    
    def generate_config_files(self):
        """Generate standard configuration files"""
        
        # requirements.txt
        requirements_content = """
# Core dependencies
pandas>=1.5.0
numpy>=1.21.0
librosa>=0.9.0
streamlit>=1.25.0
matplotlib>=3.5.0
seaborn>=0.11.0

# Audio processing
soundfile>=0.10.0
scipy>=1.9.0

# Data handling
pyyaml>=6.0
python-dotenv>=0.19.0

# Optional dependencies
plotly>=5.0.0
jupyter>=1.0.0
""".strip()

        # environment.yml
        environment_content = """
name: batgentic
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pandas>=1.5.0
  - numpy>=1.21.0
  - librosa>=0.9.0
  - matplotlib>=3.5.0
  - seaborn>=0.11.0
  - soundfile>=0.10.0
  - scipy>=1.9.0
  - pip
  - pip:
    - streamlit>=1.25.0
    - pyyaml>=6.0
    - python-dotenv>=0.19.0
    - plotly>=5.0.0
""".strip()

        # .gitignore
        gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Data
data/raw/
data/interim/
*.w4v
*.wav
*.mp3

# Jupyter
.ipynb_checkpoints

# Environment
.env
.venv
env/
venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
""".strip()

        config_files = [
            ("requirements.txt", requirements_content),
            ("environment.yml", environment_content),
            (".gitignore", gitignore_content),
        ]
        
        for filename, content in config_files:
            file_path = self.base_path / filename
            
            if not self.dry_run:
                if not file_path.exists():
                    file_path.write_text(content)
                    self.log(f"Generated: {filename}")
                else:
                    self.log(f"Config exists: {filename}")
            else:
                self.log(f"[DRY RUN] Would generate: {filename}")
                
    def run_migration(self):
        """Run the complete migration process"""
        
        self.log("Starting BatGentic migration...")
        self.log(f"Base path: {self.base_path.absolute()}")
        self.log(f"Dry run: {self.dry_run}")
        
        steps = [
            ("Setting up directory structure", self.setup_directory_structure),
            ("Migrating audio files", self.migrate_audio_files),
            ("Migrating log files", self.migrate_logs),
            ("Migrating metadata", self.migrate_metadata),
            ("Migrating external tools", self.migrate_external_tools),
            ("Migrating existing code", self.migrate_existing_code),
            ("Generating config files", self.generate_config_files),
        ]
        
        for step_name, step_func in steps:
            self.log(f"--- {step_name} ---")
            try:
                step_func()
                self.log(f"✓ {step_name} completed")
            except Exception as e:
                self.log(f"✗ {step_name} failed: {e}")
                
        self.log("Migration complete!")
        
        if self.dry_run:
            self.log("\n" + "="*50)
            self.log("This was a DRY RUN - no files were actually moved")
            self.log("Set dry_run=False to perform the actual migration")
            self.log("="*50)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate BatGentic to industry-standard structure")
    parser.add_argument("--path", default=".", help="Base path for migration (default: current directory)")
    parser.add_argument("--execute", action="store_true", help="Actually perform migration (default: dry run)")
    
    args = parser.parse_args()
    
    migrator = BatGenticMigrator(base_path=args.path)
    migrator.dry_run = not args.execute
    
    migrator.run_migration()

if __name__ == "__main__":
    main()