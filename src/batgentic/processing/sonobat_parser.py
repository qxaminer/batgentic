"""
SonoBat Parser Module

Handles parsing of SonoBat validation files in tab-separated format.
Extracts validated bat calls with confidence scores (excluding "Inf" values).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SonoBatParser:
    """
    Parser for SonoBat validation files (.txt format with tab separation).
    
    Handles extraction of validated bat calls with confidence scores,
    filtering out invalid entries and preparing data for audio processing.
    """
    
    def __init__(self):
        """Initialize the SonoBat parser."""
        self.data = None
        self.validated_calls = None
        self.file_path = None
        
    def load_validation_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load SonoBat validation file and parse its contents.
        
        Args:
            file_path: Path to the SonoBat validation file (.txt)
            
        Returns:
            pandas.DataFrame: Parsed validation data
            
        Raises:
            FileNotFoundError: If the validation file doesn't exist
            pd.errors.ParserError: If the file format is invalid
        """
        self.file_path = Path(file_path)
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Validation file not found: {self.file_path}")
            
        try:
            # Read tab-separated file with flexible handling
            self.data = pd.read_csv(
                self.file_path,
                sep='\t',
                encoding='utf-8',
                low_memory=False,
                na_values=['', 'NA', 'N/A', 'null'],
                keep_default_na=True
            )
            
            logger.info(f"Loaded {len(self.data)} records from {self.file_path}")
            logger.info(f"Columns found: {list(self.data.columns)}")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error parsing validation file: {e}")
            raise pd.errors.ParserError(f"Failed to parse {self.file_path}: {e}")
    
    def extract_validated_calls(self, 
                              confidence_column: str = None,
                              species_column: str = None,
                              filename_column: str = None,
                              min_confidence: float = 0.0,
                              confidence_column_index: int = None) -> pd.DataFrame:
        """
        Extract validated bat calls excluding infinite confidence values.
        
        Args:
            confidence_column: Name of the confidence score column (auto-detected if None)
            species_column: Name of the species identification column (auto-detected if None)
            filename_column: Name of the audio filename column (auto-detected if None)
            min_confidence: Minimum confidence threshold (default: 0.0)
            confidence_column_index: Use column index instead of name (like your working code with parts[6])
            
        Returns:
            pandas.DataFrame: Filtered dataframe with validated calls
            
        Raises:
            ValueError: If required columns are not found
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_validation_file() first.")
        
        # Handle column index approach (like your working code)
        if confidence_column_index is not None:
            # Use column index approach similar to your working code
            working_data = self.data.copy()
            
            if confidence_column_index >= len(working_data.columns):
                raise ValueError(f"Column index {confidence_column_index} out of range. Max: {len(working_data.columns)-1}")
            
            conf_col_name = working_data.columns[confidence_column_index]
            filename_col_name = working_data.columns[0]  # Assume first column is filename
            
            # Convert confidence to numeric and filter like your code
            working_data[conf_col_name] = pd.to_numeric(working_data[conf_col_name], errors='coerce')
            
            # Filter out 'Inf' and empty values (like your working code)
            mask_valid = working_data[conf_col_name].notna()
            mask_not_inf = np.isfinite(working_data[conf_col_name])
            mask_confidence = working_data[conf_col_name] >= min_confidence
            
            final_mask = mask_valid & mask_not_inf & mask_confidence
            
        else:
            # Auto-detect or use provided column names
            if confidence_column is None:
                confidence_column = self._find_confidence_column()
            if species_column is None:
                try:
                    species_column = self._find_species_column()
                except:
                    species_column = None  # Make species optional
            if filename_column is None:
                filename_column = self._find_filename_column()
            
            # Validate required columns exist
            missing_cols = []
            for col in [confidence_column, filename_column]:
                if col not in self.data.columns:
                    missing_cols.append(col)
            
            if missing_cols:
                available_cols = list(self.data.columns)
                raise ValueError(f"Missing columns: {missing_cols}. Available columns: {available_cols}")
            
            # Create a copy for processing
            working_data = self.data.copy()
            
            # Convert confidence column to numeric, handling various text representations
            working_data[confidence_column] = pd.to_numeric(
                working_data[confidence_column], 
                errors='coerce'
            )
            
            # Filter out infinite values and NaN
            mask_finite = np.isfinite(working_data[confidence_column])
            mask_confidence = working_data[confidence_column] >= min_confidence
            mask_filename = working_data[filename_column].notna()
            
            if species_column and species_column in working_data.columns:
                mask_species = working_data[species_column].notna()
                final_mask = mask_finite & mask_confidence & mask_species & mask_filename
            else:
                final_mask = mask_finite & mask_confidence & mask_filename
            
        
        self.validated_calls = working_data[final_mask].copy()
        
        # Log filtering results
        original_count = len(working_data)
        final_count = len(self.validated_calls)
        
        logger.info(f"Filtering results:")
        logger.info(f"  Original records: {original_count}")
        logger.info(f"  Final validated calls: {final_count}")
        
        if confidence_column_index is not None:
            logger.info(f"  Used column index {confidence_column_index} for confidence")
        else:
            logger.info(f"  Used column names for filtering")
        
        return self.validated_calls
    
    def get_species_summary(self) -> Dict[str, int]:
        """
        Get summary statistics of species in validated calls.
        
        Returns:
            Dict[str, int]: Species codes and their counts
        """
        if self.validated_calls is None:
            raise ValueError("No validated calls available. Run extract_validated_calls() first.")
        
        try:
            species_col = self._find_species_column()
            species_counts = self.validated_calls[species_col].value_counts().to_dict()
            
            logger.info(f"Species summary: {species_counts}")
            return species_counts
        except ValueError:
            # No species column found
            logger.info("No species column available for summary")
            return {"Unknown": len(self.validated_calls)}
    
    def get_confidence_stats(self) -> Dict[str, float]:
        """
        Get confidence score statistics for validated calls.
        
        Returns:
            Dict[str, float]: Statistical summary of confidence scores
        """
        if self.validated_calls is None:
            raise ValueError("No validated calls available. Run extract_validated_calls() first.")
        
        try:
            confidence_col = self._find_confidence_column()
            conf_series = self.validated_calls[confidence_col]
        except ValueError:
            # Try to find confidence column by index or content
            numeric_columns = self.validated_calls.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                # Use the first numeric column as confidence
                confidence_col = numeric_columns[0]
                conf_series = self.validated_calls[confidence_col]
            else:
                raise ValueError("No confidence column found and no numeric columns available")
        
        stats = {
            'mean': conf_series.mean(),
            'median': conf_series.median(),
            'std': conf_series.std(),
            'min': conf_series.min(),
            'max': conf_series.max(),
            'q25': conf_series.quantile(0.25),
            'q75': conf_series.quantile(0.75)
        }
        
        logger.info(f"Confidence statistics: {stats}")
        return stats
    
    def get_audio_file_paths(self, audio_directory: Optional[Path] = None) -> List[Tuple[str, Path]]:
        """
        Get paths to corresponding audio files for validated calls.
        
        Args:
            audio_directory: Directory containing audio files (default: same as validation file)
            
        Returns:
            List[Tuple[str, Path]]: List of (species, audio_file_path) tuples
        """
        if self.validated_calls is None:
            raise ValueError("No validated calls available. Run extract_validated_calls() first.")
        
        if audio_directory is None:
            audio_directory = self.file_path.parent
        else:
            audio_directory = Path(audio_directory)
        
        filename_col = self._find_filename_column()
        species_col = self._find_species_column()
        
        audio_paths = []
        missing_files = []
        
        for _, row in self.validated_calls.iterrows():
            filename = row[filename_col]
            species = row[species_col]
            
            # Try different common audio extensions
            audio_extensions = ['.wav', '.WAV', '.mp3', '.MP3', '.flac', '.FLAC']
            audio_file = None
            
            for ext in audio_extensions:
                # Remove existing extension and try with new one
                base_name = Path(filename).stem
                potential_file = audio_directory / f"{base_name}{ext}"
                
                if potential_file.exists():
                    audio_file = potential_file
                    break
            
            if audio_file:
                audio_paths.append((species, audio_file))
            else:
                missing_files.append(filename)
        
        if missing_files:
            logger.warning(f"Missing audio files: {missing_files[:5]}...")  # Show first 5
            
        logger.info(f"Found {len(audio_paths)} audio files out of {len(self.validated_calls)} validated calls")
        
        return audio_paths
    
    def _find_confidence_column(self) -> str:
        """Find the confidence column name (case-insensitive)."""
        possible_names = ['confidence', 'Confidence', 'CONFIDENCE', 'conf', 'Conf']
        for name in possible_names:
            if name in self.data.columns:
                return name
        raise ValueError("No confidence column found in the data")
    
    def _find_species_column(self) -> str:
        """Find the species column name (case-insensitive)."""
        possible_names = ['species', 'Species', 'SPECIES', 'spec', 'Spec']
        for name in possible_names:
            if name in self.data.columns:
                return name
        raise ValueError("No species column found in the data")
    
    def _find_filename_column(self) -> str:
        """Find the filename column name (case-insensitive)."""
        possible_names = ['filename', 'Filename', 'FILENAME', 'file', 'File', 'name', 'Name']
        for name in possible_names:
            if name in self.data.columns:
                return name
        raise ValueError("No filename column found in the data")
    
    def export_validated_calls(self, output_path: Union[str, Path]) -> None:
        """
        Export validated calls to a CSV file.
        
        Args:
            output_path: Path for the output CSV file
        """
        if self.validated_calls is None:
            raise ValueError("No validated calls available. Run extract_validated_calls() first.")
        
        output_path = Path(output_path)
        self.validated_calls.to_csv(output_path, index=False)
        logger.info(f"Exported {len(self.validated_calls)} validated calls to {output_path}")