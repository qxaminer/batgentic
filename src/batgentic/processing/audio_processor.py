"""
Audio Processor Module

Handles audio processing for bat call sonification using librosa.
Provides pitch shifting functionality to make ultrasonic bat calls audible to humans.
"""

import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Audio processor for bat call sonification.
    
    Handles loading, processing, and saving of bat call audio files
    with pitch shifting to make ultrasonic calls audible to humans.
    """
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize the audio processor.
        
        Args:
            sample_rate: Target sample rate for processing (default: 22050 Hz)
        """
        self.sample_rate = sample_rate
        self.processed_audio = {}
        
    def load_audio(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Load audio file using librosa.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple[np.ndarray, int]: (audio_data, original_sample_rate)
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio file cannot be loaded
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        try:
            # Load audio file with librosa
            audio_data, sr = librosa.load(
                str(file_path),
                sr=None,  # Keep original sample rate initially
                mono=True,  # Convert to mono
                dtype=np.float32
            )
            
            logger.debug(f"Loaded {file_path}: {len(audio_data)} samples at {sr} Hz")
            
            return audio_data, sr
            
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            raise ValueError(f"Cannot load audio file {file_path}: {e}")
    
    def pitch_shift_audio(self, 
                         audio_data: np.ndarray, 
                         sr: int,
                         cents: float,
                         preserve_length: bool = True) -> np.ndarray:
        """
        Apply pitch shifting to make ultrasonic calls audible.
        
        Args:
            audio_data: Input audio data
            sr: Sample rate of the input audio
            cents: Pitch shift amount in cents (negative values shift down)
                  Typical range: -1200 to -3600 cents (1-3 octaves down)
            preserve_length: Whether to preserve the original audio length
            
        Returns:
            np.ndarray: Pitch-shifted audio data
        """
        if cents == 0:
            logger.warning("No pitch shift applied (cents = 0)")
            return audio_data
        
        try:
            # Convert cents to semitones (100 cents = 1 semitone)
            semitones = cents / 100.0
            
            # Apply pitch shifting using librosa
            shifted_audio = librosa.effects.pitch_shift(
                audio_data,
                sr=sr,
                n_steps=semitones,
                bins_per_octave=12 * 4  # Higher resolution for better quality
            )
            
            # Normalize audio to prevent clipping
            if np.max(np.abs(shifted_audio)) > 0:
                shifted_audio = shifted_audio / np.max(np.abs(shifted_audio)) * 0.95
            
            logger.debug(f"Applied pitch shift: {cents} cents ({semitones:.2f} semitones)")
            
            return shifted_audio
            
        except Exception as e:
            logger.error(f"Error applying pitch shift: {e}")
            raise ValueError(f"Pitch shift failed: {e}")
    
    def process_single_file(self,
                           input_path: Union[str, Path],
                           output_path: Union[str, Path],
                           cents: float = -2400,
                           target_sr: Optional[int] = None) -> Dict[str, Union[str, float, int]]:
        """
        Process a single audio file with pitch shifting.
        
        Args:
            input_path: Path to input audio file
            output_path: Path for output processed file
            cents: Pitch shift amount in cents (default: -2400, 2 octaves down)
            target_sr: Target sample rate for output (default: use class default)
            
        Returns:
            Dict: Processing metadata including file paths and parameters
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if target_sr is None:
            target_sr = self.sample_rate
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load original audio
        audio_data, original_sr = self.load_audio(input_path)
        
        # Resample to target sample rate if needed
        if original_sr != target_sr:
            audio_data = librosa.resample(
                audio_data, 
                orig_sr=original_sr, 
                target_sr=target_sr
            )
        
        # Apply pitch shifting
        processed_audio = self.pitch_shift_audio(audio_data, target_sr, cents)
        
        # Save processed audio
        sf.write(
            str(output_path),
            processed_audio,
            target_sr,
            format='WAV',
            subtype='PCM_24'  # High quality output
        )
        
        # Calculate processing statistics
        original_duration = len(audio_data) / target_sr
        processed_duration = len(processed_audio) / target_sr
        
        metadata = {
            'input_path': str(input_path),
            'output_path': str(output_path),
            'original_sr': original_sr,
            'target_sr': target_sr,
            'pitch_shift_cents': cents,
            'original_duration': original_duration,
            'processed_duration': processed_duration,
            'samples_original': len(audio_data),
            'samples_processed': len(processed_audio)
        }
        
        logger.info(f"Processed {input_path.name}: {cents} cents shift, {original_duration:.2f}s duration")
        
        return metadata
    
    def batch_process(self,
                     file_paths: List[Tuple[str, Path, Path]],
                     cents: float = -2400,
                     target_sr: Optional[int] = None,
                     show_progress: bool = True) -> List[Dict]:
        """
        Process multiple audio files in batch.
        
        Args:
            file_paths: List of (species, input_path, output_path) tuples
            cents: Pitch shift amount in cents (default: -2400)
            target_sr: Target sample rate for output
            show_progress: Whether to show progress bar
            
        Returns:
            List[Dict]: List of processing metadata for each file
        """
        if target_sr is None:
            target_sr = self.sample_rate
        
        results = []
        failed_files = []
        
        # Setup progress bar
        iterator = tqdm(file_paths, desc="Processing audio files") if show_progress else file_paths
        
        for species, input_path, output_path in iterator:
            try:
                # Add species to metadata
                metadata = self.process_single_file(input_path, output_path, cents, target_sr)
                metadata['species'] = species
                metadata['status'] = 'success'
                
                results.append(metadata)
                
                if show_progress:
                    iterator.set_postfix({'current': input_path.name, 'species': species})
                    
            except Exception as e:
                error_info = {
                    'input_path': str(input_path),
                    'output_path': str(output_path),
                    'species': species,
                    'status': 'failed',
                    'error': str(e)
                }
                
                results.append(error_info)
                failed_files.append(input_path.name)
                
                logger.error(f"Failed to process {input_path}: {e}")
        
        # Log summary
        successful = len([r for r in results if r['status'] == 'success'])
        logger.info(f"Batch processing complete: {successful}/{len(file_paths)} files processed successfully")
        
        if failed_files:
            logger.warning(f"Failed files: {failed_files[:3]}{'...' if len(failed_files) > 3 else ''}")
        
        return results
    
    def get_audio_info(self, file_path: Union[str, Path]) -> Dict[str, Union[str, float, int]]:
        """
        Get information about an audio file without loading it fully.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dict: Audio file information
        """
        file_path = Path(file_path)
        
        try:
            info = sf.info(str(file_path))
            
            audio_info = {
                'file_path': str(file_path),
                'duration': info.duration,
                'sample_rate': info.samplerate,
                'channels': info.channels,
                'frames': info.frames,
                'format': info.format,
                'subtype': info.subtype,
                'file_size_mb': file_path.stat().st_size / (1024 * 1024)
            }
            
            return audio_info
            
        except Exception as e:
            logger.error(f"Error getting audio info for {file_path}: {e}")
            return {'file_path': str(file_path), 'error': str(e)}
    
    def create_preview(self,
                      audio_data: np.ndarray,
                      sr: int,
                      duration: float = 5.0,
                      start_time: float = 0.0) -> np.ndarray:
        """
        Create a preview/excerpt of audio data.
        
        Args:
            audio_data: Input audio data
            sr: Sample rate
            duration: Preview duration in seconds (default: 5.0)
            start_time: Start time for preview in seconds (default: 0.0)
            
        Returns:
            np.ndarray: Preview audio data
        """
        start_sample = int(start_time * sr)
        end_sample = int((start_time + duration) * sr)
        
        # Ensure we don't exceed audio bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_data), end_sample)
        
        preview = audio_data[start_sample:end_sample]
        
        logger.debug(f"Created preview: {len(preview)} samples ({len(preview)/sr:.2f}s)")
        
        return preview
    
    def generate_pitch_variants(self,
                               audio_data: np.ndarray,
                               sr: int,
                               cent_values: List[float]) -> Dict[float, np.ndarray]:
        """
        Generate multiple pitch-shifted versions of audio.
        
        Args:
            audio_data: Input audio data
            sr: Sample rate
            cent_values: List of pitch shift values in cents
            
        Returns:
            Dict[float, np.ndarray]: Dictionary mapping cent values to processed audio
        """
        variants = {}
        
        for cents in cent_values:
            try:
                shifted_audio = self.pitch_shift_audio(audio_data, sr, cents)
                variants[cents] = shifted_audio
                logger.debug(f"Generated variant at {cents} cents")
            except Exception as e:
                logger.warning(f"Failed to generate variant at {cents} cents: {e}")
        
        return variants