"""
BatGentic: Interactive Bioacoustic Analysis Tool for Bat Call Validation and Sonification

A comprehensive tool for processing SonoBat validation files and generating
human-audible versions of bat calls through pitch shifting.

Author: BatGentic Development Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "BatGentic Development Team"
__email__ = "contact@batgentic.org"

from .processing.sonobat_parser import SonoBatParser
from .processing.audio_processor import AudioProcessor
from .utils.utils import create_output_directory, validate_audio_file, get_system_info

__all__ = [
    "SonoBatParser",
    "AudioProcessor", 
    "create_output_directory",
    "validate_audio_file",
    "get_system_info",
]
