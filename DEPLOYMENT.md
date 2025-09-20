# BatGentic Deployment Guide

## Streamlit Community Cloud Deployment

### Prerequisites
- GitHub repository with the BatGentic code
- Streamlit Community Cloud account

### Deployment Steps

1. **Repository Setup**
   ```bash
   git init
   git add .
   git commit -m "Initial BatGentic portfolio project"
   git remote add origin https://github.com/your-username/batgentic.git
   git push -u origin main
   ```

2. **Streamlit Cloud Configuration**
   - Connect GitHub repository
   - Set Python version: 3.9+
   - Main file path: `app.py`
   - Advanced settings: Enable resource limits

3. **Environment Setup**
   - Dependencies automatically installed from `requirements.txt`
   - System packages installed from `packages.txt`
   - Configuration loaded from `.streamlit/config.toml`

### Configuration Files

#### `.streamlit/config.toml`
- Sets theme colors and server configuration
- Optimized for professional presentation

#### `packages.txt`
- System-level dependencies for audio processing
- Includes `libsndfile1` and `ffmpeg`

#### `requirements.txt`
- Python package dependencies
- Version pinned for stability
- Streamlined for cloud deployment

## Local Development Setup

### Installation
```bash
# Clone repository
git clone https://github.com/your-username/batgentic.git
cd batgentic

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

### Development Commands
```bash
# Test imports
python -c "import sys; sys.path.append('src'); from batgentic import SonoBatParser"

# Test ingestion pipeline
python ingest_data.py --help

# Run linting
# (if using development tools)
flake8 src/
black src/
```

## Production Considerations

### Performance
- **Memory usage**: ~200-500 MB for typical audio processing
- **Processing time**: 0.5-2x real-time depending on file size
- **Concurrent users**: 2-5 users supported on free tier

### Limitations
- **File size limits**: 200 MB max per upload
- **Processing timeout**: 10 minutes max per operation
- **Storage**: Temporary files only, no persistent storage

### Optimization Tips
- Use preview mode for large batch processing
- Consider chunking large audio files
- Enable caching for repeated operations

## Demo Mode Features

### Sample Data
- `/demo_data/sample_validation.txt`: 10 sample bat call records
- Species diversity from North American bats
- Realistic confidence scores and metadata

### Limited Functionality
- Audio processing requires actual .wav files
- Demo focuses on data parsing and visualization
- Full pipeline demonstration available locally

## Security & Privacy

### Data Handling
- All uploaded files are temporary
- No persistent storage of user data
- Processing happens in isolated containers

### Best Practices
- Don't upload sensitive research data to cloud demo
- Use local installation for confidential datasets
- Demo version for presentation purposes only

## Troubleshooting

### Common Issues

#### Import Errors
```python
# If experiencing import issues, verify path:
import sys
sys.path.append('src')
from batgentic import SonoBatParser
```

#### Audio Processing Errors
- Ensure audio files are valid formats (WAV recommended)
- Check file size limits (< 200 MB per file)
- Verify sample rates are supported (22050/44100/48000 Hz)

#### Deployment Issues
- Check `requirements.txt` for version conflicts
- Verify `packages.txt` system dependencies
- Review Streamlit Cloud build logs

### Support

For deployment issues:
1. Check Streamlit Community Cloud documentation
2. Review GitHub Actions/deployment logs
3. Verify all configuration files are properly formatted

For technical questions about the BatGentic implementation:
- Review the comprehensive README.md
- Check inline code documentation
- Examine the modular package structure in `src/batgentic/`

## Future Enhancements

### Cloud Infrastructure
- **AWS/GCP deployment** for production scale
- **Docker containerization** for consistent environments  
- **CI/CD pipelines** for automated testing and deployment

### Performance Scaling
- **Redis caching** for repeated operations
- **Celery task queues** for background processing
- **Database integration** for persistent storage

### Advanced Features
- **REST API endpoints** for programmatic access
- **Batch processing queues** for large datasets
- **Real-time streaming** for live acoustic monitors
