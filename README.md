# Music Chord and Melody Extractor

This Python script uses OpenAI's GPT-4 model to analyze music files and extract chord progressions, melody information, and align them with lyrics when available.

## Features

- Audio file processing and feature extraction
- Chord progression analysis
- Melody analysis
- Lyrics alignment with chords
- Key and mode detection
- Pattern recognition
- JSON output format

## Requirements

- Python 3.8+
- OpenAI API key
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Basic usage:
   ```python
   from music_analyzer import MusicAnalyzer
   
   analyzer = MusicAnalyzer()
   analysis = analyzer.analyze_music("path/to/your/audio/file.mp3")
   ```

2. With lyrics:
   ```python
   lyrics = """
   [Verse 1]
   Your lyrics here
   """
   analysis = analyzer.analyze_music("path/to/your/audio/file.mp3", lyrics)
   ```

3. Save results:
   ```python
   analyzer.save_analysis(analysis, "output.json")
   ```

## Output Format

The analysis results are saved in JSON format with the following structure:

```json
{
    "key": "string",
    "mode": "string",
    "chord_progression": [
        {
            "chord": "string",
            "start_time": "float",
            "end_time": "float",
            "lyrics": "string (if available)"
        }
    ],
    "melody_analysis": "string",
    "patterns": ["string"]
}
```

## Notes

- The script uses librosa for audio processing and feature extraction
- OpenAI's GPT-4 model is used for the musical analysis
- Processing time may vary depending on the length of the audio file
- Results are more accurate when lyrics are provided 