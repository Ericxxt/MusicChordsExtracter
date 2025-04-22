import os
import librosa
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import json
import subprocess
import soundfile as sf
import tempfile
from pathlib import Path
import torch
import re
from fpdf import FPDF
import shutil

class MusicAnalyzer:
    def __init__(self, api_key=None):
        load_dotenv()
        self.client = OpenAI(api_key='' or os.getenv('OPENAI_API_KEY'))
        
    def load_audio(self, audio_path):
        """Load and preprocess audio file."""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path)
            return y, sr
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return None, None

    def extract_features(self, y, sr):
        """Extract basic audio features."""
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        features = {
            # 'tempo': librosa.feature.tempo(y=y, sr=sr)[0],
            'tempo': float(tempo),
            'beat_frames': beat_frames,
            'beat_times': librosa.frames_to_time(beat_frames, sr=sr),
            'chroma': librosa.feature.chroma_cqt(y=y, sr=sr), # cqt preferred by music song, better frequency resolution constant-Q transform 
            'mfcc': librosa.feature.mfcc(y=y, sr=sr),
            'duration': librosa.get_duration(y=y, sr=sr)
        }
        return features

    def separate_vocals(self, audio_path):
        """Separate vocals from the music using Demucs."""
        print("Separating vocals from music...")
        try:
            audio_name = Path(audio_path).stem
            
            # Create a directory for storing separated vocals
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocal_files")
            os.makedirs(output_dir, exist_ok=True)
            
            vocals_dir = os.path.join(output_dir, audio_name)
            vocals_path = os.path.join(vocals_dir, f"{audio_name}.wav")
            
            # Check if vocals are already separated
            if os.path.exists(vocals_path):
                print(f"Using previously separated vocals from {vocals_path}")
                vocals, sr = librosa.load(vocals_path)
                return vocals, sr
            
            # Create temporary directory for Demucs processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Run Demucs
                subprocess.run([
                    'demucs', '--two-stems=vocals',
                    '-o', temp_dir,
                    audio_path
                ], check=True)

                # Get the path to the separated vocals in the temp directory
                temp_vocals_path = os.path.join(temp_dir, 'htdemucs', audio_name, 'vocals.wav')
                
                # Copy the vocals to our persistent storage
                os.makedirs(vocals_dir, exist_ok=True)
                shutil.copy(temp_vocals_path, vocals_path)
                
                # Load the separated vocals
                vocals, sr = librosa.load(vocals_path)
                return vocals, sr

        except Exception as e:
            print(f"Error in vocal separation: {e}")
            return None, None

    # Currently extracted japanese lyrics contains incorrect unicode(emojis), korean, might refine later
    # def clean_lyrics(self, lyrics):
    #     """Clean extracted lyrics to remove instrument sounds and other artifacts."""
    #     if not lyrics:
    #         return lyrics
            
    #     # Remove common non-lyric patterns
    #     # Remove sound effects, instrument sounds in brackets
    #     cleaned = re.sub(r'\[.*?\]', '', lyrics)
    #     # Remove repetitive sounds that might be instruments
    #     cleaned = re.sub(r'(?i)\b(na|la|da|dum|mmm|ooh|aah|yeah|oh|hm+|uh|ah)\b\s*(?:\1\s*)+', '', cleaned)
    #     # Remove isolated character repetitions that are likely not words
    #     cleaned = re.sub(r'(?i)(\w)\1{3,}', '', cleaned)
    #     # Remove short lines that are likely just sounds
    #     lines = [line for line in cleaned.split('\n') if len(line.strip()) > 3]
    #     cleaned = '\n'.join(lines)
        
    #     return cleaned.strip()

    def transcribe_vocals(self, vocals, sr, audio_name):
        """Transcribe vocals to lyrics using OpenAI Whisper."""
        print("Transcribing vocals to lyrics...")
        
        # Check if lyrics file already exists
        lyrics_file = f"{audio_name}_lyrics.txt"
        lyrics_json_file = f"{audio_name}_lyrics.json"
        if os.path.exists(lyrics_json_file):
            print(f"Found existing lyrics JSON file: {lyrics_json_file}")
            with open(lyrics_json_file, 'r') as f:
                lyrics_data = json.load(f)
                # Extract just the text for compatibility with existing code
                lyrics = "\n".join([segment["lyrics"] for segment in lyrics_data])
                return lyrics, lyrics_data
        elif os.path.exists(lyrics_file):
            print(f"Found existing lyrics file: {lyrics_file}")
            with open(lyrics_file, 'r') as f:
                lyrics = f.read()
                print(f"existing lyrics:{lyrics}")
                return lyrics, None
        
        try:
            # Save vocals to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                sf.write(temp_file.name, vocals, sr)
                
                # Use OpenAI's Whisper model for transcription with timestamps
                with open(temp_file.name, 'rb') as audio_file:
                    transcript = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json"  # Changed to get timestamps
                    )
            
            # Clean up temporary file
            os.unlink(temp_file.name)
            
            # Extract segments with timestamps
            lyrics_data = []
            full_text = []
            
            for segment in transcript.segments:
                lyrics_data.append({
                    "start_time": segment.start,
                    "end_time": segment.end,
                    "lyrics": segment.text
                })
                full_text.append(segment.text)
            
            # Join all lyrics text
            combined_lyrics = "\n".join(full_text)
            
            # Save the lyrics data to a JSON file
            with open(lyrics_json_file, 'w') as f:
                json.dump(lyrics_data, f, indent=2)
            print(f"Lyrics with timestamps saved to {lyrics_json_file}")
            
            # Also save the plain text for backward compatibility
            with open(lyrics_file, 'w') as f:
                print(f"extracted lyrics:{combined_lyrics}")
                f.write(combined_lyrics)
            print(f"Lyrics text saved to {lyrics_file}")
            
            return combined_lyrics, lyrics_data

        except Exception as e:
            print(f"Error in transcription: {e}")
            return None, None

    def analyze_music(self, audio_path):
        """Analyze music and extract chords with lyrics alignment."""
        y, sr = self.load_audio(audio_path)
        if y is None:
            return None
        print(f"Finished loading audio file, the duration of file is {librosa.get_duration(y=y, sr=sr)} seconds")

        # Get audio filename for saving lyrics
        audio_name = Path(audio_path).stem
        
        print("attempting to extract lyrics from audio...")
        vocals, vocals_sr = self.separate_vocals(audio_path)
        lyrics = None
        lyrics_data = None
        if vocals is not None:
            lyrics, lyrics_data = self.transcribe_vocals(vocals, vocals_sr, audio_name)

        print(f"Starting extraction of features from {audio_path}")
        features = self.extract_features(y, sr)
        
        # Use lyrics_data with timestamps directly in the prompt
        prompt = self._create_analysis_prompt(features, lyrics, lyrics_data, sr=sr)
        
        print(f"prompt before sending to Open AI:{prompt}")
        
        try:
            response_file = f"{audio_name}_openai_response.json"
            if os.path.exists(response_file):
                print(f"Found existing response file: {response_file}")
                with open(response_file, "r", encoding="utf-8") as f:
                    content = json.load(f)
                # Extract the content if needed
                return content
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a music analysis expert. Analyze the following musical data and return a valid JSON response as specified."},
                    {"role": "user", "content": prompt}
                ]
            )
            valid_content = response.choices[0].message.content
            print(f"raw valid_content: {valid_content}")
            m = re.search(r"\{.*\}", valid_content, re.DOTALL)
            if m:
                valid_content = m.group(0)   # "{ggwegwe}"
                print(f"valid_content: {valid_content}")
                analysis_content = json.loads(valid_content)
                with open(f"{audio_name}_openai_response.json", "w", encoding="utf-8") as f:
                    json.dump(analysis_content, f, indent=2, ensure_ascii=False) 
            else:
                raise Exception("failed to get response")
            print(f"Tokens used: {response.usage.total_tokens} (Prompt: {response.usage.prompt_tokens}, Completion: {response.usage.completion_tokens})")
            return json.loads(valid_content)
            
        except Exception as e:
            print(f"Error in OpenAI analysis: {e}")
            return None

    def _create_analysis_prompt(self, features, lyrics, lyrics_data=None, sr=22050, hop_length=512, max_duration=0):
        if max_duration == 0:
            max_duration = features['duration']
        chroma = features['chroma']
        mfcc = features['mfcc']
        tempo = features['tempo']
        beat_times = features.get('beat_times')
        time_per_frame = hop_length / sr

        # Estimate duration per musical phrase (2 bars of 4/4 time)
        beats_per_phrase = 4
        if beat_times is not None and len(beat_times) > beats_per_phrase:
            beat_durations = np.diff(beat_times)
            avg_beat_duration = np.mean(beat_durations[:beats_per_phrase])
            phrase_duration = avg_beat_duration * beats_per_phrase
        else:
            phrase_duration = 6.0  # fallback
        chunk_size = int(phrase_duration / time_per_frame)

        prompt = f"You are a music theory expert. Analyze this music based on the audio duration of {max_duration:.1f}s.\n\n"
        prompt += f"- Tempo: {tempo:.2f} BPM\n"
        prompt += f"- Each segment below represents ~{phrase_duration:.1f}s (approx. two bars of 4/4 time):\n\n"

        for i in range(0, chroma.shape[1], chunk_size):
            start_t = i * time_per_frame
            if start_t > max_duration:
                break
            end_t = min((i + chunk_size) * time_per_frame, max_duration)
            mean_chroma = np.mean(chroma[:, i:i+chunk_size], axis=1)
            mean_mfcc = np.mean(mfcc[:, i:i+chunk_size], axis=1)

            chroma_str = ", ".join([f"{note}:{val:.2f}" for note, val in zip(
                ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'], mean_chroma)])
            mfcc_str = ", ".join([f"{val:.2f}" for val in mean_mfcc[:8]])

            prompt += f"Segment ({start_t:.2f}s–{end_t:.2f}s):\n"
            prompt += f"Chroma: {chroma_str}\n"
            prompt += f"MFCC: {mfcc_str}\n\n"

        # Include time-stamped lyrics if available
        if lyrics_data:
            prompt += "Lyrics with timestamps:\n"
            for segment in lyrics_data:
                prompt += f"({segment['start_time']:.2f}s - {segment['end_time']:.2f}s): {segment['lyrics']}\n"
        else:
            prompt += f"\nFull lyrics:\n{lyrics.strip()}\n\n"

        prompt += """
    Instructions:
    1. Analyze musical key based on Chroma info provided and music style, this should be Bb key for sure!
    2. Assign chords to segments based on the features above.
    3. For each lyric segment, identify ALL chords that occur during that segment.
    4. When a single lyric line spans multiple chord changes, break it down into smaller parts with each chord.
    5. Long lyric lines may have multiple chord changes - identify each change point based on musical features.
    6. Identify any repeating chord patterns or structures.
    7. Since we provided full lyrics, please supplement chords for lyrics after provided segments time.
    
    IMPORTANT: A single line of lyrics often contains multiple chord changes. Divide longer lyric lines into sub-sections with appropriate chords for each part.

    Return in below JSON format, just pure json no invalid syntax words:
    {
    "key": "string",
    "mode": "string",
    "chord_progression": [
        {
        "chord": "string",
        "start_time": float,
        "end_time": float,
        "lyrics": "string"
        }
    ],
    "melody_analysis": "string",
    "patterns": ["string"]
    }

    """
        return prompt

    def save_analysis(self, analysis, output_path):
        """Save analysis results to a JSON file."""
        if analysis:
            with open(output_path, 'w') as f:
                json.dump(analysis, f, indent=4)
            print(f"Analysis saved to {output_path}")

    # output a PDF like formal guitar tab instead of json file
    # lyrics aligns with chord like normal guitar tab 
    def create_guitar_tab_pdf(self, data, pdf_path, song_title = None):
        """Create a PDF with chord and lyrics aligned similar to a guitar tab."""
        # Set up the PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Title
        pdf.set_font("Helvetica", "B", 16)
        title = song_title or data.get("title", "Guitar Tab")
        pdf.cell(0, 10, title, ln=True, align="C")
        pdf.ln(5)

        # Song info (key & mode)
        pdf.set_font("Helvetica", "", 12)
        key_mode = f"Key: {data['key']} {data['mode'].capitalize()}"
        pdf.cell(0, 8, key_mode, ln=True)
        pdf.ln(5)

        # Switch to monospaced for chord-lyric alignment
        pdf.set_font("Courier", "", 10)
        
        # Sort chord progression by start time to ensure correct ordering
        chord_progression = sorted(data["chord_progression"], key=lambda x: x["start_time"])
        
        # Group chords by lyric sections for display
        # We'll use a time-based approach to group chords that belong together
        
        # First, identify natural breaks in the chord progression
        # A new section starts when there's a significant time gap between chords
        # or when the lyrics change significantly
        sections = []
        current_section = []
        
        for i, chord in enumerate(chord_progression):
            if not current_section:
                current_section.append(chord)
                continue
                
            # Check if this is a new section based on time gap or lyric change
            prev_chord = current_section[-1]
            time_gap = chord["start_time"] - prev_chord["end_time"]
            
            # Start a new section if there's a significant time gap (e.g., > 1 second)
            # or if the current section has a lot of chords already
            if time_gap > 1.0 or len(current_section) >= 4:
                sections.append(current_section)
                current_section = [chord]
            else:
                current_section.append(chord)
                
        # Add the last section if it's not empty
        if current_section:
            sections.append(current_section)
            
        # Now render each section
        for section in sections:
            # Calculate time range for this section
            start_time = section[0]["start_time"]
            end_time = section[-1]["end_time"]
            time_range = f"[{start_time:.2f}s - {end_time:.2f}s]"
            
            # Print the time range
            pdf.set_font("Helvetica", "I", 8)
            pdf.cell(0, 4, time_range, ln=True)
            
            # Print chords above lyrics for this section
            pdf.set_font("Courier", "B", 10)  # Bold for chords
            
            max_chord_width = 8  # Minimum width for each chord
            for chord in section:
                chord_width = max(len(chord["chord"]), len(chord["lyrics"]) // 4)
                max_chord_width = max(max_chord_width, chord_width)
                
            # Print chord line
            chord_line = ""
            for chord in section:
                chord_line += chord["chord"].center(max_chord_width) + " "
            pdf.cell(0, 6, chord_line, ln=True)
            
            # Build lyrics line(s)
            pdf.set_font("Courier", "", 10)  # Regular for lyrics
            
            # Combine lyrics from the chords in this section
            full_lyrics = ""
            for chord in section:
                if chord["lyrics"] and not chord["lyrics"].isspace():
                    # Add spacing between lyrics if needed
                    if full_lyrics and not full_lyrics.endswith(" "):
                        full_lyrics += " "
                    full_lyrics += chord["lyrics"].strip()
            
            # Format lyrics to wrap within the PDF width
            # Use multi_cell to handle long lyrics with automatic wrapping
            pdf.multi_cell(0, 6, full_lyrics)
            
            pdf.ln(4)  # Space between sections
        
        # Save
        pdf.output(pdf_path)
        print(f"PDF written to {pdf_path}")

def main():

    analyzer = MusicAnalyzer()
    
    # Put MP3 audio file path
    audio_path = "" # input mp3 file, example: until_you.mp3
    
    # Analyze music
    analysis = analyzer.analyze_music(audio_path)  # Automatically extracts lyrics with timestamps
    if analysis:
        
        # Save as PDF chord sheet
        song_title = Path(audio_path).stem.replace('_', ' ').title()
        analyzer.create_guitar_tab_pdf(analysis, f"{song_title}_chord_sheet.pdf", song_title)
        
        # Print some key information
        print(f"\nKey: {analysis['key']}")
        print(f"Mode: {analysis['mode']}")
        print("\nChord Progression:")
        for chord in analysis['chord_progression']:
            print(f"{chord['chord']} ({chord['start_time']:.2f}s - {chord['end_time']:.2f}s): {chord.get('lyrics', '')}")

if __name__ == "__main__":
    main() 