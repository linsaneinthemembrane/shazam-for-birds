# bird_song_id/main.py

import os
import sys
import argparse
from audio import AudioProcessor
from database import FingerprintDatabase
from api import eBirdAPI

class BirdSongIdentifier:
    def __init__(self, db_path='bird_fingerprints.json', api_key=None):
        """Initialize the bird song identification system"""
        self.audio_processor = AudioProcessor()
        self.database = FingerprintDatabase(db_path)
        self.ebird_api = eBirdAPI(api_key)
        
        # Ensure required directories exist
        os.makedirs('recordings', exist_ok=True)
        
    def identify_bird_song(self, audio_file, plot=True):
        """
        Identify a bird from an audio recording
        
        Parameters:
        - audio_file: Path to the audio file
        - plot: Whether to visualize the process
        
        Returns:
        - List of potential matches with confidence scores
        """
        print(f"\nAnalyzing audio file: {audio_file}")
        
        # Process the audio to extract fingerprint
        self.audio_processor.load_audio(audio_file)
        frequencies, times, spectrogram = self.audio_processor.create_spectrogram()
        
        # Extract distinctive peaks
        peaks = self.audio_processor.extract_peaks(spectrogram, frequencies, times)
        
        # Create fingerprint from peaks
        fingerprint = self.audio_processor.create_fingerprint_from_peaks(peaks, frequencies, times)
        
        # Visualize if requested
        if plot:
            self.audio_processor.visualize_peaks(frequencies, times, spectrogram, peaks)
        
        # Match against database
        matches = self.database.match_fingerprint(fingerprint)
        
        return matches
    
    def add_to_database(self, audio_file, species_name, bird_id=None, metadata=None, plot=True):
        """
        Add a bird song to the database
        
        Parameters:
        - audio_file: Path to the audio file
        - species_name: Name of the bird species
        - bird_id: Optional identifier (generated if None)
        - metadata: Additional information about the recording
        - plot: Whether to visualize the spectrogram
        """
        if bird_id is None:
            # Generate ID based on species and timestamp
            bird_id = f"{species_name.replace(' ', '_').lower()}_{int(time.time())}"
            
        if metadata is None:
            metadata = {}
            
        print(f"\nProcessing {species_name} recording: {audio_file}")
        
        # Process the audio
        self.audio_processor.load_audio(audio_file)
        frequencies, times, spectrogram = self.audio_processor.create_spectrogram()
        
        # Extract peaks
        peaks = self.audio_processor.extract_peaks(spectrogram, frequencies, times)
        
        # Create fingerprint
        fingerprint = self.audio_processor.create_fingerprint_from_peaks(peaks, frequencies, times)
        
        # Visualize if requested
        if plot:
            self.audio_processor.visualize_peaks(frequencies, times, spectrogram, peaks)
        
        # Add to database
        metadata['source'] = 'ebird'  # Mark as from eBird
        self.database.add_fingerprint(bird_id, species_name, fingerprint, metadata)
        
        return bird_id
    
    def get_bird_info(self, species_name):
        """Get information about a bird species using the eBird API"""
        results = self.ebird_api.search_species(species_name)
        
        if not results:
            print(f"No information found for {species_name}")
            return None
            
        # Use the first result
        bird_info = results[0]
        print(f"Found information for {bird_info['name']} ({bird_info['scientific']})")
        
        # In a real application, you'd fetch more details using the species code
        return bird_info

def main():
    """Main function to demonstrate the bird song identification system"""
    parser = argparse.ArgumentParser(description='Bird Song Identification System')
    
    # Define command line arguments
    parser.add_argument('command', choices=['identify', 'add', 'info'],
                        help='Command to execute: identify a recording, add to database, or get bird info')
    parser.add_argument('--file', help='Path to audio file')
    parser.add_argument('--species', help='Bird species name (for add command)')
    parser.add_argument('--noplot', action='store_true', help='Disable visualization')
    parser.add_argument('--db', default='bird_fingerprints.json', help='Database file path')
    parser.add_argument('--api-key', help='eBird API key')
    
    args = parser.parse_args()
    
    # Initialize the system
    identifier = BirdSongIdentifier(db_path=args.db, api_key=args.api_key)
    
    if args.command == 'identify':
        if not args.file:
            print("Error: --file argument is required for identify command")
            return
            
        # Identify the bird song
        matches = identifier.identify_bird_song(args.file, plot=not args.noplot)
        
        # Display results
        if matches:
            print("\nIdentification Results:")
            print("======================")
            for i, match in enumerate(matches):
                print(f"{i+1}. Species: {match['species']}")
                print(f"   Confidence: {match['confidence']:.2f}")
                print(f"   Matching points: {match['match_count']}")
                print(f"   Time alignment: {match['time_offset']:.2f} seconds")
                print("")
                
                # For the top match, try to get additional info
                if i == 0:
                    bird_info = identifier.get_bird_info(match['species'])
                    if bird_info:
                        print(f"Additional information for {match['species']}:")
                        for key, value in bird_info.items():
                            print(f"   {key}: {value}")
        else:
            print("\nNo matches found in the database.")
            
    elif args.command == 'add':
        if not args.file or not args.species:
            print("Error: Both --file and --species arguments are required for add command")
            return
            
        # Add to database
        bird_id = identifier.add_to_database(args.file, args.species, plot=not args.noplot)
        print(f"\nSuccessfully added {args.species} to database with ID: {bird_id}")
        
    elif args.command == 'info':
        if not args.species:
            print("Error: --species argument is required for info command")
            return
            
        # Get bird information
        bird_info = identifier.get_bird_info(args.species)
        if bird_info:
            print(f"\nInformation for {args.species}:")
            for key, value in bird_info.items():
                print(f"   {key}: {value}")
        else:
            print(f"\nNo information found for {args.species}")

if __name__ == "__main__":
    main()
