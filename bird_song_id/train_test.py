# bird_song_id/train_test.py

import os
import random
import argparse
from audio import AudioProcessor
from database import FingerprintDatabase

def evaluate_model(db_path, test_dir, output_file=None):
    """
    Evaluate the bird song identification system with test recordings
    
    Parameters:
    - db_path: Path to the database file
    - test_dir: Directory containing test recordings
    - output_file: Path to save the evaluation results
    
    Returns:
    - Dictionary with evaluation results
    """
    # Initialize components
    audio_processor = AudioProcessor()
    database = FingerprintDatabase(db_path)
    
    # Find all audio files in the test directory
    test_files = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith(('.mp3', '.wav')):
                test_files.append(os.path.join(root, file))
    
    print(f"Found {len(test_files)} test files")
    
    # Evaluate each test file
    results = []
    correct = 0
    
    for file_path in test_files:
        # Extract species name from directory path
        species_dir = os.path.basename(os.path.dirname(file_path))
        
        print(f"\nTesting: {file_path}")
        print(f"Expected species: {species_dir}")
        
        try:
            # Process the audio file
            fingerprint = audio_processor.process_audio_file(file_path, plot=False)
            
            # Match against the database
            matches = database.match_fingerprint(fingerprint)
            
            # Determine if the prediction is correct
            is_correct = False
            predicted_species = "Unknown"
            
            if matches:
                top_match = matches[0]
                predicted_species = top_match['species']
                
                # Check if the prediction matches the expected species
                if species_dir in predicted_species or predicted_species in species_dir:
                    is_correct = True
                    correct += 1
            
            # Record the result
            result = {
                'file_path': file_path,
                'expected': species_dir,
                'predicted': predicted_species,
                'correct': is_correct,
                'matches': matches[:3] if matches else []  # Store top 3 matches
            }
            
            results.append(result)
            
            # Print the result
            print(f"Predicted: {predicted_species}")
            print(f"Correct: {is_correct}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Calculate accuracy
    accuracy = correct / len(test_files) if test_files else 0
    print(f"\nOverall accuracy: {accuracy:.2%} ({correct}/{len(test_files)})")
    
    # Save results to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(f"Accuracy: {accuracy:.2%} ({correct}/{len(test_files)})\n\n")
            
            for result in results:
                f.write(f"File: {result['file_path']}\n")
                f.write(f"Expected: {result['expected']}\n")
                f.write(f"Predicted: {result['predicted']}\n")
                f.write(f"Correct: {result['correct']}\n")
                
                if result['matches']:
                    f.write("Top matches:\n")
                    for match in result['matches']:
                        f.write(f"- {match['species']} (confidence: {match['confidence']:.2f})\n")
                
                f.write("\n")
        
        print(f"Evaluation results saved to {output_file}")
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': len(test_files),
        'results': results
    }

def main():
    parser = argparse.ArgumentParser(description='Train and test bird song identification system')
    
    # Define command line arguments
    parser.add_argument('--db', default='bird_fingerprints.json', help='Database file path')
    parser.add_argument('--test-dir', default='test_recordings', help='Test directory')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--output', default='evaluation_results.txt', help='Evaluation results file')
    
    args = parser.parse_args()
    
    # Evaluate model if requested
    if args.evaluate:
        evaluate_model(args.db, args.test_dir, args.output)

if __name__ == "__main__":
    main()
