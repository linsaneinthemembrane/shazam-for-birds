# bird_song_id/database.py
import json
import os

class Database:
    def __init__(self, db_file='bird_songs.json'):
        """Initialize the database with a file to store bird song data"""
        self.db_file = db_file
        self.bird_songs = {}
        self.load_database()
    
    def load_database(self):
        """Load the database from file if it exists"""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r') as f:
                    self.bird_songs = json.load(f)
            except json.JSONDecodeError:
                self.bird_songs = {}
    
    def save_database(self):
        """Save the database to file"""
        with open(self.db_file, 'w') as f:
            json.dump(self.bird_songs, f, indent=2)
    
    def add_bird_song(self, bird_id, species_name, hashes, metadata=None):
        """
        Add a bird song to the database
        
        Parameters:
        - bird_id: Unique identifier for the bird song
        - species_name: Name of the bird species
        - hashes: List of hash values representing the fingerprint
        - metadata: Additional information about the recording
        """
        if metadata is None:
            metadata = {}
            
        # Convert hash list to strings for JSON serialization
        str_hashes = [str(h) for h in hashes]
        
        self.bird_songs[bird_id] = {
            'species': species_name,
            'hashes': str_hashes,
            'metadata': metadata
        }
        self.save_database()
        
    def match_fingerprint(self, query_hashes, threshold=0.3):
        """
        Find matches for a query fingerprint in the database
        
        Parameters:
        - query_hashes: List of hash values from the query audio
        - threshold: Minimum similarity score to consider a match
        
        Returns:
        - List of (bird_id, species_name, similarity_score) tuples
        """
        str_query_hashes = set(str(h) for h in query_hashes)
        matches = []
        
        for bird_id, bird_data in self.bird_songs.items():
            # Convert stored hashes back to a set
            db_hashes = set(bird_data['hashes'])
            
            # Calculate Jaccard similarity
            intersection = len(str_query_hashes.intersection(db_hashes))
            union = len(str_query_hashes.union(db_hashes))
            similarity = intersection / union if union > 0 else 0
            
            if similarity >= threshold:
                matches.append((bird_id, bird_data['species'], similarity))
        
        # Sort by similarity score (highest first)
        return sorted(matches, key=lambda x: x[2], reverse=True)
