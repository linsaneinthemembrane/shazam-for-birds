# bird_song_id/api.py
import requests
import json

class BirdAPI:
    def __init__(self):
        """Initialize the API connector"""
        self.ebird_api_base = "https://api.ebird.org/v2"
        # You would need to register for an eBird API key
        self.api_key = '656nbprpa5pj'  
    
    def set_api_key(self, key):
        """Set the API key for eBird"""
        self.api_key = key
    
    def get_bird_info(self, species_code):
        """
        Get information about a bird species from eBird
        
        Parameters:
        - species_code: The eBird species code
        
        Returns:
        - Dictionary with bird information or None if not found
        """
        if not self.api_key:
            raise ValueError("API key not set. Use set_api_key() first.")
        
        # Endpoint for species information
        url = f"{self.ebird_api_base}/data/obs/{species_code}"
        
        headers = {
            "X-eBirdApiToken": self.api_key
        }
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise exception for HTTP errors
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return None
    
    def search_species(self, query):
        """
        Search for bird species based on a text query
        This is a simulated function as eBird doesn't offer this exact endpoint
        """
        # In a real implementation, you would call the appropriate API endpoint
        # For now, we'll return some mock data
        mock_results = {
            "robin": {"code": "amerob", "name": "American Robin", "scientific": "Turdus migratorius"},
            "cardinal": {"code": "norcar", "name": "Northern Cardinal", "scientific": "Cardinalis cardinalis"},
            "sparrow": {"code": "houspa", "name": "House Sparrow", "scientific": "Passer domesticus"}
        }
        
        # Simple partial matching
        results = []
        query = query.lower()
        for key, data in mock_results.items():
            if query in key or query in data["name"].lower():
                results.append(data)
        
        return results
    
    def get_recent_observations(self, location_code):
        """
        Get recent bird observations for a location
        
        Parameters:
        - location_code: eBird location code
        
        Returns:
        - List of recent observations
        """
        if not self.api_key:
            raise ValueError("API key not set. Use set_api_key() first.")
        
        url = f"{self.ebird_api_base}/data/obs/recent/{location_code}"
        
        headers = {
            "X-eBirdApiToken": self.api_key
        }
        
        params = {
            "back": 7  # Get observations from the last 7 days
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return []
