import faiss
import numpy as np
from typing import List, Dict, Any
from .clip_encoder import ClipEncoder
from event_detector.events import Event

class VideoIndexer:
    def __init__(self, embedding_dim: int = 512):
        self.clip_encoder = ClipEncoder()
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.events: List[Dict] = []  # Store event metadata
        
    def add_event(self, event: Event, frames: List[np.ndarray]):
        """Add an event and its associated frames to the index."""
        event_dict = {
            "description": event.description,
            "frames": frames,
            "start_frame": event.start_frame,
            "end_frame": event.end_frame,
            "field_zone": event.field_zone,
            "players_involved": event.players_involved,
            "teams_involved": event.teams_involved
        }
        
        # Get embedding for the event
        embedding = self.clip_encoder.encode_event(event_dict)
        
        # Add to FAISS index
        self.index.add(embedding)
        
        # Store event metadata
        self.events.append(event_dict)
        
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for events matching the query text."""
        # Encode query text
        query_embedding = self.clip_encoder.encode_text([query])
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, k)
        
        # Return matched events with their distances
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.events):  # Valid index
                result = self.events[idx].copy()
                result["similarity_score"] = 1.0 / (1.0 + distance)  # Convert distance to similarity score
                results.append(result)
                
        return results
        
    def search_by_zone(self, zone: str, k: int = 5) -> List[Dict]:
        """Search for events in a specific field zone."""
        matching_events = []
        for event in self.events:
            if event["field_zone"] == zone:
                matching_events.append(event)
                
        # Sort by time (frame number)
        matching_events.sort(key=lambda x: x["start_frame"])
        return matching_events[:k]
        
    def search_by_player(self, player_id: int, k: int = 5) -> List[Dict]:
        """Search for events involving a specific player."""
        matching_events = []
        for event in self.events:
            if player_id in event["players_involved"]:
                matching_events.append(event)
                
        # Sort by time (frame number)
        matching_events.sort(key=lambda x: x["start_frame"])
        return matching_events[:k] 