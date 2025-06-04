import cv2
import numpy as np
from typing import Dict, List
from .events import Event, PassEvent, PressureEvent, PossessionChangeEvent

class EventVisualizer:
    def __init__(self):
        self.banner_height = 120
        self.banner_font = cv2.FONT_HERSHEY_DUPLEX
        self.banner_font_scale = 1.5
        self.banner_thickness = 3
        self.arrow_thickness = 3
        self.active_pass_events = []  # Initialize list to track active pass events
        
    def draw_event_banner(self, frame: np.ndarray, text: str, color: tuple):
        """Draw a semi-transparent banner with text at the top of the frame."""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay for the banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, self.banner_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add text
        text_size = cv2.getTextSize(text, self.banner_font, self.banner_font_scale, self.banner_thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (self.banner_height + text_size[1]) // 2
        
        # Draw text with contrasting border
        cv2.putText(frame, text, (text_x, text_y), self.banner_font,
                    self.banner_font_scale, (0, 0, 0), self.banner_thickness + 2)
        cv2.putText(frame, text, (text_x, text_y), self.banner_font,
                    self.banner_font_scale, color, self.banner_thickness)
        
        return frame
        
    def draw_pass_event(self, frame: np.ndarray, event: PassEvent, tracks: Dict):
        """Visualize a pass event with arrow and banner."""
        # Get player positions
        passer = tracks["players"][event.start_frame][event.passer_id]
        if event.receiver_id and event.receiver_id in tracks["players"][event.end_frame]:
            receiver = tracks["players"][event.end_frame][event.receiver_id]
            
            # Draw arrow between players
            start_pos = tuple(map(int, passer['position']))
            end_pos = tuple(map(int, receiver['position']))
            
            # Calculate arrow properties
            angle = np.arctan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
            arrow_length = 30
            arrow_points = np.array([
                [arrow_length, 0],
                [0, arrow_length//2],
                [0, -arrow_length//2]
            ])
            
            # Rotate arrow points
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            arrow_points = np.dot(arrow_points, rotation_matrix.T)
            arrow_points += end_pos
            
            # Draw the arrow
            cv2.arrowedLine(frame, start_pos, end_pos, (0, 255, 0), self.arrow_thickness)
            
        # Draw banner
        banner_text = f"PASS IN PROGRESS: Player {event.passer_id} → Player {event.receiver_id}"
        self.draw_event_banner(frame, banner_text, (0, 255, 0))
        
    def draw_pressure_event(self, frame: np.ndarray, event: PressureEvent, tracks: Dict):
        """Visualize a pressure event with intensity indicator."""
        # Get player positions
        pressured = tracks["players"][event.start_frame][event.pressured_player_id]
        pressuring = tracks["players"][event.start_frame][event.pressuring_player_id]
        
        # Draw pressure line
        start_pos = tuple(map(int, pressuring['position']))
        end_pos = tuple(map(int, pressured['position']))
        
        # Draw pulsing circle around pressured player
        radius = int(40 + 10 * np.sin(cv2.getTickCount() * 0.0001))  # Pulsing effect
        cv2.circle(frame, end_pos, radius, (0, 0, 255), 2)
        
        # Draw pressure line with intensity-based color
        intensity_color = (0, int(255 * (1 - event.pressure_intensity)), int(255 * event.pressure_intensity))
        cv2.line(frame, start_pos, end_pos, intensity_color, self.arrow_thickness)
        
        # Draw banner
        banner_text = f"PRESSURE: Player {event.pressuring_player_id} on Player {event.pressured_player_id}"
        self.draw_event_banner(frame, banner_text, (0, 0, 255))
        
    def draw_possession_change(self, frame: np.ndarray, event: PossessionChangeEvent):
        """Visualize a possession change event."""
        banner_text = f"POSSESSION CHANGE: Team {event.previous_team} → Team {event.new_team}"
        self.draw_event_banner(frame, banner_text, (255, 165, 0))
        
    def draw_events(self, frame: np.ndarray, events: List[Event], frame_num: int, tracks: Dict) -> np.ndarray:
        """Draw all events on the frame."""
        # Update active pass events
        self.active_pass_events = [e for e in self.active_pass_events if e.end_frame >= frame_num]
        
        # Add new pass events
        for event in events:
            if isinstance(event, PassEvent):
                self.active_pass_events.append(event)
        
        # Draw pass banners
        banner_offset = 0
        for pass_event in self.active_pass_events:
            progress = (frame_num - pass_event.start_frame) / (pass_event.end_frame - pass_event.start_frame)
            if 0 <= progress <= 1:
                frame = self.draw_event_banner(
                    frame,
                    pass_event.description,
                    (0, 128, 0)  # Green color for passes
                )
        
        # Draw pressure events
        for event in events:
            if isinstance(event, PressureEvent):
                frame = self.draw_event_banner(
                    frame,
                    event.description,
                    (255, 0, 0)  # Red color for pressure
                )
            elif isinstance(event, PossessionChangeEvent):
                frame = self.draw_event_banner(
                    frame,
                    event.description,
                    (0, 0, 255)  # Blue color for possession changes
                )
        
        return frame 