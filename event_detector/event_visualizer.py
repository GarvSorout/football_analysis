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
        
    def draw_pass_event(self, frame: np.ndarray, event: PassEvent, tracks: Dict):
        """Visualize a pass event with arrow and banner."""
        # Draw banner first (this should always show)
        banner_text = f"PASS IN PROGRESS: Player {event.passer_id} → Player {event.receiver_id}"
        self.draw_event_banner(frame, banner_text, (0, 255, 0))
        
        # Try to draw arrow if both players are visible
        try:
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
        except (KeyError, IndexError):
            # If either player is not in tracking data, just show the banner without the arrow
            pass
        
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
        
        # Draw field zone highlight
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        if event.field_zone == "midfield":
            cv2.rectangle(overlay, (w//3, 0), (2*w//3, h), (255, 165, 0), -1)
        elif event.field_zone == "defensive_third":
            cv2.rectangle(overlay, (0, 0), (w//3, h), (255, 165, 0), -1)
        else:  # final_third
            cv2.rectangle(overlay, (2*w//3, 0), (w, h), (255, 165, 0), -1)
            
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
    def draw_events(self, frame: np.ndarray, events: List[Event], frame_num: int, tracks: Dict) -> np.ndarray:
        """Draw all events on the frame."""
        # Update active pass events
        self.active_pass_events = [e for e in self.active_pass_events if e.end_frame >= frame_num]
        
        # Add new pass events
        for event in events:
            if isinstance(event, PassEvent):
                self.active_pass_events.append(event)
        
        # Draw pass events
        for pass_event in self.active_pass_events:
            progress = (frame_num - pass_event.start_frame) / (pass_event.end_frame - pass_event.start_frame)
            if 0 <= progress <= 1:
                self.draw_pass_event(frame, pass_event, tracks)
        
        # Draw other events
        for event in events:
            if isinstance(event, PressureEvent):
                self.draw_pressure_event(frame, event, tracks)
            elif isinstance(event, PossessionChangeEvent):
                self.draw_possession_change(frame, event)
        
        return frame 