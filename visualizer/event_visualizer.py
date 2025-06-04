import cv2
import numpy as np
from typing import List, Dict, Any
from event_detector.events import Event, PassEvent, PressureEvent, PossessionChangeEvent

class EventVisualizer:
    def __init__(self):
        self.active_pass_events = []  # Track ongoing pass events
        self.banner_height = 40
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.line_thickness = 2
        
    def _draw_banner(self, frame: np.ndarray, text: str, color: tuple, y_offset: int = 0) -> np.ndarray:
        """Draw a semi-transparent banner with text at the top of the frame."""
        h, w = frame.shape[:2]
        banner = frame.copy()
        
        # Draw semi-transparent banner background
        cv2.rectangle(banner, (0, y_offset), (w, y_offset + self.banner_height), color, -1)
        
        # Add text
        text_size = cv2.getTextSize(text, self.font, self.font_scale, self.line_thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = y_offset + (self.banner_height + text_size[1]) // 2
        cv2.putText(banner, text, (text_x, text_y), self.font, self.font_scale, (255, 255, 255), self.line_thickness)
        
        # Blend banner with original frame
        alpha = 0.7
        frame = cv2.addWeighted(banner, alpha, frame, 1 - alpha, 0)
        return frame
        
    def _draw_pressure_indicator(self, frame: np.ndarray, event: PressureEvent, tracks: Dict) -> np.ndarray:
        """Draw pressure indicators between players."""
        pressuring_player = tracks["players"][event.start_frame].get(event.pressuring_player_id)
        pressured_player = tracks["players"][event.start_frame].get(event.pressured_player_id)
        
        if pressuring_player and pressured_player:
            p1 = pressuring_player.get('position_pixel')
            p2 = pressured_player.get('position_pixel')
            
            if p1 and p2:
                # Draw line between players
                cv2.line(frame, tuple(map(int, p1)), tuple(map(int, p2)), (0, 0, 255), 2)
                
                # Draw pressure intensity indicator
                mid_point = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                radius = int(20 * event.pressure_intensity)
                cv2.circle(frame, mid_point, radius, (0, 0, 255), 2)
                
        return frame
        
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
                frame = self._draw_banner(
                    frame,
                    pass_event.description,
                    (0, 128, 0),  # Green color for passes
                    banner_offset
                )
                banner_offset += self.banner_height
        
        # Draw pressure events
        for event in events:
            if isinstance(event, PressureEvent):
                frame = self._draw_banner(
                    frame,
                    event.description,
                    (255, 0, 0),  # Red color for pressure
                    banner_offset
                )
                frame = self._draw_pressure_indicator(frame, event, tracks)
                banner_offset += self.banner_height
            elif isinstance(event, PossessionChangeEvent):
                frame = self._draw_banner(
                    frame,
                    event.description,
                    (0, 0, 255),  # Blue color for possession changes
                    banner_offset
                )
                banner_offset += self.banner_height
        
        return frame 