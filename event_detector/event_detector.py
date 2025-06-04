import numpy as np
from typing import List, Dict, Any, Tuple
from .events import Event, PassEvent, PressureEvent, PossessionChangeEvent
from utils import measure_distance

class EventDetector:
    def __init__(self):
        self.pressure_distance_threshold = 2.0  # meters
        self.pressure_speed_threshold = 5.0  # m/s
        self.last_possession_team = None  # Track last team with possession
        self.last_player_with_ball = None  # Track last player with ball
        self.pass_start_frame = None  # Track when current pass started
        
    def _get_field_zone(self, position: Tuple[float, float]) -> str:
        """Determine the zone of the field based on coordinates."""
        x, y = position
        if x < 23.32 * 0.33:  # First third
            return "defensive_third"
        elif x < 23.32 * 0.66:  # Middle third
            return "midfield"
        else:
            return "final_third"
            
    def _detect_passes(self, tracks: Dict, frame_num: int, team_ball_control: np.ndarray, player_ball_control: np.ndarray) -> List[PassEvent]:
        """Detect passing events using player control changes."""
        passes = []
        
        if frame_num > 0:  # Skip first frame as we need previous state
            # Check if player control changed
            if player_ball_control[frame_num] != player_ball_control[frame_num - 1]:
                old_player = player_ball_control[frame_num - 1]
                new_player = player_ball_control[frame_num]
                
                # Skip if either player is not in tracking data
                if old_player not in tracks['players'][frame_num] or new_player not in tracks['players'][frame_num]:
                    return passes
                
                # Get team information for both players
                old_player_team = tracks['players'][frame_num][old_player]['team']
                new_player_team = tracks['players'][frame_num][new_player]['team']
                
                # If same team, it's a pass
                if old_player_team == new_player_team:
                    pass_event = PassEvent(
                        start_frame=frame_num - 1,  # Pass started in previous frame
                        end_frame=frame_num,        # And completed in current frame
                        description=f"Pass from Player {old_player} to Player {new_player}",
                        players_involved=[old_player, new_player],
                        teams_involved=[old_player_team],
                        field_zone=self._get_field_zone((0, 0)),  # Default to midfield
                        confidence=0.95,
                        passer_id=old_player,
                        receiver_id=new_player,
                        pass_type="pass",  # Simplified pass type
                        pass_success=True
                    )
                    passes.append(pass_event)
        
        return passes
        
    def _detect_pressure(self, tracks: Dict, frame_num: int) -> List[PressureEvent]:
        """Detect pressing situations based on player proximity and movement."""
        pressure_events = []
        
        for player_id, player in tracks["players"][frame_num].items():
            if not player.get('has_ball', False):
                continue
                
            player_pos = player.get('position_transformed')
            if not player_pos:
                continue
                
            player_team = player['team']
            
            # Find closest opponents
            for other_id, other in tracks["players"][frame_num].items():
                if other['team'] == player_team:
                    continue
                    
                other_pos = other.get('position_transformed')
                if not other_pos:
                    continue
                    
                distance = measure_distance(player_pos, other_pos)
                if distance < self.pressure_distance_threshold:
                    # Check if opponent is moving towards the player
                    if 'speed' in other and other['speed'] > self.pressure_speed_threshold:
                        pressure_event = PressureEvent(
                            start_frame=frame_num,
                            end_frame=frame_num,
                            description=f"Player {other_id} pressing Player {player_id}",
                            players_involved=[player_id, other_id],
                            teams_involved=[player_team, other['team']],
                            field_zone=self._get_field_zone(player_pos),
                            confidence=0.9,
                            pressuring_player_id=other_id,
                            pressured_player_id=player_id,
                            pressure_intensity=1.0 - (distance / self.pressure_distance_threshold)
                        )
                        pressure_events.append(pressure_event)
                        
        return pressure_events

    def _detect_possession_changes(self, tracks: Dict, frame_num: int, team_ball_control: np.ndarray) -> List[PossessionChangeEvent]:
        """Detect possession changes using the existing team_ball_control array."""
        possession_changes = []
        
        # Get current team with possession
        current_team = team_ball_control[frame_num]
        
        # Check for possession change
        if self.last_possession_team is not None and current_team != self.last_possession_team:
            # Find the player with the ball
            ball_pos = None
            for player_id, player in tracks["players"][frame_num].items():
                if player.get('has_ball', False):
                    ball_pos = player.get('position_transformed')
                    break
            
            # If no player has the ball, try to get ball position directly
            if not ball_pos and 1 in tracks["ball"][frame_num]:
                ball_pos = tracks["ball"][frame_num][1].get('position_transformed')
            
            # Default to midfield if we can't determine position
            field_zone = self._get_field_zone(ball_pos) if ball_pos else "midfield"
            
            possession_change = PossessionChangeEvent(
                start_frame=frame_num,
                end_frame=frame_num,
                description=f"Possession change from Team {self.last_possession_team} to Team {current_team}",
                players_involved=[],  # Could add involved players if needed
                teams_involved=[self.last_possession_team, current_team],
                field_zone=field_zone,
                confidence=0.95,
                previous_team=self.last_possession_team,
                new_team=current_team,
                cause="interception",  # Could be more specific with additional analysis
                field_position=ball_pos if ball_pos else (0, 0)
            )
            possession_changes.append(possession_change)
        
        # Update last possession team
        self.last_possession_team = current_team
        
        return possession_changes
        
    def detect_events(self, tracks: Dict, frame_num: int, team_ball_control: np.ndarray = None, player_ball_control: np.ndarray = None) -> List[Event]:
        """Main method to detect all types of events in the current frame."""
        events = []
        
        if frame_num > 0 and player_ball_control is not None:  # Skip first frame as we need previous state
            # Detect passes using player control array
            passes = self._detect_passes(tracks, frame_num, team_ball_control, player_ball_control)
            events.extend(passes)
            # Log passes
            for pass_event in passes:
                print(f"[Frame {frame_num}] PASS: {pass_event.description} ({pass_event.pass_type} pass in {pass_event.field_zone})")
        
        # Detect pressure situations
        pressure = self._detect_pressure(tracks, frame_num)
        events.extend(pressure)
        # Log pressure events
        for pressure_event in pressure:
            print(f"[Frame {frame_num}] PRESSURE: {pressure_event.description} (Intensity: {pressure_event.pressure_intensity:.2f})")

        # Detect possession changes if team_ball_control is provided
        if team_ball_control is not None:
            possession_changes = self._detect_possession_changes(tracks, frame_num, team_ball_control)
            events.extend(possession_changes)
            # Log possession changes
            for possession_event in possession_changes:
                print(f"[Frame {frame_num}] POSSESSION CHANGE: {possession_event.description} (Zone: {possession_event.field_zone})")
        
        return events 