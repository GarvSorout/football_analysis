 # Decompiled with PyLingual (https://pylingual.io)
# Internal filename: /Users/garvsorout/football_analysis/event_detector/events.py
# Bytecode version: 3.9.0beta5 (3425)
# Source timestamp: 2025-06-04 01:22:39 UTC (1749000159)

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np

@dataclass
class Event:
    start_frame: int
    end_frame: int
    description: str
    players_involved: List[int]
    teams_involved: List[int]
    field_zone: str
    confidence: float

@dataclass
class PassEvent(Event):
    passer_id: int
    receiver_id: Optional[int]
    pass_type: str
    pass_success: bool

@dataclass
class PressureEvent(Event):
    pressuring_player_id: int
    pressured_player_id: int
    pressure_intensity: float

@dataclass
class PossessionChangeEvent(Event):
    previous_team: int
    new_team: int
    cause: str
    field_position: Tuple[float, float]