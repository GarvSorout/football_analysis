from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from event_detector import EventDetector
from event_detector.event_visualizer import EventVisualizer


def main():
    # Read Video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    
    # Assign Ball Aquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    player_ball_control = []  # New array to track which player has the ball
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        # Only assign control if player exists in tracking data
        if assigned_player != -1 and assigned_player in tracks['players'][frame_num]:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
            player_ball_control.append(assigned_player)
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)
            player_ball_control.append(player_ball_control[-1] if player_ball_control else -1)
    
    team_ball_control = np.array(team_ball_control)
    player_ball_control = np.array(player_ball_control)
    
    print("\nPlayer control sequence:")
    print(player_ball_control)  # Print the entire array at the end
    
    # Initialize event detector and video indexer
    event_detector = EventDetector()
    event_visualizer = EventVisualizer()

    # Process events frame by frame
    print("Detecting and indexing events...")
    all_events = []  # Store events for each frame
    
    # First pass: Detect all events
    for frame_num in range(len(video_frames)):
        events = event_detector.detect_events(tracks, frame_num, team_ball_control, player_ball_control)
        all_events.append(events)

    # Second pass: Create annotated frames
    # First get base annotations from tracker
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Then add camera movement and speed/distance
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)
    
    # Finally add event visualizations frame by frame
    for frame_num in range(len(output_video_frames)):
        events = all_events[frame_num]
        frame = output_video_frames[frame_num]
        frame = event_visualizer.draw_events(frame, events, frame_num, tracks)
        output_video_frames[frame_num] = frame

    # Save video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()