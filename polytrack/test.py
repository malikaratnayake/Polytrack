import pandas as pd
import math

def is_point_inside_circle(x0, y0, cx, cy, radius):
    # Calculate the distance between the point (x0, y0) and the circle's center (cx, cy)
    distance_to_center = math.sqrt((x0 - cx)**2 + (y0 - cy)**2)
    
    # Check if the distance to the center is less than the radius
    if distance_to_center < radius:
        return True  # The point is inside the circle
    else:
        return False  # The point is outside the circle

def check_insect_flower_relationship(insect_tracks, flower_tracks, N):
    # Get the last N frames from insect_tracks and flower_tracks
    last_insect_frames = insect_tracks[insect_tracks['nframe'].isin(range(max(insect_tracks['nframe']) - N + 1, max(insect_tracks['nframe']) + 1))]
    last_flower_frames = flower_tracks[flower_tracks['nframe'].isin(range(max(flower_tracks['nframe']) - N + 1, max(flower_tracks['nframe']) + 1))]
    
    # Initialize a list to store flower numbers for each insect (default to None)
    flower_numbers = [None] * len(last_insect_frames)
    
    # Iterate over each row in the last_insect_frames dataframe
    for idx, insect_row in last_insect_frames.iterrows():
        x0 = insect_row['x0']
        y0 = insect_row['y0']
        
        # Iterate over each row in the last_flower_frames dataframe
        for flower_idx, flower_row in last_flower_frames.iterrows():
            cx = flower_row['cx']
            cy = flower_row['cy']
            radius = flower_row['radius']
            flower_num = flower_row['flower_num']
            
            # Check if the center point of the insect is inside the flower's circle
            if is_point_inside_circle(x0, y0, cx, cy, radius):
                if flower_numbers[idx] is None:
                    flower_numbers[idx] = []
                flower_numbers[idx].append(flower_num)
    
    # Add a new column 'flower_num' to the last_insect_frames dataframe
    last_insect_frames['flower_num'] = flower_numbers
    
    return last_insect_frames

# Example usage:
# Assuming insect_tracks and flower_tracks are pandas dataframes containing the specified columns

# Generate example dataframes (replace this with your actual data)
insect_tracks = pd.DataFrame({
    'nframe': [1, 1, 2, 2, 3, 3],
    'insect_num': [1, 2, 3, 4, 5, 6],
    'x0': [10, 15, 20, 25, 30, 35],
    'y0': [20, 25, 30, 35, 40, 45]
})

flower_tracks = pd.DataFrame({
    'nframe': [1, 1, 2, 2, 3, 3],
    'flower_num': [101, 102, 103, 104, 105, 106],
    'cx': [12, 22, 32, 42, 52, 62],
    'cy': [22, 32, 42, 52, 62, 72],
    'radius': [5, 8, 7, 10, 6, 9]
})

# Specify the number of frames (N) to consider
N = 3

# Check insect-flower relationship for the last N frames
result_df = check_insect_flower_relationship(insect_tracks, flower_tracks, N)

# Display the updated insect_tracks dataframe with the 'flower_num' column
print(result_df)
