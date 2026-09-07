import os
import shutil
import glob
import argparse

parser = argparse.ArgumentParser(description="Fix downloaded track files")
parser.add_argument("--maps-dir", default=os.path.join("f1tenth_gym", "maps"), help="Path to the downloaded maps directory")

def fix_downloaded_tracks():
    args = parser.parse_args()
    maps_dir = args.maps_dir

    if not os.path.exists(maps_dir):
        print(f"Maps directory {maps_dir} not found. Run train.py first to trigger the download.")
        return

    # Loop through all downloaded track folders (e.g., 'Spielberg')
    for track_name in os.listdir(maps_dir):
        track_dir = os.path.join(maps_dir, track_name)
        
        if os.path.isdir(track_dir):
            # The file the server shipped
            wrong_yaml = os.path.join(track_dir, f"{track_name}.yaml")
            # The file the loader actually wants
            right_yaml = os.path.join(track_dir, f"{track_name}_map.yaml")
            
            # If the downloaded yaml exists but the required one doesn't, copy it
            if os.path.exists(wrong_yaml) and not os.path.exists(right_yaml):
                shutil.copy(wrong_yaml, right_yaml)
                print(f"Fixed YAML configuration for {track_name}")
                
            # Do the same for the map image file if necessary
            wrong_img = os.path.join(track_dir, f"{track_name}.png")
            right_img = os.path.join(track_dir, f"{track_name}_map.png")
            
            if os.path.exists(wrong_img) and not os.path.exists(right_img):
                shutil.copy(wrong_img, right_img)
                print(f"Fixed image file for {track_name}")

if __name__ == "__main__":
    print("Running track fix shim...")
    fix_downloaded_tracks()
    print("Complete!")