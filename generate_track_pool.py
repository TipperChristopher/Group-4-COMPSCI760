import os
import shutil
import sys
import subprocess
import re

def reset_and_generate(num_tracks=100):
    print("Cleaning up old synthetic tracks...")
    dest_dir = os.path.join("f1tenth_gym", "maps") 
    if os.path.exists(dest_dir):
        for folder in os.listdir(dest_dir):
            if folder.startswith("synthetic_track_"):
                shutil.rmtree(os.path.join(dest_dir, folder))

    print(f"Generating {num_tracks} new raw tracks...")
    subprocess.run([sys.executable, "random_trackgen.py", "--n-maps", str(num_tracks)])
    
    source_dir = "maps"
    os.makedirs(dest_dir, exist_ok=True)
    
    print("Packaging tracks and patching 4-column centerlines...")
    for i in range(num_tracks):
        src_yaml = os.path.join(source_dir, f"map{i}_map.yaml")
        src_pgm = os.path.join(source_dir, f"map{i}_map.pgm")
        
        if not os.path.exists(src_yaml) or not os.path.exists(src_pgm):
            continue
            
        track_name = f"synthetic_track_{i}"
        track_folder = os.path.join(dest_dir, track_name)
        os.makedirs(track_folder, exist_ok=True)
        
        dest_pgm_name = f"{track_name}_map.pgm"
        dest_yaml_name = f"{track_name}_map.yaml"
        
        with open(src_yaml, 'r') as f:
            content = f.read()
            
        # FORCE the YAML to point to the exact .pgm file
        content = re.sub(r"image:\s*.*", f"image: {dest_pgm_name}", content)
        
        with open(os.path.join(track_folder, dest_yaml_name), 'w') as f:
            f.write(content)
            
        shutil.copy(src_pgm, os.path.join(track_folder, dest_pgm_name))
        
        # PATCH THE CENTERLINE CSV TO 4 COLUMNS
        src_csv = os.path.join(source_dir, f"map{i}_centerline.csv")
        dest_csv = os.path.join(track_folder, f"{track_name}_centerline.csv")
        if os.path.exists(src_csv):
            with open(src_csv, 'r') as infile, open(dest_csv, 'w') as outfile:
                for line in infile:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('#'):
                        outfile.write("#x,y,w_left,w_right\n")
                    else:
                        # Append 2.0 meters for left and right track widths
                        outfile.write(f"{line}, 2.0, 2.0\n")

    if os.path.exists(source_dir):
        shutil.rmtree(source_dir)
            
    print(f"Success! {num_tracks} tracks are perfectly packaged in {dest_dir}.")

if __name__ == "__main__":
    reset_and_generate(100)