import os
import shutil
import sys
import subprocess
import re
import stat

def remove_readonly(func, path, excinfo):
    """Clear the read-only attribute and retry removal on Windows."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as e:
        print(f"Warning: Could not remove {path}: {e}")

def reset_and_generate(num_tracks=100):
    print("Cleaning up old synthetic tracks...")
    dest_dir = os.path.join("f1tenth_gym", "maps") 
    if os.path.exists(dest_dir):
        for folder in os.listdir(dest_dir):
            if folder.startswith("synthetic_track_"):
                folder_path = os.path.join(dest_dir, folder)
                shutil.rmtree(folder_path, onerror=remove_readonly)

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
            
        content = re.sub(r"image:\s*.*", f"image: {dest_pgm_name}", content)
        
        with open(os.path.join(track_folder, dest_yaml_name), 'w') as f:
            f.write(content)
            
        shutil.copy(src_pgm, os.path.join(track_folder, dest_pgm_name))
        
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
                        outfile.write(f"{line}, 2.0, 2.0\n")

    if os.path.exists(source_dir):
        shutil.rmtree(source_dir, onerror=remove_readonly)
            
    print(f"Success! {num_tracks} tracks are packaged in {dest_dir}.")

if __name__ == "__main__":
    reset_and_generate(100)