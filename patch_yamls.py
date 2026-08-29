import os

possible_dirs = [
    os.path.join("f1tenth_gym", "maps"),
    os.path.join("f1tenth_gym", "f1tenth_gym", "maps")
]

patched_count = 0

for maps_dir in possible_dirs:
    if not os.path.exists(maps_dir):
        continue
        
    for i in range(100):
        track_name = f"synthetic_track_{i}"
        yaml_path = os.path.join(maps_dir, track_name, f"{track_name}_map.yaml")
        
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as file:
                content = file.read()
            
            # Edits the explicit text inside the configuration file
            new_content = content.replace("map0", f"{track_name}_map")
            
            with open(yaml_path, 'w') as file:
                file.write(new_content)
            patched_count += 1

print(f"Successfully patched the internal text of {patched_count} YAML files!")