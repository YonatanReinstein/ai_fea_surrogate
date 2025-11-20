import os
import re

folder = r"data/beam/dataset/screenshots"

# Regex to extract number from filenames like mesh_123.png
pattern = re.compile(r"(.*?)(\d+)(\.\w+)$")

files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

# Sort by the numeric part (so rename doesn't reorder incorrectly)
def extract_number(filename):
    m = pattern.match(filename)
    return int(m.group(2)) if m else -1

files_sorted = sorted(files, key=extract_number)

# Determine padding width
max_num = max(extract_number(f) for f in files_sorted)
pad = len(str(max_num))

for f in files_sorted:
    match = pattern.match(f)
    if not match:
        continue

    prefix, number, ext = match.groups()
    new_name = f"{prefix}{int(number):0{pad}d}{ext}"

    old_path = os.path.join(folder, f)
    new_path = os.path.join(folder, new_name)

    if old_path != new_path:
        print(f"Renaming: {f} → {new_name}")
        os.rename(old_path, new_path)

print("Done.")
