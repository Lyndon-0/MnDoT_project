import xml.etree.ElementTree as ET
import csv

# Input and output files
INPUT_XML = 'metro_config.xml'   # Change if needed
OUTPUT_CSV = 'detectors_i35_i94_i494_i694.csv'

# Exact routes (non-I-35)
EXACT_ROUTES = {'I-94', 'I-494', 'I-694'}

rows = []

tree = ET.parse(INPUT_XML)
root = tree.getroot()

for corridor in root.findall('.//corridor'):
    route_full = corridor.get('route')
    if route_full is None:
        continue

    dir_attr = corridor.get('dir', '')

    # Determine route and direction
    if route_full.startswith('I-35'):
        route_upper = route_full.upper()
        if 'E' in route_upper:
            route = 'I-35E'
        elif 'W' in route_upper:
            route = 'I-35W'
        else:
            continue
            # route = 'I-35'  # Fallback for plain I-35

        # Normalize direction to NB/SB for I-35 variants
        dir_upper = dir_attr.upper()
        if dir_upper == 'N':
            direction = 'NB'
        elif dir_upper == 'S':
            direction = 'SB'
        else:
            direction = dir_attr  # Keep as-is (e.g., NB, SB, or rare others)
    elif route_full in EXACT_ROUTES:
        route = route_full
        direction = dir_attr
    else:
        continue  # Skip unrelated routes

    for r_node in corridor.findall('r_node'):
        lat = r_node.get('lat')
        lon = r_node.get('lon')
        if lat is None or lon is None:
            continue  # Skip if no coordinates
        
        detectors = r_node.findall('detector')
        if not detectors:  # Skip nodes with no detectors
            continue
        
        for detector in detectors:
            det_name = detector.get('name')
            if det_name is None:
                continue
            lane = detector.get('lane', '')  # Empty string if not present
            
            rows.append({
                'detector_name': det_name,
                'route': route,
                'direction': direction,
                'lat': lat,
                'lon': lon,
                'lane': lane
            })

# Sort for consistent output
rows.sort(key=lambda x: (
    x['route'],
    x['direction'],
    float(x['lat']),
    float(x['lon']),
    x['detector_name'],
    x['lane']
))

# Write CSV
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['detector_name', 'route', 'direction', 'lat', 'lon', 'lane'])
    writer.writeheader()
    writer.writerows(rows)

print(f"Done! Extracted {len(rows)} detectors.")
print(f"CSV saved as: {OUTPUT_CSV}")