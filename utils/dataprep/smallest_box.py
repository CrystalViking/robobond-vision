smallest_width = float('inf')
smallest_height = float('inf')

with open('pos_br.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()
        num_boxes = int(parts[1])
        if len(parts) < 2 + num_boxes * 4:
            print(f"Skipping line with unexpected format: {line}")
            continue
        for i in range(num_boxes):
            width = int(parts[2 + i * 4 + 2])
            height = int(parts[2 + i * 4 + 3])
            smallest_width = min(smallest_width, width)
            smallest_height = min(smallest_height, height)

print(f"Smallest width: {smallest_width}")
print(f"Smallest height: {smallest_height}")