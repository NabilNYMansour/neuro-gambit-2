from constants import DATA_PATH

count = 0
print(f"Counting games in {DATA_PATH}...")
with open(DATA_PATH) as src:
    for line in src:
        if line.startswith("[Event "):
            count += 1
print(f"Total games: {count}")
