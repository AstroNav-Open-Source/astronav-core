import sqlite3
from catalog import catalog
from index_data import pair_db
import random

# Create a small test database with only 10 stars
conn = sqlite3.connect('star_catalog_small_test_500_stars.db')
c = conn.cursor()

c.execute('''create table if not exists stars (
    hip integer primary key,
    raicrs real,
    deicrs real,
    vmag real
)''')

c.execute('''create table if not exists pairs (
    id integer primary key autoincrement,
    star1_id integer,
    star2_id integer,
    angle real,
    foreign key (star1_id) references stars(hip),
    foreign key (star2_id) references stars(hip)
)''')

c.execute('create index if not exists idx_pairs_star1_id on pairs(star1_id)')
c.execute('create index if not exists idx_pairs_star2_id on pairs(star2_id)')
c.execute('create index if not exists idx_pairs_angle on pairs(angle)')
c.execute('create index if not exists idx_stars_vmag on stars(vmag)')

# Required HIP stars
required_hips = [24436, 33579, 30438, 34444, 32349]

# Find these stars in the catalog and add 5 random ones
selected_stars = []
available_hips = []

# First, find the required stars and collect all available HIPs
for row in catalog:
    hip_id = int(row["HIP"])
    if hip_id in required_hips:
        selected_stars.append(row)
        print(f"Found required star HIP {hip_id}")
    else:
        available_hips.append((hip_id, row))

# Add 5 random stars from the remaining catalog
random.seed(42)  # For reproducible results
random_stars = random.sample(available_hips, 500)
for hip_id, row in random_stars:
    selected_stars.append(row)
    print(f"Added random star HIP {hip_id}")

print(f"\nTotal stars selected: {len(selected_stars)}")

def insert_stars():
    for row in selected_stars:
        c.execute('insert or ignore into stars (hip, raicrs, deicrs, vmag) values (?, ?, ?, ?)',
                  (int(row["HIP"]), float(row["RAICRS"]), float(row["DEICRS"]), float(row["Vmag"])))
        print(f"Inserted star HIP {int(row['HIP'])}")

def insert_pairs():
    # Get the HIP IDs of our selected stars
    selected_hip_ids = {int(row["HIP"]) for row in selected_stars}
    
    pair_count = 0
    for (i, j), angle in pair_db.items():
        # Only include pairs where both stars are in our selected set
        if i in selected_hip_ids and j in selected_hip_ids:
            c.execute('insert or ignore into pairs (star1_id, star2_id, angle) values (?, ?, ?)', (i, j, angle))
            pair_count += 1
    
    print(f"Inserted {pair_count} star pairs")

insert_stars()
insert_pairs()
conn.commit()
conn.close()
print("Small test database created successfully!")
print("Database name: star_catalog_small_test.db")
