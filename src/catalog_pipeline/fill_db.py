import sqlite3
from catalog import catalog
from index_data import pair_db

conn = sqlite3.connect('star_catalog.db')
c = conn.cursor()

c.execute('''create table if not exists stars (
    hip integer primary key,
    raicrs real,
    deicrs real,
    vmag real
)''')
c.execute('''create table if not exists pairs (
    star1_id integer,
    star2_id integer,
    angle real,
    primary key (star1_id, star2_id),
    foreign key (star1_id) references stars(hip),
    foreign key (star2_id) references stars(hip)
)''')

c.execute('create index if not exists idx_pairs_star1_id on pairs(star1_id)')
c.execute('create index if not exists idx_pairs_star2_id on pairs(star2_id)')
c.execute('create index if not exists idx_pairs_angle on pairs(angle)')
c.execute('create index if not exists idx_stars_vmag on stars(vmag)')

def insert_stars():
    for row in catalog:
        c.execute('insert or ignore into stars (hip, raicrs, deicrs, vmag) values (?, ?, ?, ?)',
                  (int(row["HIP"]), float(row["RAICRS"]), float(row["DEICRS"]), float(row["Vmag"])))

def insert_pairs():
    for (i, j), angle in pair_db.items():
        c.execute('insert or ignore into pairs (star1_id, star2_id, angle) values (?, ?, ?)', (i, j, angle))

insert_stars()
insert_pairs()
conn.commit()
conn.close()
print("Database filled successfully.")
