from astroquery.vizier import Vizier
import astropy.units as u
from astropy.coordinates import SkyCoord

# Limit magnitude to ~6 (naked eye visible, typical for star trackers)
Vizier.ROW_LIMIT = -1
catalog = Vizier(columns=["HIP", "RA_ICRS", "DE_ICRS", "Vmag"], 
                 column_filters={"Vmag": "<6"}).get_catalogs("I/239/hip_main")[0]

print(catalog[:5])
