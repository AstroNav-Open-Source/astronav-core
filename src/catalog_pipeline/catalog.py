from astroquery.vizier import Vizier
import astropy.units as u
from astropy.coordinates import SkyCoord

# Limit magnitude to ~6 (naked eye visible, typical for star trackers)
catalog = Vizier(columns=["HIP", "RAICRS", "DEICRS", "Vmag"], 
                 column_filters={"Vmag": "<6"}).get_catalogs("I/239/hip_main")[0]

print(catalog)
