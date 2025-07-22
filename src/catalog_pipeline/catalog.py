from astroquery.vizier import Vizier
import astropy.units as u
from astropy.coordinates import SkyCoord

# Limit magnitude to ~6 (naked eye visible, typical for star trackers)
vizier = Vizier(columns=["HIP", "RAICRS", "DEICRS", "Vmag"], 
                column_filters={"Vmag": "<6"}, row_limit=-1)
catalog = vizier.get_catalogs("I/239/hip_main")[0]

print(catalog)
