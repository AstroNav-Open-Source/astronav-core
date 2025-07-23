from astroquery.vizier import Vizier
import astropy.units as u
from astropy.coordinates import SkyCoord

# Limit magnitude to ~6 (naked eye visible, typical for star trackers)
vizier = Vizier(columns=["HIP", "RAICRS", "DEICRS", "Vmag"], 
                column_filters={"Vmag": "<6"}, row_limit=-1)
catalog = vizier.get_catalogs("I/239/hip_main")[0]

# Filter out rows with masked RAICRS or DEICRS
if hasattr(catalog['RAICRS'], 'mask') and hasattr(catalog['DEICRS'], 'mask'):
    valid = ~catalog['RAICRS'].mask & ~catalog['DEICRS'].mask
    catalog = catalog[valid]

if __name__ == "__main__":
    print(catalog)
