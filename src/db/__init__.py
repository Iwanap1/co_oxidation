import warnings
from cryptography.utils import CryptographyDeprecationWarning

warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message="You appear to be connected to a CosmosDB cluster.*",
    category=UserWarning,
)

from .database import Database
from .migrator import Migrator