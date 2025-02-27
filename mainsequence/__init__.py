
#order matters are vam_endpoint comes from tdag
import os
from .tdag_client import CONSTANTS as TDAG_CONSTANTS
os.environ["VAM_ENDPOINT"] = TDAG_CONSTANTS.VAM_ENDPOINT
from .vam_client import CONSTANTS as VAM_CONSTANTS

