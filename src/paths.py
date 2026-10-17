"""Central path configuration. Override any location with an environment variable."""
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = os.environ.get('GALATEA_DATA', os.path.join(REPO, 'data', 'lunarsim'))      # rendered images + features
RESULTS = os.environ.get('GALATEA_RESULTS', os.path.join(REPO, 'results'))
WEIGHTS = os.environ.get('GALATEA_WEIGHTS', os.path.join(REPO, 'weights'))          # downloaded checkpoints
EXT = os.environ.get('GALATEA_EXT', os.path.join(REPO, 'third_party'))              # cloned model code
PAPER = os.environ.get('GALATEA_PAPER', os.path.join(REPO, 'paper'))
