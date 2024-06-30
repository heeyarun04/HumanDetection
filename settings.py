from pathlib import Path
import sys

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
# Get the parent directory of the current file
ROOT = FILE.parent
# Add the root path to the sys.path list if it is not already there
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
# Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

# Sources
WEBCAM = 'Webcam'
RTSP = 'RTSP'

SOURCES_LIST = [WEBCAM, RTSP]

# ML Model config
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'yolov8m.pt'
# In case of your custome model comment out the line above and
# Place your custom model pt file name at the line below 
# DETECTION_MODEL = MODEL_DIR / 'my_detection_model.pt'

# Webcam
WEBCAM_PATH = 0
