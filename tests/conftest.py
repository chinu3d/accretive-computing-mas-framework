import sys
import os

# Add the build directory to the path so pytest can find the accretive_mas library
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build'))
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)
