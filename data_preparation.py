import sys
sys.path.insert(0, "./libraries")
from image_extractor import ImgToCsv

x = ImgToCsv()
x.get_image_info("./workspace/images/test/")