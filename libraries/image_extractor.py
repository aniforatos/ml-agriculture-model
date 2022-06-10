
from exif import Image
import os
import pandas as pd
class ImgToCsv:
    def __init__(self):
        self.gps_dict = {"Image": [], "Latitude": [], "Longitude":[], "Altitude": []}
        pass
    
    def get_image_info(self, path):
        for file in os.listdir(path):
            image = Image(open(os.path.join(path,file), "rb"))
            self.gps_dict["Image"].append(os.path.join(path,file))
            self.gps_dict["Latitude"].append(image.get('gps_latitude'))
            self.gps_dict["Longitude"].append(image.get('gps_longitude'))
            self.gps_dict["Altitude"].append(image.get('gps_altitude'))
            # print(f"GPS Info: Lat: {image.get('gps_latitude')}, Long: {image.get('gps_longitude')}\
            #       Alt: {image.get('gps_altitude')}")
        
        df = pd.DataFrame.from_dict(self.gps_dict)
        df.to_csv("Image_GPS_Mapping.csv")
        print("Saved CSV...")
            
