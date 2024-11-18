
# oauth -> https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
import requests

from sentinelhub import bbox_to_dimensions, CRS, BBox   

import json
import os
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv() # Load .env file

class OAuthConfig:
    """Handles OAuth configuration for the session."""

    # Reference: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/Authentication.html#python

    def __init__(self, token_cache_file = "token.json", token_expiration_buffer_time = 60):
        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET")
        self.token_url = os.getenv("TOKEN_URL")
        self.token_cache_file = token_cache_file
        self.token_expiration_buffer_time = token_expiration_buffer_time # seconds
        self.oauth_session = None

    def create_session(self):
        """Create and authorize an OAuth session."""
        client = BackendApplicationClient(client_id=self.client_id)
        token = self.load_token()

        if token and not self.token_expired(token):
            # Reuse the existing token
            self.oauth_session = OAuth2Session(client=client, token=token)
            print("Token loaded from cache.")
        else:
            # Fetch a new token
            self.oauth_session = OAuth2Session(client=client)
            token = self.fetch_token()
            print("Token fetched and saved to cache.")

        return self.oauth_session, token

    def fetch_token(self):
        """Fetches token for the OAuth session."""
        token = self.oauth_session.fetch_token(
            token_url=self.token_url,
            client_secret=self.client_secret,
            include_client_id=True
        )
        self.save_token(token)
        return token
    
    def token_expired(self, token):
        """Checks if the token has expired."""
        return token.get('expires_at', 0) - time.time() < self.token_expiration_buffer_time
    
    def save_token(self, token):
        """Saves the token to a file."""
        with open(self.token_cache_file, 'w') as f:
            json.dump(token, f)
    
    def load_token(self):
        """Loads the token from a file if it exists."""
        if os.path.exists(self.token_cache_file):
            with open(self.token_cache_file, 'r') as f:
                token = json.load(f)
            return token
        return None



class DataFetcher:
    """Fetches data from given URLs using an authenticated OAuth session."""

    def __init__(self, 
                 oauth_session, 
                 token, 
                 process_url = "https://sh.dataspace.copernicus.eu/api/v1/process", 
                 catalog_url = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"):
        self.oauth_session = oauth_session
        self.token = token
        self.catalog_url = catalog_url
        self.process_url = process_url

    def query_catalog(self, query_params):
        response = self.oauth_session.get(self.catalog_url, params=query_params)
        return response.json()
    
    def get_process_data(self, headers, data):
        response = self.oauth_session.post(self.process_url, headers=headers, json=data)
        return response
        
    def get_data(self, url, headers, data):
        """Fetches data from a specified URL."""
        response = self.oauth_session.post(url, headers=headers, json=data)
        return response.json()


class FileHandler:
    """Handles file operations."""

    @staticmethod
    def save_to_file(data, filename):
        """Saves data to a specified file."""
        with open(filename, 'wb') as f:
            f.write(data)

    @staticmethod
    def generate_process_image_name(data, evalscript_name):
        
        # Extracting satellite type and band information
        satellite_type = data["input"]["data"][0]["type"]
        # bands = ", ".join(data["evalscript"].split('input: ["')[1].split('"],')[0].split(", "))
        
        # Extracting width and height, rounding height to the nearest integer
        width = int(data["output"]["width"])
        height = round(float(data["output"]["height"]))
        
        # Extracting time range and formatting dates
        time_from = datetime.fromisoformat(data["input"]["data"][0]["dataFilter"]["timeRange"]["from"][:-1])
        time_to = datetime.fromisoformat(data["input"]["data"][0]["dataFilter"]["timeRange"]["to"][:-1])
        time_range = f"{time_from.strftime('%b')}-{time_to.strftime('%b')}_{time_from.year}"
        
        # Adding geographical coordinates for context (could translate bbox to location if desired)
        bbox = data["input"]["bounds"]["bbox"]
        location = f"Lat_{bbox[1]:.2f}_{bbox[3]:.2f}_Lon_{bbox[0]:.2f}_{bbox[2]:.2f}"
        
        # Creating the file name
        image_extension = data["output"]["responses"][0]["format"]["type"].split("/")[1]
        image_name = f"{satellite_type}_{evalscript_name}_{width}x{height}px_{time_range}_{location}.{image_extension}"


        return image_name
    
    def save_process_image(self, data, query_data, evalscript_name, output_folder = os.path.join(os.getcwd(), "output")):
        """Saves process data to a specified file."""
        image_name = self.generate_process_image_name(query_data, evalscript_name)
        file = os.path.join(output_folder, image_name)
        self.save_to_file(data, file)


class EvalscriptLoader:
    """Loads evalscripts from a specified directory."""

    def __init__(self, folder_path = os.path.join(os.getcwd(), "evalscripts")):
        self.folder_path = folder_path

    def load_evalscript(self, sattelite_name, script_name):
        """Loads the evalscript content from a text file."""
        script_path = os.path.join(self.folder_path, sattelite_name, script_name, "script.js")
        
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"The evalscript '{script_name}' was not found in {self.folder_path}")
        
        with open(script_path, 'r') as file:
            evalscript = file.read()
        
        return evalscript
    

def main():

    # Initialize OAuth session
    oauth_config = OAuthConfig()
    oauth_session, token = oauth_config.create_session()
    print("Token:", token)

    # Data fetching
    data_fetcher = DataFetcher(oauth_session, token)

    # Catalog
    # # Reference: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Catalog.html
    # bbox = "13,45,14,46"
    # datetime_range = "2019-12-10T00:00:00Z/2019-12-11T00:00:00Z"
    # collections = "sentinel-1-grd"
    # limit = 1
    # query = {
    #     "bbox": bbox,
    #     "datetime": datetime_range,
    #     "collections": collections,
    #     "limit": int(limit),
    # }
    # catalog_response = data_fetcher.query_catalog(query)
    # print("Catalog Response:", catalog_response)
    # FileHandler.save_to_file(str(catalog_response).encode(), f"catalog_data_bbox_{query['bbox']}_datetime_{query['datetime']}_collections_{query['collections']}_limit_{query['limit']}.json")


    # Process data

    evalscript_loader = EvalscriptLoader()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer f{token}"
    }

    #  The bounding box in WGS84 coordinate system is [46.16, -16.15, 46.51, -15.58] (longitude and latitude coordinates of lower left and upper right corners)
    # INPUT PARAMETERS:
    type = 'sentinel-2-l2a' # must be from ['sentinel-1-grd', 'sentinel-2-l2a', 'sentinel-2-l1c']
    bbox = (12.44693, 41.870072, 12.541001, 41.917096)
    timeRange = ['2024-08-07T00:00:00Z', '2024-10-07T23:59:59Z']
    evalscript_name = 'lai'
    image_format = 'JPEG' # must be from ['JPEG', 'PNG', 'TIFF', 'APP/JSON']    

    # Query creation
    bbox = BBox(bbox=bbox, crs=CRS.WGS84)
    print(bbox)
    resolution = 15
    size = bbox_to_dimensions(bbox, resolution=resolution)
    width = size[0]
    height = size[1]
    if(image_format == 'JPEG'):
        image_format = 'image/jpeg'
    elif(image_format == 'PNG'):
        image_format = 'image/png'
    elif(image_format == 'TIFF'):
        image_format = 'image/tiff'
    elif(image_format == 'APP/JSON'):
        image_format = 'application/json'
    if(type == 'sentinel-1-grd'):
        sattelite_name = 'sentinel-1'
    else:
        sattelite_name = 'sentinel-2'
    evalscript_content = evalscript_loader.load_evalscript(sattelite_name, evalscript_name)

    query_data = {
        "input": {
            "bounds": {
                "bbox": [
                    bbox.lower_left[0],
                    bbox.lower_left[1],
                    bbox.upper_right[0],
                    bbox.upper_right[1]
                ]
            },
            "data": [
                {
                    "dataFilter": {
                        "timeRange": {
                            "from": timeRange[0],
                            "to": timeRange[1]
                        }
                    },
                    "type": type
                }
            ],
            "temporal": True  # Required for multi-temporal processing
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [
                {
                    "identifier": "default",
                    "format": {
                        "type": f"image/{image_format}"
                    }
                }
            ]
        },
        "evalscript": evalscript_content
    }

    response = data_fetcher.get_process_data(headers, query_data)
    file_handler = FileHandler()
    if response.status_code == 200:
        file_handler.save_process_image(response.content, query_data, evalscript_name)
    else:
        print("Failed to download image.")
    return response

    

if __name__ == "__main__":

    main()
