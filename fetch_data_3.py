from fastapi import FastAPI, Query, HTTPException
from typing import List
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
import requests
from sentinelhub import bbox_to_dimensions, CRS, BBox
import json
import os
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()


class OAuthConfig:
    """Handles OAuth configuration and token management for the session."""

    def __init__(self, token_cache_file="token.json", token_expiration_buffer_time=60):
        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET")
        self.token_url = os.getenv("TOKEN_URL")
        self.token_cache_file = token_cache_file
        self.token_expiration_buffer_time = token_expiration_buffer_time  # seconds
        self.oauth_session = None

    def create_session(self):
        """Create and authorize an OAuth session."""
        client = BackendApplicationClient(client_id=self.client_id)
        token = self._load_token()

        if token and not self._token_expired(token):
            # Reuse the existing token
            self.oauth_session = OAuth2Session(client=client, token=token)
            print("Token loaded from cache.")
        else:
            # Fetch a new token
            self.oauth_session = OAuth2Session(client=client)
            token = self._fetch_token()
            print("Token fetched and saved to cache.")

        return self.oauth_session, token

    def _fetch_token(self):
        """Fetches token for the OAuth session."""
        token = self.oauth_session.fetch_token(
            token_url=self.token_url,
            client_secret=self.client_secret,
            include_client_id=True
        )
        self._save_token(token)
        return token

    def _token_expired(self, token):
        """Checks if the token has expired."""
        return token.get('expires_at', 0) - time.time() < self.token_expiration_buffer_time

    def _save_token(self, token):
        """Saves the token to a file."""
        with open(self.token_cache_file, 'w') as f:
            json.dump(token, f)

    def _load_token(self):
        """Loads the token from a file if it exists."""
        if os.path.exists(self.token_cache_file):
            with open(self.token_cache_file, 'r') as f:
                token = json.load(f)
            return token
        return None


class DataFetcher:
    """Fetches data using an authenticated OAuth session."""

    def __init__(self, oauth_session, token,
                 process_url="https://sh.dataspace.copernicus.eu/api/v1/process",
                 catalog_url="https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"):
        self.oauth_session = oauth_session
        self.token = token
        self.catalog_url = catalog_url
        self.process_url = process_url

    def query_catalog(self, query_params):
        response = self.oauth_session.get(self.catalog_url, params=query_params)
        response.raise_for_status()
        return response.json()

    def get_process_data(self, headers, data):
        response = self.oauth_session.post(self.process_url, headers=headers, json=data)
        response.raise_for_status()
        return response


class FileSaver:
    """Handles file saving operations."""

    @staticmethod
    def save_to_file(data, filename):
        """Saves data to a specified file."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            f.write(data)


class ImageNameGenerator:
    """Generates image file names based on query data."""

    @staticmethod
    def generate_process_image_name(data, evalscript_name):
        satellite_type = data["input"]["data"][0]["type"]
        width = int(data["output"]["width"])
        height = int(data["output"]["height"])
        time_from = datetime.fromisoformat(data["input"]["data"][0]["dataFilter"]["timeRange"]["from"].rstrip('Z'))
        time_to = datetime.fromisoformat(data["input"]["data"][0]["dataFilter"]["timeRange"]["to"].rstrip('Z'))
        time_range = f"{time_from.strftime('%b')}-{time_to.strftime('%b')}_{time_from.year}"
        bbox = data["input"]["bounds"]["bbox"]
        location = f"Lat_{bbox[1]:.2f}_{bbox[3]:.2f}_Lon_{bbox[0]:.2f}_{bbox[2]:.2f}"
        image_extension = data["output"]["responses"][0]["format"]["type"].split("/")[-1]
        image_name = f"{satellite_type}_{evalscript_name}_{width}x{height}px_{time_range}_{location}.{image_extension}"
        return image_name


class EvalscriptLoader:
    """Loads evalscripts from a specified directory."""

    def __init__(self, folder_path=os.path.join(os.getcwd(), "evalscripts")):
        self.folder_path = folder_path

    def load_evalscript(self, satellite_name, script_name):
        """Loads the evalscript content from a text file."""
        script_path = os.path.join(self.folder_path, satellite_name, script_name, "script.js")

        if not os.path.exists(script_path):
            raise FileNotFoundError(f"The evalscript '{script_name}' was not found in {self.folder_path}")

        with open(script_path, 'r') as file:
            evalscript = file.read()

        return evalscript


class QueryDataBuilder:
    """Builds query data for the process API."""

    def __init__(self, bbox_coords, time_range, data_type, evalscript_content, image_format, resolution=15):
        self.bbox_coords = bbox_coords
        self.time_range = time_range
        self.data_type = data_type
        self.evalscript_content = evalscript_content
        self.image_format = image_format
        self.resolution = resolution

    def build_query_data(self):
        bbox = BBox(bbox=self.bbox_coords, crs=CRS.WGS84)
        size = bbox_to_dimensions(bbox, resolution=self.resolution)
        width, height = size

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
                                "from": self.time_range[0],
                                "to": self.time_range[1]
                            }
                        },
                        "type": self.data_type
                    }
                ],
                "temporal": True
            },
            "output": {
                "width": width,
                "height": height,
                "responses": [
                    {
                        "identifier": "default",
                        "format": {
                            "type": self.image_format
                        }
                    }
                ]
            },
            "evalscript": self.evalscript_content
        }

        return query_data


# We need to adjust the main function to accept parameters
def process_data(
    bbox_coords: List[float],
    time_range: List[str],
    data_type: str,
    evalscript_name: str,
    image_format: str,
    resolution: int = 15
):
    # Initialize OAuth session
    oauth_config = OAuthConfig()
    oauth_session, token = oauth_config.create_session()
    print("Token:", token)

    # Data fetching
    data_fetcher = DataFetcher(oauth_session, token)

    # Load evalscript
    evalscript_loader = EvalscriptLoader()
    satellite_name = 'sentinel-2' if 'sentinel-2' in data_type else 'sentinel-1'
    if(image_format == 'JPEG'):
        image_format = 'image/jpeg'
    elif(image_format == 'PNG'):
        image_format = 'image/png'
    elif(image_format == 'TIFF'):
        image_format = 'image/tiff'
    elif(image_format == 'APP/JSON'):
        image_format = 'application/json'
    evalscript_content = evalscript_loader.load_evalscript(satellite_name, evalscript_name)

    # Build query data
    query_builder = QueryDataBuilder(
        bbox_coords=bbox_coords,
        time_range=time_range,
        data_type=data_type,
        evalscript_content=evalscript_content,
        image_format=image_format,
        resolution=resolution
    )
    query_data = query_builder.build_query_data()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token['access_token']}"
    }

    try:
        response = data_fetcher.get_process_data(headers, query_data)
    except requests.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)

    if response.status_code == 200:
        image_name = ImageNameGenerator.generate_process_image_name(query_data, evalscript_name)
        output_folder = os.path.join(os.getcwd(), "output")
        file_path = os.path.join(output_folder, image_name)
        FileSaver.save_to_file(response.content, file_path)
        print(f"Image saved to {file_path}")
        return {"message": f"Image saved to {file_path}"}
    else:
        print("Failed to download image.")
        raise HTTPException(status_code=500, detail="Failed to download image.")

# Define the FastAPI endpoint
@app.get("/process")
def process_endpoint(
    bbox_coords: str = Query(..., description="Bounding box coordinates as comma-separated values: min_lon,min_lat,max_lon,max_lat - (longitude and latitude coordinates of lower left and upper right corners)"),
    time_from: str = Query(..., description="Start time in ISO format, e.g., '2024-08-07T00:00:00Z'"),
    time_to: str = Query(..., description="End time in ISO format, e.g., '2024-10-07T23:59:59Z'"),
    data_type: str = Query('sentinel-2-l2a', description="Data type, Options: 'sentinel-1-grd', 'sentinel-2-l2a', 'sentinel-2-l1c'"),
    evalscript_name: str = Query('lai', description="Name of the evalscript"),
    image_format: str = Query('JPEG', description="Image format, must be from ['JPEG', 'PNG', 'TIFF', 'APP/JSON']"),
    resolution: int = Query(15, description="Resolution in meters per pixel (default: 15)")
):
    # Parse bbox_coords string into a list of floats
    try:
        bbox_list = [float(coord) for coord in bbox_coords.split(',')]
        if len(bbox_list) != 4:
            raise ValueError()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid bbox_coords format. Expected format: min_lon,min_lat,max_lon,max_lat")

    # Validate time_range
    try:
        datetime_from = datetime.fromisoformat(time_from.rstrip('Z'))
        datetime_to = datetime.fromisoformat(time_to.rstrip('Z'))
        if datetime_from >= datetime_to:
            raise ValueError()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid time range format or 'from' date is after 'to' date.")
    time_range = [time_from, time_to]
    
    try:
        if(data_type not in ['sentinel-1-grd', 'sentinel-2-l2a', 'sentinel-2-l1c']):
            raise ValueError()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid data_type. Options: 'sentinel-1-grd', 'sentinel-2-l2a', 'sentinel-2-l1c'")
    
    try:
        if(image_format not in ['JPEG', 'PNG', 'TIFF', 'APP/JSON']):
            raise ValueError()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid image_format. Must be from ['JPEG', 'PNG', 'TIFF', 'APP/JSON']")
    
    # Call the processing function
    result = process_data(
        bbox_coords=bbox_list,
        time_range=time_range,
        data_type=data_type,
        evalscript_name=evalscript_name,
        image_format=image_format,
        resolution=resolution
    )

    return result
