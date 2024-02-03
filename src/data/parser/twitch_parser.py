from typing import Dict

import requests


class TwitchParser:
    def __init__(self, client_id: str, secret_key: str):
        self.client_id = client_id
        self.secret_key = secret_key
        self.access_token = self.get_token()

    def get_token(self) -> str:
        """
        Loads an OAuth client credentials flow App token for an associated client id
        to an environment variables.
        """
        auth_url = "https://id.twitch.tv/oauth2/token"
        # parameters for token request with credentials
        auth_params = {
            "client_id": self.client_id,
            "client_secret": self.secret_key,
            "grant_type": "client_credentials",
            "scope": "chat:read",
        }
        # Request response, using the url base and params to structure the request
        auth_response = requests.post(auth_url, params=auth_params)
        token = auth_response.json()["access_token"]

        return token

    def request_get(self, query: str, fields: Dict) -> Dict:
        """
        Makes a GET request from the Twitch.tv helix API.
        Receives a string for the query ex: users, games/top, search/categories...,
        and a dict for the field parameter ex: {'login': 'gmhikaru'}
        """
        # the base url for making api requests
        base_url = "https://api.twitch.tv/helix/"
        # authorization information
        headers = {
            "client-id": self.client_id,
            "Authorization": f"Bearer {self.access_token}",
        }
        # type of query, users gets the
        response = requests.get(base_url + query, headers=headers, params=fields)
        response.raise_for_status()

        return response.json()

    def get_video(self, video_id: str = None) -> Dict:
        """Getting all clips of the specified VOD
        Args:
            video_id (str): Twitch VOD ID whose clips are requested
        Returns:
            dict: video data in JSON structure
        """
        if video_id is None:
            raise TypeError("'None' value provided for video_id")
        # GET request to Twitch API /videos
        response = self.request_get("videos", {"id": video_id})
        return response["data"][0]
