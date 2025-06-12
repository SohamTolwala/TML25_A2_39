import sys
import io
import base64
import json
import requests
from PIL import Image
import numpy as np

class APIClient:
    def __init__(self, token: str):
        self.token = token
        self.port = None
        self.seed = None

    def launch(self):
        """
        Launch a new stealing session.
        Sets self.seed and self.port from the response.
        """
        url = "http://34.122.51.94:9090/stealing_launch"
        resp = requests.get(url, headers={"token": self.token})
        data = resp.json()
        if 'detail' in data:
            print("Error launching API:", data['detail'])
            sys.exit(1)
        self.seed = data.get('seed')
        self.port = data.get('port')
        print(f"[API] Launched: seed={self.seed}, port={self.port}")

    def query(self, images: list[Image.Image]) -> np.ndarray:
        """
        Query the encoder with a list of up to 1000 PIL Images.
        Returns:
            embeddings: numpy array of shape (N, 1024)
        """
        if self.port is None:
            raise RuntimeError("APIClient.launch() must be called first")

        # Prepare payload
        b64_list = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            b64_list.append(base64.b64encode(buf.getvalue()).decode('utf-8'))

        payload = json.dumps(b64_list)
        url = f"http://34.122.51.94:{self.port}/query"
        resp = requests.get(url, files={"file": payload}, headers={"token": self.token})

        if resp.status_code != 200:
            raise RuntimeError(f"Query failed ({resp.status_code}): {resp.text}")

        reps = resp.json().get("representations", [])
        return np.array(reps, dtype=np.float32)
