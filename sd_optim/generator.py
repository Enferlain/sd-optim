# generator.py - Version 1.0

import base64
import io
import asyncio
import aiohttp
import requests

from dataclasses import dataclass
from typing import Dict, List, AsyncGenerator  # Import AsyncGenerator
from PIL import Image, PngImagePlugin
from aiohttp import ClientSession
from omegaconf import DictConfig

@dataclass
class Generator:
    url: str
    batch_size: int
    webui: str

    async def generate(self, payload: Dict, cfg: DictConfig) -> AsyncGenerator[Image.Image, None]: # Changed
        if self.webui.lower() in ["a1111", "forge", "reforge"]:
            async for image in self.generate_a1111_forge(payload): # Changed
                yield image
        elif self.webui.lower() == "swarm":
            async for image in self.generate_swarm(payload): # Changed
                yield image
        elif self.webui.lower() == "comfy":
            #  Implement
            raise NotImplementedError
        else:
            raise ValueError(f"Unsupported webui: {cfg.webui}")

    async def generate_a1111_forge(self, payload: Dict) -> AsyncGenerator[Image.Image, None]: # Changed
        async with aiohttp.ClientSession() as session: # Use aiohttp
            async with session.post(url=f"{self.url}/sdapi/v1/txt2img", json=payload) as resp: # Changed
                resp.raise_for_status()
                r_json = await resp.json() # Changed
                for img_str in r_json["images"]:
                    image = Image.open(io.BytesIO(base64.b64decode(img_str.split(",", 1)[0])))
                    yield image

    async def generate_swarm(self, payload: Dict) -> AsyncGenerator[Image.Image, None]: # Changed
        async with aiohttp.ClientSession() as session:
            if "session_id" not in payload:
                session_id = await self.get_swarm_session(session)
                payload["session_id"] = session_id

            async with session.post(url=f"{self.url}/API/GenerateText2Image", json=payload) as response:
                response.raise_for_status()
                response_json = await response.json()

                if "images" not in response_json:
                    raise ValueError("Invalid response from SwarmUI: 'images' key not found.")

                images_data = response_json["images"]
                for img_data in images_data:
                    image_url = img_data["image"]
                    image_path = image_url.split("View/")[1]
                    async with session.get(f"{self.url}/View/{image_path}") as img_response:
                        img_response.raise_for_status()
                        image_bytes = await img_response.read()
                        image = Image.open(io.BytesIO(image_bytes))
                        yield image # Changed

    async def get_swarm_session(self, session):
        async with session.post(url=f"{self.url}/API/GetNewSession", json={}) as response:
            response.raise_for_status()
            return (await response.json())["session_id"]

    def generate_comfy(self, payload: Dict) -> List[Image.Image]:
        # Implement ComfyUI image generation
        raise NotImplementedError("ComfyUI image generation not implemented yet")