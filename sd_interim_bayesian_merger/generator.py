import base64
import io
import asyncio
import aiohttp
import requests

from dataclasses import dataclass
from typing import Dict, List
from PIL import Image, PngImagePlugin
from aiohttp import ClientSession

@dataclass
class Generator:
    url: str
    batch_size: int
    webui: str

    def generate(self, payload: Dict) -> List[Image.Image]:
        if self.webui.lower() in ["a1111", "forge", "reforge"]:
            return self.generate_a1111_forge(payload)
        elif self.webui.lower() == "swarm":
            return asyncio.run(self.generate_swarm(payload))
        elif self.webui.lower() == "comfy":
            return self.generate_comfy(payload)
        else:
            raise ValueError(f"Unsupported webui: {self.cfg.webui}")

    def generate_a1111_forge(self, payload: Dict) -> List[Image.Image]:
        # Assuming A1111 and Forge use the same txt2img endpoint
        r = requests.post(
            url=f"{self.url}/sdapi/v1/txt2img",
            json=payload,
        )
        r.raise_for_status()

        r_json = r.json()
        images = r_json["images"]

        return [
            Image.open(io.BytesIO(base64.b64decode(img.split(",", 1)[0])))
            for img in images
        ]

    async def generate_swarm(self, payload: Dict) -> List[Image.Image]:
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
                images = []
                for img_data in images_data:
                    image_url = img_data["image"]
                    image_path = image_url.split("View/")[1]
                    async with session.get(f"{self.url}/View/{image_path}") as img_response:
                        img_response.raise_for_status()
                        image_bytes = await img_response.read()
                        image = Image.open(io.BytesIO(image_bytes))
                        images.append(image)

                return images

    async def get_swarm_session(self, session):
        async with session.post(url=f"{self.url}/API/GetNewSession", json={}) as response:
            response.raise_for_status()
            return (await response.json())["session_id"]

    def generate_comfy(self, payload: Dict) -> List[Image.Image]:
        # Implement ComfyUI image generation
        raise NotImplementedError("ComfyUI image generation not implemented yet")