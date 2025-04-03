# generator.py - Version 1.0 (Assumed Structure)

import base64
import binascii
import io
import asyncio
import logging

import aiohttp
import requests # Keep requests for Swarm sync calls

from dataclasses import dataclass
from typing import Dict, List, AsyncGenerator, Optional
from PIL import Image, PngImagePlugin, UnidentifiedImageError
from aiohttp import ClientSession # For async requests
from omegaconf import DictConfig # Keep if cfg is used

logger = logging.getLogger(__name__)


@dataclass
class Generator:
    url: str
    batch_size: int # Note: This batch_size might be redundant if Optimizer controls looping
    webui: str

    async def generate(self, payload: Dict, cfg: DictConfig) -> AsyncGenerator[Image.Image, None]:
        """
        Main generation function, routes to specific WebUI methods.
        Yields PIL Images asynchronously.
        """
        webui_lower = self.webui.lower()
        # logger.debug(f"Generator called for WebUI: {webui_lower}")

        if webui_lower in ["a1111", "forge", "reforge"]:
            async for image in self.generate_a1111_forge(payload):
                yield image
        elif webui_lower == "swarm":
            async for image in self.generate_swarm(payload):
                yield image
        elif webui_lower == "comfy":
            logger.error("ComfyUI generation not implemented in Generator.")
            raise NotImplementedError("ComfyUI generation not implemented yet")
            # <<< REMOVED 'yield' here
        else:
            logger.error(f"Unsupported webui type in Generator: {self.webui}")
            raise ValueError(f"Unsupported webui: {self.webui}")
            # <<< REMOVED 'yield' here

    async def generate_a1111_forge(self, payload: Dict) -> AsyncGenerator[Image.Image, None]:
        """Generates images using A1111/Forge API (async)."""
        api_url = f"{self.url.rstrip('/')}/sdapi/v1/txt2img"
        logger.info(f"Sending request to A1111/Forge API: {api_url}")
        # logger.debug(f"Payload (first 5 keys): {dict(list(payload.items())[:5])}...")

        # Ensure n_iter or batch_size in payload is 1 if optimizer handles looping
        # payload['n_iter'] = 1
        # payload['batch_size'] = 1

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, json=payload, timeout=300) as resp: # Long timeout for generation
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"A1111/Forge API Error ({resp.status}): {error_text[:500]}") # Log first 500 chars
                        resp.raise_for_status() # Raise HTTPError

                    r_json = await resp.json()
                    if "images" not in r_json or not r_json["images"]:
                         logger.warning("A1111/Forge API response missing 'images' or images list is empty.")
                         return # Stop generation if no images

                    logger.info(f"Received {len(r_json['images'])} image(s) from A1111/Forge.")
                    for i, img_str in enumerate(r_json["images"]):
                        try:
                            # Handle potential info string prepended by some UIs
                            if "," in img_str:
                                img_data = base64.b64decode(img_str.split(",", 1)[1])
                            else:
                                img_data = base64.b64decode(img_str)
                            image = Image.open(io.BytesIO(img_data))
                            # logger.debug(f"Yielding image {i+1} from A1111/Forge batch.")
                            yield image
                        except (binascii.Error, ValueError, UnidentifiedImageError) as decode_err:
                             logger.error(f"Failed to decode/open image {i} from A1111/Forge response: {decode_err}")
                             # Continue to next image? Or raise?

        except aiohttp.ClientError as http_err:
            logger.error(f"HTTP error during A1111/Forge generation: {http_err}", exc_info=True)
            # Don't raise here, allow optimizer to handle lack of images
        except asyncio.TimeoutError:
             logger.error(f"Timeout during A1111/Forge image generation request to {api_url}.")
        except Exception as e:
            logger.error(f"Unexpected error in generate_a1111_forge: {e}", exc_info=True)
            # Don't raise, allow optimizer to handle


    async def generate_swarm(self, payload: Dict) -> AsyncGenerator[Image.Image, None]:
        """Generates images using SwarmUI API (async)."""
        gen_api_url = f"{self.url.rstrip('/')}/API/GenerateText2Image"
        session_api_url = f"{self.url.rstrip('/')}/API/GetNewSession"
        logger.info(f"Sending request to Swarm API: {gen_api_url}")
        # logger.debug(f"Payload (first 5 keys): {dict(list(payload.items())[:5])}...")

        try:
            async with aiohttp.ClientSession() as session:
                # Get Session ID asynchronously
                session_id = await self.get_swarm_session(session)
                if not session_id:
                     logger.error("Failed to get Swarm session ID.")
                     return # Cannot proceed

                payload["session_id"] = session_id
                # Ensure Swarm generates the correct number (usually 1 if optimizer loops)
                # payload['num_images'] = 1 # Example key, check Swarm API docs

                # Generate Image Request
                async with session.post(gen_api_url, json=payload, timeout=300) as response: # Long timeout
                    if response.status != 200:
                         error_text = await response.text()
                         logger.error(f"Swarm API Generate Error ({response.status}): {error_text[:500]}")
                         response.raise_for_status()

                    response_json = await response.json()
                    if "images" not in response_json or not isinstance(response_json["images"], list):
                        logger.warning(f"Invalid Swarm API response format or missing 'images': {response_json}")
                        return

                    logger.info(f"Received {len(response_json['images'])} image reference(s) from Swarm.")
                    # Fetch each image asynchronously
                    for i, img_ref in enumerate(response_json["images"]):
                         # Check if img_ref is a dict and has 'image' key (newer Swarm) or just a URL string (older?)
                         if isinstance(img_ref, dict) and 'url' in img_ref:
                             image_url_path = img_ref['url'] # Use 'url' field if present
                         elif isinstance(img_ref, str):
                              image_url_path = img_ref # Assume it's the URL path directly
                         else:
                              logger.warning(f"Unrecognized image reference format from Swarm: {img_ref}")
                              continue

                         # Construct full URL if needed (sometimes relative path is given)
                         if image_url_path.startswith('/'):
                              view_url = f"{self.url.rstrip('/')}{image_url_path}"
                         else:
                              # Assume it needs View/ prefix if just filename? Check Swarm API response structure.
                              # This might need adjustment based on actual Swarm output.
                              view_url = f"{self.url.rstrip('/')}/View/{image_url_path}"

                         logger.debug(f"Fetching Swarm image {i+1} from: {view_url}")
                         try:
                             async with session.get(view_url, timeout=60) as img_response: # Timeout for image download
                                 if img_response.status != 200:
                                     error_text = await img_response.text()
                                     logger.error(f"Failed to fetch Swarm image {i+1} ({img_response.status}): {error_text[:500]}")
                                     continue # Skip this image

                                 image_bytes = await img_response.read()
                                 image = Image.open(io.BytesIO(image_bytes))
                                 yield image
                         except (aiohttp.ClientError, asyncio.TimeoutError) as fetch_err:
                              logger.error(f"Error fetching Swarm image {i+1} from {view_url}: {fetch_err}")
                         except (UnidentifiedImageError) as img_err:
                              logger.error(f"Error opening fetched Swarm image {i+1}: {img_err}")

        except aiohttp.ClientError as http_err:
            logger.error(f"HTTP error during Swarm generation: {http_err}", exc_info=True)
        except asyncio.TimeoutError:
             logger.error(f"Timeout during Swarm image generation request to {gen_api_url}.")
        except Exception as e:
            logger.error(f"Unexpected error in generate_swarm: {e}", exc_info=True)


    async def get_swarm_session(self, session: aiohttp.ClientSession) -> Optional[str]:
        """Helper to get a new session ID from SwarmUI."""
        session_api_url = f"{self.url.rstrip('/')}/API/GetNewSession"
        try:
            async with session.post(session_api_url, json={}, timeout=10) as response:
                if response.status != 200:
                     logger.error(f"Swarm GetNewSession failed ({response.status}): {await response.text()}")
                     return None
                session_data = await response.json()
                session_id = session_data.get("session_id")
                if not session_id:
                     logger.error(f"Swarm GetNewSession response missing 'session_id': {session_data}")
                     return None
                # logger.debug(f"Obtained Swarm session ID: {session_id}")
                return session_id
        except Exception as e:
             logger.error(f"Error getting Swarm session ID: {e}")
             return None

    # generate_comfy remains NotImplementedError
    def generate_comfy(self, payload: Dict) -> List[Image.Image]:
        raise NotImplementedError("ComfyUI image generation not implemented yet")