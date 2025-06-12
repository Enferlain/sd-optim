# generator.py - Version 1.1 (Yield one image at a time)

import base64
import binascii
import io
import asyncio
import logging

import aiohttp
import requests  # Keep for Swarm sync calls if needed

from dataclasses import dataclass
from typing import Dict, List, AsyncGenerator, Optional
from PIL import Image, PngImagePlugin, UnidentifiedImageError
from aiohttp import ClientSession
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclass
class Generator:
    url: str
    batch_size: int
    webui: str

    # --- MODIFIED: Accept session object ---
    async def generate(
            self,
            payload: Dict,
            cfg: DictConfig,
            session: aiohttp.ClientSession  # <<< ADD session parameter
    ) -> AsyncGenerator[Image.Image, None]:
        """
        Main generation function, uses the provided session.
        Yields PIL Images asynchronously, ONE AT A TIME.
        """
        webui_lower = self.webui.lower()
        logger.debug(f"Generator called for WebUI: {webui_lower}")

        # Ensure API batch size is 1
        if webui_lower in ["a1111", "forge", "reforge"]:
            payload['batch_size'] = 1
            payload['n_iter'] = 1
        elif webui_lower == "swarm":
            if 'images_per_prompt' in payload:
                payload['images_per_prompt'] = 1
            elif 'num_images' in payload:
                payload['num_images'] = 1

        # Route to specific method, PASSING the session
        if webui_lower in ["a1111", "forge", "reforge"]:
            async for image in self.generate_a1111_forge(payload, session):
                yield image  # Already yields one by one
        elif webui_lower == "swarm":
            async for image in self.generate_swarm(payload, session):
                yield image  # Already yields one by one
        elif webui_lower == "comfy":
            logger.error("ComfyUI generation not implemented in Generator.")
            raise NotImplementedError("ComfyUI generation not implemented yet")
        else:
            logger.error(f"Unsupported webui type in Generator: {self.webui}")
            raise ValueError(f"Unsupported webui: {self.webui}")

    # --- MODIFIED: Accept session object ---
    async def generate_a1111_forge(
            self,
            payload: Dict,
            session: aiohttp.ClientSession  # <<< ADD session parameter
    ) -> AsyncGenerator[Image.Image, None]:
        """Generates images using A1111/Forge API (async) with provided session."""
        api_url = f"{self.url.rstrip('/')}/sdapi/v1/txt2img"
        try:
            # <<< USE the provided session >>>
            async with session.post(api_url, json=payload) as resp:  # Removed timeout here, handled by session
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"A1111/Forge API Error ({resp.status}): {error_text[:500]}")
                    resp.raise_for_status()

                r_json = await resp.json()
                if "images" not in r_json or not r_json["images"]:
                    logger.warning("A1111/Forge API response missing 'images' or list is empty.")
                    return

                # logger.debug(f"Received {len(r_json['images'])} image string(s) from A1111/Forge.")
                for i, img_str in enumerate(r_json["images"]):
                    try:
                        if "," in img_str:
                            img_data = base64.b64decode(img_str.split(",", 1)[1])
                        else:
                            img_data = base64.b64decode(img_str)
                        image = Image.open(io.BytesIO(img_data))
                        # logger.debug(f"Yielding image {i+1} from A1111/Forge response.")
                        yield image
                    except (binascii.Error, ValueError, UnidentifiedImageError) as decode_err:
                        logger.error(f"Failed to decode/open image {i}: {decode_err}")

        # <<< REMOVED explicit TimeoutError catch, should be handled by session timeout >>>
        except aiohttp.ClientError as http_err:
            logger.error(f"HTTP error during A1111/Forge generation: {http_err}", exc_info=False)
        except Exception as e:
            logger.error(f"Unexpected error in generate_a1111_forge: {e}", exc_info=True)

    # --- MODIFIED: Accept session object ---
    async def generate_swarm(
            self,
            payload: Dict,
            session: aiohttp.ClientSession  # <<< ADD session parameter
    ) -> AsyncGenerator[Image.Image, None]:
        """Generates images using SwarmUI API (async) with provided session."""
        gen_api_url = f"{self.url.rstrip('/')}/API/GenerateText2Image"
        logger.info(f"Sending request to Swarm API: {gen_api_url}")
        try:
            # <<< USE provided session for getting Swarm session ID >>>
            session_id = await self.get_swarm_session(session)  # Pass session here
            if not session_id: logger.error("Failed to get Swarm session ID."); return

            payload["session_id"] = session_id

            # <<< USE provided session for generating image >>>
            async with session.post(gen_api_url, json=payload) as response:  # Removed timeout, handled by session
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Swarm API Generate Error ({response.status}): {error_text[:500]}")
                    response.raise_for_status()

                response_json = await response.json()
                # ... (rest of swarm image fetching logic using the SAME session object) ...
                if "images" not in response_json or not isinstance(response_json["images"], list):
                    logger.warning(f"Invalid Swarm API response format: {response_json}")
                    return

                logger.debug(f"Received {len(response_json['images'])} image ref(s) from Swarm.")
                for i, img_ref in enumerate(response_json["images"]):
                    # determine image URL path
                    image_url_path = None
                    if isinstance(img_ref, dict) and 'url' in img_ref:
                        image_url_path = img_ref['url']
                    elif isinstance(img_ref, str):
                        image_url_path = img_ref
                    else:
                        logger.warning(f"Unrecognized Swarm image ref: {img_ref}")
                        continue

                    if image_url_path.startswith('/'):
                        view_url = f"{self.url.rstrip('/')}{image_url_path}"
                    else:
                        view_url = f"{self.url.rstrip('/')}/View/{image_url_path}"

                    # logger.debug(f"Fetching Swarm image {i+1} from: {view_url}")
                    try:
                        # <<< USE provided session for fetching image data >>>
                        async with session.get(view_url) as img_response:  # Removed timeout, handled by session
                            if img_response.status != 200:
                                logger.error(
                                    f"Failed Swarm image fetch {i + 1} ({img_response.status})")
                                continue
                            image_bytes = await img_response.read()
                            image = Image.open(io.BytesIO(image_bytes))
                            yield image
                    except aiohttp.ClientError as fetch_err:
                        logger.error(f"Error fetching Swarm image {i + 1}: {fetch_err}")
                    except UnidentifiedImageError as img_err:
                        logger.error(f"Error opening Swarm image {i + 1}: {img_err}")

        # <<< REMOVED explicit TimeoutError catch >>>
        except aiohttp.ClientError as http_err:
            logger.error(f"HTTP error during Swarm generation: {http_err}", exc_info=False)
        except Exception as e:
            logger.error(f"Unexpected error in generate_swarm: {e}", exc_info=True)

    # --- MODIFIED: Accept session object ---
    async def get_swarm_session(self, session: aiohttp.ClientSession) -> Optional[str]:  # <<< ADD session parameter
        """Helper to get a new session ID from SwarmUI using provided session."""
        session_api_url = f"{self.url.rstrip('/')}/API/GetNewSession"
        try:
            # <<< USE provided session >>>
            async with session.post(session_api_url, json={}) as response:  # Removed timeout, handled by session
                if response.status != 200:
                    logger.error(f"Swarm GetNewSession fail ({response.status})")
                    return None
                session_data = await response.json()
                session_id = session_data.get("session_id")
                if not session_id: logger.error(f"Swarm GetNewSession missing 'session_id'"); return None
                return session_id
        except Exception as e:
            logger.error(f"Error getting Swarm session ID: {e}")
            return None

    # generate_comfy remains NotImplementedError
    def generate_comfy(self, payload: Dict) -> List[Image.Image]:
        raise NotImplementedError("ComfyUI image generation not implemented yet")
