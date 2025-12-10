# sd_optim/generator.py
import logging
import aiohttp

from dataclasses import dataclass
from typing import Dict, AsyncGenerator, Optional
from PIL import Image
from omegaconf import DictConfig

from sd_optim.gen_adapters import BackendAdapter, A1111Adapter, ComfyUIAdapter

logger = logging.getLogger(__name__)


@dataclass
class Generator:
    url: str
    batch_size: int
    webui: str
    adapter: Optional[BackendAdapter] = None

    def __post_init__(self):
        self._initialize_adapter()

    def _initialize_adapter(self):
        """Factory method to create the correct backend adapter."""
        ui_type = self.webui.lower()
        
        if ui_type in ["a1111", "forge", "reforge"]:
            logger.info(f"Initializing Adapter for {ui_type} at {self.url}")
            self.adapter = A1111Adapter(self.url, webui_type=ui_type)
            
        elif ui_type == "comfy":
            logger.info(f"Initializing Adapter for ComfyUI at {self.url}")
            self.adapter = ComfyUIAdapter(self.url)
            
        elif ui_type == "swarm":
            # TODO: Implement SwarmAdapter in adapters.py if needed. 
            # For now, we can either raise error or fallback if you port the code.
            logger.error("SwarmUI adapter not yet implemented in refactor.")
            raise NotImplementedError("SwarmUI support requires SwarmAdapter implementation.")
            
        else:
            raise ValueError(f"Unsupported WebUI type: {self.webui}")

    async def generate(
            self,
            payload: Dict,
            cfg: DictConfig, # Kept for signature compatibility if needed, but unused
            session: aiohttp.ClientSession
    ) -> AsyncGenerator[Image.Image, None]:
        """
        Delegates generation to the active adapter.
        """
        if not self.adapter:
            raise RuntimeError("Backend Adapter not initialized.")

        # Ensure batch size is 1 for optimization safety (overriding payload)
        # This is strictly for the Optimizer's "One Sample" logic.
        # If payload needs specific batching, the Adapter handles it.
        # payload['batch_size'] = 1 
        
        # The adapter returns an async generator
        async for image in self.adapter.generate(payload, session):
            yield image
