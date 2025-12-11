import json
import logging
import uuid
import random
import io
import asyncio
import aiohttp

from abc import ABC, abstractmethod
from typing import Dict, AsyncGenerator, Optional, List, Any
from pathlib import Path
from PIL import Image
from hydra.utils import get_original_cwd

logger = logging.getLogger(__name__)


class BackendAdapter(ABC):
    """
    Abstract interface for interacting with different WebUI backends.
    Handles Model Loading, Unloading, and Image Generation.
    """
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    @abstractmethod
    async def load_model(self, model_path: Path, session: aiohttp.ClientSession, **kwargs):
        """Loads a model. For Comfy, this just sets the target for the next workflow."""
        pass

    @abstractmethod
    async def unload_model(self, session: aiohttp.ClientSession, **kwargs):
        """Unloads models to free VRAM."""
        pass

    @abstractmethod
    async def generate(self, payload: Dict, session: aiohttp.ClientSession) -> AsyncGenerator[Image.Image, None]:
        """Generates images based on the payload."""
        pass


class A1111Adapter(BackendAdapter):
    """
    Adapter for Automatic1111, Forge, and ReForge.
    Requires scripts/api.py to be installed in the WebUI extensions folder.
    """
    
    def __init__(self, base_url: str, webui_type: str = "a1111"):
        super().__init__(base_url)
        self.webui_type = webui_type

    async def load_model(self, model_path: Path, session: aiohttp.ClientSession, **kwargs):
        api_url = f"{self.base_url}/sd_optim/load-model"
        payload = {
            "model_path": str(model_path.resolve()),
            "webui": self.webui_type
        }
        async with session.post(api_url, json=payload) as resp:
            if resp.status != 200:
                err = await resp.text()
                raise RuntimeError(f"A1111 Load Error {resp.status}: {err}")
            logger.info(f"A1111: Loaded model {model_path.name}")

    async def unload_model(self, session: aiohttp.ClientSession, **kwargs):
        api_url = f"{self.base_url}/sd_optim/unload-model"
        params = {"webui": self.webui_type}
        async with session.post(api_url, params=params) as resp:
             if resp.status != 200:
                logger.warning(f"A1111 Unload Warning {resp.status}")

    async def generate(self, payload: Dict, session: aiohttp.ClientSession) -> AsyncGenerator[Image.Image, None]:
        api_url = f"{self.base_url}/sdapi/v1/txt2img"
        
        # Enforce batch size 1 for optimization safety
        payload['batch_size'] = 1
        payload['n_iter'] = 1

        async with session.post(api_url, json=payload) as resp:
            if resp.status != 200:
                err = await resp.text()
                raise RuntimeError(f"A1111 Generate Error {resp.status}: {err}")

            r_json = await resp.json()
            if "images" not in r_json:
                return

            import base64
            for img_str in r_json["images"]:
                image_data = base64.b64decode(img_str.split(",", 1)[1] if "," in img_str else img_str)
                yield Image.open(io.BytesIO(image_data))


class ComfyUIAdapter(BackendAdapter):
    """
    Adapter for ComfyUI using WebSocket streaming.
    Dynamically swaps 'SaveImage' nodes for 'SaveImageWebsocket' to avoid disk I/O.
    """
    
    def __init__(self, base_url: str):
        super().__init__(base_url)
        self.client_id = str(uuid.uuid4())
        self.current_model_filename: Optional[str] = None
        self.websocket_save_nodes: List[str] = [] # <<< NEW: Track save nodes

    async def load_model(self, model_path: Path, session: aiohttp.ClientSession, **kwargs):
        # In Comfy, we just store the filename. It gets injected into the workflow later.
        self.current_model_filename = model_path.name
        logger.info(f"ComfyAdapter: Targeted model set to '{self.current_model_filename}'")

    async def unload_model(self, session: aiohttp.ClientSession, **kwargs):
        # ComfyUI has a specific endpoint to free memory
        api_url = f"{self.base_url}/free"
        payload = {"unload_models": True, "free_memory": True}
        async with session.post(api_url, json=payload) as resp:
             if resp.status != 200:
                logger.warning(f"ComfyUI Free Memory Warning {resp.status}")

    async def generate(self, payload: Dict, session: aiohttp.ClientSession) -> AsyncGenerator[Image.Image, None]:
        template_path_str = payload.get("workflow_json")
        if not template_path_str:
            raise ValueError("ComfyUI payload missing 'workflow_json' path")

        # --- NEW: Path Resolution ---
        template_path = Path(template_path_str)
        if not template_path.is_absolute():
            # Resolve relative paths against the ORIGINAL project root, not the Hydra log dir
            project_root = get_original_cwd()
            template_path = Path(project_root) / template_path
        # --- END NEW ---

        # Load Template
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                workflow = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"ComfyUI workflow template not found: {template_path}")

        # 1. Inject Optimization Parameters (Model, Prompt, Seed)
        self._inject_parameters(workflow, payload)
        
        # 2. Hot-Swap: Enable Zero-Disk Mode
        # We find nodes saving to disk and make them stream to WebSocket instead.
        self._enable_websocket_streaming(workflow)

        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/ws?clientId={self.client_id}"

        try:
            async with session.ws_connect(ws_url) as ws:
                # Submit Prompt
                api_payload = {"prompt": workflow, "client_id": self.client_id}
                async with session.post(f"{self.base_url}/prompt", json=api_payload) as resp:
                    if resp.status != 200:
                        err = await resp.text()
                        raise RuntimeError(f"ComfyUI Prompt Error {resp.status}: {err}")
                    resp_json = await resp.json()
                    prompt_id = resp_json['prompt_id']

                # Execution Loop
                current_node_id = None
                
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = msg.json()
                        if data['type'] == 'executing':
                            exec_data = data['data']
                            if exec_data['prompt_id'] == prompt_id:
                                # If node is None, execution is finished
                                if exec_data['node'] is None:
                                    break 
                                current_node_id = exec_data['node']

                    elif msg.type == aiohttp.WSMsgType.BINARY:
                        # --- NEW LOGIC ---
                        # Only process the image if it comes from a node we're watching
                        if current_node_id in self.websocket_save_nodes:
                            image_data = msg.data[8:] 
                            try:
                                image = Image.open(io.BytesIO(image_data))
                                image.load()
                                logger.debug(f"Identified FINAL image from save node {current_node_id}")
                                yield image
                            except Exception as e:
                                logger.error(f"Failed to decode streaming image: {e}")
                        else:
                            logger.debug(f"Ignoring preview image from node {current_node_id}")
                            
        except asyncio.CancelledError:
            # This is the key: When the Optimizer cancels us, we log it cleanly and exit.
            logger.debug("ComfyUI Generator was cancelled by the consumer. This is expected.")
        finally:
            # This ensures we don't leave the generator in a suspended state.
            pass

    def _enable_websocket_streaming(self, workflow: Dict):
        """
        Finds all 'SaveImage' and 'SaveImageWebsocket' nodes, converts the former,
        and records all their IDs for intelligent listening.
        """
        self.websocket_save_nodes = [] # Reset for this run
        for node_id, node in workflow.items():
            class_type = node.get("class_type")
            if class_type == "SaveImage":
                logger.debug(f"Swapping Node {node_id} (SaveImage) -> SaveImageWebsocket")
                node["class_type"] = "SaveImageWebsocket"
                if "filename_prefix" in node["inputs"]:
                    del node["inputs"]["filename_prefix"]
                self.websocket_save_nodes.append(node_id)
            elif class_type == "SaveImageWebsocket":
                self.websocket_save_nodes.append(node_id)
        
        if not self.websocket_save_nodes:
            logger.warning("No SaveImage or SaveImageWebsocket nodes found in workflow. May not receive any final images.")
        else:
            logger.debug(f"Listening for final images from nodes: {self.websocket_save_nodes}")

    def _inject_parameters(self, workflow: Dict, payload: Dict):
        """
        Crawls the ComfyUI graph to find nodes by class_type and injects values.
        """
        def find_nodes(class_list: List[str]) -> List[str]:
            return [nid for nid, node in workflow.items() if node.get("class_type") in class_list]

        # 1. Inject Model
        if self.current_model_filename:
            loaders = find_nodes(["CheckpointLoaderSimple", "CheckpointLoader"]) 
            for nid in loaders:
                workflow[nid]["inputs"]["ckpt_name"] = self.current_model_filename

        # 2. Inject Dimensions (Latent)
        latents = find_nodes(["EmptyLatentImage"])
        for nid in latents:
            if "width" in payload: workflow[nid]["inputs"]["width"] = payload["width"]
            if "height" in payload: workflow[nid]["inputs"]["height"] = payload["height"]
            workflow[nid]["inputs"]["batch_size"] = 1

        # 3. Inject Model Sampling (V-Pred / ZSNR) - NEW!
        sampling_nodes = find_nodes(["ModelSamplingDiscrete"])
        for nid in sampling_nodes:
            inputs = workflow[nid]["inputs"]
            # 'sampling' expects string: "v_prediction", "eps", "lcm", "x0"
            if "model_sampling" in payload: 
                inputs["sampling"] = payload["model_sampling"]
            
            # 'zsnr' expects boolean: true/false
            if "zsnr" in payload: 
                inputs["zsnr"] = payload["zsnr"]

        # 4. Inject Sampler Settings (Standard KSampler)
        # Note: If you use custom samplers (SamplerCustom), these won't be found, 
        # protecting your custom JSON settings from being overwritten.
        samplers = find_nodes(["KSampler", "KSamplerAdvanced"])
        if samplers:
            nid = samplers[0]
            inputs = workflow[nid]["inputs"]
            
            if "steps" in payload: inputs["steps"] = payload["steps"]
            if "cfg" in payload: inputs["cfg"] = payload["cfg"]
            if "sampler_name" in payload: inputs["sampler_name"] = payload["sampler_name"]
            if "scheduler" in payload: inputs["scheduler"] = payload["scheduler"]

            # Seed logic
            seed = payload.get("seed", -1)
            if seed == -1: seed = random.randint(0, 0xffffffffffffffff)
            
            if "noise_seed" in inputs: inputs["noise_seed"] = seed
            elif "seed" in inputs: inputs["seed"] = seed

            # 5. Trace Prompts from Sampler
            if "prompt" in payload and "positive" in inputs:
                self._inject_text(workflow, inputs["positive"], payload["prompt"])
            
            if "negative_prompt" in payload and "negative" in inputs:
                self._inject_text(workflow, inputs["negative"], payload["negative_prompt"])

    def _inject_text(self, workflow: Dict, link: List, text: str):
        """Helper to follow a link and inject text into the target node."""
        if not isinstance(link, list) or len(link) < 1: return
        node_id = link[0]
        if node_id not in workflow: return
        
        node = workflow[node_id]
        inputs = node["inputs"]
        
        # Heuristic for text input fields
        if "text" in inputs: inputs["text"] = text
        elif "string" in inputs: inputs["string"] = text
        elif "prompt" in inputs: inputs["prompt"] = text

# Add SwarmAdapter here if needed following the same pattern
