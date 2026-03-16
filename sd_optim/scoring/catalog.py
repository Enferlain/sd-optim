"""Metadata for scorer implementations shipped with sd-optim."""

from __future__ import annotations

SCORER_CLASS_PATHS: dict[str, tuple[str, str]] = {
    "laion": ("sd_optim.extensions.bundled.scorers.models.Laion", "Laion"),
    "chad": ("sd_optim.extensions.bundled.scorers.models.Laion", "Laion"),
    "clip": ("sd_optim.extensions.bundled.scorers.models.CLIPScore", "CLIPScore"),
    "pick": ("sd_optim.extensions.bundled.scorers.models.PickScore", "PickScore"),
    "wdaes": ("sd_optim.extensions.bundled.scorers.models.WDAes", "WDAes"),
    "shadowv2": ("sd_optim.extensions.bundled.scorers.models.ShadowScore", "ShadowScore"),
    "cafe": ("sd_optim.extensions.bundled.scorers.models.CafeScore", "CafeScore"),
    "noai": ("sd_optim.extensions.bundled.scorers.models.NoAIScore", "NoAIScore"),
    "cityaes": (
        "sd_optim.extensions.bundled.scorers.models.CityAesthetics",
        "CityAestheticsScorer",
    ),
    "aestheticv25": ("sd_optim.extensions.bundled.scorers.models.AestheticV25", "AestheticV25"),
    "luminaflex": (
        "sd_optim.extensions.bundled.scorers.models.LumiAnatomyv2",
        "Dinov3AnatomyScorer",
    ),
    "lumidinov3": (
        "sd_optim.extensions.bundled.scorers.models.LumiAnatomyv2",
        "Dinov3AnatomyScorer",
    ),
    "lumidinov2l": (
        "sd_optim.extensions.bundled.scorers.models.LumiAnatomyv2",
        "Dinov3AnatomyScorer",
    ),
    "lumidinov2g": (
        "sd_optim.extensions.bundled.scorers.models.LumiAnatomyv2",
        "Dinov3AnatomyScorer",
    ),
    "simplequality": (
        "sd_optim.extensions.bundled.scorers.models.SimpleQuality",
        "SimpleQualityScorer",
    ),
    "hybridnoise": (
        "sd_optim.extensions.bundled.scorers.models.HybridNoiseScorer",
        "HybridNoiseScorer",
    ),
    "hybridnoise_fullimg": (
        "sd_optim.extensions.bundled.scorers.models.HybridNoiseScorer",
        "HybridNoiseFullImageScorer",
    ),
    "backgroundblackness": (
        "sd_optim.extensions.bundled.scorers.models.BackgroundBlacknessScorer",
        "BackgroundBlacknessScorer",
    ),
    "pcascorer": ("sd_optim.extensions.bundled.scorers.models.PCAScorer", "PCAScorer"),
    "textureclean": ("sd_optim.extensions.bundled.scorers.models.TextureScorer", "TextureScorer"),
    "textureclean_fullimg": (
        "sd_optim.extensions.bundled.scorers.models.TextureScorer",
        "TextureScorerFullImage",
    ),
}

MODEL_DATA: dict[str, dict[str, str | None]] = {
    "laion": {
        "url": "https://github.com/grexzen/SD-Chad/blob/main/sac+logos+ava1-l14-linearMSE.pth?raw=true",
        "file_name": "Laion.pth",
    },
    "chad": {
        "url": "https://github.com/grexzen/SD-Chad/blob/main/chadscorer.pth?raw=true",
        "file_name": "Chad.pth",
    },
    "wdaes": {
        "url": "https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/models/aes-B32-v0.pth?download=true",
        "file_name": "WD_Aes.pth",
    },
    "imagereward": {
        "url": "https://huggingface.co/THUDM/ImageReward/resolve/main/ImageReward.pt?download=true",
        "file_name": "ImageReward.pt",
    },
    "clip": {
        "url": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt?raw=true",
        "file_name": "CLIP-ViT-L-14.pt",
    },
    "blip": {
        "url": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth?raw=true",
        "file_name": "BLIP_Large.safetensors",
    },
    "hpsv21": {
        "url": "https://huggingface.co/xswu/HPSv2/resolve/main/HPS_v2.1_compressed.pt?download=true",
        "file_name": "HPS_v2.1.pt",
    },
    "hpsv3": {
        "url": "https://huggingface.co/MizzenAI/HPSv3/resolve/main/HPSv3.safetensors?download=true",
        "file_name": "HPSv3.safetensors",
    },
    "pick": {
        "url": "https://huggingface.co/yuvalkirstain/PickScore_v1/resolve/main/model.safetensors?download=true",
        "file_name": "Pick-A-Pic.safetensors",
    },
    "shadowv2": {
        "url": "https://huggingface.co/shadowlilac/aesthetic-shadow-v2/resolve/main/model.safetensors?download=true",
        "file_name": "ShadowV2.safetensors",
    },
    "cafe": {
        "url": "https://huggingface.co/cafeai/cafe_aesthetic/resolve/3bca27c5c0b6021056b1e84e5a18cf1db9fe5d4c/model.safetensors?download=true",
        "file_name": "Cafe.safetensors",
    },
    "class": {
        "url": "https://huggingface.co/cafeai/cafe_style/resolve/d5ae1a7ac05a12ab84732c25f2ea7225d35ac81b/model.safetensors?download=true",
        "file_name": "CLASS.safetensors",
    },
    "real": {
        "url": "https://huggingface.co/Sumsub/Sumsub-ffs-synthetic-2.0/resolve/main/synthetic.pt?download=true",
        "file_name": "REAL.pt",
    },
    "anime": {
        "url": "https://huggingface.co/saltacc/anime-ai-detect/resolve/e175bb6b5e19cda40bc6c9ad85b138ee7c7ce23a/model.safetensors?download=true",
        "file_name": "ANIME.safetensors",
    },
    "cityaes": {
        "url": "https://huggingface.co/city96/CityAesthetics/resolve/main/CityAesthetics-Anime-v1.8.safetensors?download=true",
        "file_name": "CityAesthetics-Anime-v1.8.safetensors",
    },
    "aestheticv25": {
        "url": None,
        "file_name": "aesthetic-predictor-v2-5",
    },
    "luminaflex": {
        "url": None,
        "file_name": "AnatomyFlaws-v14.7_adabelief_fl_naflex_4670_s1K.safetensors",
        "config_name": "AnatomyFlaws-v14.7_adabelief_fl_naflex_4670.config.json",
    },
    "lumidinov3": {
        "url": None,
        "file_name": "AnatomyFlaws-v15.5_dinov3_7b_bnb_fl_s3K_best_val.safetensors",
        "config_name": "AnatomyFlaws-v15.5_dinov3_7b_bnb_fl.config.json",
    },
    "lumidinov2l": {
        "url": None,
        "file_name": "AnatomyFlaws-v6.3_adabeleif_fl_sigmoid_dinov2_large_efinal_s10K_final.safetensors",
        "config_name": "AnatomyFlaws-v6.3_adabeleif_fl_sigmoid_dinov2_large.config.json",
    },
    "lumidinov2g": {
        "url": None,
        "file_name": "AnatomyFlaws-v6.4_adabeleif_fl_sigmoid_dinov2_giant_efinal_s10K_final.safetensors",
        "config_name": "AnatomyFlaws-v6.4_adabeleif_fl_sigmoid_dinov2_giant.config.json",
    },
    "simplequality": {"url": None, "file_name": None},
    "hybridnoise": {"url": None, "file_name": None},
    "hybridnoise_fullimg": {"url": None, "file_name": None},
    "backgroundblackness": {"url": None, "file_name": None},
    "pcascorer": {"url": None, "file_name": None},
    "textureclean": {"url": None, "file_name": None},
    "textureclean_fullimg": {"url": None, "file_name": None},
}

__all__ = ["MODEL_DATA", "SCORER_CLASS_PATHS"]
