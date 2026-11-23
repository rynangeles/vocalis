"""JSON form of inference parameters stored on a generation row."""

from __future__ import annotations

import json

from vocalis.core.exceptions import PersistenceError
from vocalis.engine.types import InferenceParameters


def dump_parameters(parameters: InferenceParameters) -> str:
    payload = {
        "cfg_value": parameters.cfg_value,
        "inference_timesteps": parameters.inference_timesteps,
        "normalize": parameters.normalize,
        "denoise": parameters.denoise,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def load_parameters(raw: str) -> InferenceParameters:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PersistenceError("Stored inference parameters are not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise PersistenceError("Stored inference parameters are not valid JSON.")
    try:
        cfg_value = payload["cfg_value"]
        timesteps = payload["inference_timesteps"]
        normalize = payload["normalize"]
        denoise = payload["denoise"]
    except KeyError as exc:
        raise PersistenceError("Stored inference parameters are incomplete.") from exc
    if isinstance(cfg_value, bool) or not isinstance(cfg_value, (int, float)):
        raise PersistenceError("Stored inference parameters are incomplete.")
    if isinstance(timesteps, bool) or not isinstance(timesteps, int):
        raise PersistenceError("Stored inference parameters are incomplete.")
    if not isinstance(normalize, bool) or not isinstance(denoise, bool):
        raise PersistenceError("Stored inference parameters are incomplete.")
    return InferenceParameters(
        cfg_value=float(cfg_value),
        inference_timesteps=timesteps,
        normalize=normalize,
        denoise=denoise,
    )
