from __future__ import annotations

import tempfile
from pathlib import Path

from .sequence_engine import Observation, Step


def analyse_video_zero_shot(video_bytes: bytes, steps: list[Step], sample_every_seconds: float = 1.0):
    """Classify sampled video frames locally with an open-source vision-language model.

    This optional adapter downloads model weights on first use. Later runs can be
    fully local when the weights are cached. It is deliberately isolated from the
    deterministic sequence engine so the model can later be replaced by a
    Snapdragon-optimised ONNX/QNN action-recognition model.
    """
    try:
        import cv2
        import torch
        from PIL import Image
        from transformers import AutoProcessor, CLIPModel
    except ImportError as exc:
        raise RuntimeError(
            "Video AI dependencies are missing. Install requirements-ai.txt or use Demo mode."
        ) from exc

    if not video_bytes:
        raise ValueError("No video data was provided.")

    model_id = "openai/clip-vit-base-patch32"
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id, use_safetensors=True)
        model.eval()
    except Exception as exc:
        raise RuntimeError(
            "CLIP could not be loaded. Close StepSense, update transformers, "
            "safetensors and huggingface-hub, delete the incomplete CLIP cache, "
            "then restart the app. Original error: " + str(exc)
        ) from exc
    prompts = [step.prompt or f"a person performing {step.name}" for step in steps]
    prompt_to_id = dict(zip(prompts, [step.id for step in steps]))

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
        handle.write(video_bytes)
        temp_path = Path(handle.name)

    capture = cv2.VideoCapture(str(temp_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(fps * sample_every_seconds))
    detections = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
                with torch.inference_mode():
                    outputs = model(**inputs)
                    scores = outputs.logits_per_image.softmax(dim=1)[0]
                best_index = int(scores.argmax().item())
                detections.append({
                    "step_id": prompt_to_id[prompts[best_index]],
                    "time": frame_index / fps,
                    "confidence": float(scores[best_index].item()),
                })
            frame_index += 1
    finally:
        capture.release()
        temp_path.unlink(missing_ok=True)

    observations = []
    for item in detections:
        if observations and observations[-1].step_id == item["step_id"]:
            previous = observations[-1]
            observations[-1] = Observation(
                step_id=previous.step_id,
                start_seconds=previous.start_seconds,
                end_seconds=item["time"] + sample_every_seconds,
                confidence=max(previous.confidence, item["confidence"]),
            )
        else:
            observations.append(Observation(
                step_id=item["step_id"],
                start_seconds=item["time"],
                end_seconds=item["time"] + sample_every_seconds,
                confidence=item["confidence"],
            ))
    return observations
