# StepSense AI

StepSense AI is an on-device practical-skills assessment prototype for Snapdragon-powered PCs. It evaluates the order and completeness of a practical procedure and produces an explainable, step-level report.

## Current MVP

The included demonstration uses a five-step laboratory safety preparation routine:

1. Sanitise hands
2. Wear laboratory coat
3. Wear face mask
4. Wear safety goggles
5. Wear protective gloves

### Why this procedure was selected

Laboratory safety preparation was selected as the first MVP demonstration because its actions are visible through an ordinary laptop camera, follow a clear expected order and can be recorded safely with easily available objects. The five actions are also different enough to test whether the system can distinguish steps, recognise omissions and detect sequence errors.

Hand sanitising is included as the opening step because it provides an observable start to the procedure and makes a missed-first-step error easy to demonstrate. The coat, mask, goggles and gloves then create a natural sequence with distinct visual cues. This procedure is used only to prove the StepSense workflow; the product is not restricted to laboratory safety or PPE assessment.

The primary product domain remains education and vocational training. A trainer can later replace this sample procedure with another structured practical task by editing the procedure configuration and supplying suitable training examples.

The working sequence engine detects correct completion, missing steps, wrong order, repeated steps, excessive duration and low-confidence observations. Four pre-labelled demo attempts are included so the complete interface can be tested without model downloads.

An experimental local video adapter samples frames and uses an open-source CLIP model to classify actions. It is a prototype integration, not a validated safety system. The final competition version should replace it with a fine-tuned lightweight action-recognition model exported to ONNX/QNN and profiled on the target Snapdragon PC.

## Run the reliable demo

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown by Streamlit, select **Run assessment**, choose a demo attempt and generate the report.

## Enable experimental video analysis

```bash
pip install -r requirements-ai.txt
streamlit run app.py
```

The open-source model weights are downloaded on first use. Once cached, video processing can run locally. The adapter is deliberately separate from the sequence engine so it can be replaced with a Snapdragon-optimised model.

### If the CLIP model download is incomplete

Stop the application, update the model-loading packages and remove the incomplete cached copy before restarting:

```cmd
python -m pip install --upgrade "transformers<5" safetensors huggingface-hub hf-xet
rmdir /s /q "%USERPROFILE%\.cache\huggingface\hub\models--openai--clip-vit-base-patch32"
python -m streamlit run app.py
```

If the cache folder does not exist, Windows will display a harmless “system cannot find the file” message. The model download is approximately 600 MB and must finish completely during the next analysis attempt.

## Test

```bash
pip install pytest
pytest -q
```

For a dependency-light smoke test:

```bash
python scripts/smoke_test.py
```

## Project structure

```text
StepSense_AI/
├── app.py
├── core/
│   ├── io.py
│   ├── sequence_engine.py
│   └── video_ai.py
├── data/
│   ├── demo_attempts/
│   └── procedures/
├── tests/
├── requirements.txt
└── requirements-ai.txt
```

## Snapdragon optimisation plan

1. Fine-tune a lightweight action-recognition model on recorded procedure clips.
2. Export the model to ONNX with a fixed input shape and batch size one.
3. Quantise the model where accuracy permits.
4. Convert or profile it using Qualcomm-compatible deployment tools.
5. Measure model size, latency, throughput, memory use and power behaviour on the target Snapdragon PC.
6. Replace `core/video_ai.py` while keeping the procedure graph and report layers unchanged.

## Responsible use

StepSense supports trainers; it does not replace qualified supervision in high-risk environments. The current MVP is not medically validated and does not provide diagnosis or safety certification. Low-confidence observations should always be reviewed by a person.
