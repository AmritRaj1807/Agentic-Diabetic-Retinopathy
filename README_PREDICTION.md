# Single-Image Prediction

This project now includes a standalone inference script:

- `predict.py`

It does not retrain the model and does not change `Train.py`, `Test.py`, or the trained architecture.

## Default Checkpoint

By default, `predict.py` loads:

```powershell
outputs\exp_parallel_effb4_swinb384_corn_fusion_head\models\model_epoch_015_qwk_0.8943.pth
```

The path is resolved relative to the project root, so the project folder can be moved without hard-coding a Windows username.

## Run A Prediction

From the project root:

```powershell
cd "D:\Projects\Diabetic Retinopathy\Diabetic-Retinopathy-Grading"
python predict.py --image "D:\Projects\Diabetic Retinopathy\Diabetic-Retinopathy-Grading\sample.jpg"
```

To use a custom checkpoint:

```powershell
python predict.py --image "sample.jpg" --checkpoint "outputs\exp_parallel_effb4_swinb384_corn_fusion_head\models\model_epoch_015_qwk_0.8943.pth"
```

To predict multiple images:

```powershell
python predict.py --image "sample1.jpg" "sample2.jpg" "sample3.jpg"
```

## Preprocessing Behavior

By default, `predict.py` treats the input as a raw fundus image and reuses the existing `Preprocessing.py` pipeline:

- automatic fundus crop
- square padding
- resize to `384 x 384`
- CLAHE enhancement

It then applies the same deterministic tensor transform used by `Test.py`:

- `Resize((384, 384))`
- `ToTensor()`
- ImageNet normalization

If the image has already been processed by `Preprocessing.py`, add:

```powershell
python predict.py --image "dataset\processed_validation\some_image.jpg" --input-preprocessed
```

## Python API

A future Streamlit, FastAPI, or Flask app can import the prediction function:

```python
from predict import load_inference_bundle, predict_image

bundle = load_inference_bundle()
result = predict_image("sample.jpg", bundle=bundle)

print(result["predicted_class"])
print(result["predicted_label"])
print(result["confidence"])
print(result["class_scores"])
```

`load_inference_bundle()` loads the checkpoint once. Reuse the bundle for web apps so the model is not reloaded for every request.

## Safety Note

This is for research/education use only. The model output is a screening prediction and is not a substitute for professional medical evaluation.
