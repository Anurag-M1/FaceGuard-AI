# FaceGuard‑AI

Flask‑based web app for detecting facial/skin anomalies and returning a matched condition label with supplement info. Upload an image, run the model, and get a result page with the predicted condition.

## Features
- Image upload + instant inference
- Condition label mapped from model output
- Result page with supplement info from CSV
- Simple Flask frontend

## Project structure
```
FaceGuard-AI/
├─ app.py
├─ predict.py
├─ data_files/
│  └─ supplement_info.csv
├─ trained_model/        # expected at runtime (see below)
│  ├─ best_model.h5
│  └─ datafile.json
├─ static/
├─ templates/
└─ model/                # training notebook
```

## Model files (required)
`predict.py` expects a folder named `trained_model/` with:
- `best_model.h5`
- `datafile.json` (label mapping)

If your model files are stored elsewhere, update this line in `predict.py`:
```python
baseDir = os.path.join(os.getcwd(), 'trained_model')
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `requirements.txt` is missing, install the basics:
```bash
python -m pip install flask keras tensorflow pillow numpy pandas
```

## Run
```bash
python app.py
```
Open http://localhost:5000
live https://faceguard-ai-gg8d.onrender.com

## API
### POST `/analyze`
Form‑data field: `file`

Example:
```bash
curl -X POST http://localhost:5000/analyze   -F "file=@/path/to/image.jpg"
```

Response:
```json
{ "product_id": 12 }
```

Then open:
```
/result?id=12
```

## Notes
- Uploaded images are saved temporarily to `images/` and deleted after inference.
- `supplement_info.csv` drives the label + supplement metadata used by the result page.

## Author
Anurag Kumar Singh
