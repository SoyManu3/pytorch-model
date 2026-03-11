from pathlib import Path
from importlib import import_module

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    torch = import_module("torch")
    nn = import_module("torch.nn")
except ImportError:
    torch = None
    nn = None


if nn is not None:
    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(3, 2), nn.Sigmoid())
            self.decoder = nn.Sequential(nn.Linear(2, 3), nn.Sigmoid())

        def forward(self, x):
            z = self.encoder(x)
            x_hat = self.decoder(z)
            return x_hat
else:
    Autoencoder = None


class InputData(BaseModel):
    values: list[float] = Field(..., min_length=3, max_length=3)


app = FastAPI(title="Autoencoder API")

MODEL_PATH = Path(__file__).with_name("autoencoder.pth")
model = Autoencoder() if Autoencoder is not None else None
model_loaded = False


@app.on_event("startup")
def load_model() -> None:
    global model_loaded
    if torch is None or model is None:
        model_loaded = False
        return

    if not MODEL_PATH.exists():
        model_loaded = False
        return

    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    model_loaded = True


@app.get("/")
def root():
    return {"status": "ok", "model_loaded": model_loaded}


@app.post("/predict")
def predict(payload: InputData):
    if torch is None:
        raise HTTPException(status_code=500, detail="Dependencia faltante: instala torch")

    if not model_loaded:
        raise HTTPException(status_code=500, detail="Modelo no cargado: falta autoencoder.pth")

    x = torch.tensor([payload.values], dtype=torch.float32)
    with torch.no_grad():
        reconstructed = model(x).squeeze(0).tolist()

    return {"input": payload.values, "reconstruction": reconstructed}