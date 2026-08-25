import io
import os

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

from model import get_model


app = FastAPI()

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    ),
])


def get_checkpoint_path():
    path = os.environ.get("CHECKPOINT_PATH")
    if path is None:
        path = "/app/checkpoints/classifier_v1.pt"
    return path


def load_model():
    path = get_checkpoint_path()
    m = get_model("resnet18", 10)
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint["model_state_dict"]
    m.load_state_dict(state)
    m.eval()
    return m


try:
    model = load_model()
except Exception as e:
    print("could not load model:", e, flush=True)
    model = None


@app.get("/health")
def health():
    if model is None:
        return JSONResponse(status_code=503, content={"status": "model not loaded"})
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=503, content={"error": "model not loaded"})

    raw = await image.read()
    img = Image.open(io.BytesIO(raw))
    img = img.convert("RGB")

    x = transform(img)
    x = x.unsqueeze(0)

    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)

    probs = probs[0]

    result = {}
    for i in range(len(class_names)):
        name = class_names[i]
        value = probs[i].item()
        result[name] = round(value, 4)

    best_index = int(torch.argmax(probs))
    best_name = class_names[best_index]

    return {"predicted_class": best_name, "probabilities": result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
