from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

import shutil
import uuid
import os

app = FastAPI()


# agregar CORS

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# se Carga modelo YOLO

model = YOLO("models/best.pt")


# Ruta principal

@app.get("/")
def home():
    return {"message": "API IA funcionando"}


# Predicción IA

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Crear carpeta uploads
    os.makedirs("uploads", exist_ok=True)

    # Obtener extensión
    extension = file.filename.split(".")[-1]

    # Nombre único
    filename = f"{uuid.uuid4()}.{extension}"

    # Ruta archivo
    file_path = f"uploads/{filename}"

    # Guardar imagen
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Predicción YOLO
    results = model(file_path)

    detections = []

    # Extraer resultados
    for r in results:
        for box in r.boxes:

            clase_id = int(box.cls[0])

            confidence = float(box.conf[0])

            class_name = model.names[clase_id]

            detections.append({
                "class": class_name,
                "confidence": confidence
            })

    return {
        "success": True,
        "detections": detections
    }
    