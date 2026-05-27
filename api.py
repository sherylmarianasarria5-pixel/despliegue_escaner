from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import uuid
import os

app = FastAPI()

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# CARGAR MODELO YOLO
# =========================
model = YOLO("models/best.pt")  # 👈 asegúrate que sea tu modelo entrenado

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# =========================
# ENDPOINT PREDICT
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # guardar imagen
    file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.jpg")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # =========================
    # PREDICCIÓN
    # =========================
    results = model(file_path, conf=0.70)

    detections = []

    # clases válidas del sistema cafetalero
    valid_classes = ["Hoja_Sana", "Enfermedad_ROYA", "arbol_cafe"]

    # =========================
    # CASO 1: no detecta nada
    # =========================
    if len(results[0].boxes) == 0:
        return {
            "detections": [],
            "message": "Imagen no válida. Carga otra imagen de hoja de café."
        }

    # =========================
    # PROCESAR DETECCIONES
    # =========================
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]

        detections.append({
            "class": class_name,
            "confidence": conf
        })

    # =========================
    # CASO 2: no es café (fuera de dominio)
    # =========================
    is_cafe = any(d["class"] in valid_classes for d in detections)

    if not is_cafe:
        return {
            "detections": [],
            "message": "Imagen no válida. Carga otra imagen de hoja de café."
        }

    # =========================
    # RESPUESTA FINAL
    # =========================
    return {
        "detections": detections
    }
    
    