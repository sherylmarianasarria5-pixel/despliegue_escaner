from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

import shutil
import uuid
import os

# =========================
# FASTAPI
# =========================

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

model = YOLO("models/best.pt")

# =========================
# RUTA PRINCIPAL
# =========================

@app.get("/")
def home():

    return {
        "message": "API IA funcionando"
    }

# =========================
# PREDICCIÓN IA
# =========================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        # =========================
        # CREAR CARPETA UPLOADS
        # =========================

        os.makedirs("uploads", exist_ok=True)

        # =========================
        # EXTENSIÓN ARCHIVO
        # =========================

        extension = file.filename.split(".")[-1]

        # =========================
        # NOMBRE ÚNICO
        # =========================

        filename = f"{uuid.uuid4()}.{extension}"

        # =========================
        # RUTA ARCHIVO
        # =========================

        file_path = f"uploads/{filename}"

        # =========================
        # GUARDAR IMAGEN
        # =========================

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # =========================
        # PREDICCIÓN YOLO
        # =========================

        results = model(file_path)

        detections = []

        # =========================
        # EXTRAER RESULTADOS
        # =========================

        for r in results:

            for box in r.boxes:

                clase_id = int(box.cls[0])

                confidence = float(box.conf[0])

                class_name = model.names[clase_id]

                # =========================
                # RECOMENDACIONES
                # =========================

                recommendations = []

                message = ""

                # =========================
                # ENFERMEDAD ROYA
                # =========================

                if class_name == "Enfermedad_ROYA":

                    message = "Se detectó roya en la hoja de café"

                    recommendations = [
                        "Aplicar tratamiento antifúngico",
                        "Monitorear hojas cercanas",
                        "Eliminar hojas muy afectadas",
                        "Reducir exceso de humedad"
                    ]

                # =========================
                # HOJA SANA
                # =========================

                elif class_name == "Hoja_Sana":

                    message = "La hoja analizada se encuentra saludable"

                    recommendations = [
                        "Mantener monitoreo constante",
                        "Continuar buenas prácticas agrícolas",
                        "Revisar hojas semanalmente"
                    ]

                # =========================
                # ÁRBOL CAFÉ
                # =========================

                elif class_name == "arbol_cafe":

                    message = "Se detectó un árbol de café"

                    recommendations = [
                        "Verificar estado de hojas",
                        "Controlar humedad del cultivo",
                        "Realizar inspecciones periódicas"
                    ]

                # =========================
                # DETECCIÓN
                # =========================

                detections.append({
                    "class": class_name,
                    "confidence": confidence,
                    "message": message,
                    "recommendations": recommendations
                })

        # =========================
        # VALIDAR SI NO DETECTA
        # =========================

        if len(detections) == 0:

            return {
                "success": False,
                "message": "Esto no es una hoja de café",
                "detections": []
            }

        # =========================
        # RESPUESTA EXITOSA
        # =========================

        return {
            "success": True,
            "message": "Análisis completado",
            "detections": detections
        }

    except Exception as e:

        return {
            "success": False,
            "message": "Error analizando imagen",
            "error": str(e)
        }