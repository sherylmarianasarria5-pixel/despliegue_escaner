from fastapi import FastAPI, UploadFile, File, Form
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

model = YOLO("models/best.pt")

UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# =========================
# VARIEDADES VÁLIDAS
# =========================

VARIEDADES_VALIDAS = [
    "Caturra",
    "Castillo",
    "Colombia",
    "Borbón"
]

# =========================
# TRATAMIENTOS DE ROYA POR VARIEDAD
# =========================

TRATAMIENTOS_ROYA = {

    "Caturra": [
        "Aplicar fungicida sistémico a base de triazoles (ciproconazol)",
        "Caturra es altamente susceptible a la roya: priorizar control preventivo",
        "Realizar podas de renovación para reducir densidad foliar",
        "Aumentar frecuencia de monitoreo a cada 8-10 días",
        "Contacta a un profesional agrónomo"
    ],

    "Castillo": [
        "Aplicar fungicida a base de cobre (oxicloruro de cobre)",
        "Castillo tiene resistencia genética moderada: reforzar solo en focos activos",
        "Eliminar hojas con pústulas esporuladas para reducir propagación",
        "Mantener monitoreo cada 15 días",
        "Contacta a un profesional agrónomo"
    ],

    "Colombia": [
        "Aplicar fungicida preventivo a base de cobre en bajas dosis",
        "Variedad Colombia es resistente: enfocar en manejo cultural antes que químico",
        "Mejorar ventilación de las plantas reduciendo sombra excesiva",
        "Monitorear cada 20 días dado su nivel de resistencia",
        "Contacta a un profesional agrónomo"
    ],

    "Borbón": [
        "Aplicar fungicida sistémico a base de triazoles de forma inmediata",
        "Borbón es muy susceptible a la roya: actuar con tratamiento curativo urgente",
        "Eliminar y destruir hojas severamente afectadas fuera del cultivo",
        "Aumentar frecuencia de monitoreo a cada 7 días",
        "Contacta a un profesional agrónomo"
    ],

}

# Recomendaciones genéricas si no se especifica variedad o no es reconocida
TRATAMIENTO_ROYA_GENERICO = [
    "Aplicar tratamiento antifúngico",
    "Monitorear hojas cercanas",
    "Eliminar hojas muy afectadas",
    "Reducir exceso de humedad",
    "Contacta a un profesional"
]

# =========================
# ENDPOINT PRINCIPAL
# =========================

@app.get("/")
def home():

    return {
        "message": "API IA funcionando"
    }

# =========================
# ENDPOINT PREDICT
# =========================

@app.post("/predict")
async def predict(file: UploadFile = File(...), variedad: str = Form(None)):

    try:

        # =========================
        # NORMALIZAR VARIEDAD RECIBIDA
        # =========================

        variedad_normalizada = None

        if variedad:

            for v in VARIEDADES_VALIDAS:
                if v.lower() == variedad.strip().lower():
                    variedad_normalizada = v
                    break

        # =========================
        # GUARDAR IMAGEN
        # =========================

        file_path = os.path.join(
            UPLOAD_DIR,
            f"{uuid.uuid4()}.jpg"
        )

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # =========================
        # PREDICCIÓN YOLO
        # =========================

        results = model(file_path, conf=0.40)

        detections = []

        # =========================
        # CLASES VÁLIDAS
        # =========================

        valid_classes = [
            "Hoja_Sana",
            "Enfermedad_ROYA",
            "arbol_cafe"
        ]

        # =========================
        # NO DETECTA NADA
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

            # =========================
            # IGNORAR CLASES INVÁLIDAS
            # =========================

            if class_name not in valid_classes:
                continue

            # =========================
            # VARIABLES
            # =========================

            recommendations = []

            message = ""

            # =========================
            # ENFERMEDAD ROYA
            # =========================

            if class_name == "Enfermedad_ROYA":

                if variedad_normalizada:

                    message = f"Se detectó roya en la hoja de café (variedad {variedad_normalizada})."

                    recommendations = TRATAMIENTOS_ROYA[variedad_normalizada]

                else:

                    message = "Se detectó roya en la hoja de café."

                    recommendations = TRATAMIENTO_ROYA_GENERICO

            # =========================
            # HOJA SANA
            # =========================

            elif class_name == "Hoja_Sana":

                message = "La hoja analizada se encuentra saludable."

                recommendations = [
                    "Mantener monitoreo constante",
                    "Continuar buenas prácticas agrícolas",
                    "Revisar hojas semanalmente"
                ]

            # =========================
            # ÁRBOL CAFÉ
            # =========================

            elif class_name == "arbol_cafe":

                message = "Se detectó un árbol de café."

                recommendations = [
                    "Verificar estado de hojas",
                    "Controlar humedad del cultivo",
                    "Realizar inspecciones periódicas"
                ]

            # =========================
            # AGREGAR DETECCIÓN
            # =========================

            detections.append({
                "class": class_name,
                "confidence": conf,
                "message": message,
                "recommendations": recommendations,
                "variedad": variedad_normalizada
            })

        # =========================
        # VALIDAR SI NO ES CAFÉ
        # =========================

        if len(detections) == 0:

            return {
                "detections": [],
                "message": "Imagen no válida. Carga otra imagen de hoja de café."
            }

        # =========================
        # RESPUESTA FINAL
        # =========================

        return {
            "detections": detections,
            "message": "Análisis completado"
        }

    except Exception as e:

        return {
            "detections": [],
            "message": "Error analizando imagen",
            "error": str(e)
        }
        