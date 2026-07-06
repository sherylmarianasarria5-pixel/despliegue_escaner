from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import uuid
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("models/best.pt")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

VARIEDADES_VALIDAS = ["Caturra", "Castillo", "Colombia", "Borbon"]

SEVERIDAD_BAJA  = (0.40, 0.60)
SEVERIDAD_MEDIA = (0.60, 0.80)

def obtener_severidad(confidence: float) -> str:
    if SEVERIDAD_BAJA[0] <= confidence < SEVERIDAD_BAJA[1]:
        return "Baja"
    elif SEVERIDAD_MEDIA[0] <= confidence < SEVERIDAD_MEDIA[1]:
        return "Media"
    return "Alta"

TRATAMIENTOS_ROYA = {
    "Caturra": {
        "Baja": [
            "Aplicar fungicida preventivo a base de cobre (oxicloruro de cobre)",
            "Caturra es altamente susceptible: iniciar control preventivo de inmediato",
            "Realizar podas de ventilacion para reducir humedad foliar",
            "Monitorear cada 10 dias para detectar avance",
            "Contacta a un profesional agronomo"
        ],
        "Media": [
            "Aplicar fungicida sistemico a base de triazoles (ciproconazol)",
            "Eliminar hojas con pustulas visibles para reducir propagacion",
            "Caturra es altamente susceptible: reforzar monitoreo cada 7 dias",
            "Asegurar cobertura completa del follaje en la aplicacion",
            "Contacta a un profesional agronomo"
        ],
        "Alta": [
            "Aplicar fungicida sistemico + cobre de forma combinada e inmediata",
            "Caturra es muy susceptible: esta severidad requiere accion urgente",
            "Eliminar y destruir hojas severamente afectadas fuera del lote",
            "Realizar una segunda aplicacion a los 10-14 dias",
            "Contacta a un profesional agronomo urgentemente"
        ]
    },
    "Castillo": {
        "Baja": [
            "Aplicar fungicida preventivo a base de cobre en dosis moderadas",
            "Castillo tiene resistencia genetica moderada: el control preventivo es suficiente",
            "Monitorear cada 15 dias para verificar evolucion",
            "Mantener cobertura vegetal equilibrada",
            "Contacta a un profesional agronomo"
        ],
        "Media": [
            "Aplicar fungicida a base de cobre + triazol en dosis curativa",
            "Aunque Castillo tiene resistencia, la infeccion esta activa: actuar con oportunidad",
            "Eliminar hojas con pustulas esporuladas manualmente",
            "Monitorear cada 10 dias hasta controlar el foco",
            "Contacta a un profesional agronomo"
        ],
        "Alta": [
            "Aplicar fungicida sistemico de accion triazol + cobre de forma inmediata",
            "La resistencia de Castillo no es suficiente para este nivel de infeccion",
            "Eliminar y destruir hojas severamente afectadas fuera del lote",
            "Realizar monitoreo cada 7 dias y programar segunda aplicacion",
            "Contacta a un profesional agronomo urgentemente"
        ]
    },
    "Colombia": {
        "Baja": [
            "Aplicar fungicida preventivo a base de cobre en bajas dosis",
            "Variedad Colombia es resistente: manejo cultural es prioridad",
            "Mejorar ventilacion reduciendo sombra excesiva",
            "Monitorear cada 20 dias dado su buen nivel de resistencia",
            "Contacta a un profesional agronomo"
        ],
        "Media": [
            "Aplicar fungicida a base de cobre en dosis media como refuerzo",
            "Variedad Colombia tiene resistencia, pero requiere apoyo quimico ante avance",
            "Podar ramas bajas para mejorar circulacion de aire",
            "Monitorear cada 15 dias para asegurar control",
            "Contacta a un profesional agronomo"
        ],
        "Alta": [
            "Aplicar fungicida sistemico a base de triazoles de forma inmediata",
            "La resistencia de Colombia esta siendo superada: actuar con firmeza",
            "Eliminar hojas severamente afectadas y destruirlas fuera del lote",
            "Evaluar posible origen de inoculo externo cercano",
            "Contacta a un profesional agronomo urgentemente"
        ]
    },
    "Borbon": {
        "Baja": [
            "Aplicar fungicida preventivo a base de cobre + triazol en dosis baja",
            "Borbon es muy susceptible a la roya: no bajar la guardia aunque sea severidad baja",
            "Eliminar hojas con primeros sintomas visibles",
            "Monitorear cada 7-8 dias por su alta susceptibilidad",
            "Contacta a un profesional agronomo"
        ],
        "Media": [
            "Aplicar fungicida sistemico a base de triazoles de forma curativa",
            "Borbon es muy susceptible: actuar antes de que avance a severidad alta",
            "Eliminar y destruir hojas afectadas fuera del cultivo",
            "Monitorear cada 5-7 dias con atencion a hojas nuevas",
            "Contacta a un profesional agronomo"
        ],
        "Alta": [
            "Aplicar fungicida sistemico triazol + cobre de forma combinada e inmediata",
            "Borbon es extremadamente susceptible: esta severidad pone en riesgo el lote",
            "Eliminar y destruir hojas severamente afectadas fuera del lote",
            "Programar aplicaciones cada 7-10 dias y evaluar renovacion si persiste",
            "Contacta a un profesional agronomo URGENTEMENTE"
        ]
    }
}

TRATAMIENTO_ROYA_GENERICO = {
    "Baja": [
        "Aplicar fungicida preventivo a base de cobre",
        "Monitorear hojas cercanas cada 15 dias",
        "Eliminar hojas con sintomas iniciales",
        "Reducir exceso de humedad en el cultivo",
        "Contacta a un profesional agronomo"
    ],
    "Media": [
        "Aplicar fungicida sistemico a base de triazoles",
        "Eliminar y destruir hojas afectadas fuera del lote",
        "Monitorear cada 10 dias para evitar propagacion",
        "Asegurar cobertura completa en la aplicacion",
        "Contacta a un profesional agronomo"
    ],
    "Alta": [
        "Aplicar fungicida sistemico + cobre combinado de inmediato",
        "Eliminar y destruir hojas severamente afectadas fuera del lote",
        "Monitorear cada 7 dias y programar segunda aplicacion",
        "Evaluar necesidad de renovacion si el foco es extenso",
        "Contacta a un profesional agronomo URGENTEMENTE"
    ]
}

@app.get("/")
def home():
    return {"message": "API IA funcionando"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), variedad: str = Form(None)):
    try:
        variedad_normalizada = None
        if variedad:
            for v in VARIEDADES_VALIDAS:
                if v.lower() == variedad.strip().lower():
                    variedad_normalizada = v
                    break

        file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.jpg")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        results = model(file_path, conf=0.40)
        detections = []

        valid_classes = ["Hoja_Sana", "Enfermedad_ROYA", "arbol_cafe"]

        if len(results[0].boxes) == 0:
            return {"detections": [], "message": "Imagen no valida. Carga otra imagen de hoja de cafe."}

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]

            if class_name not in valid_classes:
                continue

            recommendations = []
            message = ""

            if class_name == "Enfermedad_ROYA":
                severidad = obtener_severidad(conf)
                if variedad_normalizada:
                    message = f"Se detecto roya en la hoja de cafe (variedad {variedad_normalizada}) - Severidad {severidad}."
                    recommendations = TRATAMIENTOS_ROYA[variedad_normalizada][severidad]
                else:
                    message = f"Se detecto roya en la hoja de cafe - Severidad {severidad}."
                    recommendations = TRATAMIENTO_ROYA_GENERICO[severidad]

            elif class_name == "Hoja_Sana":
                message = "La hoja analizada se encuentra saludable."
                recommendations = ["Mantener monitoreo constante", "Continuar buenas practicas agricolas", "Revisar hojas semanalmente"]

            elif class_name == "arbol_cafe":
                message = "Se detecto un arbol de cafe."
                recommendations = ["Verificar estado de hojas", "Controlar humedad del cultivo", "Realizar inspecciones periodicas"]

            severidad_out = obtener_severidad(conf) if class_name == "Enfermedad_ROYA" else ("Baja" if class_name == "Hoja_Sana" else None)

            detections.append({
                "class": class_name,
                "confidence": conf,
                "severidad": severidad_out,
                "message": message,
                "recommendations": recommendations,
                "variedad": variedad_normalizada
            })

        if len(detections) == 0:
            return {"detections": [], "message": "Imagen no valida. Carga otra imagen de hoja de cafe."}

        return {"detections": detections, "message": "Analisis completado"}

    except Exception as e:
        return {"detections": [], "message": "Error analizando imagen", "error": str(e)}
