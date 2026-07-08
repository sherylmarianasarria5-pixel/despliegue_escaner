from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import uuid
import os
import hashlib
import random

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

SEVERIDAD_BAJA = (0.40, 0.60)
SEVERIDAD_MEDIA = (0.60, 0.80)


def obtener_severidad(confidence: float) -> str:
    if SEVERIDAD_BAJA[0] <= confidence < SEVERIDAD_BAJA[1]:
        return "Baja"
    elif SEVERIDAD_MEDIA[0] <= confidence < SEVERIDAD_MEDIA[1]:
        return "Media"
    return "Alta"


POOL_RECOMENDACIONES = {
    "ROYA_Alta": [
        {
            "titulo": "Fungicida cúprico urgente",
            "descripcion": "Aplicar fungicida cúprico 300g/200L cada 7 días",
            "detalle_general": "El fungicida cúprico a base de oxicloruro de cobre actúa como protectante y bactericida. Crea una barrera sobre la hoja que impide la germinación de esporas de roya.",
            "tratamientos": [
                {"paso": 1, "titulo": "Preparar la mezcla", "descripcion": "Mezclar 300g en 200L de agua", "detalle": "Llene el tanque de fumigación hasta la mitad con agua limpia. Agregue los 300g de fungicida cúprico lentamente mientras agita constantemente. Complete con agua hasta llegar a los 200L.", "consejo": "Prepare solo la cantidad que usará en el día. La mezcla pierde efectividad después de 24 horas.", "precaucion": "Use guantes de nitrilo, mascarilla N95 y gafas de protección."},
                {"paso": 2, "titulo": "Aplicar en el momento adecuado", "descripcion": "Aplicar cada 7 días, en horas frescas", "detalle": "Use bomba de espalda con boquilla de cono ajustable calibrada a 200L/ha. Aplique entre 6-10am o después de 4pm. Dirija la aspersión al envés de las hojas.", "consejo": "No aplique si hay lluvia en las próximas 6 horas o si el viento supera 10 km/h.", "precaucion": "Manténgase a favor del viento para no recibir la aspersión."},
                {"paso": 3, "titulo": "Monitorear la evolución", "descripcion": "Evaluar cada 3-4 días durante el tratamiento", "detalle": "Seleccione 10 plantas al azar en el lote tratado y revise 4 hojas por planta. Cuente cuántas hojas tienen pústulas activas.", "consejo": "Marque 3 plantas con cinta de colores y evalúe siempre las mismas.", "precaucion": "Respete el período de carencia mínimo de 7 días antes de cosechar."},
            ],
        },
        {
            "titulo": "Poda fitosanitaria severa",
            "descripcion": "Eliminar ramas con más del 50% de hojas afectadas",
            "detalle_general": "La poda fitosanitaria reduce la carga de inóculo en el cafetal. Al eliminar las partes más afectadas se disminuye la presión de la enfermedad.",
            "tratamientos": [
                {"paso": 1, "titulo": "Identificar ramas a podar", "descripcion": "Seleccionar ramas con más del 50% de hojas con pústulas", "detalle": "Recorra el lote y evalúe cada planta visualmente. Marque las ramas con más de la mitad de hojas afectadas.", "consejo": "Haga la evaluación temprano en la mañana cuando la luz es perpendicular.", "precaucion": "No pode más del 30% del follaje total de la planta."},
                {"paso": 2, "titulo": "Ejecutar la poda correctamente", "descripcion": "Cortar ramas con herramienta desinfectada", "detalle": "Use tijeras de podar bien afiladas. Desinfecte con alcohol al 70% entre cada planta.", "consejo": "Lleve dos tijeras: mientras una se desinfecta, use la otra.", "precaucion": "Use guantes de carnaza para evitar cortes."},
                {"paso": 3, "titulo": "Disponer del material podado", "descripcion": "Retirar y eliminar ramas cortadas del lote", "detalle": "Recoja todo el material podado inmediatamente. No lo deje en el suelo.", "consejo": "Tenga costales extendidos en el suelo mientras poda.", "precaucion": "Si quema, hágalo en horas de la mañana en un lugar despejado."},
            ],
        },
        {
            "titulo": "Fungicida sistémico",
            "descripcion": "Aplicar triazol o azoxistrobina para control avanzado",
            "detalle_general": "Los fungicidas sistémicos penetran los tejidos de la hoja y protegen el crecimiento nuevo. Ideales para infecciones avanzadas.",
            "tratamientos": [
                {"paso": 1, "titulo": "Seleccionar el producto adecuado", "descripcion": "Usar triazol (200cc/200L) o azoxistrobina según disponibilidad", "detalle": "Elija entre triazol o azoxistrobina. Alterne entre familias químicas para evitar resistencia.", "consejo": "Lleve un registro de qué producto aplicó.", "precaucion": "No mezcle triazoles con productos alcalinos."},
                {"paso": 2, "titulo": "Preparar y aplicar", "descripcion": "Preparar la dosis exacta y aplicar con buena cobertura", "detalle": "Mida exactamente 200cc de triazol por cada 200L de agua. Use surfactante no iónico para mejorar penetración.", "consejo": "Agregue el surfactante al último.", "precaucion": "Use mascarilla con filtro para vapores orgánicos."},
                {"paso": 3, "titulo": "Rotar para evitar resistencia", "descripcion": "Alternar con fungicida cúprico cada 14 días", "detalle": "Aplique el sistémico, espere 14 días y aplique fungicida de contacto.", "consejo": "Configure recordatorios cada 14 días en la app.", "precaucion": "Respete el período de carencia de 14-21 días."},
            ],
        },
        {
            "titulo": "Control biológico con Trichoderma",
            "descripcion": "Aplicar Trichoderma harzianum 10g/L al suelo",
            "detalle_general": "Trichoderma es un hongo benéfico que parasita hongos patógenos y estimula el crecimiento de raíces.",
            "tratamientos": [
                {"paso": 1, "titulo": "Preparar la solución", "descripcion": "Mezclar 10g por litro de agua", "detalle": "Disuelva 10g de Trichoderma por litro de agua limpia sin cloro.", "consejo": "Prepare máximo 2 horas antes de aplicar.", "precaucion": "No mezcle con fungicidas químicos."},
                {"paso": 2, "titulo": "Aplicar al suelo y follaje", "descripcion": "Aplicar en la base de la planta y al follaje", "detalle": "Aplique 200cc al suelo en la base de cada planta y 100cc al follaje.", "consejo": "Aplique después de una lluvia ligera.", "precaucion": "Guarde en lugar fresco y oscuro."},
                {"paso": 3, "titulo": "Repetir la aplicación", "descripcion": "Repetir cada 15 días durante 3 aplicaciones", "detalle": "Necesita al menos 3 aplicaciones con intervalo de 15 días para establecerse.", "consejo": "Agregue melaza al 1% para alimentar el Trichoderma.", "precaucion": "Limpie bien la bomba después de usar Trichoderma."},
            ],
        },
    ],
    "ROYA_Media": [
        {
            "titulo": "Fungicida cúprico preventivo",
            "descripcion": "Fungicida cúprico 250g/200L cada 10-14 días",
            "detalle_general": "En niveles moderados, una dosis más baja de fungicida cúprico es suficiente para controlar la roya.",
            "tratamientos": [
                {"paso": 1, "titulo": "Preparar dosis moderada", "descripcion": "Mezclar 250g en 200L de agua", "detalle": "Agregue 250g de fungicida cúprico en polvo o 500cc si es líquido.", "consejo": "Si el agua es muy dura, agregue 50cc de vinagre blanco por cada 100L.", "precaucion": "Use guantes y mascarilla."},
                {"paso": 2, "titulo": "Aplicar con buena cobertura", "descripcion": "Cubrir toda la planta especialmente el envés", "detalle": "Calibre la bomba para 200L/ha. Aplique en círculos alrededor de cada planta.", "consejo": "Agregue adherente agrícola a 0.5cc/L.", "precaucion": "No aplique con temperatura superior a 30°C."},
                {"paso": 3, "titulo": "Programar próximas aplicaciones", "descripcion": "Repetir cada 10-14 días según condiciones", "detalle": "En lluvias, cada 10 días. En sequía, cada 14 días.", "consejo": "Configure alarma en su celular.", "precaucion": "No aplique si hay floración abundante."},
            ],
        },
        {
            "titulo": "Eliminar hojas afectadas manualmente",
            "descripcion": "Retirar hojas con síntomas visibles de roya",
            "detalle_general": "La remoción manual reduce el inóculo disponible y no tiene costo de insumos.",
            "tratamientos": [
                {"paso": 1, "titulo": "Identificar hojas con síntomas", "descripcion": "Buscar pústulas anaranjadas en el envés", "detalle": "Revise hoja por hoja. Busque pústulas anaranjadas en el envés.", "consejo": "Haga esta labor después de una lluvia.", "precaucion": "Use guantes desechables."},
                {"paso": 2, "titulo": "Arrancar hojas correctamente", "descripcion": "Desprender la hoja desde la base del pecíolo", "detalle": "Jale la hoja hacia abajo con movimiento firme.", "consejo": "Si hay muchas hojas afectadas, pode la rama completa.", "precaucion": "No arranque más del 20% de hojas de una planta."},
                {"paso": 3, "titulo": "Recolectar y desechar", "descripcion": "No dejar hojas en el suelo", "detalle": "Use un costal para recolectar. Queme o entierre a 40cm con cal.", "consejo": "Lleve dos costales: uno para hojas enfermas y otro para sanas.", "precaucion": "Limpie sus botas antes de salir del lote."},
            ],
        },
        {
            "titulo": "Poda selectiva para ventilación",
            "descripcion": "Podar ramas bajas y chupones para mejorar aireación",
            "detalle_general": "La poda selectiva mejora la circulación de aire creando un microclima menos favorable para la roya.",
            "tratamientos": [
                {"paso": 1, "titulo": "Identificar qué podar", "descripcion": "Seleccionar ramas bajas, chupones y ramas entrecruzadas", "detalle": "Identifique ramas que tocan el suelo, chupones del tronco y ramas entrecruzadas.", "consejo": "Agáchese al nivel de la base y mire hacia arriba.", "precaucion": "No pode más del 25% del follaje total."},
                {"paso": 2, "titulo": "Podar con precisión", "descripcion": "Cortar en bisel y desinfectar herramientas", "detalle": "Corte en bisel a 45° a 1-2cm del tallo. Desinfecte entre plantas.", "consejo": "Tenga un recipiente con alcohol al 70%.", "precaucion": "Afíle las tijeras cada semana."},
                {"paso": 3, "titulo": "Mantener densidad adecuada", "descripcion": "Dejar distancia entre plantas para circulación de aire", "detalle": "Asegure 30-40cm de espacio entre follaje de plantas contiguas.", "consejo": "Surcos orientados de oriente a occidente tienen mejor ventilación.", "precaucion": "Aplique fungicida preventivo después de podar."},
            ],
        },
        {
            "titulo": "Fertilización potásica",
            "descripcion": "Aplicar potasio para fortalecer la planta contra la roya",
            "detalle_general": "El potasio fortalece las paredes celulares y activa mecanismos de defensa natural.",
            "tratamientos": [
                {"paso": 1, "titulo": "Calcular la dosis", "descripcion": "Aplicar KCl 200kg/ha fraccionado en 2 veces", "detalle": "Use 200kg de KCl por hectárea, dividido en dos aplicaciones de 100kg.", "consejo": "Tome una muestra de suelo para análisis.", "precaucion": "Fraccione el potasio para evitar pérdidas."},
                {"paso": 2, "titulo": "Aplicar correctamente", "descripcion": "Distribuir en corona alrededor de la planta", "detalle": "Haga un hoyo de 10-15cm a 30-40cm del tallo. Cubra con tierra.", "consejo": "Aplique después de una lluvia.", "precaucion": "Use guantes, el KCl es corrosivo."},
                {"paso": 3, "titulo": "Complementar con otros nutrientes", "descripcion": "Asegurar nitrógeno, fósforo y micronutrientes", "detalle": "Aplique NPK completo (15-15-15) a 250g/planta 30 días después del potasio.", "consejo": "Aplique cal dolomítica 500g/planta una vez al año.", "precaucion": "No aplique nitrógeno en exceso."},
            ],
        },
    ],
    "ROYA_Baja": [
        {
            "titulo": "Fungicida preventivo",
            "descripcion": "Aplicar mancozeb cada 15 días",
            "detalle_general": "El mancozeb actúa como protector de amplio espectro. Ideal para mantenimiento preventivo.",
            "tratamientos": [
                {"paso": 1, "titulo": "Preparar la dosis preventiva", "descripcion": "Mezclar 200g en 200L de agua", "detalle": "Use 200g de mancozeb por cada 200L de agua.", "consejo": "El mancozeb es compatible con la mayoría de fungicidas.", "precaucion": "Puede irritar los ojos. Use gafas."},
                {"paso": 2, "titulo": "Aplicar en momentos clave", "descripcion": "Aplicar antes de lluvias intensas", "detalle": "Aplique antes del inicio de temporada de lluvias (marzo-abril y septiembre-octubre).", "consejo": "Consulte el pronóstico del tiempo.", "precaucion": "Evite vientos fuertes para evitar deriva."},
                {"paso": 3, "titulo": "Mantener frecuencia", "descripcion": "Repetir cada 15 días durante temporada de riesgo", "detalle": "Mantenga cada 15 días durante los 3 meses de mayor precipitación.", "consejo": "Si aparece roya, cambie a fungicida cúprico o sistémico.", "precaucion": "No use mancozeb más de 6 veces al año."},
            ],
        },
        {
            "titulo": "Monitoreo constante cada 8 días",
            "descripcion": "Revisar el cultivo cada 8 días para detectar signos tempranos",
            "detalle_general": "El monitoreo regular permite detectar la enfermedad en etapas tempranas cuando el control es más fácil.",
            "tratamientos": [
                {"paso": 1, "titulo": "Inspeccionar hojas estratégicas", "descripcion": "Revisar hojas del tercio medio e inferior", "detalle": "Revise 5 hojas por planta en 10 plantas al azar. Busque puntos amarillos en el haz.", "consejo": "Haga el monitoreo siempre el mismo día de la semana.", "precaucion": "No se confíe en temporada seca."},
                {"paso": 2, "titulo": "Usar escala de severidad", "descripcion": "Clasificar lo observado bajo/medio/alto", "detalle": "0 pústulas = sano, 1-3 = bajo, 4-6 = medio, 7+ = alto.", "consejo": "Use la app para registrar cada monitoreo.", "precaucion": "Revise también hojas que parecen sanas."},
                {"paso": 3, "titulo": "Registrar condiciones ambientales", "descripcion": "Anotar lluvia, temperatura y humedad", "detalle": "Registre días desde última lluvia, temperatura media y presencia de neblina.", "consejo": "Instale un pluviómetro casero.", "precaucion": "Si encuentra roya generalizada, actúe de inmediato."},
            ],
        },
        {
            "titulo": "Fertilización balanceada",
            "descripcion": "Fortalecer la planta con nutrición completa",
            "detalle_general": "Una nutrición equilibrada fortalece las defensas naturales de la planta contra enfermedades.",
            "tratamientos": [
                {"paso": 1, "titulo": "Planificar la fertilización", "descripcion": "Aplicar NPK 15-3-31 a 250g/planta más micronutrientes", "detalle": "Aplique 250g/planta de NPK completo al inicio de lluvias (abril) y agosto.", "consejo": "Haga análisis de suelo cada 2 años.", "precaucion": "No fertilice en suelo seco."},
                {"paso": 2, "titulo": "Aplicar correctamente", "descripcion": "Distribuir en zona de goteo", "detalle": "Haga surco circular a 30-40cm del tallo. Cubra con tierra.", "consejo": "Divida la dosis en 2 hoyos opuestos.", "precaucion": "En pendiente, aplique del lado arriba."},
                {"paso": 3, "titulo": "Complementar con abonos orgánicos", "descripcion": "Agregar materia orgánica para mejorar el suelo", "detalle": "Aplique 3-5kg de compost o gallinaza por planta cada 6 meses.", "consejo": "Haga su propia composta con pulpa de café.", "precaucion": "Verifique que la gallinaza esté bien composteada."},
            ],
        },
        {
            "titulo": "Manejo de sombra",
            "descripcion": "Regular el porcentaje de sombra en el cafetal",
            "detalle_general": "La sombra moderada (30-40%) es beneficiosa; el exceso favorece la roya.",
            "tratamientos": [
                {"paso": 1, "titulo": "Medir la sombra actual", "descripcion": "Evaluar el porcentaje de sombra en el lote", "detalle": "Mida la sombra a las 12 del mediodía. Ideal: 30-40% de sombra.", "consejo": "Use app de luxómetro en el celular.", "precaucion": "No elimine todos los árboles de golpe."},
                {"paso": 2, "titulo": "Regular los árboles de sombra", "descripcion": "Podar árboles para ajustar al 30-40% ideal", "detalle": "Si hay exceso, pode ramas bajas. Si falta, siembre árboles temporales.", "consejo": "Los árboles de guamo son excelentes para sombra.", "precaucion": "No pode en época de cosecha."},
                {"paso": 3, "titulo": "Monitorear el efecto", "descripcion": "Evaluar respuesta del cafetal al ajuste de sombra", "detalle": "Monitoree 2-3 meses: la incidencia de roya debe disminuir.", "consejo": "Mida producción antes y después del ajuste.", "precaucion": "En zonas cálidas, no baje del 35% de sombra."},
            ],
        },
    ],
    "Hoja_Sana": [
        {
            "titulo": "Planta en buen estado",
            "descripcion": "Continúa con el manejo habitual",
            "detalle_general": "Tus plantas están sanas. Sigue con las buenas prácticas que te han funcionado.",
            "tratamientos": [
                {"paso": 1, "titulo": "Mantener las prácticas actuales", "descripcion": "Sigue con tu programa de manejo", "detalle": "Continúa con fertilización, control de malezas y podas.", "consejo": "Documenta todo en la app.", "precaucion": "No te confíes, la roya puede aparecer después de lluvias."},
                {"paso": 2, "titulo": "Revisar calendario de fertilización", "descripcion": "Programar próxima fertilización", "detalle": "Revisa cuándo fue la última fertilización. Los meses críticos son abril y agosto.", "consejo": "Ten los insumos listos con anticipación.", "precaucion": "No sobre-fertilices."},
                {"paso": 3, "titulo": "Programar próxima poda", "descripcion": "Planificar poda de mantenimiento post-cosecha", "detalle": "Después de cosecha, poda ramas secas, chupones y ramas bajas.", "consejo": "La poda también ayuda a controlar broca.", "precaucion": "No podes en plena época de lluvias."},
            ],
        },
        {
            "titulo": "Fertilización preventiva con bioestimulantes",
            "descripcion": "Aplicar bioestimulantes foliares para fortalecer defensas",
            "detalle_general": "Los bioestimulantes activan mecanismos de defensa natural de la planta.",
            "tratamientos": [
                {"paso": 1, "titulo": "Preparar la mezcla", "descripcion": "Diluir 3cc/L de aminoácidos + algas marinas", "detalle": "Mezcle 3cc de aminoácidos + 2cc de extracto de algas marinas por litro.", "consejo": "Compre productos concentrados y mézclelos usted mismo.", "precaucion": "Agite suavemente, no enérgicamente."},
                {"paso": 2, "titulo": "Aplicar al follaje", "descripcion": "Cada 20 días, dirigido al envés", "detalle": "Aplique con bomba de espalda, dirigiendo al envés. Use boquilla de cono fino.", "consejo": "Agregue surfactante agrícola.", "precaucion": "No mezcle con fungicidas cúpricos."},
                {"paso": 3, "titulo": "Programar aplicaciones estratégicas", "descripcion": "Aplicar antes de periodos de estrés", "detalle": "Aplique 10-15 días antes de lluvias intensas y antes de floración.", "consejo": "Marque calendario fijo cada 20 días.", "precaucion": "No aplique con calor >30°C."},
            ],
        },
    ],
}


def generar_recomendaciones_desde_pool(pool_key: str, image_bytes: bytes) -> list:
    pool = POOL_RECOMENDACIONES.get(pool_key, [])
    if not pool:
        return []

    seed = hashlib.md5(image_bytes).hexdigest()
    rng = random.Random(seed)

    cantidad = 2 if pool_key in ("ROYA_Media", "ROYA_Baja", "Hoja_Sana") else 3
    seleccionadas = rng.sample(pool, k=min(cantidad, len(pool)))

    return [dict(r) for r in seleccionadas]


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

        with open(file_path, "rb") as f:
            image_bytes = f.read()

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
            recomendaciones_detalladas = []
            message = ""

            if class_name == "Enfermedad_ROYA":
                severidad = obtener_severidad(conf)

                if variedad_normalizada:
                    message = f"Se detecto roya en la hoja de cafe (variedad {variedad_normalizada}) - Severidad {severidad}."
                    recommendations = TRATAMIENTOS_ROYA[variedad_normalizada][severidad]
                else:
                    message = f"Se detecto roya en la hoja de cafe - Severidad {severidad}."
                    recommendations = TRATAMIENTO_ROYA_GENERICO[severidad]

                recomendaciones_detalladas = generar_recomendaciones_desde_pool(f"ROYA_{severidad}", image_bytes)

            elif class_name == "Hoja_Sana":
                message = "La hoja analizada se encuentra saludable."
                recommendations = ["Mantener monitoreo constante", "Continuar buenas practicas agricolas", "Revisar hojas semanalmente"]
                recomendaciones_detalladas = generar_recomendaciones_desde_pool("Hoja_Sana", image_bytes)

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
                "recomendaciones_detalladas": recomendaciones_detalladas,
                "variedad": variedad_normalizada
            })

        if len(detections) == 0:
            return {"detections": [], "message": "Imagen no valida. Carga otra imagen de hoja de cafe."}

        return {"detections": detections, "message": "Analisis completado"}

    except Exception as e:
        return {"detections": [], "message": "Error analizando imagen", "error": str(e)}
