from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

import shutil
import uuid
import os
import hashlib
import random

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
# INFO DE ENFERMEDADES
# =========================

ENFERMEDADES_INFO = {
    "Enfermedad_ROYA": {
        "nombre_cientifico": "Hemileia vastatrix",
        "nombre_comun": "Roya del café",
    },
    "Hoja_Sana": {
        "nombre_cientifico": "Sin patógenos detectados",
        "nombre_comun": "Planta sana",
    },
}

# =========================
# CÁLCULO DE SEVERIDAD
# =========================

def calcular_severidad(class_name: str, confidence: float) -> str:
    if class_name == "Hoja_Sana":
        return "Ninguna"
    if class_name == "Enfermedad_ROYA":
        if confidence >= 0.80:
            return "Alta"
        if confidence >= 0.50:
            return "Media"
        return "Baja"
    return ""

# =========================
# RECOMENDACIONES
# =========================

# =========================
# POOL DE RECOMENDACIONES
# =========================

POOL_RECOMENDACIONES = {
    "ROYA_Alta": [
        {
            "titulo": "Fungicida cúprico urgente",
            "descripcion": "Aplicar fungicida cúprico 300g/200L cada 7 días",
            "detalle_general": "El fungicida cúprico a base de oxicloruro de cobre actúa como protectante y bactericida. Crea una barrera sobre la hoja que impide la germinación de esporas de roya. Es el producto de primera línea para control químico en cafetales.",
            "tratamientos": [
                {"paso": 1, "titulo": "Preparar la mezcla", "descripcion": "Mezclar 300g en 200L de agua", "detalle": "Llene el tanque de fumigación hasta la mitad con agua limpia. Agregue los 300g de fungicida cúprico lentamente mientras agita constantemente. Complete con agua hasta llegar a los 200L. Siga mezclando por 5 minutos adicionales hasta que el producto esté completamente disuelto. Use agua con pH entre 5.5 y 6.5, si es necesario ajústelo con un acidificante.", "consejo": "Prepare solo la cantidad que usará en el día. La mezcla pierde efectividad después de 24 horas. No la almacene.", "precaucion": "Use guantes de nitrilo, mascarilla N95 y gafas de protección. No mezcle con productos alcalinos."},
                {"paso": 2, "titulo": "Aplicar en el momento adecuado", "descripcion": "Aplicar cada 7 días, en horas frescas", "detalle": "Use bomba de espalda con boquilla de cono ajustable calibrada a 200L/ha. Aplique entre 6-10am o después de 4pm, evitando el sol fuerte. Dirija la aspersión al envés de las hojas donde la roya se desarrolla. Cubra la planta completa, priorizando los tercios medio e inferior que concentran mayor humedad.", "consejo": "No aplique si hay lluvia en las próximas 6 horas o si el viento supera 10 km/h. Un palo con una bolsa plástica atada le ayuda a ver la dirección del viento.", "precaucion": "Manténgase a favor del viento para no recibir la aspersión. Use ropa manga larga y botas de caucho."},
                {"paso": 3, "titulo": "Monitorear la evolución", "descripcion": "Evaluar cada 3-4 días durante el tratamiento", "detalle": "Seleccione 10 plantas al azar en el lote tratado y revise 4 hojas por planta (2 del tercio medio, 2 del inferior). Cuente cuántas hojas tienen pústulas activas (polvo anaranjado). Si después de 3 aplicaciones no hay reducción significativa, el hongo podría ser resistente y necesita consultar a un agrónomo.", "consejo": "Marque 3 plantas con cinta de colores y evalúe siempre las mismas para tener una comparación exacta.", "precaucion": "Respete el período de carencia mínimo de 7 días antes de cosechar los frutos."},
            ],
        },
        {
            "titulo": "Poda fitosanitaria severa",
            "descripcion": "Eliminar ramas con más del 50% de hojas afectadas",
            "detalle_general": "La poda fitosanitaria reduce la carga de inóculo en el cafetal. Al eliminar las partes más afectadas de la planta, se disminuye la presión de la enfermedad y se facilita la penetración de los fungicidas. Además mejora la aireación del cultivo.",
            "tratamientos": [
                {"paso": 1, "titulo": "Identificar ramas a podar", "descripcion": "Seleccionar ramas con más del 50% de hojas con pústulas", "detalle": "Recorra el lote y evalúe cada planta visualmente. Marque las ramas que tengan más de la mitad de sus hojas con pústulas anaranjadas activas, hojas secas o defoliación severa. Priorice las ramas del tercio inferior que suelen estar más sombreadas y húmedas. Use cinta de colores para marcar antes de empezar a cortar.", "consejo": "Haga la evaluación temprano en la mañana cuando la luz es perpendicular y se ven mejor las pústulas en el envés.", "precaucion": "No pode más del 30% del follaje total de la planta para evitar estresarla demasiado."},
                {"paso": 2, "titulo": "Ejecutar la poda correctamente", "descripcion": "Cortar ramas con herramienta desinfectada", "detalle": "Use tijeras de podar bien afiladas para hacer cortes limpios en bisel a 2cm del tallo principal. Desinfecte las tijeras con alcohol al 70% o solución de cloro al 10% entre cada planta, sumergiendo las cuchillas por 30 segundos. No deje muñones porque son entrada de enfermedades. Cubra los cortes mayores a 3cm con pasta cicatrizante a base de cobre.", "consejo": "Lleve dos tijeras: mientras una se desinfecta, use la otra. Así no pierde tiempo entre planta y planta.", "precaucion": "Use guantes de carnaza para evitar cortes. Las tijeras recién desinfectadas con cloro pueden corroerse, enjuáguelas con agua."},
                {"paso": 3, "titulo": "Disponer del material podado", "descripcion": "Retirar y eliminar ramas cortadas del lote", "detalle": "Recoja todo el material podado inmediatamente después del corte. No lo deje en el suelo entre los surcos porque las esporas de roya pueden reinfectar las plantas sanas. Queme el material lejos del cafetal o entiérrelo a mínimo 50cm de profundidad cubriendo con cal viva. Si no puede quemar, sáquelo del lote y cubra con plástico negro por 2 semanas para que se descomponga.", "consejo": "Tenga costales o lonas extendidas en el suelo mientras poda para que las ramas caigan directamente sobre ellas y sea más fácil recogerlas.", "precaucion": "Si quema, hágalo en horas de la mañana y en un lugar despejado. Avise a los vecinos. Tenga agua o tierra cerca por si el fuego se extiende."},
            ],
        },
        {
            "titulo": "Fungicida sistémico",
            "descripcion": "Aplicar triazol o azoxistrobina para control avanzado",
            "detalle_general": "Los fungicidas sistémicos penetran en los tejidos de la hoja y se traslocan a través de la planta, protegiendo el crecimiento nuevo. Son ideales para infecciones avanzadas porque atacan al hongo desde adentro. Deben alternarse con fungicidas de contacto para evitar resistencia.",
            "tratamientos": [
                {"paso": 1, "titulo": "Seleccionar el producto adecuado", "descripcion": "Usar triazol (200cc/200L) o azoxistrobina según disponibilidad", "detalle": "Elija entre triazol (ej. ciproconazol, tebuconazol) a 200cc/200L o azoxistrobina a 300cc/200L. Los triazoles actúan inhibiendo la síntesis de esteroles del hongo. Las estrobilurinas bloquean la respiración mitocondrial. No los use de forma consecutiva: alterne entre familias químicas. Consulte la etiqueta del producto específico que tenga disponible.", "consejo": "Lleve un registro escrito de qué producto aplicó y en qué fecha para saber cuándo cambiar de familia química.", "precaucion": "No mezcle triazoles con productos alcalinos (caldo bordelés). Deje al menos 5 días entre aplicaciones."},
                {"paso": 2, "titulo": "Preparar y aplicar", "descripcion": "Preparar la dosis exacta y aplicar con buena cobertura", "detalle": "Mida exactamente 200cc de triazol por cada 200L de agua. Agregue primero un tercio del agua, luego el fungicida, agite y complete el agua. Use un surfactante no iónico (0.5cc/L) para mejorar la penetración. Aplique con boquilla de cono vacío a presión media (40-60 PSI). Cubra especialmente las hojas jóvenes del tercio superior donde la enfermedad progresa.", "consejo": "Agregue el surfactante al último después de llenar el tanque, para evitar excesiva espuma.", "precaucion": "Los fungicidas sistémicos son más tóxicos que los cúpricos. Use mascarilla con filtro para vapores orgánicos."},
                {"paso": 3, "titulo": "Rotar para evitar resistencia", "descripcion": "Alternar con fungicida cúprico cada 14 días", "detalle": "Aplique el sistémico, espere 14 días y aplique un fungicida de contacto como el cúprico. Luego otros 14 días y vuelva al sistémico. Esta rotación evita que el hongo desarrolle resistencia. No use el mismo sistémico más de 3 veces consecutivas. Combine con poda y control biológico para un manejo integrado.", "consejo": "Use el calendario de la app para programar las rotaciones. Configure recordatorios cada 14 días.", "precaucion": "Respete el período de carencia de 14-21 días para fungicidas sistémicos antes de cosechar."},
            ],
        },
        {
            "titulo": "Control biológico con Trichoderma",
            "descripcion": "Aplicar Trichoderma harzianum 10g/L al suelo",
            "detalle_general": "Trichoderma harzianum es un hongo benéfico que parasita y compite con los hongos patógenos como la roya. Además produce enzimas que degradan las paredes celulares del patógeno y estimula el crecimiento de raíces. Es una alternativa ecológica y sostenible.",
            "tratamientos": [
                {"paso": 1, "titulo": "Preparar la solución", "descripcion": "Mezclar 10g por litro de agua", "detalle": "Disuelva 10g de Trichoderma en polvo por cada litro de agua limpia (sin cloro). Use agua de lluvia o agua reposada por 24 horas si el agua es clorada. Revuelva suavemente con una varilla de madera o plástico (el metal puede afectar el hongo). Deje reposar la mezcla 15-20 minutos antes de aplicar para activar las esporas.", "consejo": "Prepare la solución máximo 2 horas antes de aplicar. El Trichoderma vivo necesita aplicarse fresco.", "precaucion": "No mezcle con fungicidas químicos porque matarían el Trichoderma. Espere 10 días después de aplicar fungicidas químicos."},
                {"paso": 2, "titulo": "Aplicar al suelo y follaje", "descripcion": "Aplicar en la base de la planta y al follaje", "detalle": "Aplique 200cc de la solución directamente al suelo en la base de cada planta, en un radio de 30cm alrededor del tallo. Luego aplique 100cc al follaje con una bomba de espalda, dirigiendo al envés de las hojas. Haga esto en horas frescas de la mañana o al atardecer. La luz UV directa mata las esporas de Trichoderma.", "consejo": "Aplique después de una lluvia ligera cuando el suelo está húmedo, así el hongo benéfico se establece mejor.", "precaucion": "Guarde el producto en un lugar fresco y oscuro. Las esporas de Trichoderma pueden causar alergia respiratoria en personas sensibles."},
                {"paso": 3, "titulo": "Repetir la aplicación", "descripcion": "Repetir cada 15 días durante 3 aplicaciones", "detalle": "Para que el Trichoderma se establezca en el suelo y la planta, necesita al menos 3 aplicaciones con intervalo de 15 días. Después de este ciclo inicial, puede aplicar una vez al mes como mantenimiento. Combine con materia orgánica (compost, gallinaza) para que el Trichoderma tenga sustrato donde reproducirse.", "consejo": "Agregue melaza al 1% (10cc/L) a la solución en la segunda aplicación para alimentar el Trichoderma.", "precaucion": "Limpie bien la bomba de fumigación después de usar Trichoderma. No la use inmediatamente para fungicidas químicos."},
            ],
        },
        {
            "titulo": "Eliminar hojas caídas del suelo",
            "descripcion": "Retirar hojas infectadas del suelo para evitar reinfección",
            "detalle_general": "Las hojas caídas con roya contienen esporas que salpican con la lluvia a las hojas sanas de la parte baja de la planta. Eliminarlas rompe el ciclo de la enfermedad. Es una práctica sencilla pero muy efectiva en el manejo integrado.",
            "tratamientos": [
                {"paso": 1, "titulo": "Recolectar las hojas caídas", "descripcion": "Rastrillar en un radio de 2m alrededor de cada planta", "detalle": "Use un rastrillo de jardín para juntar todas las hojas caídas en un radio de al menos 2 metros alrededor de la base de cada cafeto. Incluya hojas secas, hojas con pústulas y cualquier resto vegetal en descomposición. Si hay hojas enterradas superficialmente, remueva la capa superior del suelo (1-2cm) con cuidado de no dañar raíces superficiales.", "consejo": "Haga esta labor después de una lluvia cuando las hojas están mojadas y se juntan más fácilmente sin deshacerse.", "precaucion": "Use guantes y evite el contacto directo porque las esporas de roya pueden causar irritación respiratoria al movilizarse."},
                {"paso": 2, "titulo": "Eliminar el material recolectado", "descripcion": "Sacar del lote y enterrar o compostear", "detalle": "Recoja el material en costales y sáquelo del cafetal. Tiene tres opciones: 1) Entierre en una fosa de 50cm de profundidad cubriendo con una capa de cal viva y tierra. 2) Compostee fuera del lote en una pila cubierta con plástico negro por 45-60 días, volteando cada 15 días. 3) Queme en un lugar seguro y despejado lejos de árboles y viviendas.", "consejo": "Si compostea, agregue una capa de pasto seco entre las capas de hojas de café para mejorar la relación carbono-nitrógeno.", "precaucion": "No use este material para mulch o cobertura del mismo cafetal porque las esporas pueden reinfectar."},
                {"paso": 3, "titulo": "Monitorear y repetir", "descripcion": "Repetir cada semana durante la temporada de lluvias", "detalle": "Durante la temporada de lluvias (abril-junio, octubre-noviembre en Colombia), las hojas caen más rápido y las esporas se dispersan por salpique. Repita esta labor cada 7-8 días. En temporada seca, cada 15-20 días es suficiente. Registre en la app cada vez que realiza la limpieza para llevar control.", "consejo": "Vincule a una persona de la finca específicamente para esta labor, asignándole surcos fijos cada semana.", "precaucion": "Después de manipular hojas con roya, lávese las manos y la ropa antes de comer o beber algo."},
            ],
        },
        {
            "titulo": "Caldo bordelés",
            "descripcion": "Aplicar caldo bordelés al 1% cada 10 días",
            "detalle_general": "El caldo bordelés es una mezcla artesanal de sulfato de cobre y cal hidratada. Es uno de los fungicidas más antiguos y efectivos para el control de roya. Su acción es protectante y debe aplicarse antes de que la enfermedad esté muy avanzada o en combinación con otras medidas.",
            "tratamientos": [
                {"paso": 1, "titulo": "Preparar el caldo bordelés", "descripcion": "Mezclar 1kg sulfato de cobre + 1kg cal + 100L agua", "detalle": "En un recipiente plástico de 120L, disuelva 1kg de sulfato de cobre en 50L de agua caliente (no hirviendo). En otro recipiente, disuelva 1kg de cal hidratada en 50L de agua fría. Cuando ambas soluciones estén a temperatura ambiente, vierta lentamente la solución de cobre sobre la de cal (nunca al revés) mientras revuelve con una paleta de madera. Debe quedar una mezcla de color azul lechoso. Pruebe el pH: debe ser neutro o ligeramente alcalino (pH 7-8).", "consejo": "Pruebe la calidad del caldo sumergiendo un cuchillo limpio — si queda una capa uniforme azul, está bien. Si no se adhiere, agregue más cal.", "precaucion": "Use recipientes de plástico o barro. El metal reacciona con el sulfato de cobre y arruina la mezcla. El proceso genera vapores, hágalo al aire libre."},
                {"paso": 2, "titulo": "Aplicar oportunamente", "descripcion": "Aplicar en horas frescas, cubriendo bien el envés", "detalle": "Use la mezcla dentro de las primeras 8 horas después de preparada porque pierde efectividad. Aplique con bomba de espalda con boquilla de cono a presión media (40-50 PSI). Gastar aproximadamente 100L de caldo por hectárea. Dirija la aspersión al envés de las hojas, moviendo la boquilla hacia arriba y barriendo de abajo hacia arriba. La capa azulada que queda sobre las hojas es la protección.", "consejo": "Agregue 100cc de melaza por cada 100L de caldo como adherente natural para que se fije mejor a las hojas.", "precaucion": "El caldo bordelés mancha la ropa y los equipos permanentemente. Use ropa vieja y limpia la bomba inmediatamente después."},
                {"paso": 3, "titulo": "Rotar con otros productos", "descripcion": "Alternar cada 10 días con fungicidas orgánicos", "detalle": "Aplique caldo bordelés cada 10 días durante 2 aplicaciones, luego alterne con fungicidas orgánicos (Bacillus subtilis, extracto de cola de caballo) para no acumular cobre en el suelo. No use caldo bordelés por más de 4 aplicaciones consecutivas. El cobre se acumula en el suelo y puede afectar la microbiología del mismo a largo plazo.", "consejo": "Use la app para llevar un calendario de rotación y evitar aplicar caldo bordelés más de 6 veces por año en el mismo lote.", "precaucion": "No aplique caldo bordelés en floración porque puede afectar las abejas y otros polinizadores."},
            ],
        },
    ],
    "ROYA_Media": [
        {
            "titulo": "Fungicida cúprico preventivo",
            "descripcion": "Fungicida cúprico 250g/200L cada 10-14 días",
            "detalle_general": "En niveles moderados de infección, una dosis más baja de fungicida cúprico es suficiente para controlar la roya y prevenir su avance. La aplicación oportuna evita que la severidad suba a niveles críticos.",
            "tratamientos": [
                {"paso": 1, "titulo": "Preparar dosis moderada", "descripcion": "Mezclar 250g en 200L de agua", "detalle": "Llene el tanque hasta la mitad con agua limpia. Agregue 250g de fungicida cúprico en polvo o 500cc si es líquido. Agite constantemente mientras completa los 200L de agua. Revuelva por 3-4 minutos hasta que el producto esté uniforme. Verifique que no queden grumos en el fondo del tanque.", "consejo": "Si el agua es muy dura (pH alcalino), agregue 50cc de vinagre blanco por cada 100L para acidificar ligeramente.", "precaucion": "Use guantes y mascarilla. Evite inhalar el polvo del fungicida al medirlo."},
                {"paso": 2, "titulo": "Aplicar con buena cobertura", "descripcion": "Cubrir toda la planta especialmente el envés", "detalle": "Calibre la bomba para un gasto de 200L/ha. Aplique en círculos alrededor de cada planta, empezando desde la parte baja y subiendo. Asegure que el envés de las hojas quede bien mojado. No escatime en las hojas del tercio inferior donde la roya suele empezar. La aplicación debe dejar las hojas brillantes pero sin goteo excesivo.", "consejo": "Agregue un adherente agrícola a 0.5cc/L para mejorar la cobertura, especialmente si las hojas están muy cerosas.", "precaucion": "No aplique con temperatura superior a 30°C porque el producto se evapora rápido y puede quemar las hojas."},
                {"paso": 3, "titulo": "Programar próximas aplicaciones", "descripcion": "Repetir cada 10-14 días según condiciones climáticas", "detalle": "Si está en temporada de lluvias, aplique cada 10 días. Si es temporada seca, cada 14 días. Si llueve dentro de las 6 horas posteriores a la aplicación, debe repetirla. Lleve un registro en la app de cada aplicación: fecha, dosis, lote y condiciones climáticas. Esto le ayuda a planificar y evaluar efectividad.", "consejo": "Configure una alarma en su celular los días de aplicación para no olvidar. La constancia es clave en el control de roya.", "precaucion": "No aplique si hay floración abundante para proteger abejas y otros polinizadores."},
            ],
        },
        {
            "titulo": "Eliminar hojas afectadas manualmente",
            "descripcion": "Retirar hojas con síntomas visibles de roya",
            "detalle_general": "La remoción manual de hojas enfermas reduce el inóculo disponible para infectar hojas sanas. Es una labor que requiere mano de obra pero es efectiva y no tiene costo de insumos. Funciona mejor cuando la infección es moderada y focalizada.",
            "tratamientos": [
                {"paso": 1, "titulo": "Identificar hojas con síntomas", "descripcion": "Buscar pústulas anaranjadas o manchas cloróticas", "detalle": "Revise hoja por hoja en cada planta. Busque pústulas de color anaranjado brillante en el envés de la hoja (signo activo de roya) o manchas cloróticas (amarillentas) en el haz. También retire hojas que ya están secas pero aún adheridas. Concéntrese en las hojas del tercio medio e inferior que son las más afectadas.", "consejo": "Haga esta labor después de una lluvia, ya que las pústulas se ven más nítidas y las hojas enfermas se distinguen mejor.", "precaucion": "Use guantes desechables si es posible. Si no, lávese bien las manos al terminar cada surco."},
                {"paso": 2, "titulo": "Arrancar hojas correctamente", "descripcion": "Desprender la hoja desde la base del pecíolo", "detalle": "Tome la hoja afectada por el pecíolo y jálela hacia abajo con un movimiento firme. Debe desprenderse limpiamente. No arranque hojas sanas adyacentes. Si la hoja no se desprende fácilmente, use tijeras de podar desinfectadas. No deje restos de pecíolo porque pueden pudrirse y atraer hongos.", "consejo": "Si encuentra una rama con muchas hojas afectadas consecutivas, pode la rama completa en lugar de hoja por hoja.", "precaucion": "Evite arrancar más del 20% de las hojas de una sola planta para no estresarla."},
                {"paso": 3, "titulo": "Recolectar y desechar", "descripcion": "No dejar hojas en el suelo, sacarlas del lote", "detalle": "Use un costal o canasto para recolectar las hojas a medida que las arranca. No las deje caer al suelo porque las esporas pueden salpicar a otras plantas. Al final de la jornada, queme las hojas recolectadas o entiérrelas a 40cm de profundidad con cal. No las deposite en la composta cerca del cafetal.", "consejo": "Lleve dos costales: uno para hojas enfermas y otro para hojas sanas que caigan accidentalmente (estas pueden ir a composta).", "precaucion": "Limpie sus botas antes de salir del lote tratado para no llevar esporas a otras áreas de la finca."},
            ],
        },
        {
            "titulo": "Poda selectiva para ventilación",
            "descripcion": "Podar ramas bajas y chupones para mejorar aireación",
            "detalle_general": "La humedad alta y la poca circulación de aire favorecen el desarrollo de la roya. Una poda selectiva mejora la entrada de luz y aire al interior de la planta, creando un microclima menos favorable para el hongo. Es una práctica preventiva fundamental.",
            "tratamientos": [
                {"paso": 1, "titulo": "Identificar qué podar", "descripcion": "Seleccionar ramas bajas, chupones y ramas entrecruzadas", "detalle": "Identifique tres tipos de ramas a podar: 1) Ramas bajas que tocan el suelo o están a menos de 40cm de la superficie, 2) Chupones o brotes verticales que crecen del tronco principal y roban nutrientes, 3) Ramas que se entrecruzan o crecen hacia adentro de la planta. Marque cada una con cinta antes de empezar a podar para tener una visión clara.", "consejo": "Agáchese al nivel de la base de la planta y mire hacia arriba para ver la estructura interna con claridad.", "precaucion": "No pode más del 25% del follaje total. La planta necesita suficiente área foliar para fotosíntesis."},
                {"paso": 2, "titulo": "Podar con precisión", "descripcion": "Cortar en bisel y desinfectar herramientas", "detalle": "Use tijeras de podar afiladas. Haga cortes limpios en bisel (45°) a 1-2cm del tallo o rama principal. Los cortes lisos cicatrizan más rápido y evitan entrada de patógenos. Desinfecte las tijeras entre planta y planta con alcohol al 70%. Aplique pasta cicatrizante en cortes mayores a 2cm de diámetro para sellar la herida.", "consejo": "Tenga un recipiente con la solución desinfectante y sumerja las tijeras por 30 segundos entre cada planta.", "precaucion": "Las tijeras sin afilar hacen cortes desgarrados que demoran en cicatrizar. Afílelas cada semana si está podando seguido."},
                {"paso": 3, "titulo": "Mantener la densidad adecuada", "descripcion": "Dejar distancia entre plantas para circulación de aire", "detalle": "Revise la distancia entre plantas dentro del surco. Si están muy juntas (menos de 1.5m), considere podar más severamente las ramas laterales que se tocan entre plantas vecinas. La meta es que haya al menos 30-40cm de espacio entre el follaje de plantas contiguas para que circule el aire. En la calle entre surcos, mantenga libre de vegetación alta que bloquee el viento.", "consejo": "En surcos orientados de oriente a occidente (sentido del sol), la ventilación es naturalmente mejor.", "precaucion": "Después de podar, aplique un fungicida preventivo para proteger los cortes de posibles infecciones."},
            ],
        },
        {
            "titulo": "Fertilización potásica",
            "descripcion": "Aplicar potasio para fortalecer la planta contra la roya",
            "detalle_general": "El potasio fortalece las paredes celulares de las hojas y activa los mecanismos de defensa natural de la planta contra patógenos. Una planta bien nutrida con potasio es más resistente a la roya y se recupera más rápido.",
            "tratamientos": [
                {"paso": 1, "titulo": "Calcular la dosis", "descripcion": "Aplicar KCl 200kg/ha fraccionado en 2 veces", "detalle": "Calcule la dosis según el análisis de suelo idealmente. Si no tiene análisis, use 200kg de KCl (cloruro de potasio) por hectárea como dosis general. Divida esta cantidad en dos aplicaciones de 100kg cada una. La primera aplicación al inicio de la temporada de lluvias y la segunda 30 días después. Si su finca es menor a 1 hectárea, calcule 20g por planta.", "consejo": "Tome una muestra de suelo (1kg de tierra mezclada de 10 puntos del lote) y envíela a análisis. Saber exactamente qué necesita su suelo ahorra dinero.", "precaucion": "No aplique todo el potasio de una sola vez. Fraccionado se aprovecha mejor y evita pérdidas por lixiviación."},
                {"paso": 2, "titulo": "Aplicar correctamente", "descripcion": "Distribuir en corona alrededor de la planta", "detalle": "Haga un hoyo de 10-15cm de profundidad en forma de media luna alrededor de cada planta, a 30-40cm del tallo en el lado de mayor goteo (donde cae más agua de lluvia). Coloque la dosis de KCl en el hoyo y cubra con tierra. No lo deje expuesto al sol porque se volatiliza. No lo ponga directo al tallo porque puede quemar las raíces superficiales.", "consejo": "Aplique después de una lluvia cuando el suelo está húmedo para que el potasio se disuelva y esté disponible más rápido.", "precaucion": "Use guantes al manipular KCl porque es corrosivo para la piel, especialmente si tiene cortaduras."},
                {"paso": 3, "titulo": "Complementar con otros nutrientes", "descripcion": "Asegurar nitrógeno, fósforo y micronutrientes", "detalle": "El potasio funciona mejor cuando hay buen balance de nutrientes. 30 días después de la primera aplicación de potasio, aplique un fertilizante completo NPK (15-15-15 o 17-6-18-2) a 250g por planta. Agregue boro (20g/planta) y zinc (15g/planta) si su suelo es deficiente. Los micronutrientes ayudan a que la planta produzca compuestos de defensa contra hongos.", "consejo": "Aplique cal dolomítica (500g/planta) una vez al año para corregir acidez del suelo y aportar calcio y magnesio.", "precaucion": "No aplique fertilizantes nitrogenados en exceso — el exceso de nitrógeno hace hojas suculentas más susceptibles a roya."},
            ],
        },
        {
            "titulo": "Monitoreo intensivo cada 5 días",
            "descripcion": "Revisión constante de la evolución de la enfermedad",
            "detalle_general": "En nivel medio de infección, el monitoreo frecuente permite detectar cambios y actuar rápido antes de que suba a severidad alta. La prevención es más efectiva y económica que el control curativo.",
            "tratamientos": [
                {"paso": 1, "titulo": "Establecer puntos de muestreo fijos", "descripcion": "Seleccionar 10 plantas representativas del lote", "detalle": "Camine el lote en zigzag y seleccione 10 plantas distribuidas uniformemente que representen las diferentes condiciones del lote (parte alta, baja, borde, centro). Márquelas con cinta de colores o pintura en el tronco. Use siempre las mismas plantas para tener datos comparables. Divida el lote en 4 cuadrantes y seleccione 2-3 plantas por cuadrante.", "consejo": "Use estacas de madera pintadas de un color llamativo para marcar las plantas de monitoreo y encontrarlas fácilmente.", "precaucion": "Registre las coordenadas GPS de cada planta marcada en la app para que otros trabajadores también las encuentren."},
                {"paso": 2, "titulo": "Realizar la evaluación", "descripcion": "Revisar 10 hojas por planta y contar afectadas", "detalle": "De cada planta marcada, seleccione 10 hojas al azar (5 del tercio medio, 5 del inferior). Revise el envés de cada hoja buscando pústulas anaranjadas. Cuente cuántas de las 10 hojas tienen síntomas activos. También registre si hay hojas secas, defoliación o si aparecieron hojas nuevas sanas. Tome una foto de referencia de 2 hojas por planta.", "consejo": "Use la app para llenar los datos directamente en campo. Lleve la plantilla de monitoreo descargada por si no hay señal.", "precaucion": "No confunda manchas de otro origen (quemadura de sol, deficiencia nutricional) con roya. Las pústulas de roya tienen polvo anaranjado."},
                {"paso": 3, "titulo": "Registrar y decidir", "descripcion": "Anotar resultados y ajustar plan de manejo", "detalle": "Anote en la app los resultados: fecha, número de hojas afectadas por planta, severidad estimada (baja: 1-3 hojas, media: 4-6, alta: 7+). Si en dos monitoreos consecutivos la cantidad de hojas afectadas aumenta, refuerce con fungicida inmediatamente. Si disminuye, mantenga el plan actual. Comparta los datos con el asistente técnico de su finca.", "consejo": "Grafique los resultados cada mes para visualizar la tendencia. Puede usar Excel o la app para ver si la enfermedad está subiendo o bajando.", "precaucion": "No espere a que la enfermedad esté muy avanzada para actuar. Actuar en fase media siempre es más barato y efectivo."},
            ],
        },
    ],
    "ROYA_Baja": [
        {
            "titulo": "Fungicida preventivo",
            "descripcion": "Aplicar mancozeb cada 15 días",
            "detalle_general": "En nivel bajo de roya, el mancozeb actúa como protector de amplio espectro. No cura la enfermedad pero evita que las esporas germinen sobre la hoja. Es más económico que los fungicidas curativos y ideal para mantenimiento.",
            "tratamientos": [
                {"paso": 1, "titulo": "Preparar la dosis preventiva", "descripcion": "Mezclar 200g en 200L de agua", "detalle": "Use 200g de mancozeb por cada 200L de agua. Agregue el producto cuando el tanque tenga la mitad del agua, agite vigorosamente y complete con agua. El mancozeb se disuelve fácilmente, no requiere agitación prolongada. Use agua limpia de preferencia. Si el agua está turbia, déjela reposar 1 hora antes de usarla.", "consejo": "El mancozeb es compatible con la mayoría de fungicidas e insecticidas. Puede mezclarlo con el cúprico para ampliar el espectro.", "precaucion": "El mancozeb puede irritar los ojos. Use gafas de protección al preparar y aplicar. Lávese inmediatamente si le cae en los ojos."},
                {"paso": 2, "titulo": "Aplicar en momentos clave", "descripcion": "Aplicar preventivamente antes de lluvias intensas", "detalle": "El mejor momento es justo antes del inicio de la temporada de lluvias (marzo-abril y septiembre-octubre). También aplique después de una poda o cosecha. Use boquilla de abanico para mayor cobertura en aplicaciones preventivas. La presión debe ser media (30-40 PSI). Un gasto de 200-250L/ha es suficiente para cobertura completa.", "consejo": "Consulte el pronóstico del tiempo en la app del clima antes de aplicar. Idealmente aplique 1-2 días antes de la lluvia pronosticada.", "precaucion": "Evite aplicar con vientos fuertes (>15 km/h) para evitar deriva hacia cultivos vecinos o fuentes de agua."},
                {"paso": 3, "titulo": "Mantener frecuencia", "descripcion": "Repetir cada 15 días durante temporada de riesgo", "detalle": "Mantenga la aplicación cada 15 días durante los 3 meses de mayor precipitación en su zona. Después puede espaciar a cada 20-25 días. Si las condiciones son muy lluviosas, reduzca el intervalo a 12 días. El mancozeb tiene un período de protección de 7-10 días, por eso la frecuencia de 15 días es adecuada como preventivo.", "consejo": "Si ve que la roya empieza a aparecer a pesar del mancozeb, cambie a un fungicida sistémico o cúprico inmediatamente.", "precaucion": "No use mancozeb más de 6 veces por año en el mismo lote para evitar resistencia del hongo."},
            ],
        },
        {
            "titulo": "Monitoreo constante cada 8 días",
            "descripcion": "Revisar el cultivo cada 8 días para detectar signos tempranos",
            "detalle_general": "En fase preventiva, el monitoreo regular permite detectar la enfermedad en sus primeras etapas cuando el control es más fácil. Una revisión semanal bien hecha es la mejor herramienta de prevención.",
            "tratamientos": [
                {"paso": 1, "titulo": "Inspeccionar hojas estratégicas", "descripcion": "Revisar hojas del tercio medio e inferior", "detalle": "Cada 8 días revise 5 hojas por planta en 10 plantas al azar del lote. Elija hojas del tercio medio (3) y tercio inferior (2) porque ahí se concentra la humedad y empieza la roya. Mire el envés de cada hoja con cuidado. Busque puntos amarillos pequeños en el haz y pústulas naranja claras en el envés. Use una lupa de 10x si tiene para ver esporas tempranas.", "consejo": "Haga el monitoreo siempre el mismo día de la semana (ej. todos los miércoles) para crear rutina.", "precaucion": "No se confíe en la temporada seca — la roya puede sobrevivir en hojas viejas y reactivarse con la primera lluvia."},
                {"paso": 2, "titulo": "Usar la escala de severidad", "descripcion": "Clasificar lo observado en bajo, medio o alto", "detalle": "Use esta escala simple: 0 pústulas = sano, 1-3 hojas con pústulas = bajo, 4-6 hojas = medio, 7+ hojas = alto. Si encuentra nivel bajo registre y siga monitoreando. Si encuentra nivel medio, aplique la recomendación de severidad media. Si encuentra nivel alto en alguna planta, active el protocolo de roya alta inmediatamente.", "consejo": "Tenga una libreta o use la app para registrar cada monitoreo. Con los datos de 3 meses puede predecir los picos de la enfermedad en su finca.", "precaucion": "Revise también hojas que parecen sanas a simple vista. Las infecciones tempranas no se ven bien hasta que pasan 10-15 días."},
                {"paso": 3, "titulo": "Registrar condiciones ambientales", "descripcion": "Anotar lluvia, temperatura y humedad con cada monitoreo", "detalle": "Cada vez que monitoree, registre: días desde la última lluvia, temperatura media, presencia de neblina o rocío intenso. La roya necesita 6-12 horas de hoja mojada para infectar. Si registra lluvias frecuentes (cada 2-3 días) y temperaturas entre 18-25°C, las condiciones son óptimas para roya y debe reforzar la prevención.", "consejo": "Instale un pluviómetro casero (botella plástica cortada y marcada) en el lote para medir lluvia semanal.", "precaucion": "Si después de un periodo lluvioso encuentra roya generalizada en el lote, no espere al monitoreo siguiente — actúe de inmediato."},
            ],
        },
        {
            "titulo": "Fertilización balanceada",
            "descripcion": "Fortalecer la planta con una nutrición completa",
            "detalle_general": "Una nutrición equilibrada fortalece las defensas naturales de la planta. Las deficiencias nutricionales hacen al cafeto más vulnerable a enfermedades. Un plan de fertilización basado en análisis de suelo optimiza la producción y reduce la incidencia de roya.",
            "tratamientos": [
                {"paso": 1, "titulo": "Planificar la fertilización", "descripcion": "Aplicar NPK 15-3-31 a 250g/planta más micronutrientes", "detalle": "Aplique 250g por planta de un fertilizante completo NPK (15-3-31 o 17-6-18-2) en dos épocas del año: al inicio de la temporada de lluvias (abril) y a mediados del año (agosto). Si su suelo es ácido (pH < 5.5), aplique 200g de cal dolomítica por planta 30 días antes de la fertilización. Los micronutrientes como boro (5g/planta) y zinc (3g/planta) se pueden aplicar foliarmente.", "consejo": "Haga un análisis de suelo cada 2 años. No fertilice sin saber qué necesita realmente su suelo.", "precaucion": "No aplique fertilizantes granulados cuando el suelo esté seco. Necesitan humedad para disolverse. Aplique después de una lluvia."},
                {"paso": 2, "titulo": "Aplicar correctamente", "descripcion": "Distribuir en la zona de goteo, no al pie del tallo", "detalle": "Haga un surco circular de 10cm de profundidad alrededor de la planta, a 30-40cm del tallo en la zona de goteo (donde cae el agua de lluvia de las hojas). Coloque el fertilizante uniformemente en el surco y cubra con tierra. No lo deje expuesto porque los nutrientes se evaporan o se pierden con la escorrentía.", "consejo": "Divida la dosis en 2 hoyos opuestos alrededor de la planta para distribuir mejor las raíces.", "precaucion": "Si el suelo está muy inclinado (pendiente >15%), aplique en media luna del lado de arriba de la pendiente para evitar pérdida por escorrentía."},
                {"paso": 3, "titulo": "Complementar con abonos orgánicos", "descripcion": "Agregar materia orgánica para mejorar el suelo", "detalle": "Aplique 3-5kg de compost o gallinaza bien descompuesta por planta cada 6 meses. La materia orgánica mejora la retención de agua, la estructura del suelo y alimenta los microorganismos benéficos. Incorpore al suelo superficialmente (5-10cm) en la zona de goteo. No use estiércol fresco porque puede quemar raíces y atraer plagas.", "consejo": "Haga su propia composta con pulpa de café, hojas secas y gallinaza. En 3-4 meses tiene abono de alta calidad.", "precaucion": "Si compra gallinaza, verifique que esté bien composteada (olor a tierra, no a amoníaco). La gallinaza fresca es muy fuerte y quema las raíces."},
            ],
        },
        {
            "titulo": "Manejo de sombra",
            "descripcion": "Regular el porcentaje de sombra en el cafetal",
            "detalle_general": "La sombra moderada (30-40%) es beneficiosa para el café porque reduce el estrés por calor y mantiene la humedad equilibrada. Pero el exceso de sombra crea un microclima muy húmedo y oscuro que favorece la roya. El manejo correcto de la sombra es una herramienta preventiva clave.",
            "tratamientos": [
                {"paso": 1, "titulo": "Medir la sombra actual", "descripcion": "Evaluar el porcentaje de sombra en el lote", "detalle": "Mida la sombra a las 12 del mediodía usando una aplicación de medidor de luz en su celular o simplemente observando el patrón de luz en el suelo. Idealmente debe tener 30-40% de sombra (pasa luz filtrada, no sol directo ni sombra densa). Si camina por el lote al mediodía y ve parches grandes de sol directo >50%, hay poca sombra. Si parece de noche, hay demasiada sombra.", "consejo": "Tome 5 fotos del cielo desde el suelo en diferentes puntos del lote y compárelas visualmente para estimar el porcentaje de sombra.", "precaucion": "No elimine todos los árboles de sombra de golpe. El cambio brusco estresa el cafetal y reduce la producción por 1-2 años."},
                {"paso": 2, "titulo": "Regular los árboles de sombra", "descripcion": "Podar árboles para ajustar al 30-40% ideal", "detalle": "Si hay exceso de sombra (>40%), pode los árboles de sombra: elimine ramas bajas, aclare el interior del dosel y si es necesario tale algunos árboles dejando los mejores. Si hay poca sombra (<30%), siembre árboles de sombra temporal (guandul, plátano) mientras crecen los permanentes (guamo, nogal, carbonero). Los árboles de sombra deben tener raíces profundas para no competir con el café.", "consejo": "Los árboles de guamo (Inga spp.) son excelentes para sombra de café: dan sombra filtrada, fijan nitrógeno y producen leña.", "precaucion": "No pode los árboles de sombra en época de cosecha de café porque los frutos se estresan y pueden caerse prematuramente."},
                {"paso": 3, "titulo": "Monitorear el efecto", "descripcion": "Evaluar cómo responde el cafetal al ajuste de sombra", "detalle": "Después de ajustar la sombra, monitoree durante 2-3 meses: la incidencia de roya debería disminuir, las hojas deben verse más verdes y vigorosas, y la producción no debe caer. Las plantas bajo sombra excesiva suelen tener hojas grandes y delgadas (más susceptibles a roya). Después del ajuste, las hojas nuevas deben ser más gruesas y firmes.", "consejo": "Mida la producción por planta (kg de café cereza) antes y después del ajuste de sombra para ver el beneficio real.", "precaucion": "En zonas muy cálidas (>30°C), no reduzca la sombra por debajo del 35% porque el sol directo puede estresar las plantas."},
            ],
        },
        {
            "titulo": "Aplicar micorrizas",
            "descripcion": "Fortalecer el sistema radicular con hongos benéficos",
            "detalle_general": "Las micorrizas son hongos benéficos que se asocian simbióticamente con las raíces del café, mejorando la absorción de agua y nutrientes (especialmente fósforo). Plantas con micorrizas desarrolladas son más resistentes al estrés y a enfermedades como la roya.",
            "tratamientos": [
                {"paso": 1, "titulo": "Adquirir y almacenar correctamente", "descripcion": "Comprar inoculante micorrícico de calidad", "detalle": "Compre inoculante micorrícico comercial (Glomus spp., Rhizophagus spp.) en presentación de polvo o granulado. Verifique la fecha de vencimiento y la cantidad de esporas viables por gramo (debe tener mínimo 100 esporas/g). Almacene en lugar fresco (18-25°C) y seco, nunca al sol. Una vez abierto el empaque, use todo en máximo 30 días.", "consejo": "Los productos nacionales suelen ser más frescos y adaptados a las condiciones locales. Pregunte en su almacén agrícola de confianza.", "precaucion": "No mezcle micorrizas con fungicidas químicos — el fungicida mata el hongo benéfico. Espere al menos 15 días entre aplicaciones."},
                {"paso": 2, "titulo": "Aplicar al suelo", "descripcion": "Aplicar 5g por planta en la zona radicular", "detalle": "Haga un hoyo de 5-8cm de profundidad a 20cm del tallo en 3 puntos alrededor de la planta. Deposite 5g del inoculante en cada hoyo (15g total por planta) y cubra con tierra inmediatamente — las esporas son sensibles a la luz UV. Aplique en suelo húmedo, preferiblemente después de una lluvia. El mejor momento es al inicio de la temporada de lluvias.", "consejo": "Aplique las micorrizas junto con abono orgánico (compost) para que las esporas tengan sustrato donde desarrollarse.", "precaucion": "No aplique fertilizantes con alto contenido de fósforo (P) al mismo tiempo que las micorrizas. El fósforo químico inhibe la simbiosis."},
                {"paso": 3, "titulo": "Mantener el sistema establecido", "descripcion": "Repetir cada 60 días y mantener buenas prácticas de suelo", "detalle": "Repita la aplicación cada 60 días durante la primera temporada (2-3 aplicaciones). A partir del segundo año, una aplicación al inicio de lluvias es suficiente si el suelo se maneja bien. Evite la quema de residuos, la labranza excesiva y los fungicidas químicos cerca de las raíces porque destruyen las redes de micorrizas. Mantenga cobertura vegetal entre los surcos para proteger el suelo.", "consejo": "Si tiene plantas de café jóvenes (menos de 2 años), inocularlas con micorrizas desde el vivero da plantas más fuertes desde el inicio.", "precaucion": "Si nota que las plantas tienen raíces poco desarrolladas o crecen lentamente, las micorrizas ayudarán pero debe corregir primero problemas de drenaje o compactación."},
            ],
        },
    ],
    "Hoja_Sana": [
        {
            "titulo": "Planta en buen estado",
            "descripcion": "Continúa con el manejo habitual que has mantenido",
            "detalle_general": "Tus plantas están sanas. Sigue aplicando las buenas prácticas que te han funcionado hasta ahora. La prevención y el monitoreo constante son la clave para mantener el cafetal sano y productivo.",
            "tratamientos": [
                {"paso": 1, "titulo": "Mantener las prácticas actuales", "descripcion": "Sigue con el programa de manejo que tienes", "detalle": "Continúa con la fertilización, el control de malezas y las podas que has estado haciendo. Si tu cafetal está sano, significa que tu manejo está funcionando. No bajes la guardia: la roya puede aparecer cuando cambian las condiciones climáticas. Mantén la frecuencia de monitoreo cada 15 días para detectar cualquier cambio a tiempo.", "consejo": "Documenta todo lo que haces en la app para tener un registro de lo que funciona en tu finca.", "precaucion": "No te confíes por completo — la roya puede aparecer repentinamente después de lluvias intensas o cambios de temperatura."},
                {"paso": 2, "titulo": "Revisar el calendario de fertilización", "descripcion": "Asegurarse de que la próxima fertilización está programada", "detalle": "Revisa en la app o en tus registros cuándo fue la última fertilización y cuándo toca la próxima. Las plantas sanas necesitan nutrientes constantes para mantenerse fuertes. No esperes a que aparezcan deficiencias. Si tienes dudas sobre el plan de fertilización, consulta con un asistente técnico cada 6 meses para ajustar según la etapa del cultivo.", "consejo": "Los meses críticos para fertilizar son abril (inicio lluvias) y agosto (mitad de año). Ten los insumos listos con anticipación.", "precaucion": "No sobre-fertilices. El exceso de nitrógeno hace las hojas más suculentas y atractivas para plagas y enfermedades."},
                {"paso": 3, "titulo": "Programar la próxima poda", "descripcion": "Planificar poda de mantenimiento para después de cosecha", "detalle": "Después de la cosecha principal, planifica una poda de mantenimiento: elimina ramas secas, chupones y ramas bajas que tocan el suelo. Esto mantendrá el cafetal ventilado y facilitará las labores. También es buen momento para evaluar si alguna planta necesita ser renovada (zoca) por baja producción o daño estructural.", "consejo": "La poda después de cosecha también ayuda a controlar la broca del café al eliminar ramas con frutos secos que quedaron.", "precaucion": "No podes en plena época de lluvias — los cortes tardan más en cicatrizar y son entrada de enfermedades. Espera a que pase el pico de lluvias."},
            ],
        },
        {
            "titulo": "Riego adecuado",
            "descripcion": "Mantener la humedad ideal según la temporada",
            "detalle_general": "El agua es esencial para el cafeto, pero tanto el exceso como el déficit hídrico estresan la planta y la hacen más vulnerable a enfermedades. Un riego bien manejado mantiene las defensas de la planta activas.",
            "tratamientos": [
                {"paso": 1, "titulo": "Evaluar la necesidad de riego", "descripcion": "Revisar la humedad del suelo y el clima", "detalle": "Revisa la humedad del suelo introduciendo un palo de madera (como un pincho de bambú) a 20cm de profundidad. Si sale limpio y seco, necesita riego. Si sale con tierra húmeda adherida, tiene suficiente humedad. También revisa el pronóstico del clima: si hay lluvia en los próximos 2-3 días, espera. El cafeto necesita entre 1,200-1,800mm de lluvia al año, distribuidos uniformemente.", "consejo": "Un pluviómetro casero te ayuda a saber exactamente cuánta lluvia está cayendo en tu finca.", "precaucion": "No riegues en exceso. El encharcamiento prolongado pudre las raíces y favorece hongos del suelo como Phytophthora."},
                {"paso": 2, "titulo": "Aplicar riego eficiente", "descripcion": "Riego por goteo o aspersión baja en horas frescas", "detalle": "El riego por goteo es el más eficiente para café: usa menos agua y la aplica directamente a la zona radicular. Si no tienes goteo, usa aspersión baja (microaspersores) en horas de la mañana temprano para que las hojas se sequen rápido. El riego por la noche moja las hojas por muchas horas, creando condiciones perfectas para la roya. Riega cada 8-12 días en temporada seca, ajustando según el tipo de suelo.", "consejo": "Un sistema de riego por goteo casero con manguera de cintilla y un tanque elevado de 200L puede regar hasta 50 plantas.", "precaucion": "Si riegas por aspersión, hazlo siempre en la mañana (6-9am) para que las hojas estén secas antes del atardecer."},
                {"paso": 3, "titulo": "Mantener cobertura del suelo", "descripcion": "Mulch o cobertura vegetal para conservar humedad", "detalle": "Mantén una capa de 5-10cm de materia orgánica (hojarasca, pulpa de café composteada, pasto seco) alrededor de las plantas, sin tocar el tallo. Esto reduce la evaporación del agua del suelo, mantiene la temperatura más estable y alimenta los microorganismos benéficos. También evita que el sol golpee directamente el suelo y que las malezas crezcan.", "consejo": "La hojarasca de los mismos árboles de sombra es el mejor mulch: es gratuita, local y se descompone lentamente.", "precaucion": "No pongas el mulch en contacto directo con el tallo del cafeto para evitar pudrición del cuello de la raíz."},
            ],
        },
        {
            "titulo": "Fertilización preventiva con bioestimulantes",
            "descripcion": "Aplicar bioestimulantes foliares para fortalecer defensas",
            "detalle_general": "Los bioestimulantes foliares (aminoácidos, algas marinas, ácidos húmicos) activan los mecanismos de defensa natural de la planta. No son fertilizantes convencionales sino potenciadores del metabolismo vegetal que preparan al cafeto para resistir estrés biótico (plagas, enfermedades).",
            "tratamientos": [
                {"paso": 1, "titulo": "Preparar la mezcla de bioestimulantes", "descripcion": "Diluir 3cc/L de aminoácidos + algas marinas", "detalle": "Mezcla 3cc de aminoácidos por litro de agua más 2cc de extracto de algas marinas (Ascophyllum nodosum) por litro. Use agua limpia, preferiblemente de lluvia o reposada. Complete el volumen en la bomba de fumigación y agite suavemente. Los aminoácidos estimulan la síntesis de proteínas de defensa y las algas marinas aportan hormonas vegetales naturales (citoquininas, auxinas).", "consejo": "Compra los productos concentrados y mézclalos tú mismo. Sale más económico que comprar mezclas listas y puedes ajustar las dosis.", "precaucion": "Agita suavemente, no enérgicamente. La espuma excesiva puede indicar degradación del producto."},
                {"paso": 2, "titulo": "Aplicar al follaje", "descripcion": "Aplicar cada 20 días, dirigido al envés de las hojas", "detalle": "Aplica la mezcla con bomba de espalda, dirigiendo al envés de las hojas donde los estomas absorben mejor. Use boquilla de cono fino. La dosis recomendada es 200L de mezcla por hectárea. Aplica en horas frescas (6-9am). Los bioestimulantes funcionan mejor cuando hay buena humedad en el suelo (después de una lluvia o riego).", "consejo": "Agrega un surfactante agrícola (0.5cc/L) para que las gotas se extiendan mejor sobre la hoja cerosa.", "precaucion": "No mezcles bioestimulantes con fungicidas cúpricos en la misma bomba — el cobre puede oxidar los aminoácidos y reducir su efectividad."},
                {"paso": 3, "titulo": "Programar aplicaciones estratégicas", "descripcion": "Aplicar antes de periodos de estrés (lluvias intensas, floración)", "detalle": "Aplica una dosis de bioestimulantes 10-15 días antes del inicio de la temporada de lluvias intensas para preparar las defensas de la planta. También aplica antes de la floración principal (enero-febrero en Colombia) para asegurar una floración vigorosa. Durante la cosecha, las plantas están estresadas y se benefician de una aplicación de recuperación después de la recolección.", "consejo": "Marque en la app un calendario fijo: cada 20 días desde abril hasta diciembre. Prepare la mezcla la noche anterior para ahorrar tiempo.", "precaucion": "No apliques bioestimulantes en horas de calor intenso (>30°C). Las hojas cierran los estomas y no absorben el producto."},
            ],
        },
        {
            "titulo": "Sombra adecuada",
            "descripcion": "Mantener el equilibrio de sombra en el cafetal",
            "detalle_general": "La sombra es uno de los factores más importantes en un cafetal sostenible. Regula la temperatura, mantiene la humedad del suelo, aporta materia orgánica y crea un hábitat para enemigos naturales de plagas. El reto es encontrar el punto justo donde la sombra beneficia sin favorecer enfermedades.",
            "tratamientos": [
                {"paso": 1, "titulo": "Evaluar el nivel de sombra", "descripcion": "Verificar que el cafetal tenga 30-40% de sombra", "detalle": "Usa el método del mediodía: párate en el centro del lote a las 12pm y mira el suelo. Deberías ver una mezcla de luz filtrada y pequeñas manchas de sol directo (30-40% de la superficie iluminada, 60-70% sombreada). Si ves sombra muy densa donde apenas entra luz, hay exceso de sombra. Si ves casi todo iluminado, falta sombra. Complementa con una foto del dosel para tener referencia.", "consejo": "Descarga una app de luxómetro en tu celular. Mide 500-800 lux a nivel del suelo al mediodía para un nivel ideal de sombra.", "precaucion": "No compares la sombra de tu finca con la de otros sin considerar la altitud. En zonas más bajas (más cálidas) se necesita más sombra que en zonas altas."},
                {"paso": 2, "titulo": "Mantener los árboles de sombra", "descripcion": "Podar y cuidar los árboles existentes", "detalle": "Poda los árboles de sombra una vez al año después de la cosecha principal. Elimina ramas secas, enfermas o muy bajas que estorben el paso. Los árboles de sombra también necesitan fertilización — aplicar abono orgánico alrededor de su base cada año. Si hay árboles muy viejos o propensos a caerse, reemplázalos gradualmente por plántulas de especies adecuadas.", "consejo": "Los mejores árboles de sombra para café: guamo (Inga edulis), carbonero (Calliandra), nogal (Cordia alliodora) y plátano como sombra temporal.", "precaucion": "No use árboles que tengan raíces superficiales agresivas (como eucalipto) que compiten fuertemente con el café por agua y nutrientes."},
                {"paso": 3, "titulo": "Monitorear la sombra durante el año", "descripcion": "Ajustar según la época del año", "detalle": "La sombra no es estática — cambia con las estaciones. En verano, los árboles de sombra (especialmente los deciduos) pierden hojas y entra más luz, lo cual es bueno. En invierno, los árboles tienen más hojas y la sombra es más densa. Monitorea la sombra al menos 4 veces al año (inicio y final de cada temporada de lluvias) y ajusta con podas ligeras si es necesario.", "consejo": "Lleva un calendario fotográfico: toma la misma foto del dosel desde el mismo punto cada 3 meses para ver la evolución de la sombra.", "precaucion": "No hagas cambios drásticos de sombra en menos de 6 meses. El cafetal necesita tiempo para adaptarse a los cambios de luminosidad."},
            ],
        },
    ],
}

# =========================
# GENERAR RECOMENDACIONES
# =========================

def generar_recomendaciones(class_name: str, severidad: str, image_bytes: bytes) -> list:
    if class_name == "arbol_cafe":
        return []

    if class_name == "Hoja_Sana":
        pool_key = "Hoja_Sana"
        cantidad = 2
    elif class_name == "Enfermedad_ROYA":
        pool_key = f"ROYA_{severidad}"
        cantidad = 2 if severidad in ("Media", "Baja") else 3
    else:
        return []

    pool = POOL_RECOMENDACIONES.get(pool_key, [])
    if not pool:
        return []

    seed = hashlib.md5(image_bytes).hexdigest()
    rng = random.Random(seed)
    seleccionadas = rng.sample(pool, k=min(cantidad, len(pool)))

    return [dict(r) for r in seleccionadas]

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
        # LEER BYTES DE LA IMAGEN
        # =========================

        with open(file_path, "rb") as f:
            image_bytes = f.read()

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

                severidad = calcular_severidad(class_name, confidence)
                info = ENFERMEDADES_INFO.get(class_name, {})
                recomendaciones = generar_recomendaciones(class_name, severidad, image_bytes)

                detections.append({
                    "class": class_name,
                    "confidence": confidence,
                    "severidad": severidad,
                    "nombre_cientifico": info.get("nombre_cientifico", ""),
                    "nombre_comun": info.get("nombre_comun", ""),
                    "recomendaciones": recomendaciones,
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
        