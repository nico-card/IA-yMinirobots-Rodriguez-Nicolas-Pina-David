"""
Robot con Control de Voz por IA (Red Neuronal / NLP) — Pygame + Inteligencia Artificial
=======================================================================================
"""

import math
import sys
import threading
import pygame
import speech_recognition as sr

# NUEVO: Librerías para el procesamiento de lenguaje y red neuronal/clasificador
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# 1. Configuración de la pantalla y cuadrícula
# ---------------------------------------------------------------------------
GRID_SIZE = 20  # Matriz de 20x20 casillas
CELL_SIZE = 32  # Cada casilla mide 32x32 píxeles
UI_HEIGHT = 130  # Espacio inferior para el panel de control de la IA

WIN_W = GRID_SIZE * CELL_SIZE
WIN_H = GRID_SIZE * CELL_SIZE + UI_HEIGHT

# Paleta de colores (RGB)
C_BG = (245, 245, 250)
C_GRID = (210, 210, 220)
C_GRID_ALT = (238, 238, 245)
C_BODY = (41, 98, 205)
C_BODY_RIM = (20, 55, 140)
C_ARROW = (255, 215, 0)
C_TRAIL = (170, 200, 255)
C_PANEL = (30, 32, 48)
C_TEXT_HI = (255, 255, 255)
C_TEXT_LO = (160, 165, 190)
C_GREEN = (72, 199, 116)
C_RED = (255, 80, 80)
C_YELLOW = (255, 200, 50)
C_BLUE_LT = (100, 160, 255)

# Direcciones representadas de forma matricial (X, Y)
# En Pygame, el eje Y crece hacia ABAJO. Por eso Norte es (0, -1)
DIR_DELTA = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
DIR_LABEL = {0: "Norte ↑", 1: "Este →", 2: "Sur ↓", 3: "Oeste ←"}
DIR_ANGLE = {0: 90, 1: 0, 2: 270, 3: 180}


# ---------------------------------------------------------------------------
# 2. Cerebro de la IA: Clasificador de Intenciones (NLP)
# ---------------------------------------------------------------------------
class IACommandClassifier:
    """Esta clase actúa como el cerebro NLP del robot.

    Se entrena con ejemplos y aprende a generalizar sinónimos.
    """

    def __init__(self):
        # El Vectorizador convierte el texto ("avanza por favor") en números basados en la importancia de las palabras
        self.vectorizer = TfidfVectorizer()
        # Usamos Regresión Logística (equivalente a una red neuronal de una capa, ideal para texto ligero)
        self.model = LogisticRegression()

        # Dataset de entrenamiento: Mapeamos frases comunes en español a "etiquetas" (comandos)
        self.training_data = {
            "adelante": [
                "adelante",
                "avanza",
                "avanzar",
                "camina",
                "da un paso",
                "sigue derecho",
                "mueve hacia el frente",
                "anda",
                "dale",
                "recto",
            ],
            "atras": [
                "atrás",
                "atras",
                "retrocede",
                "retroceder",
                "camina hacia atrás",
                "regresa",
                "un paso atrás",
                "reversa",
            ],
            "izquierda": [
                "gira a la izquierda",
                "izquierda",
                "mira a la izquierda",
                "voltea a la izquierda",
                "rotar a la izquierda",
                "orientación izquierda",
            ],
            "derecha": [
                "gira a la derecha",
                "derecha",
                "mira a la derecha",
                "voltea a la derecha",
                "rotar a la derecha",
                "orientación derecha",
            ],
            "mueve_izquierda": [
                "mueve izquierda",
                "mover izquierda",
                "muévete hacia la izquierda",
                "muévete a la izquierda",
                "avanza hacia la izquierda",
                "gira y mueve izquierda",
                "gira a la izquierda y avanza",
                "cruza a la izquierda",
                "desplázate a la izquierda"
            ],
            "mueve_derecha": [
                "mueve derecha",
                "mover derecha",
                "muévete hacia la derecha",
                "muévete a la derecha",
                "avanza hacia la derecha",
                "gira y mueve derecha",
                "gira a la derecha y avanza",
                "cruza a la derecha",
                "desplázate a la derecha"
            ],
        }
        self._train()

    def _train(self):
        #Compila los datos y entrena el modelo matemático."""
        X_texts = []
        y_labels = []

        for label, sentences in self.training_data.items():
            for sentence in sentences:
                X_texts.append(sentence.lower())
                y_labels.append(label)

        # Matriz de características (numérica)
        X_train = self.vectorizer.fit_transform(X_texts)
        self.model.fit(X_train, y_labels)

    def predict(self, text: str) -> tuple[str, float]:
        """Recibe una frase de voz, calcula las probabilidades y devuelve la

        intención más probable.
        """
        text_cleaned = text.lower().strip()
        # Convertir la frase del usuario al formato numérico que entiende el modelo
        X_vec = self.vectorizer.transform([text_cleaned])

        # Predecir la etiqueta y obtener la certeza (probabilidad)
        prediction = self.model.predict(X_vec)[0]
        probabilities = self.model.predict_proba(X_vec)
        max_prob = max(probabilities[0])

        # Umbral de confianza: Si la IA no está al menos 35% segura, prefiere no adivinar
        if max_prob < 0.35:
            return "desconocido", max_prob

        return prediction, max_prob


# ---------------------------------------------------------------------------
# 3. Clase Robot (Lógica del entorno virtual)
# ---------------------------------------------------------------------------
class Robot:

    def __init__(self):
        # Inicia en el centro de la cuadrícula
        self.x = GRID_SIZE // 2
        self.y = GRID_SIZE // 2
        self.direction = 0  # 0=Norte, 1=Este, 2=Sur, 3=Oeste
        self.trail: list[tuple[int, int]] = []  # Historial de posiciones (rastro)

    def _step(self, forward: bool) -> bool:
        """Desplaza físicamente al robot en la matriz cuidando los límites."""
        dx, dy = DIR_DELTA[self.direction]
        if not forward:
            dx, dy = -dx, -dy  # Invierte las coordenadas para ir en reversa

        nx, ny = self.x + dx, self.y + dy
        # Validación de fronteras
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            self.trail.append((self.x, self.y))
            if len(self.trail) > 30:  # El rastro solo guarda los últimos 30 pasos
                self.trail.pop(0)
            self.x, self.y = nx, ny
            return True
        return False

    def move_forward(self) -> bool:
        return self._step(True)

    def move_backward(self) -> bool:
        return self._step(False)

    def turn_left(self) -> None:
        self.direction = (self.direction - 1) % 4

    def turn_right(self) -> None:
        self.direction = (self.direction + 1) % 4

    def turn_and_move_left(self) -> bool:
        self.turn_left()
        return self.move_forward()

    def turn_and_move_right(self) -> bool:
        self.turn_right()
        return self.move_forward()


# ---------------------------------------------------------------------------
# 4. Enrutador de Intenciones
# ---------------------------------------------------------------------------
def execute_ia_intent(intent: str, robot: Robot) -> str:

    if intent == "mueve_izquierda":
        ok = robot.turn_and_move_left()
        return (
            "↖ Cort: Girar + Avanzar Izquierda" + ("" if ok else " (Límite mapa)")
        )

    elif intent == "mueve_derecha":
        ok = robot.turn_and_move_right()
        return "↗ Cort: Girar + Avanzar Derecha" + ("" if ok else " (Límite mapa)")

    elif intent == "adelante":
        ok = robot.move_forward()
        return "⬆ Cort: Adelante" + ("" if ok else " (Límite mapa)")

    elif intent == "atras":
        ok = robot.move_backward()
        return "⬇ Cort: Atrás" + ("" if ok else " (Límite mapa)")

    elif intent == "izquierda":
        robot.turn_left()
        return "↺ Cort: Rotación Izquierda (In situ)"

    elif intent == "derecha":
        robot.turn_right()
        return "↻ Cort: Rotación Derecha (In situ)"

    return " Cortana: Comando no comprendido o inseguro"


# ---------------------------------------------------------------------------
# 5. Controlador de voz por hilos (Multithreading)
# ---------------------------------------------------------------------------
class VoiceController:

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.commands: list[str] = []  # Cola de comandos de texto recibidos
        self.status = "Iniciando micrófono…"
        self.ready = False
        self.running = True

    def listen_loop(self):
        try:
            with sr.Microphone() as src:
                # Calibración inicial para ignorar el ruido ambiental (ventiladores, clics)
                self.recognizer.energy_threshold = 300
                self.recognizer.adjust_for_ambient_noise(src, duration=1.5)
                self.status = "🎤 Escuchando... Di un comando (Cortana conectada)"
                self.ready = True

                while self.running:
                    try:
                        # Escucha fragmentos de voz de hasta 5 segundos
                        audio = self.recognizer.listen(
                            src, timeout=3, phrase_time_limit=4
                        )
                        self.status = "⏳ Procesando audio por Red..."

                        # Transcripción del audio a texto (Speech to Text)
                        text = self.recognizer.recognize_google(
                            audio, language="es-ES"
                        )
                        self.commands.append(text)
                        self.status = f'✅ Comando aceptado "{text}"'
                    except sr.WaitTimeoutError:
                        self.status = "🎤 Escuchando..¿Qué debería hacer?"
                    except sr.UnknownValueError:
                        self.status = "🔇 Las interferencias de la comunicación no nos dejan conectarnos, por favor vuelva a decir el comando señor"
                    except sr.RequestError as e:
                        self.status = f"❌ Nos destruyeron las comunicaciones: {e}"
        except OSError as e:
            self.status = f"❌ Ya no existen comunicaciones, intente de nuevo: {e}"
            self.ready = False


# ---------------------------------------------------------------------------
# 6. Funciones Gráficas (Renderizado de interfaz)
# ---------------------------------------------------------------------------
def draw_robot(surf: pygame.Surface, robot: Robot):

    # Dibujar rastro difuminado
    for i, (tx, ty) in enumerate(robot.trail):
        alpha = int(80 * (i + 1) / len(robot.trail)) if robot.trail else 80
        cx = tx * CELL_SIZE + CELL_SIZE // 2
        cy = ty * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(surf, C_TRAIL, (cx, cy), CELL_SIZE // 6)

    # Centro geométrico actual
    cx = robot.x * CELL_SIZE + CELL_SIZE // 2
    cy = robot.y * CELL_SIZE + CELL_SIZE // 2
    r = CELL_SIZE // 2 - 3

    # Sombra del robot
    pygame.draw.circle(surf, (180, 185, 210), (cx + 2, cy + 3), r)
    # Chasis principal
    pygame.draw.circle(surf, C_BODY, (cx, cy), r)
    pygame.draw.circle(surf, C_BODY_RIM, (cx, cy), r, 2)

    # Cálculo vectorial de la flecha de dirección (Trigonometría en radianes)
    a_rad = math.radians(DIR_ANGLE[robot.direction])
    tip_dist = r - 4
    base_dist = r // 2
    perp_rad = a_rad + math.pi / 2

    tip = (
        cx + int(tip_dist * math.cos(a_rad)),
        cy - int(tip_dist * math.sin(a_rad)),
    )
    b1 = (
        cx
        + int(base_dist * math.cos(perp_rad))
        - int((tip_dist * 0.3) * math.cos(a_rad)),
        cy
        - int(base_dist * math.sin(perp_rad))
        + int((tip_dist * 0.3) * math.sin(a_rad)),
    )
    b2 = (
        cx
        - int(base_dist * math.cos(perp_rad))
        - int((tip_dist * 0.3) * math.cos(a_rad)),
        cy
        + int(base_dist * math.sin(perp_rad))
        + int((tip_dist * 0.3) * math.sin(a_rad)),
    )

    pygame.draw.polygon(surf, C_ARROW, [tip, b1, b2])

    # Ojo visor (Frente del robot)
    eye_x = cx + int((r - 8) * math.cos(a_rad))
    eye_y = cy - int((r - 8) * math.sin(a_rad))
    pygame.draw.circle(surf, C_TEXT_HI, (eye_x, eye_y), 3)


def draw_panel(
    surf: pygame.Surface,
    robot: Robot,
    voice: VoiceController,
    last_action: str,
    fonts: dict,
    ia_info: str,
):
    """Renderiza el HUD informativo inferior."""
    panel_y = GRID_SIZE * CELL_SIZE
    pygame.draw.rect(surf, C_PANEL, (0, panel_y, WIN_W, UI_HEIGHT))
    pygame.draw.line(surf, C_BLUE_LT, (0, panel_y), (WIN_W, panel_y), 1)

    # Led indicador de micrófono
    mic_color = C_GREEN if voice.ready else C_RED
    pygame.draw.circle(surf, mic_color, (14, panel_y + 18), 7)

    # Mostrar estados en pantalla
    surf.blit(fonts["md"].render(voice.status, True, C_TEXT_HI), (28, panel_y + 10))
    surf.blit(fonts["lg"].render(last_action, True, C_YELLOW), (10, panel_y + 36))
    surf.blit(fonts["sm"].render(ia_info, True, C_GREEN), (10, panel_y + 62))

    info = f"Coordenadas: ({robot.x}, {robot.y})  |  Vector: {DIR_LABEL[robot.direction]}"
    surf.blit(fonts["sm"].render(info, True, C_TEXT_LO), (10, panel_y + 82))

    help_txt = "Prueba decir sinónimos: 'camina', 'reversa', 'voltea a la derecha', 'da un paso hacia adelante'"
    surf.blit(fonts["xs"].render(help_txt, True, C_TEXT_LO), (10, panel_y + 108))

# ---------------------------------------------------------------------------
# 6. Funciones Gráficas (Renderizado de interfaz)
# ---------------------------------------------------------------------------
def draw_robot(surf: pygame.Surface, robot: Robot):
    """Renderiza el jugador con forma de Nave Espacial vectorial."""
    
    # 1. Dibujar estela de propulsión (rastro en el mapa)
    for i, (tx, ty) in enumerate(robot.trail):
        # La estela se desvanece y se hace más pequeña mientras más antigua sea
        factor = (i + 1) / len(robot.trail)
        cx = tx * CELL_SIZE + CELL_SIZE // 2
        cy = ty * CELL_SIZE + CELL_SIZE // 2
        radio_estela = max(2, int((CELL_SIZE // 6) * factor))
        
        # Color naranja/amarillento para la energía del motor
        pygame.draw.circle(surf, (255, 170, 80), (cx, cy), radio_estela)

    # 2. Centro geométrico de la celda actual
    cx = robot.x * CELL_SIZE + CELL_SIZE // 2
    cy = robot.y * CELL_SIZE + CELL_SIZE // 2

    # Ángulo actual en radianes
    a_rad = math.radians(DIR_ANGLE[robot.direction])

    # Función auxiliar para rotar los puntos 2D de la nave según la dirección
    def rot_pt(px, py):
        # px, py son coordenadas relativas asumiendo que la nave mira hacia la derecha (Este)
        # Rotamos usando la matriz de rotación 2D y ajustamos para el eje Y invertido de Pygame
        rx = cx + px * math.cos(a_rad) - py * math.sin(a_rad)
        ry = cy - (px * math.sin(a_rad) + py * math.cos(a_rad))
        return int(rx), int(ry)

    # 3. Calcular los vértices de la Nave (Chasis tipo Ala Delta)
    punta         = rot_pt(14, 0)      # Nariz de la nave
    ala_izq       = rot_pt(-12, -14)   # Punta del ala izquierda (superior en la pantalla)
    base_izq      = rot_pt(-6, -4)     # Unión ala-fuselaje izquierdo
    motor_trasero = rot_pt(-10, 0)     # Muesca trasera para el motor
    base_der      = rot_pt(-6, 4)      # Unión ala-fuselaje derecho
    ala_der       = rot_pt(-12, 14)    # Punta del ala derecha (inferior en la pantalla)

    forma_nave = [punta, ala_der, base_der, motor_trasero, base_izq, ala_izq]

    # 4. Dibujar sombra (desfasada ligeramente hacia abajo a la derecha)
    sombra_nave = [(x + 3, y + 4) for x, y in forma_nave]
    pygame.draw.polygon(surf, (180, 185, 210), sombra_nave)

    # 5. Dibujar propulsor (Llama principal)
    llama_1 = rot_pt(-10, -4)
    llama_2 = rot_pt(-20, 0)  # Punta extrema de la llama
    llama_3 = rot_pt(-10, 4)
    pygame.draw.polygon(surf, C_YELLOW, [llama_1, llama_2, llama_3])
    pygame.draw.polygon(surf, C_RED, [llama_1, llama_2, llama_3], 1) # Borde de la llama

    # 6. Dibujar el chasis principal
    C_NAVE = (40, 45, 60)      # Gris oscuro espacial
    C_BORDES = (100, 150, 255) # Azul tecnológico para los bordes
    
    pygame.draw.polygon(surf, C_NAVE, forma_nave)
    pygame.draw.polygon(surf, C_BORDES, forma_nave, 2) # Contorno metálico

    # 7. Dibujar la cabina de cristal (Cockpit)
    cabina_pts = [rot_pt(6, 0), rot_pt(-2, -4), rot_pt(-5, 0), rot_pt(-2, 4)]
    C_CRISTAL = (0, 200, 255)
    
    pygame.draw.polygon(surf, C_CRISTAL, cabina_pts)
    # Reflejo blanco en la punta de la cabina
    pygame.draw.circle(surf, C_TEXT_HI, rot_pt(2, -1), 1)
# ---------------------------------------------------------------------------
# 7. Ciclo Principal de Ejecución (Main Loop)
# ---------------------------------------------------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("🤖 Robot Inteligente NLP con Red Neuronal")

    fonts = {
        "xs": pygame.font.SysFont("Arial", 11, italic=True),
        "sm": pygame.font.SysFont("Arial", 13),
        "md": pygame.font.SysFont("Arial", 14, bold=True),
        "lg": pygame.font.SysFont("Arial", 17, bold=True),
    }

    clock = pygame.time.Clock()
    robot = Robot()
    voice = VoiceController()

    # NUEVO: Instanciar y entrenar la Inteligencia Artificial al arrancar
    print("Entrenando cerebro NLP...")
    brain = IACommandClassifier()
    print("¡Cortana está lista y cargada con éxito!")

    # Desplegar el hilo para la escucha asíncrona del micrófono
    voice_thread = threading.Thread(target=voice.listen_loop, daemon=True)
    voice_thread.start()

    last_action = "Esperando entrada de voz o teclado..."
    ia_info = "Cortana : En reposo"

    # Caché gráfico de la cuadrícula para optimizar FPS
    grid_surf = pygame.Surface((WIN_W, GRID_SIZE * CELL_SIZE))
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            color = C_GRID_ALT if (x + y) % 2 == 0 else C_BG
            pygame.draw.rect(
                grid_surf,
                color,
                (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE),
            )
            pygame.draw.rect(
                grid_surf,
                C_GRID,
                (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                1,
            )

    running = True
    while running:
        # --- Captura de Interrupciones de Usuario (Teclado/Mouse) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_UP:
                    robot.move_forward()
                    last_action = "Teclado manual: Adelante"
                elif event.key == pygame.K_DOWN:
                    robot.move_backward()
                    last_action = "Teclado manual: Atrás"
                elif event.key == pygame.K_LEFT:
                    robot.turn_left()
                    last_action = "Teclado manual: Giro Izquierda"
                elif event.key == pygame.K_RIGHT:
                    robot.turn_right()
                    last_action = "Teclado manual: Giro Derecha"

        # --- Lógica de la Inteligencia Artificial (Procesamiento de Voz) ---
        if voice.commands:
            raw_text = voice.commands.pop(0)

            # 1. Pasamos el texto transcrito por la Red Neuronal de clasificación
            intent, confidence = brain.predict(raw_text)

            # 2. Generamos métricas para mostrar en la interfaz
            ia_info = f"Cortana deducción: '{intent}' | Certeza matemática: {confidence * 100:.1f}%"

            # 3. Mandamos el comando deducido al ejecutor físico del Robot
            last_action = execute_ia_intent(intent, robot)

        # --- Dibujado de Cuadros (Gráficos) ---
        screen.blit(grid_surf, (0, 0))
        draw_robot(screen, robot)
        draw_panel(screen, robot, voice, last_action, fonts, ia_info)

        pygame.display.flip()
        clock.tick(60)  # Forzar ejecución constante a 60 fotogramas por segundo

    # Cierre limpio de procesos e hilos
    voice.running = False
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()