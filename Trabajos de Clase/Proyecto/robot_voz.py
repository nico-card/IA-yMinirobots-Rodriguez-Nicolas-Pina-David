"""
Robot con Control de Voz — Pygame + IA de reconocimiento de voz
================================================================
Comandos de voz disponibles (español):
  adelante         → avanza una casilla hacia adelante
  atrás / atras    → retrocede una casilla
  izquierda        → GIRA a la izquierda en el puesto (sin moverse)
  derecha          → GIRA a la derecha en el puesto (sin moverse)
  mueve izquierda  → gira a la izquierda Y avanza
  mueve derecha    → gira a la derecha Y avanza
  para / stop      → (futuro)

Controles de teclado (respaldo):
  ↑ ↓     → adelante / atrás
  ← →     → girar en el puesto
  A D     → girar y moverse (izquierda/derecha)
  ESC     → salir
"""

import pygame
import speech_recognition as sr
import threading
import sys
import math

# ---------------------------------------------------------------------------
# Configuración de la cuadrícula
# ---------------------------------------------------------------------------
GRID_SIZE = 20          # número de celdas
CELL_SIZE = 32          # píxeles por celda
UI_HEIGHT = 130         # panel de estado inferior

WIN_W = GRID_SIZE * CELL_SIZE
WIN_H = GRID_SIZE * CELL_SIZE + UI_HEIGHT

# ---------------------------------------------------------------------------
# Paleta de colores
# ---------------------------------------------------------------------------
C_BG        = (245, 245, 250)
C_GRID      = (210, 210, 220)
C_GRID_ALT  = (238, 238, 245)
C_BODY      = (41, 98, 205)
C_BODY_RIM  = (20, 55, 140)
C_ARROW     = (255, 215, 0)
C_TRAIL     = (170, 200, 255)
C_PANEL     = (30, 32, 48)
C_TEXT_HI   = (255, 255, 255)
C_TEXT_LO   = (160, 165, 190)
C_GREEN     = (72, 199, 116)
C_RED       = (255, 80, 80)
C_YELLOW    = (255, 200, 50)
C_BLUE_LT   = (100, 160, 255)

# ---------------------------------------------------------------------------
# Direcciones: 0=Norte, 1=Este, 2=Sur, 3=Oeste
# ---------------------------------------------------------------------------
DIR_DELTA = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
DIR_LABEL = {0: "Norte ↑", 1: "Este →", 2: "Sur ↓", 3: "Oeste ←"}
# Ángulo en grados (desde el este, antihorario) para dibujar la flecha
DIR_ANGLE = {0: 90, 1: 0, 2: 270, 3: 180}


# ---------------------------------------------------------------------------
# Clase Robot
# ---------------------------------------------------------------------------
class Robot:
    def __init__(self):
        self.x = GRID_SIZE // 2
        self.y = GRID_SIZE // 2
        self.direction = 0          # Norte
        self.trail: list[tuple[int, int]] = []

    def _step(self, forward: bool) -> bool:
        dx, dy = DIR_DELTA[self.direction]
        if not forward:
            dx, dy = -dx, -dy
        nx, ny = self.x + dx, self.y + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            self.trail.append((self.x, self.y))
            if len(self.trail) > 30:
                self.trail.pop(0)
            self.x, self.y = nx, ny
            return True
        return False

    def move_forward(self)  -> bool: return self._step(True)
    def move_backward(self) -> bool: return self._step(False)
    def turn_left(self)  -> None: self.direction = (self.direction - 1) % 4
    def turn_right(self) -> None: self.direction = (self.direction + 1) % 4

    def turn_and_move_left(self)  -> bool:
        self.turn_left()
        return self.move_forward()

    def turn_and_move_right(self) -> bool:
        self.turn_right()
        return self.move_forward()


# ---------------------------------------------------------------------------
# Procesador de comandos de voz
# ---------------------------------------------------------------------------
def process_command(text: str, robot: Robot) -> str:
    t = text.lower().strip()

    # Mueve + dirección  (gira Y avanza)
    if any(p in t for p in ["mueve izquierda", "mover izquierda",
                             "avanza izquierda", "gira y mueve izquierda"]):
        ok = robot.turn_and_move_left()
        return "↖  Giró + movió → izquierda" + ("" if ok else "  (límite)")

    if any(p in t for p in ["mueve derecha", "mover derecha",
                             "avanza derecha", "gira y mueve derecha"]):
        ok = robot.turn_and_move_right()
        return "↗  Giró + movió → derecha" + ("" if ok else "  (límite)")

    # Adelante / atrás
    if any(p in t for p in ["adelante", "avanza", "avanzar", "forward"]):
        ok = robot.move_forward()
        return "⬆  Adelante" + ("" if ok else "  (límite)")

    if any(p in t for p in ["atrás", "atras", "retrocede",
                             "retroceder", "backward", "back"]):
        ok = robot.move_backward()
        return "⬇  Atrás" + ("" if ok else "  (límite)")

    # Girar en el puesto
    if any(p in t for p in ["izquierda", "girar izquierda", "gira izquierda"]):
        robot.turn_left()
        return "↺  Giró izquierda (en el puesto)"

    if any(p in t for p in ["derecha", "girar derecha", "gira derecha"]):
        robot.turn_right()
        return "↻  Giró derecha (en el puesto)"

    return f'❓  No reconocí: "{text}"'


# ---------------------------------------------------------------------------
# Controlador de voz (hilo aparte)
# ---------------------------------------------------------------------------
class VoiceController:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.commands: list[str] = []
        self.status    = "Iniciando micrófono…"
        self.ready     = False
        self.running   = True
        self.last_heard = ""

    def listen_loop(self):
        try:
            with sr.Microphone() as src:
                self.recognizer.energy_threshold = 300
                self.recognizer.adjust_for_ambient_noise(src, duration=1.5)
                self.status = "🎤  Escuchando…  di un comando"
                self.ready  = True
                while self.running:
                    try:
                        audio = self.recognizer.listen(
                            src, timeout=4, phrase_time_limit=5
                        )
                        self.status = "⏳  Procesando…"
                        text = self.recognizer.recognize_google(
                            audio, language="es-ES"
                        )
                        self.last_heard = text
                        self.commands.append(text)
                        self.status = f'✅  Escuché: "{text}"'
                    except sr.WaitTimeoutError:
                        self.status = "🎤  Escuchando…  di un comando"
                    except sr.UnknownValueError:
                        self.status = "🔇  No entendí, intenta de nuevo"
                    except sr.RequestError as e:
                        self.status = f"❌  Error de red: {e}"
        except OSError as e:
            self.status = f"❌  Micrófono no disponible: {e}"
            self.ready  = False


# ---------------------------------------------------------------------------
# Dibujo del robot
# ---------------------------------------------------------------------------
def draw_robot(surf: pygame.Surface, robot: Robot):
    # Rastro
    for i, (tx, ty) in enumerate(robot.trail):
        alpha = int(80 * (i + 1) / len(robot.trail)) if robot.trail else 80
        cx = tx * CELL_SIZE + CELL_SIZE // 2
        cy = ty * CELL_SIZE + CELL_SIZE // 2
        r  = max(3, CELL_SIZE // 6)
        pygame.draw.circle(surf, C_TRAIL, (cx, cy), r)

    cx = robot.x * CELL_SIZE + CELL_SIZE // 2
    cy = robot.y * CELL_SIZE + CELL_SIZE // 2
    r  = CELL_SIZE // 2 - 3

    # Sombra suave
    pygame.draw.circle(surf, (180, 185, 210), (cx + 2, cy + 3), r)
    # Cuerpo
    pygame.draw.circle(surf, C_BODY, (cx, cy), r)
    pygame.draw.circle(surf, C_BODY_RIM, (cx, cy), r, 2)

    # Flecha de dirección
    a_rad = math.radians(DIR_ANGLE[robot.direction])
    tip_dist  = r - 4
    base_dist = r // 2
    perp_rad  = a_rad + math.pi / 2

    tip = (cx + int(tip_dist * math.cos(a_rad)),
           cy - int(tip_dist * math.sin(a_rad)))
    b1  = (cx + int(base_dist * math.cos(perp_rad)) - int((tip_dist * 0.3) * math.cos(a_rad)),
           cy - int(base_dist * math.sin(perp_rad)) + int((tip_dist * 0.3) * math.sin(a_rad)))
    b2  = (cx - int(base_dist * math.cos(perp_rad)) - int((tip_dist * 0.3) * math.cos(a_rad)),
           cy + int(base_dist * math.sin(perp_rad)) + int((tip_dist * 0.3) * math.sin(a_rad)))

    pygame.draw.polygon(surf, C_ARROW, [tip, b1, b2])

    # "Ojo" (círculo pequeño en el frente)
    eye_x = cx + int((r - 8) * math.cos(a_rad))
    eye_y = cy - int((r - 8) * math.sin(a_rad))
    pygame.draw.circle(surf, C_TEXT_HI, (eye_x, eye_y), 3)


# ---------------------------------------------------------------------------
# Panel de estado inferior
# ---------------------------------------------------------------------------
def draw_panel(surf: pygame.Surface, robot: Robot, voice: VoiceController,
               last_action: str, fonts: dict):
    panel_y = GRID_SIZE * CELL_SIZE
    pygame.draw.rect(surf, C_PANEL, (0, panel_y, WIN_W, UI_HEIGHT))
    pygame.draw.line(surf, C_BLUE_LT, (0, panel_y), (WIN_W, panel_y), 1)

    # Indicador de micrófono
    mic_color = C_GREEN if voice.ready else C_RED
    pygame.draw.circle(surf, mic_color, (14, panel_y + 18), 7)

    # Estado del reconocedor
    status_s = fonts["md"].render(voice.status, True, C_TEXT_HI)
    surf.blit(status_s, (28, panel_y + 10))

    # Último comando ejecutado
    action_s = fonts["lg"].render(last_action, True, C_YELLOW)
    surf.blit(action_s, (10, panel_y + 38))

    # Posición y dirección
    info = (f"Posición: ({robot.x}, {robot.y})    "
            f"Dirección: {DIR_LABEL[robot.direction]}")
    info_s = fonts["sm"].render(info, True, C_TEXT_LO)
    surf.blit(info_s, (10, panel_y + 72))

    # Ayuda de comandos
    help1 = "Voz: adelante · atrás · izquierda · derecha · mueve izquierda · mueve derecha"
    help2 = "Teclado: ↑↓ mover  ←→ girar  A/D girar+mover  ESC salir"
    surf.blit(fonts["xs"].render(help1, True, C_TEXT_LO), (10, panel_y + 92))
    surf.blit(fonts["xs"].render(help2, True, C_TEXT_LO), (10, panel_y + 110))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("🤖 Robot con Control de Voz")

    fonts = {
        "xs": pygame.font.SysFont("Arial", 12),
        "sm": pygame.font.SysFont("Arial", 14),
        "md": pygame.font.SysFont("Arial", 15, bold=True),
        "lg": pygame.font.SysFont("Arial", 18, bold=True),
    }

    clock = pygame.time.Clock()
    robot = Robot()
    voice = VoiceController()

    # Lanzar hilo de voz
    voice_thread = threading.Thread(target=voice.listen_loop, daemon=True)
    voice_thread.start()

    last_action = "Esperando comando…"

    # Pre-dibujar la cuadrícula en una superficie cachée
    grid_surf = pygame.Surface((WIN_W, GRID_SIZE * CELL_SIZE))
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            color = C_GRID_ALT if (x + y) % 2 == 0 else C_BG
            pygame.draw.rect(grid_surf, color,
                             (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(grid_surf, C_GRID,
                             (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

    running = True
    while running:
        # -------- Eventos --------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_UP:
                    robot.move_forward()
                    last_action = "Teclado ↑  Adelante"
                elif event.key == pygame.K_DOWN:
                    robot.move_backward()
                    last_action = "Teclado ↓  Atrás"
                elif event.key == pygame.K_LEFT:
                    robot.turn_left()
                    last_action = "Teclado ←  Giró izquierda (en el puesto)"
                elif event.key == pygame.K_RIGHT:
                    robot.turn_right()
                    last_action = "Teclado →  Giró derecha (en el puesto)"
                elif event.key == pygame.K_a:
                    robot.turn_and_move_left()
                    last_action = "Teclado A  Giró + movió izquierda"
                elif event.key == pygame.K_d:
                    robot.turn_and_move_right()
                    last_action = "Teclado D  Giró + movió derecha"

        # -------- Comandos de voz --------
        if voice.commands:
            text   = voice.commands.pop(0)
            result = process_command(text, robot)
            last_action = result

        # -------- Dibujo --------
        screen.blit(grid_surf, (0, 0))
        draw_robot(screen, robot)
        draw_panel(screen, robot, voice, last_action, fonts)

        pygame.display.flip()
        clock.tick(60)

    voice.running = False
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
