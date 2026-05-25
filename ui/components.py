import flet as ft

# --- CONSTANTES DE TEMA CYBERPUNK ---
COLOR_FONDO = "#050505"
COLOR_SUPERFICIE = "#111111"
COLOR_ACENTO = "#00ff41"
COLOR_TEXTO = "#e0e0e0"
COLOR_GRIS_CLARO = "#BDBDBD"
COLOR_GRIS_MEDIO = "#9E9E9E"
FUENTE_PRINCIPAL = "Roboto Mono"

def add_opacity(hex_color: str, opacity: float) -> str:
    """Añade opacidad a un color HEX sin usar métodos deprecados de Flet."""
    if not hex_color.startswith("#"): 
        return hex_color 
    hex_color = hex_color.lstrip("#")
    alpha = int(opacity * 255)
    return f"#{alpha:02x}{hex_color}"

class MatrixButton(ft.Container):
    """Botón con estilo Cyberpunk original y capacidad de BLOQUEO seguro para hilos."""
    def __init__(self, text, icon, on_click, width=450, color=COLOR_ACENTO):
        super().__init__()
        self.on_click_action = on_click
        self.base_color = color
        self.is_locked = False
        
        self.content = ft.Row([
            ft.Icon(icon, color=color),
            ft.Text(text, color=color, size=16, font_family=FUENTE_PRINCIPAL, weight="bold"),
        ], alignment=ft.MainAxisAlignment.START)
        
        self.padding = 10
        self.border = ft.border.all(1, color)
        self.bgcolor = add_opacity(color, 0.05) 
        self.on_click = self.animar_click
        self.on_hover = lambda e: self.animar_hover(e, color)
        self.width = width
        self.height = 60
        # Animación nativa de Flet
        self.animate = ft.Animation(300, ft.AnimationCurve.EASE_OUT)

    def toggle_lock(self, locked: bool):
        self.is_locked = locked
        if locked:
            self.opacity = 0.5
            self.border = ft.border.all(1, "grey")
            self.content.controls[0].color = "grey"
            self.content.controls[1].color = "grey"
        else:
            self.opacity = 1.0
            self.border = ft.border.all(1, self.base_color)
            self.content.controls[0].color = self.base_color
            self.content.controls[1].color = self.base_color
        self.update()

    def animar_hover(self, e, color):
        if self.is_locked: return
        if e.data == "true":
            self.bgcolor = add_opacity(color, 0.2)
            self.content.controls[1].color = "white" 
            self.border = ft.border.all(2, color) 
        else:
            self.bgcolor = add_opacity(color, 0.05)
            self.content.controls[1].color = color
            self.border = ft.border.all(1, color)
        self.update()

    def animar_click(self, e):
        if self.is_locked: return
        if self.on_click_action: 
            self.on_click_action(e)

class TerminalHeader(ft.Container):
    """Encabezado de Terminal estándar para las vistas."""
    def __init__(self, version="10.5.2"):
        super().__init__()
        self.content = ft.Column([
            ft.Text(f">>> Hydrological Data System v{version}", color=COLOR_ACENTO, size=22, font_family=FUENTE_PRINCIPAL),
            ft.Divider(color=COLOR_ACENTO, thickness=0.8),
        ], spacing=2)
        self.margin = ft.margin.only(bottom=10)