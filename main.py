import flet as ft
import imputacion_app
import analisis_app
import gastos_app
import os

# --- ESTILOS PERSONALIZADOS (CONSTANTES) ---
COLOR_FONDO = "#050505"       # Casi negro
COLOR_SUPERFICIE = "#111111"  # Gris muy oscuro para contenedores
COLOR_ACENTO = "#00ff41"      # Verde Matrix clásico
COLOR_TEXTO = "#e0e0e0"       # Blanco suave
COLOR_GRIS_CLARO = "#BDBDBD"  # Reemplazo de Grey 400
COLOR_GRIS_MEDIO = "#9E9E9E"  # Reemplazo de Grey 500

FUENTE_PRINCIPAL = "Roboto Mono"

# --- HELPER PARA OPACIDAD ---
def add_opacity(hex_color, opacity):
    """Agrega canal Alpha a un color Hex (#RRGGBB -> #AARRGGBB)"""
    if not hex_color.startswith("#"):
        return hex_color 
    
    hex_color = hex_color.lstrip("#")
    alpha = int(opacity * 255)
    return f"#{alpha:02x}{hex_color}"

# --- COMPONENTES UI PERSONALIZADOS (ESTILO MATRIX) ---

class MatrixButton(ft.Container):
    """Botón personalizado con estilo Cyberpunk/Terminal"""
    def __init__(self, text, icon, on_click):
        super().__init__()
        self.on_click_action = on_click
        self.content = ft.Row(
            [
                ft.Icon(icon, color=COLOR_ACENTO),
                ft.Text(text, color=COLOR_ACENTO, size=16, font_family=FUENTE_PRINCIPAL, weight="bold"),
            ],
            alignment=ft.MainAxisAlignment.START,
        )
        self.padding = 20
        self.border = ft.border.all(1, COLOR_ACENTO)
        self.border_radius = 0 # Bordes cuadrados
        
        self.bgcolor = add_opacity(COLOR_ACENTO, 0.05) 
        
        self.on_click = self.animar_click
        self.on_hover = self.animar_hover
        
        self.animate = ft.Animation(300, ft.AnimationCurve.EASE_OUT)
        
        self.width = 450
        self.height = 70

    def animar_hover(self, e):
        if e.data == "true":
            self.bgcolor = add_opacity(COLOR_ACENTO, 0.2)
            self.content.controls[1].color = "white" 
            self.border = ft.border.all(2, COLOR_ACENTO) 
        else:
            self.bgcolor = add_opacity(COLOR_ACENTO, 0.05)
            self.content.controls[1].color = COLOR_ACENTO
            self.border = ft.border.all(1, COLOR_ACENTO)
        self.update()

    def animar_click(self, e):
        if self.on_click_action:
            self.on_click_action(e)

class TerminalHeader(ft.Container):
    """Encabezado estilo consola"""
    def __init__(self):
        super().__init__()
        self.content = ft.Column([
            ft.Text(">>> SISTEMA DE ANÁLISIS HIDROLÓGICO v5.0.1", color=COLOR_ACENTO, size=12, font_family=FUENTE_PRINCIPAL),
            ft.Text("INITIALIZING CORE MODULES...", color=COLOR_GRIS_MEDIO, size=10, font_family=FUENTE_PRINCIPAL),
            ft.Divider(color=COLOR_ACENTO, thickness=0.5),
        ], spacing=2)
        self.margin = ft.margin.only(bottom=20)

# --- VISTAS PRINCIPALES ---
menu_view = ft.Container()
imputacion_view_container = ft.Container(visible=False, expand=True)
analisis_view_container = ft.Container(visible=False, expand=True)
gastos_view_container = ft.Container(visible=False, expand=True)

def main(page: ft.Page):
    # 1. CONFIGURACIÓN DEL TEMA (DARK MODE GLOBAL)
    page.title = "Hydrological Data System"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 20
    page.bgcolor = COLOR_FONDO
    
    # --- DIMENSIONES DE LA VENTANA (NUEVO) ---
    page.window_width = 1280
    page.window_height = 1200
    page.window_resizable = False
    page.window_maximizable = False
    #page.window_center() # Centra la ventana en el monitor al abrir
    
    page.update()
    
    page.fonts = {
        "Roboto Mono": "https://github.com/google/fonts/raw/main/apache/robotomono/RobotoMono%5Bwght%5D.ttf"
    }
    
    # Tema Simplificado y Seguro
    page.theme = ft.Theme(
        font_family=FUENTE_PRINCIPAL,
        color_scheme=ft.ColorScheme(
            primary=COLOR_ACENTO,       # Esto pintará los TextFields y Botones activos de Verde
            on_primary=COLOR_FONDO,
            surface=COLOR_SUPERFICIE,
            background=COLOR_FONDO,
            on_surface=COLOR_TEXTO,
        ),
        # DataTableTheme seguro (sin border)
        data_table_theme=ft.DataTableTheme(
            heading_text_style=ft.TextStyle(font_family=FUENTE_PRINCIPAL, color=COLOR_ACENTO, weight="bold"),
            data_text_style=ft.TextStyle(font_family=FUENTE_PRINCIPAL, color=COLOR_TEXTO),
        )
    )

    # --- NAVEGACIÓN ---
    def go_to_menu(e):
        page.vertical_alignment = ft.MainAxisAlignment.CENTER
        page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        menu_view.visible = True
        imputacion_view_container.visible = False
        analisis_view_container.visible = False
        gastos_view_container.visible = False
        page.update()

    def go_to_imputacion(e):
        page.vertical_alignment = ft.MainAxisAlignment.START
        page.horizontal_alignment = ft.CrossAxisAlignment.START
        menu_view.visible = False
        imputacion_view_container.visible = True
        analisis_view_container.visible = False
        gastos_view_container.visible = False
        page.update()

    def go_to_analisis(e):
        page.vertical_alignment = ft.MainAxisAlignment.START
        page.horizontal_alignment = ft.CrossAxisAlignment.START
        menu_view.visible = False
        imputacion_view_container.visible = False
        analisis_view_container.visible = True
        gastos_view_container.visible = False
        page.update()
        
    def go_to_gastos(e):
        page.vertical_alignment = ft.MainAxisAlignment.START
        page.horizontal_alignment = ft.CrossAxisAlignment.START
        menu_view.visible = False
        imputacion_view_container.visible = False
        analisis_view_container.visible = False
        gastos_view_container.visible = True
        page.update()

    # --- CONSTRUCCIÓN DEL MENÚ PRINCIPAL ---
    menu_view.content = ft.Container(
        content=ft.Column(
            [
                TerminalHeader(),
                # Imagen (asegúrate que Gota.jpg exista en assets)
                ft.Image(src="path19.jpg", width=180, fit=ft.ImageFit.CONTAIN), 
                ft.Text("ANÁLISIS HIDROLÓGICO", size=30, weight=ft.FontWeight.BOLD, color="white"),
                ft.Text("Seleccione protocolo de operación:", size=14, color=COLOR_GRIS_CLARO),
                ft.Divider(height=40, color="transparent"),
                
                # Botones Estilizados
                MatrixButton(
                    "1. PROCESAMIENTO DE DATOS (IMPUTACIÓN)",
                    ft.Icons.MEMORY,
                    go_to_imputacion
                ),
                ft.Divider(height=10, color="transparent"),
                MatrixButton(
                    "2. ANÁLISIS DE PRECIPITACIÓN",
                    ft.Icons.SHOW_CHART,
                    go_to_analisis
                ),
                ft.Divider(height=10, color="transparent"),
                MatrixButton(
                    "3. CÁLCULO DE GASTOS (CUENCAS)",
                    ft.Icons.CALCULATE_OUTLINED, 
                    go_to_gastos
                ),
                
                ft.Container(height=50),
                ft.Text("SYSTEM STATUS: ONLINE", size=10, color=COLOR_ACENTO),
                ft.Text("SOFTWARE SOLUTION BY GEOGRAFICA S.A. DE C.V.", size=10, color=COLOR_ACENTO),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER
        ),
        border=ft.border.all(1, add_opacity(COLOR_ACENTO, 0.2)),
        padding=40,
        border_radius=0,
        bgcolor=COLOR_SUPERFICIE,
        
        width=None,    # Dejar que se ajuste o usar un ancho máximo relativo si prefieres
        expand=False,  # No forzar expansión máxima si queremos que esté centrado
        alignment=ft.alignment.center # Asegura que el contenido interno esté centrado
    )
    
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    # --- CARGA DE VISTAS ---
    imputacion_view_container.content = imputacion_app.build_imputacion_view(page, go_to_menu)
    analisis_view_container.content = analisis_app.build_analisis_view(page, go_to_menu)
    gastos_view_container.content = gastos_app.build_gastos_view(page, go_to_menu)
    
    # Contenedor principal
    main_layout = ft.Container(
        content=ft.Stack([
            menu_view,
            imputacion_view_container,
            analisis_view_container,
            gastos_view_container
        ]),
        expand=True,
        padding=10
    )

    page.add(main_layout)
    page.update()

if __name__ == "__main__":
    ft.app(target=main, assets_dir="assets")