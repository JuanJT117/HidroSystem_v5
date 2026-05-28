import flet as ft
from ui.components import MatrixButton, TerminalHeader, COLOR_ACENTO, COLOR_SUPERFICIE, FUENTE_PRINCIPAL, add_opacity

def build_welcome_view(page: ft.Page, on_new_project, on_open_project):
    """
    Construye la vista inicial de la aplicación.
    Fuerza al usuario a crear o abrir un proyecto antes de acceder al ecosistema.
    """
    
    # --- EASTER EGG: PROTOCOLO TLÁLOC ---
    egg_clicks = [0] # Usamos una lista para mantener la referencia en el closure
    
    def trigger_tlaloc_protocol(e):
        egg_clicks[0] += 1
        
        if egg_clicks[0] >= 125: # Detonador
            egg_clicks[0] = 0 
            
            egg_dialog = ft.AlertDialog(
                modal=True,
                title=ft.Text("⚠️ PROTOCOLO DON JUAN ACTIVADO ⚠️", color=COLOR_ACENTO, font_family=FUENTE_PRINCIPAL, text_align=ft.TextAlign.CENTER),
                content=ft.Container(
                    content=ft.Image(src="tlaloc_egg.gif", fit=ft.ImageFit.CONTAIN, width=600, height=400, border_radius=10),
                    alignment=ft.alignment.center,
                    padding=10,
                    border=ft.border.all(1, COLOR_ACENTO), 
                    border_radius=10,
                    bgcolor="black"
                ),
                actions=[
                    ft.TextButton("CERRAR CONEXIÓN", on_click=lambda e: page.close(egg_dialog), style=ft.ButtonStyle(color=COLOR_ACENTO))
                ],
                actions_alignment=ft.MainAxisAlignment.CENTER,
                bgcolor=COLOR_SUPERFICIE, 
            )
            page.open(egg_dialog)
            page.snack_bar = ft.SnackBar(ft.Text("Conexión con el núcleo establecida..."), bgcolor=COLOR_ACENTO)
            page.snack_bar.open = True
            page.update()

    # --- MODAL "ACERCA DE" ---
    def open_about_modal(e):
        about_dialog = ft.AlertDialog(
            modal=False, # Permite cerrar haciendo clic fuera
            content=ft.Container(
                content=ft.Column([
                    ft.Icon(ft.Icons.INFO_OUTLINE, size=60, color=COLOR_ACENTO),
                    ft.Text("ACERCA DE", size=30, weight="bold", color="white"),
                    ft.Divider(color=COLOR_ACENTO),
                    ft.Text("HyDaS", size=20, color=COLOR_ACENTO),
                    ft.Text("Versión 10.9", color="#BDBDBD"),
                    ft.Container(height=20),
                    ft.Text("Nemi, Yocoya, Nica", color="#9E9E9E"),
                    ft.Text("Polihui", size=25, weight="bold", color="white"),
                    ft.Text("Por: Ing. Juan Jesús Torres Solano", color="#9E9E9E"),
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, tight=True),
                padding=20,
                width=450,
            ),
            bgcolor=COLOR_SUPERFICIE,
            shape=ft.RoundedRectangleBorder(radius=10),
        )
        page.open(about_dialog)

    # --- COMPONENTES PRINCIPALES ---
    btn_new = MatrixButton("CREAR NUEVO PROYECTO", ft.Icons.ADD_BOX, on_new_project, width=350, color=COLOR_ACENTO)
    btn_open = MatrixButton("ABRIR PROYECTO (.hds)", ft.Icons.FOLDER_OPEN, on_open_project, width=350, color="#FFC107") # Amarillo para distinguir acción
    btn_info = ft.IconButton(icon=ft.Icons.INFO_OUTLINE, icon_color="grey", icon_size=30, on_click=open_about_modal, tooltip="Acerca del Sistema")

    # Contenedor central
    welcome_container = ft.Container(
        content=ft.Column([
            TerminalHeader(version="10.9"),
            ft.Container(height=25),
            
            # Logotipo con Easter Egg
            ft.Container(
                content=ft.Image(src="path19.jpg", width=130, fit=ft.ImageFit.CONTAIN),
                on_click=trigger_tlaloc_protocol,
                tooltip="Iniciando sistema central...",
                padding=5,
                border_radius=60, # Hacerlo circular si la imagen es cuadrada
                ink=True 
            ),
            
            ft.Text("Nican neltocahuani in Tláloc", size=13, weight="bold", color="white", font_family=FUENTE_PRINCIPAL),
            ft.Divider(height=13, color="transparent"),
            
            # Botones de Acción
            btn_new,
            ft.Container(height=10),
            btn_open,
            
            ft.Divider(height=40, color=add_opacity(COLOR_ACENTO, 0.2)),
            
            # Botón de Información inferior
            btn_info
            
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, tight=True),
        padding=40, 
        border=ft.border.all(1, add_opacity(COLOR_ACENTO, 0.2)), 
        bgcolor=COLOR_SUPERFICIE, 
        alignment=ft.alignment.center,
        border_radius=10
    )

    return ft.Container(
        content=welcome_container,
        expand=True,
        alignment=ft.alignment.center
    )