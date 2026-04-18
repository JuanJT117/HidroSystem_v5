##############################################################################################################
##############################################################################################################
#
#   HidroSistem/
#   ├── main.py                     # Punto de entrada estricto (App Shell)
#   ├── core/                       # BACKEND: Lógica pura, matemáticas y algoritmos
#   │   ├── __init__.py
#   │   ├── descarga_logic.py        # Motor de descarga masiva de datos de CONAGUA
#   │   ├── hidrologia_mx.py        # Constantes y diccionarios de dominio
#   │   ├── imputacion_logic.py     # Motor de interpolación y ML
#   │   ├── lluvias_logic.py        # (Fusión/Limpieza de Analisis.py, lluvias.py y analisis_cuenca.py)
#   │   └── gastos_logic.py         # (Renombrado de gastos.py)
#   ├── ui/                         # FRONTEND: Vistas y componentes Flet
#   │   ├── __init__.py
#   │   ├── components.py           # Reutilizables: MatrixButton, TerminalHeader, etc.
#   │   └── views/
#   │       ├── welcome_view.py     # NUEVA: Pantalla inicial (Nuevo/Abrir Proyecto) + Easter Egg
#   │       ├── descarga_view.py    # (Modulo 1)
#   │       ├── imputacion_view.py  # (Modulo 2)
#   │       ├── analisis_view.py    # (Modulo 3)
#   │       └── gastos_view.py      # (Modulo 4)
#   ├── infrastructure/             # PUERTOS Y ADAPTADORES: Persistencia de datos
#   │   ├── __init__.py
#   │   └── project_manager.py      # NUEVO: Motor de base de datos comprimida (.hds via .xz)
#   └── assets/                     # Recursos estáticos
#       ├── catalogo_tlaloc.csv
#       ├── cuencas.cpg
#       ├── cuencas.dbf
#       ├── cuencas.prj
#       ├── cuencas.qmd
#       ├── cuencas.shp
#       ├── cuencas.shx
#       ├── environment.yml  
#       ├── estados.cpg
#       ├── estados.dbf
#       ├── estados.prj
#       ├── estados.qmd
#       ├── estados.shp
#       ├── estados.shx
#       ├── tlaloc_egg.gif
#       ├── Tlaloc_BD_Nacional_Comprimida.tar.xz
#       ├── icon.ico
#       └── path19.jpg
#
##############################################################################################################
##############################################################################################################

import flet as ft
import os
import traceback

# --- IMPORTACIÓN DE INFRAESTRUCTURA Y COMPONENTES ---
from infrastructure.project_manager import ProjectManager
from ui.components import COLOR_FONDO, COLOR_ACENTO, COLOR_SUPERFICIE, FUENTE_PRINCIPAL
from ui.views.welcome_view import build_welcome_view

# --- IMPORTACIÓN DE MÓDULOS (FRONTEND) ---
from ui.views.descarga_view import build_descarga_view
import ui.views.imputacion_view as imputacion_app
import ui.views.analisis_view as analisis_app
import ui.views.gastos_view as gastos_app

def main(page: ft.Page):
    # --- CONFIGURACIÓN DE PÁGINA ---
    page.title = "HidroSistem v9.2 - Gestión de Proyectos"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = COLOR_FONDO
    page.window_width = 1200
    page.window_height = 900
    page.fonts = {"Roboto Mono": "https://github.com/google/fonts/raw/main/apache/robotomono/RobotoMono%5Bwght%5D.ttf"}
    page.theme = ft.Theme(font_family=FUENTE_PRINCIPAL)

    # --- ESTADO DEL PROYECTO ---
    page.session.set("current_project_path", None)

    # --- CONTENEDOR PRINCIPAL (ROUTER) ---
    main_container = ft.Container(expand=True)

    # --- LISTA GLOBAL DE BOTONES PARA BLOQUEO ---
    # Esta lista guardará las referencias a los botones de Matrix que queremos controlar
    menu_buttons = [] 

    # --- DEFINICIÓN DE UTILIDADES DE BLOQUEO ---
    def lock_ui():
        """Desactiva la interacción de todos los botones registrados."""
        for btn in menu_buttons:
            try:
                btn.toggle_lock(True)
            except:
                pass
        page.update()

    def unlock_ui():
        """Reactiva la interacción de todos los botones registrados."""
        for btn in menu_buttons:
            try:
                btn.toggle_lock(False)
            except:
                pass
        page.update()
    
    # --- LÓGICA DE NAVEGACIÓN Y CARGA ---
    
    def show_dashboard():
        """Cambia la vista al menú principal de herramientas una vez cargado el proyecto."""
        page.clean()
        
        # --- LÓGICA DE RETORNO A INICIO CORREGIDA ---
        def return_to_welcome_screen():
            # Forzamos la reactivación de todo el sistema de navegación
            main_rail = page.session.get("main_rail")
            if main_rail:
                main_rail.disabled = False
                if main_rail.trailing:
                    for ctrl in main_rail.trailing.controls:
                        ctrl.disabled = False
            # 1. Liberamos el bloqueo de botones global si existe
            try: unlock_ui() 
            except: pass
            
            # 2. Limpiamos la ruta del proyecto
            page.session.set("current_project_path", None)
            
            # 3. Limpiamos la pantalla
            page.clean()
            
            # 4. Re-inyectamos la vista de bienvenida
            # IMPORTANTE: Asegúrate de usar los nombres exactos de tus pickers (file_picker_open/save)
            page.add(
                build_welcome_view(
                    page, 
                    on_new_project=lambda _: file_picker_save.save_file(file_name="ProyectoNuevo.hds"),
                    on_open_project=lambda _: file_picker_open.pick_files(allowed_extensions=["hds"])
                )
            )

        def prompt_exit_dialog(e):
            def exit_without_saving(_):
                # Cerramos SOLO el diálogo, NO limpiamos el overlay entero
                page.close(exit_dialog)
                return_to_welcome_screen()

            def save_and_exit(_):
                save_current_project(None) # Guarda en .hds vía LZMA
                page.close(exit_dialog)
                return_to_welcome_screen()

            exit_dialog = ft.AlertDialog(
                modal=True,
                title=ft.Text("⚠️ CERRAR PROYECTO", color="#ffdd00", font_family=FUENTE_PRINCIPAL),
                content=ft.Text("¿Deseas guardar los cambios antes de volver a la pantalla de inicio?"),
                actions=[
                    ft.TextButton("Cancelar", on_click=lambda _: page.close(exit_dialog), style=ft.ButtonStyle(color="white")),
                    ft.TextButton("Cerrar sin Guardar", on_click=exit_without_saving, style=ft.ButtonStyle(color="red")),
                    ft.ElevatedButton("Guardar y Cerrar", on_click=save_and_exit, bgcolor=COLOR_ACENTO, color="black"),
                ],
                actions_alignment=ft.MainAxisAlignment.END,
                bgcolor=COLOR_SUPERFICIE,
            )
            page.open(exit_dialog)
        # -----------------------------------------------

        # Definimos el NavigationRail con los 4 módulos solicitados
        rail = ft.NavigationRail(
            selected_index=0,
            label_type=ft.NavigationRailLabelType.ALL,
            min_width=100,
            min_extended_width=200,
            group_alignment=-0.9,
            destinations=[
                ft.NavigationRailDestination(icon=ft.Icons.CLOUD_DOWNLOAD, label="1. EXTRACCIÓN"),
                ft.NavigationRailDestination(icon=ft.Icons.MEMORY, label="2. IMPUTACIÓN"),
                ft.NavigationRailDestination(icon=ft.Icons.SHOW_CHART, label="3. ANÁLISIS"),
                ft.NavigationRailDestination(icon=ft.Icons.CALCULATE, label="4. GASTOS"),
            ],
            on_change=lambda e: change_module(e.control.selected_index),
            trailing=ft.Column([
                ft.Divider(),
                ft.IconButton(ft.Icons.SAVE, tooltip="GUARDAR PROYECTO", icon_color=COLOR_ACENTO, on_click=save_current_project),
                ft.IconButton(ft.Icons.LOGOUT, tooltip="CERRAR PROYECTO", icon_color="red", on_click=prompt_exit_dialog), # <--- AQUÍ ENLAZAMOS LA FUNCIÓN
            ], alignment=ft.MainAxisAlignment.END, spacing=10)
        )
        page.session.set("main_rail", rail)
        content_area = ft.Container(expand=True, padding=10)

        def change_module(index):
            # Aquí inyectamos las vistas de los módulos
            # Se pasan las funciones de construcción que ya tienes en tus .py actuales
            if index == 0:
                content_area.content = build_descarga_view(page, back_callback=None, nav_to_imputacion=None)
            elif index == 1:
                content_area.content = imputacion_app.build_imputacion_view(page, on_back_to_menu=None)
            elif index == 2:
                content_area.content = analisis_app.build_analisis_view(page, on_back_to_menu=None)
            elif index == 3:
                content_area.content = gastos_app.build_gastos_view(page, on_back_to_menu=None)
            page.update()

        # Inicializar en el primer módulo
        change_module(0)

        page.add(
            ft.Row([
                rail,
                ft.VerticalDivider(width=1),
                content_area
            ], expand=True)
        )

    # --- HANDLERS DE PROYECTO ---

    def handle_open_project(e: ft.FilePickerResultEvent):
        if e.files:
            try:
                path = e.files[0].path
                # Usamos el motor LZMA de la Fase 1
                data = ProjectManager.load_project(path)
                
                # --- NUEVO: PURGA DE MEMORIA (PREVENIR FANTASMAS) ---
                page.session.clear() 
                # ----------------------------------------------------
                
                # Hidratar la sesión con TODOS los datos recuperados
                for key, value in data.items():
                    page.session.set(key, value)
                
                page.session.set("current_project_path", path)
                show_dashboard()
                
            except Exception as ex:
                page.snack_bar = ft.SnackBar(ft.Text(f"Error al cargar: {str(ex)}"), bgcolor="red")
                page.snack_bar.open = True
                page.update()

    def handle_new_project(e: ft.FilePickerResultEvent):
        if e.path:
            try:
                # Crear un archivo inicial vacío pero con la estructura .hds (.tar.xz)
                ProjectManager.save_project(e.path, {"status": "new_project"})
                page.session.set("current_project_path", e.path)
                show_dashboard()
            except Exception as ex:
                page.snack_bar = ft.SnackBar(ft.Text(f"Error al crear: {str(ex)}"), bgcolor="red")
                page.snack_bar.open = True
                page.update()

    def save_current_project(e):
        path = page.session.get("current_project_path")
        if path:
            try:
                # Extraemos todas las llaves de la sesión para persistirlas
                # Podrías filtrar aquí qué llaves quieres guardar específicamente
                session_dict = {k: page.session.get(k) for k in page.session._Session__data.keys()}
                ProjectManager.save_project(path, session_dict)
                
                page.snack_bar = ft.SnackBar(ft.Text("Proyecto Guardado (Compresión LZMA OK)"), bgcolor=COLOR_ACENTO)
                page.snack_bar.open = True
                page.update()
            except Exception as ex:
                page.snack_bar = ft.SnackBar(ft.Text(f"Error al guardar: {str(ex)}"), bgcolor="red")
                page.snack_bar.open = True
                page.update()

    # --- PICKERS ---
    file_picker_open = ft.FilePicker(on_result=handle_open_project)
    file_picker_save = ft.FilePicker(on_result=handle_new_project)
    page.overlay.extend([file_picker_open, file_picker_save])

    # --- ARRANQUE ---
    # Cargamos la pantalla de bienvenida pasándole los disparadores de los pickers
    page.add(
        build_welcome_view(
            page, 
            on_new_project=lambda _: file_picker_save.save_file(file_name="ProyectoNuevo.hds"),
            on_open_project=lambda _: file_picker_open.pick_files(allowed_extensions=["hds"])
        )
    )

if __name__ == "__main__":
    ft.app(target=main, assets_dir="assets")