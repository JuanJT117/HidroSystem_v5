import flet as ft
import matplotlib
# --- CRÍTICO: Forzar backend no interactivo globalmente al inicio ---
matplotlib.use('Agg') 

import imputacion_app
import analisis_app
import gastos_app
import os
import pickle
import time
import traceback
import zipfile
import json
import numpy as np
import pandas as pd

# --- CONFIGURACIÓN Y ESTILOS ---
COLOR_FONDO = "#050505"
COLOR_SUPERFICIE = "#111111"
COLOR_ACENTO = "#00ff41"
COLOR_TEXTO = "#e0e0e0"
COLOR_GRIS_CLARO = "#BDBDBD"
COLOR_GRIS_MEDIO = "#9E9E9E"
FUENTE_PRINCIPAL = "Roboto Mono"

# --- LISTA MAESTRA DE PERSISTENCIA ---
SESSION_KEYS_TO_PERSIST = [
    # 1. IMPUTACIÓN
    "imput_folder_path", "imput_output_folder", "imput_station_files", "imput_map_path",
    "df_imputado_resultado", 
    
    # 2. ANÁLISIS
    "df_procesado", "df_filtrado", "df_estadisticas",
    "df_homogeneidad", "df_acf", "df_ajustes", "df_weibull", 
    "df_maximos_mensuales", "best_fit_name",
    "df_altura", "df_intensidad", 
    
    # 3. GASTOS
    "datos_cuencas_config", "df_cuencas_base", "df_cotas", "df_hms", 
    "df_intensidad_gastos", "df_altura_gastos", 
    "res_racional", "res_chow", "df_variables"
]

# --- UTILERÍAS VISUALES ---

def add_opacity(hex_color, opacity):
    if not hex_color.startswith("#"): return hex_color 
    hex_color = hex_color.lstrip("#")
    alpha = int(opacity * 255)
    return f"#{alpha:02x}{hex_color}"

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class MatrixButton(ft.Container):
    """Botón con estilo Cyberpunk original y capacidad de BLOQUEO."""
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
        self.animate = ft.Animation(300, ft.AnimationCurve.EASE_OUT)
        self.width = width
        self.height = 60

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
        if self.on_click_action: self.on_click_action(e)

class TerminalHeader(ft.Container):
    def __init__(self):
        super().__init__()
        self.content = ft.Column([
            ft.Text(">>> SISTEMA DE ANÁLISIS HIDROLÓGICO v6.2.1", color=COLOR_ACENTO, size=13, font_family=FUENTE_PRINCIPAL),
            ft.Divider(color=COLOR_ACENTO, thickness=0.8),
        ], spacing=2)
        self.margin = ft.margin.only(bottom=20)

# --- VISTAS GLOBALES ---
menu_view = ft.Container()
about_view = ft.Container(visible=False, expand=True)
imputacion_view_container = ft.Container(visible=False, expand=True)
analisis_view_container = ft.Container(visible=False, expand=True)
gastos_view_container = ft.Container(visible=False, expand=True)

def main(page: ft.Page):
    page.title = "Hydrological Data System"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 20
    page.bgcolor = COLOR_FONDO
    page.window_width = 1000
    page.window_height = 900
    page.window_resizable = False
    
    page.fonts = {"Roboto Mono": "https://github.com/google/fonts/raw/main/apache/robotomono/RobotoMono%5Bwght%5D.ttf"}
    page.theme = ft.Theme(font_family=FUENTE_PRINCIPAL, color_scheme=ft.ColorScheme(primary=COLOR_ACENTO, on_primary=COLOR_FONDO, surface=COLOR_SUPERFICIE, background=COLOR_FONDO, on_surface=COLOR_TEXTO))

    menu_buttons = [] # Referencia para bloqueo

    # --- LOADING UI ---
    loading_bar = ft.ProgressBar(width=400, color=COLOR_ACENTO, bgcolor="#222222", value=0)
    loading_text = ft.Text("Procesando...", font_family=FUENTE_PRINCIPAL, color=COLOR_ACENTO)
    
    loading_dialog = ft.AlertDialog(
        modal=True,
        title=ft.Text("SISTEMA OCUPADO", color=COLOR_ACENTO, font_family=FUENTE_PRINCIPAL, size=14),
        content=ft.Container(
            content=ft.Column([loading_text, ft.Container(height=10), loading_bar], height=80, alignment=ft.MainAxisAlignment.CENTER),
            width=450, height=100, padding=10
        ),
        bgcolor=COLOR_SUPERFICIE,
        shape=ft.RoundedRectangleBorder(radius=0)
    )
    page.overlay.append(loading_dialog)

    # --- UTILS BLOQUEO ---
    def lock_ui():
        for btn in menu_buttons: btn.toggle_lock(True)
    def unlock_ui():
        for btn in menu_buttons: btn.toggle_lock(False)

    # --- LÓGICA DE GUARDADO ---
    def save_project_state(e: ft.FilePickerResultEvent):
        unlock_ui()
        if not e.path: return
        loading_text.value = "Guardando estado..."; loading_bar.value = 0.0; loading_dialog.open = True; page.update()

        path = e.path if e.path.endswith(".hds") else f"{e.path}.hds"
        manifest = {}

        try:
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                total = len(SESSION_KEYS_TO_PERSIST)
                for idx, key in enumerate(SESSION_KEYS_TO_PERSIST):
                    loading_bar.value = (idx + 1) / total; page.update()
                    if not hasattr(page.session, key): continue
                    val = getattr(page.session, key)
                    if val is None: continue

                    if isinstance(val, pd.DataFrame):
                        filename = f"{key}.pkl"
                        with zf.open(filename, "w") as f: pickle.dump(val, f)
                        manifest[key] = {"file": filename, "type": "dataframe"}
                    else:
                        filename = f"{key}.json"
                        try:
                            json_str = json.dumps(val, cls=NumpyEncoder)
                            with zf.open(filename, "w") as f: f.write(json_str.encode("utf-8"))
                            manifest[key] = {"file": filename, "type": "json_obj"}
                        except:
                            filename = f"{key}.obj"
                            with zf.open(filename, "w") as f: pickle.dump(val, f)
                            manifest[key] = {"file": filename, "type": "pickle_obj"}
                
                zf.writestr("manifest.json", json.dumps(manifest, indent=4))
            
            loading_dialog.open = False
            page.snack_bar = ft.SnackBar(ft.Text("Proyecto guardado correctamente."), bgcolor="green", open=True)
        except Exception as ex:
            loading_dialog.open = False
            print(traceback.format_exc())
            page.snack_bar = ft.SnackBar(ft.Text(f"Error guardando: {ex}"), bgcolor="red", open=True)
        page.update()

    # --- LÓGICA DE CARGA (LAZY) ---
    def load_project_state(e: ft.FilePickerResultEvent):
        unlock_ui()
        if not e.files: return
        loading_text.value = "Cargando datos..."; loading_bar.value = 0.0; loading_dialog.open = True; page.update()

        try:
            path = e.files[0].path
            with zipfile.ZipFile(path, "r") as zf:
                manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
                total = len(manifest)
                
                for i, (key, meta) in enumerate(manifest.items()):
                    loading_bar.value = (i / total) * 0.95; page.update()
                    filename, ftype = meta["file"], meta["type"]
                    
                    if ftype == "dataframe" or ftype == "pickle_obj":
                        with zf.open(filename, "r") as f: setattr(page.session, key, pickle.load(f))
                    elif ftype == "json_obj":
                        with zf.open(filename, "r") as f: setattr(page.session, key, json.loads(f.read().decode("utf-8")))

            # Limpiar vistas para obligar a regenerar
            imputacion_view_container.content = None
            analisis_view_container.content = None
            gastos_view_container.content = None

            loading_bar.value = 1.0; page.update(); time.sleep(0.5)
            loading_dialog.open = False
            page.snack_bar = ft.SnackBar(ft.Text("Proyecto Cargado."), bgcolor="green", open=True)
            go_to_menu()
            
        except Exception as ex:
            loading_dialog.open = False
            print(f"Error carga: {traceback.format_exc()}")
            page.snack_bar = ft.SnackBar(ft.Text(f"Error carga: {ex}"), bgcolor="red", open=True)
        page.update()

    # Helpers de Picker seguros
    def safe_open_picker(picker, mode="load"):
        lock_ui()
        if mode=="load": picker.pick_files(allowed_extensions=["hds"])
        else: picker.save_file(file_name="Proyecto.hds")

    pk_save_proj = ft.FilePicker(on_result=save_project_state)
    pk_load_proj = ft.FilePicker(on_result=load_project_state)
    page.overlay.extend([pk_save_proj, pk_load_proj])

    # --- NAVEGACIÓN SEGURA ---
    def reset_views():
        menu_view.visible = False
        about_view.visible = False
        imputacion_view_container.visible = False
        analisis_view_container.visible = False
        gastos_view_container.visible = False

    def safe_nav(target_func):
        lock_ui()
        time.sleep(0.05) # Pequeña pausa para que la UI se actualice
        try:
            target_func(None)
        except Exception as e:
            print(f"Error Nav: {e}")
        finally:
            unlock_ui()

    def go_to_menu(e=None):
        reset_views()
        page.vertical_alignment = ft.MainAxisAlignment.CENTER
        page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        menu_view.visible = True
        page.update()

    def go_to_imputacion(e):
        reset_views()
        if imputacion_view_container.content is None:
            imputacion_view_container.content = imputacion_app.build_imputacion_view(page, go_to_menu)
        page.vertical_alignment = ft.MainAxisAlignment.START
        page.horizontal_alignment = ft.CrossAxisAlignment.START
        imputacion_view_container.visible = True
        page.update()

    def go_to_analisis(e):
        reset_views()
        if analisis_view_container.content is None:
            analisis_view_container.content = analisis_app.build_analisis_view(page, go_to_menu)
        page.vertical_alignment = ft.MainAxisAlignment.START
        page.horizontal_alignment = ft.CrossAxisAlignment.START
        analisis_view_container.visible = True
        page.update()
        
    def go_to_gastos(e):
        reset_views()
        if gastos_view_container.content is None:
            gastos_view_container.content = gastos_app.build_gastos_view(page, go_to_menu)
        page.vertical_alignment = ft.MainAxisAlignment.START
        page.horizontal_alignment = ft.CrossAxisAlignment.START
        gastos_view_container.visible = True
        page.update()
    
    def go_to_about(e):
        reset_views()
        page.vertical_alignment = ft.MainAxisAlignment.CENTER
        page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        about_view.visible = True
        page.update()

    about_view.content = ft.Container(
        content=ft.Column([
            ft.Icon(ft.Icons.INFO_OUTLINE, size=60, color=COLOR_ACENTO),
            ft.Text("ACERCA DE", size=30, weight="bold", color="white"),
            ft.Divider(color=COLOR_ACENTO),
            ft.Text("SISTEMA DE ANÁLISIS HIDROLÓGICO", size=20, color=COLOR_ACENTO),
            ft.Text("Versión 6.2.1", color=COLOR_GRIS_CLARO),
            ft.Container(height=20),
            ft.Text("Desarrollado para:", color=COLOR_GRIS_MEDIO),
            ft.Text("GEOGRAFICA S.A. DE C.V.", size=25, weight="bold", color="white"),
            ft.Text("Por: Ing. Juan Jesús Torres Solano", color=COLOR_GRIS_MEDIO),
            ft.Container(height=30),
            MatrixButton("VOLVER AL MENÚ", ft.Icons.ARROW_BACK, go_to_menu, width=250)
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        padding=40, border=ft.border.all(1, COLOR_ACENTO), bgcolor=COLOR_SUPERFICIE, width=600
    )

    # --- BOTONES DE MENÚ (Registrados para bloqueo) ---
    btn_imp = MatrixButton("1. IMPUTACIÓN DE ESTACIONES", ft.Icons.MEMORY, lambda _: safe_nav(go_to_imputacion))
    btn_ana = MatrixButton("2. ANÁLISIS PRECIPITACIÓN", ft.Icons.SHOW_CHART, lambda _: safe_nav(go_to_analisis))
    btn_gas = MatrixButton("3. CÁLCULO DE GASTOS", ft.Icons.CALCULATE_OUTLINED, lambda _: safe_nav(go_to_gastos))
    btn_sav = MatrixButton("GUARDAR", ft.Icons.SAVE, lambda _: safe_open_picker(pk_save_proj, "save"), width=220, color="#2196F3")
    btn_lod = MatrixButton("CARGAR", ft.Icons.UPLOAD_FILE, lambda _: safe_open_picker(pk_load_proj, "load"), width=220, color="#FFC107")
    btn_abt = MatrixButton("ACERCA DE", ft.Icons.INFO, lambda _: safe_nav(go_to_about), width=450, color="#9E9E9E")
    
    menu_buttons.extend([btn_imp, btn_ana, btn_gas, btn_sav, btn_lod, btn_abt])

    menu_view.content = ft.Container(
        content=ft.Column([
            TerminalHeader(),
            ft.Image(src="path19.jpg", width=30, fit=ft.ImageFit.CONTAIN), 
            ft.Text("ANÁLISIS HIDROLÓGICO", size=20, weight="bold", color="white"),
            ft.Divider(height=2, color="transparent"),
            btn_imp,
            ft.Divider(height=2, color="transparent"),
            btn_ana,
            ft.Divider(height=2, color="transparent"),
            btn_gas,
            ft.Divider(height=6, color=add_opacity(COLOR_ACENTO, 0.2)),
            ft.Row([btn_sav, btn_lod], alignment=ft.MainAxisAlignment.CENTER, spacing=10),
            ft.Divider(height=2, color="transparent"),
            btn_abt,
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        padding=40, border=ft.border.all(1, add_opacity(COLOR_ACENTO, 0.2)), bgcolor=COLOR_SUPERFICIE, alignment=ft.alignment.center
    )
    
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    try:
        imputacion_view_container.content = None 
        analisis_view_container.content = None
        gastos_view_container.content = None
        
        page.add(ft.Container(content=ft.Stack([menu_view, about_view, imputacion_view_container, analisis_view_container, gastos_view_container]), expand=True, padding=10))
        go_to_menu()
    except Exception as e:
        print(f"❌ ERROR AL CONSTRUIR VISTAS INICIALES: {e}")
        page.add(ft.Text(f"Error fatal al iniciar: {e}", color="red"))

if __name__ == "__main__":
    ft.app(target=main, assets_dir="assets")