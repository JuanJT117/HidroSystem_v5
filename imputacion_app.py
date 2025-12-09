import flet as ft
import os
import imputacion_logic 
import webbrowser

# --- Controles Globales ---
input_file_picker = ft.FilePicker()
output_file_picker = ft.FilePicker()

def build_imputacion_view(page: ft.Page, on_back_to_menu):
    
    # --- 1. INICIALIZACIÓN DE SESIÓN ---
    if not hasattr(page.session, "imput_folder_path"): page.session.imput_folder_path = None
    if not hasattr(page.session, "imput_output_folder"): page.session.imput_output_folder = None
    if not hasattr(page.session, "imput_station_files"): page.session.imput_station_files = {}
    if not hasattr(page.session, "imput_map_path"): page.session.imput_map_path = None
    if not hasattr(page.session, "df_imputado_resultado"): page.session.df_imputado_resultado = None

    # --- 2. ELEMENTOS VISUALES (ESTILO ORIGINAL) ---
    selected_input_folder = ft.Text("Ninguna carpeta seleccionada", color="grey")
    station_count_text = ft.Text("", visible=False, color="#00ff41") 
    open_map_button = ft.ElevatedButton("Abrir Mapa de Estaciones", icon=ft.Icons.MAP, visible=False)

    dd_estaciones = ft.Dropdown(
        label="Seleccione Estación Objetivo",
        hint_text="Elija la estación a imputar...",
        width=400,
        options=[],
        disabled=True,
        focused_border_color="#00ff41"
    )

    dd_radio = ft.Dropdown(
        label="Radio de Búsqueda (km)",
        hint_text="Distancia para buscar vecinos...",
        width=200,
        value="150",
        options=[
            ft.dropdown.Option("50"),
            ft.dropdown.Option("100"),
            ft.dropdown.Option("150"),
            ft.dropdown.Option("200"),
            ft.dropdown.Option("250"),
            ft.dropdown.Option("300"),
        ],
        focused_border_color="#00ff41"
    )

    selected_output_folder = ft.Text("Salida: Carpeta 'Imputado' (Automático)")
    impute_button = ft.ElevatedButton(
        "INICIAR IMPUTACIÓN",
        disabled=True,
        icon=ft.Icons.AUTO_FIX_HIGH,
        style=ft.ButtonStyle(color="#00ff41")
    )
    pb = ft.ProgressBar(value=0, visible=False, color="#00ff41", bgcolor="#111111")
    pbl = ft.Text("", visible=False)
    log_result = ft.Text("", visible=False, font_family="Roboto Mono", size=12)
    view_results_button = ft.ElevatedButton("Abrir carpeta de resultados", visible=False, icon=ft.Icons.FOLDER_OPEN)

    # --- 3. LÓGICA INTERNA ---

    def regenerate_map_if_needed():
        """Regenera el mapa HTML si tenemos estaciones cargadas pero no el archivo físico."""
        if page.session.imput_station_files and not page.session.imput_map_path:
             map_path = imputacion_logic.generar_mapa_html(page.session.imput_station_files, ".")
             page.session.imput_map_path = map_path
        
        if page.session.imput_map_path:
            open_map_button.visible = True
            open_map_button.on_click = lambda _: webbrowser.open(f"file:///{page.session.imput_map_path}")

    def on_input_folder_selected(e):
        if e.path: 
            page.session.imput_folder_path = e.path
            page.session.imput_output_folder = os.path.join(e.path, 'Imputado')
            selected_input_folder.value = f"Fuente: {e.path}"
            
            pbl.value = "Escaneando estaciones..."; pbl.visible = True; page.update()
            
            station_files = imputacion_logic.leer_estaciones(e.path)
            page.session.imput_station_files = station_files
            
            restore_station_ui(station_files)
            
            pbl.visible = False
            page.update()

    def restore_station_ui(station_files):
        """Actualiza el dropdown y textos basado en los archivos cargados."""
        dd_estaciones.options = []
        if station_files:
            keys = sorted(list(station_files.keys()))
            for k in keys: dd_estaciones.options.append(ft.dropdown.Option(k))
            dd_estaciones.disabled = False
            station_count_text.value = f"{len(keys)} estaciones encontradas."
            station_count_text.visible = True
            regenerate_map_if_needed()
        else:
            station_count_text.value = "No se encontraron archivos .txt válidos."

    def on_click_impute(e):
        target_id = dd_estaciones.value
        if not target_id: return
        
        try: radius_km = int(dd_radio.value)
        except: radius_km = 150 
        
        impute_button.disabled = True; pb.visible = True; pbl.visible = True
        log_result.visible = False; view_results_button.visible = False
        page.update()
        
        try:
            if not page.session.imput_output_folder: # Fallback seguro
                 page.session.imput_output_folder = os.path.join(page.session.imput_folder_path, 'Imputado')

            if not os.path.exists(page.session.imput_output_folder):
                os.makedirs(page.session.imput_output_folder)

            df_res, log_msg = imputacion_logic.impute_target_station(
                target_id, 
                page.session.imput_station_files, 
                page, pb, pbl,
                radius_km 
            )
            
            if df_res is not None:
                # Guardamos resultado en sesión para persistencia
                page.session.df_imputado_resultado = df_res
                
                out_path = imputacion_logic.save_target_csv(df_res, target_id, page.session.imput_output_folder)
                msg_display = log_msg if len(log_msg) < 2000 else log_msg[:2000] + "\n... [Log truncado]"
                log_result.value = f"✅ ÉXITO (Radio {radius_km}km):\n{msg_display}\nGuardado en: {out_path}"
                log_result.color = "green"
                view_results_button.visible = True
                page.snack_bar = ft.SnackBar(ft.Text("Proceso finalizado y guardado correctamente."), bgcolor="green", open=True)
            else:
                log_result.value = f"❌ ERROR LÓGICO:\n{log_msg}"
                log_result.color = "red"
                page.snack_bar = ft.SnackBar(ft.Text("Error durante el cálculo."), bgcolor="red", open=True)
        
        except Exception as ex:
             log_result.value = f"❌ ERROR CRÍTICO:\n{ex}"
             log_result.color = "red"
             page.snack_bar = ft.SnackBar(ft.Text(f"Error crítico: {ex}"), bgcolor="red", open=True)

        pb.visible = False; log_result.visible = True; impute_button.disabled = False
        page.update()

    # --- CALLBACKS ---
    input_file_picker.on_result = on_input_folder_selected
    output_file_picker.on_result = lambda e: setattr(page.session, 'imput_output_folder', e.path) or setattr(selected_output_folder, 'value', f"Salida: {e.path}") or page.update()
    dd_estaciones.on_change = lambda e: setattr(impute_button, 'disabled', False) or setattr(impute_button, 'text', f"IMPUTAR: {dd_estaciones.value}") or page.update()
    impute_button.on_click = on_click_impute
    view_results_button.on_click = lambda _: os.startfile(page.session.imput_output_folder) if os.path.exists(page.session.imput_output_folder) else None

    page.overlay.extend([input_file_picker, output_file_picker])

    # --- 4. VISTAS (LAYOUT ORIGINAL) ---
    
    view_fuente = ft.Container(
        content=ft.Column([
            ft.Text("Paso 1: Selección de Fuente de Datos", style=ft.TextThemeStyle.HEADLINE_SMALL, color="#00ff41"),
            ft.Divider(),
            ft.Text("Seleccione la carpeta que contiene los archivos .txt de las estaciones."),
            ft.Row([
                ft.ElevatedButton("Seleccionar Carpeta", icon=ft.Icons.FOLDER_OPEN, on_click=lambda _: input_file_picker.get_directory_path()),
                selected_input_folder
            ]),
            station_count_text,
            open_map_button
        ]), padding=20, visible=True
    )

    view_objetivo = ft.Container(
        content=ft.Column([
            ft.Text("Paso 2: Selección de Estación Objetivo", style=ft.TextThemeStyle.HEADLINE_SMALL, color="#00ff41"),
            ft.Divider(),
            ft.Text("El sistema usará las estaciones vecinas para rellenar los datos faltantes."),
            dd_estaciones,
            ft.Divider(),
            ft.Text("Configuración Geográfica:", weight="bold"),
            ft.Text("Seleccione el radio de búsqueda para zonas de llanura o dispersas."),
            dd_radio 
        ]), padding=20, visible=False
    )

    view_exec = ft.Container(
        content=ft.Column([
            ft.Text("Paso 3: Ejecución y Resultados", style=ft.TextThemeStyle.HEADLINE_SMALL, color="#00ff41"),
            ft.Divider(),
            ft.Row([
                ft.ElevatedButton("Carpeta Salida", icon=ft.Icons.DRIVE_FILE_MOVE_OUTLINE, on_click=lambda _: output_file_picker.get_directory_path()),
                selected_output_folder
            ]),
            ft.Divider(),
            impute_button,
            pbl, pb,
            ft.Container(content=log_result, bgcolor="#111111", padding=10, border_radius=5),
            view_results_button
        ], scroll=ft.ScrollMode.ADAPTIVE), padding=20, visible=False, expand=True
    )

    views = [view_fuente, view_objetivo, view_exec]

    def change_view(e):
        idx = e.control.selected_index
        for i, v in enumerate(views): v.visible = (i == idx)
        page.update()

    rail = ft.NavigationRail(
        selected_index=0,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=100, min_extended_width=400,
        leading=ft.Column([
            ft.IconButton(ft.Icons.ARROW_BACK, on_click=on_back_to_menu, tooltip="Volver al Menú"),
            ft.Text("Imputación", weight="bold")
        ], spacing=10),
        group_alignment=-0.9,
        destinations=[
            ft.NavigationRailDestination(icon=ft.Icons.FOLDER, label="Fuente"),
            ft.NavigationRailDestination(icon=ft.Icons.GPS_FIXED, label="Objetivo"),
            ft.NavigationRailDestination(icon=ft.Icons.TERMINAL, label="Proceso"),
        ],
        on_change=change_view
    )

    # --- 5. RESTAURACIÓN DE ESTADO (AUTO-CARGA) ---
    # Si existen datos en sesión, restauramos la UI al estado previo
    if page.session.imput_folder_path:
        selected_input_folder.value = f"Fuente: {page.session.imput_folder_path}"
        if page.session.imput_station_files:
            restore_station_ui(page.session.imput_station_files)
    
    if page.session.df_imputado_resultado is not None:
        log_result.value = "✅ Datos recuperados de la sesión anterior. Puede revisar los resultados o iniciar un nuevo cálculo."
        log_result.color = "green"
        log_result.visible = True
        # Saltamos a la vista de resultados si ya hay datos
        rail.selected_index = 2
        change_view(type('obj', (object,), {'control': type('obj', (object,), {'selected_index': 2})}))

    return ft.Row([rail, ft.VerticalDivider(width=1), ft.Column(views, expand=True, scroll=ft.ScrollMode.ADAPTIVE)], expand=True)