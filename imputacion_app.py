import flet as ft
import os
import imputacion_logic 
import webbrowser

# --- Controles Globales ---
input_file_picker = ft.FilePicker()
output_file_picker = ft.FilePicker()
shp_export_picker = ft.FilePicker()      # MODIFICADO: Para Paso 1 (SHP)
shp_filtered_picker = ft.FilePicker()    # MODIFICADO: Para Paso 2 (SHP)

def build_imputacion_view(page: ft.Page, on_back_to_menu):
    
    # --- 1. INICIALIZACIÓN DE SESIÓN ---
    if not hasattr(page.session, "imput_folder_path"): page.session.imput_folder_path = None
    if not hasattr(page.session, "imput_output_folder"): page.session.imput_output_folder = None
    if not hasattr(page.session, "imput_station_files"): page.session.imput_station_files = {}
    if not hasattr(page.session, "imput_map_path"): page.session.imput_map_path = None
    if not hasattr(page.session, "df_imputado_resultado"): page.session.df_imputado_resultado = None

    # --- 2. ELEMENTOS VISUALES ---
    selected_input_folder = ft.Text("Ninguna carpeta seleccionada", color="grey")
    station_count_text = ft.Text("", visible=False, color="#00ff41") 
    open_map_button = ft.ElevatedButton("Abrir Mapa de Estaciones", icon=ft.Icons.MAP, visible=False)
    
    # Botón Paso 1: Exportar TODAS como SHP
    export_qgis_button = ft.ElevatedButton(
        "Exportar Shapefile (QGIS)", 
        icon=ft.Icons.MAP_SHARP, # Icono cambiado
        visible=False,
        on_click=lambda _: shp_export_picker.save_file(file_name="estaciones.shp", allowed_extensions=["shp"])
    )

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

    # Botón Paso 2: Exportar FILTRADAS como SHP
    export_filtered_button = ft.ElevatedButton(
        "Exportar Vecinas en Rango (SHP)", 
        icon=ft.Icons.FILTER_CENTER_FOCUS,
        disabled=True, 
        on_click=lambda _: shp_filtered_picker.save_file(file_name=f"vecinas_{dd_estaciones.value}.shp", allowed_extensions=["shp"])
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
            export_qgis_button.visible = True 

    # --- LÓGICA PASO 1: Guardar TODAS (SHP) ---
    def save_metadata_shp(e):
        if e.path and page.session.imput_station_files:
            try:
                data_list = []
                for s_id, data in page.session.imput_station_files.items():
                    data_list.append({
                        "ID": s_id,
                        "LATITUD": data['lat'],
                        "LONGITUD": data['lon'],
                        "ALTITUD": data.get('alt', 0.0),
                        "ROL": "GENERICO"
                    })
                
                # LLAMADA A LOGICA SHAPEFILE
                final_path = imputacion_logic.exportar_shapefile(data_list, e.path)
                
                page.snack_bar = ft.SnackBar(ft.Text(f"Shapefile creado: {final_path}"), bgcolor="green", open=True)
            except Exception as ex:
                page.snack_bar = ft.SnackBar(ft.Text(f"Error generando SHP: {ex}"), bgcolor="red", open=True)
            page.update()

    # --- LÓGICA PASO 2: Guardar FILTRADAS (SHP) ---
    def save_filtered_shp(e):
        if not e.path or not dd_estaciones.value: return
        
        target_id = dd_estaciones.value
        try: radius = float(dd_radio.value)
        except: radius = 150.0
        
        stations = page.session.imput_station_files
        if target_id not in stations: return

        try:
            target_data = stations[target_id]
            lat_t, lon_t = target_data['lat'], target_data['lon']
            
            filtered_list = []
            
            # 1. Agregar la Estación Objetivo
            filtered_list.append({
                "ID": target_id,
                "LATITUD": lat_t,
                "LONGITUD": lon_t,
                "ALTITUD": target_data.get('alt', 0.0),
                "DISTANCIA_KM": 0.0,
                "ROL": "OBJETIVO"
            })
            
            # 2. Filtrar Vecinas
            count = 0
            for s_id, data in stations.items():
                if s_id == target_id: continue 
                dist = imputacion_logic.calculate_distance(lat_t, lon_t, data['lat'], data['lon'])
                
                if dist <= radius:
                    filtered_list.append({
                        "ID": s_id,
                        "LATITUD": data['lat'],
                        "LONGITUD": data['lon'],
                        "ALTITUD": data.get('alt', 0.0),
                        "DISTANCIA_KM": round(dist, 2),
                        "ROL": "VECINA"
                    })
                    count += 1
            
            if filtered_list:
                # LLAMADA A LOGICA SHAPEFILE
                final_path = imputacion_logic.exportar_shapefile(filtered_list, e.path)
                
                page.snack_bar = ft.SnackBar(ft.Text(f"SHP Generado con {count} vecinas."), bgcolor="green", open=True)
            else:
                page.snack_bar = ft.SnackBar(ft.Text("No se encontraron estaciones en ese radio."), bgcolor="orange", open=True)
            
            page.update()

        except Exception as ex:
             page.snack_bar = ft.SnackBar(ft.Text(f"Error filtro SHP: {ex}"), bgcolor="red", open=True)
             page.update()

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
                page.session.df_imputado_resultado = df_res
                out_path = imputacion_logic.save_target_csv(df_res, target_id, page.session.imput_output_folder)
                msg_display = log_msg if len(log_msg) < 2000 else log_msg[:2000] + "\n... [Log truncado]"
                log_result.value = f"✅ ÉXITO (Radio {radius_km}km):\n{msg_display}\nGuardado en: {out_path}"
                log_result.color = "green"
                view_results_button.visible = True
                page.snack_bar = ft.SnackBar(ft.Text("Proceso finalizado."), bgcolor="green", open=True)
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

    # --- CALLBACKS Y UTILS ---
    def on_station_change(e):
        impute_button.disabled = False
        impute_button.text = f"IMPUTAR: {dd_estaciones.value}"
        try: rad = dd_radio.value 
        except: rad = "150"
        export_filtered_button.text = f"Exportar Vecinas en {rad}km (SHP)"
        export_filtered_button.disabled = False
        page.update()

    def on_radius_change(e):
        if dd_estaciones.value:
            export_filtered_button.text = f"Exportar Vecinas en {dd_radio.value}km (SHP)"
            page.update()

    input_file_picker.on_result = on_input_folder_selected
    output_file_picker.on_result = lambda e: setattr(page.session, 'imput_output_folder', e.path) or setattr(selected_output_folder, 'value', f"Salida: {e.path}") or page.update()
    
    shp_export_picker.on_result = save_metadata_shp     # Paso 1
    shp_filtered_picker.on_result = save_filtered_shp   # Paso 2
    
    dd_estaciones.on_change = on_station_change
    dd_radio.on_change = on_radius_change
    impute_button.on_click = on_click_impute
    view_results_button.on_click = lambda _: os.startfile(page.session.imput_output_folder) if os.path.exists(page.session.imput_output_folder) else None

    page.overlay.extend([input_file_picker, output_file_picker, shp_export_picker, shp_filtered_picker])
    
    def on_reset(e):
        # 1. Limpiar Datos de Sesión
        page.session.imput_folder_path = None
        page.session.imput_output_folder = None
        page.session.imput_station_files = {}
        page.session.imput_map_path = None
        page.session.df_imputado_resultado = None

        # 2. Reiniciar UI (Controles Visuales)
        selected_input_folder.value = "Ninguna carpeta seleccionada"
        selected_input_folder.color = "grey"
        
        station_count_text.value = ""
        station_count_text.visible = False
        
        open_map_button.visible = False
        
        dd_estaciones.options = []
        dd_estaciones.value = None
        dd_estaciones.disabled = True
        
        dd_radio.value = "50" # Valor por defecto
        
        selected_output_folder.value = "Ninguna carpeta seleccionada"
        selected_output_folder.color = "grey"
        
        impute_button.disabled = True
        view_results_button.disabled = True
        
        log_result.value = "Esperando ejecución..."
        pb.value = 0
        pbl.value = ""

        page.snack_bar = ft.SnackBar(ft.Text("Módulo de Imputación Reiniciado..Torpe"), bgcolor="red")
        page.snack_bar.open = True
        page.update()

    # --- 4. VISTAS (LAYOUT ORIGINAL) ---
    
    view_fuente = ft.Container(
        content=ft.Column([
            ft.Text("Paso 1: Selección de Fuente de Datos", style=ft.TextThemeStyle.HEADLINE_SMALL, color="#00ff41"),
            ft.Divider(),
            ft.Text("Seleccione la carpeta que contiene los archivos .txt de las estaciones, Daa!!."),
            ft.Row([
                ft.ElevatedButton("Seleccionar Carpeta", icon=ft.Icons.FOLDER_OPEN, on_click=lambda _: input_file_picker.get_directory_path()),
                selected_input_folder
            ]),
            station_count_text,
            ft.Row([open_map_button, export_qgis_button], spacing=10)
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
            dd_radio,
            ft.Container(height=10),
            export_filtered_button 
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
        on_change=change_view,
        trailing=ft.Column([
            ft.Divider(),
            ft.IconButton(
                icon=ft.Icons.RESTART_ALT, 
                tooltip="Reiniciar Módulo", 
                on_click=on_reset, # Llamar a la función de limpieza local
                icon_color="red"
            ),
            ft.Text("Reiniciar", size=10, color="red")
        ], alignment=ft.MainAxisAlignment.END, spacing=5)
    )

    if page.session.imput_folder_path:
        selected_input_folder.value = f"Fuente: {page.session.imput_folder_path}"
        if page.session.imput_station_files:
            restore_station_ui(page.session.imput_station_files)
    
    if page.session.df_imputado_resultado is not None:
        log_result.value = "✅ Datos recuperados."
        log_result.color = "green"
        log_result.visible = True
        rail.selected_index = 2
        change_view(type('obj', (object,), {'control': type('obj', (object,), {'selected_index': 2})}))

    return ft.Row([rail, ft.VerticalDivider(width=1), ft.Column(views, expand=True, scroll=ft.ScrollMode.ADAPTIVE)], expand=True)