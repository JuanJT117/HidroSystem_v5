import flet as ft
import os
import threading
import webbrowser
import asyncio
import pandas as pd
import folium
import tempfile

# --- ARQUITECTURA HEXAGONAL: Importación desde el Core ---
from core import imputacion_logic 
from ui.components import COLOR_ACENTO, COLOR_SUPERFICIE, FUENTE_PRINCIPAL, add_opacity

def build_imputacion_view(page: ft.Page):
    
    # ==========================================
    # 1. INICIALIZACIÓN DE SESIÓN (Protegida)
    # ==========================================
    if not page.session.get("imput_folder_path"): page.session.set("imput_folder_path", None)
    if not page.session.get("imput_output_folder"): page.session.set("imput_output_folder", None)
    if not page.session.get("imput_station_files"): page.session.set("imput_station_files", {})
    if not page.session.get("imput_map_path"): page.session.set("imput_map_path", None)
    
    # ESTRUCTURA DE DATOS PARA PROCESAMIENTO MÚLTIPLE
    if not page.session.get("db_series_crudas"): page.session.set("db_series_crudas", {})
    if not page.session.get("cola_imputacion"): page.session.set("cola_imputacion", [])

    # ==========================================
    # 2. VARIABLES DE ESTADO Y UI
    # ==========================================
    # Variable de control de hilos (Global dentro de la vista)
    state = {"imputing": False}
    
    # --- SISTEMA DE BLOQUEO ZERO-TRUST ---
    
    # 1. Nuevos controles de progreso lineal y numérico
    texto_porcentaje = ft.Text("0%", color="#00f0ff", weight="bold", size=50, font_family="Roboto Mono")
    pb = ft.ProgressBar(width=600, color="#00f0ff", bgcolor="#111111", value=0)
    pbl = ft.Text("Preparando entorno matemático...", color="#00f0ff", size=12, italic=True, font_family="Roboto Mono")
    
    terminal_radiactiva = ft.ListView(expand=True, spacing=5, auto_scroll=True)
    
    bloqueo_radiactivo = ft.Container(
        content=ft.Column([
            texto_porcentaje,  # <--- El porcentaje gigante
            pb,                # <--- La barra lineal llenándose
            pbl,               # <--- El texto que dice en qué paso va
            ft.Divider(color="transparent", height=10),
            ft.Text("SISTEMA DE IMPUTACIÓN ACTIVO", color="#00f0ff", weight="bold", size=22, font_family="Roboto Mono"),
            ft.Text("Procesando Tensores SARIMAX y Regresión Múltiple. No cierre la ventana.", color="grey", size=12),
            ft.Container(
                content=terminal_radiactiva, height=250, width=700,
                bgcolor="#050505", border=ft.border.all(1, "#00f0ff"), border_radius=5, padding=10
            )
        ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        bgcolor=add_opacity("#050505", 0.95), expand=True, alignment=ft.alignment.center, visible=False
    )
    page.overlay.append(bloqueo_radiactivo)

    # 2. Clase Interceptora (Proxy Pattern): Traduce el progreso crudo a porcentaje visual
    class ProgressInterceptor:
        def __init__(self, pb_real, txt_porc):
            self.pb = pb_real
            self.txt = txt_porc
        @property
        def value(self): 
            return self.pb.value
        @value.setter
        def value(self, v):
            self.pb.value = v
            if v is not None: 
                self.txt.value = f"{int(v * 100)}%"

    pb_interceptado = ProgressInterceptor(pb, texto_porcentaje)
    
    def log_radiactivo(msg):
        def _update():
            terminal_radiactiva.controls.append(ft.Text(f"> {msg}", color="#00f0ff", font_family="monospace", size=11))
            page.update()
        page.run_thread(_update)

    # --- SISTEMA DE EXPORTACIÓN CSV INDIVIDUAL ---
    estacion_a_exportar = [None]
    def manejar_exportacion(e: ft.FilePickerResultEvent):
        if e.path and estacion_a_exportar[0] is not None:
            try:
                estacion_a_exportar[0].to_csv(e.path, index=False)
                page.snack_bar = ft.SnackBar(ft.Text("✅ Archivo CSV Exportado con Éxito"), bgcolor="#00f0ff", open=True)
            except Exception as ex:
                page.snack_bar = ft.SnackBar(ft.Text(f"❌ Error al guardar: {ex}"), bgcolor="red", open=True)
            page.update()

    picker_exportar_csv = ft.FilePicker(on_result=manejar_exportacion)
    page.overlay.append(picker_exportar_csv)
    
    # --- SISTEMA DE EXPORTACIÓN LOG INDIVIDUAL (QA) ---
    log_a_exportar = [None]
    def manejar_exportacion_log(e: ft.FilePickerResultEvent):
        if e.path and log_a_exportar[0] is not None:
            try:
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(e.path, 'w', encoding='utf-8') as f:
                    f.write("=================================================================\n")
                    f.write(" REPORTE DE CONTROL DE CALIDAD (QA) - IMPUTACIÓN MATEMÁTICA\n")
                    f.write("=================================================================\n")
                    f.write(f" Fecha de Descarga : {timestamp}\n")
                    f.write("=================================================================\n\n")
                    f.write(log_a_exportar[0])
                    f.write("\n\n=================================================================\n")
                    f.write(" FIN DEL REPORTE\n")
                page.snack_bar = ft.SnackBar(ft.Text("✅ Informe QA Exportado con Éxito"), bgcolor="#00f0ff", open=True)
            except Exception as ex:
                page.snack_bar = ft.SnackBar(ft.Text(f"❌ Error al guardar log: {ex}"), bgcolor="red", open=True)
            page.update()

    picker_exportar_log = ft.FilePicker(on_result=manejar_exportacion_log)
    page.overlay.append(picker_exportar_log)
    # ------------------------------------------------
    
    # Controles de Texto y Estado
    selected_input_folder = ft.Text("Ninguna carpeta seleccionada", color="grey", size=12)
    selected_output_folder = ft.Text("Carpeta de salida: Pendiente", color="grey", size=12)
    station_count_text = ft.Text("", visible=False, color=COLOR_ACENTO, weight="bold")
    
    # Consola y Progreso
    # Consola y Progreso
    log_result = ft.Text("", visible=False, size=11, font_family="monospace")
    terminal_list = ft.ListView(expand=True, spacing=5, auto_scroll=True)

    # Contenedores Dinámicos
    station_files_container = ft.Column(spacing=10, scroll=ft.ScrollMode.AUTO)
    cola_espera_ui = ft.Column(spacing=5, scroll=ft.ScrollMode.AUTO)
    visor_resultados_grid = ft.Row(wrap=True, spacing=10, scroll=ft.ScrollMode.AUTO)

    # ==========================================
    # 3. MANEJADORES DE ARCHIVOS (PICKERS)
    # ==========================================
    def auto_procesar_fuente(ruta_fuente):
        """Motor central de lectura. Puede ser llamado por el Picker o por auto-detección."""
        if not ruta_fuente or not os.path.exists(ruta_fuente): return
        
        page.session.set("imput_folder_path", ruta_fuente)
        selected_input_folder.value = f"Fuente (Auto-detectada): {ruta_fuente}"
        
        estaciones = imputacion_logic.leer_estaciones(ruta_fuente)
        if estaciones:
            page.session.set("imput_station_files", estaciones)
            map_path = imputacion_logic.generar_mapa_html(estaciones, ruta_fuente)
            if map_path:
                page.session.set("imput_map_path", {"type": "file", "path": map_path})
            
            restore_station_ui(estaciones)
            open_map_button.disabled = False
            page.snack_bar = ft.SnackBar(ft.Text(f"✅ {len(estaciones)} estaciones enlazadas desde la memoria del proyecto."), bgcolor="#00ff41", open=True)
        else:
            page.snack_bar = ft.SnackBar(ft.Text("⚠️ La carpeta detectada no contiene archivos .txt válidos"), bgcolor="orange", open=True)
        page.update()

    def on_input_dir_result(e: ft.FilePickerResultEvent):
        if e.path:
            auto_procesar_fuente(e.path)

    def on_output_dir_result(e: ft.FilePickerResultEvent):
        if e.path:
            page.session.set("imput_output_folder", e.path)
            selected_output_folder.value = f"Destino: {e.path}"
            page.update()

    file_picker_input = ft.FilePicker(on_result=on_input_dir_result)
    file_picker_output = ft.FilePicker(on_result=on_output_dir_result)
    page.overlay.extend([file_picker_input, file_picker_output])

    # ==========================================
    # 4. LÓGICA DE COLA, MEMORIA E INSPECCIÓN
    # ==========================================
    def agregar_a_cola(sid):
        cola = page.session.get("cola_imputacion")
        if sid not in cola:
            cola.append(sid)
            page.session.set("cola_imputacion", cola)
            render_cola_visual()
            
            # --- FEEDBACK VISUAL EN TIEMPO REAL ---
            indicador_estacion.value = f"🟢 [LISTO] Estación {sid} añadida a la cola de procesamiento."
            indicador_estacion.color = "#00f0ff"
            
            # --- RE-RENDER REACTIVO: Forzamos el rediseño de las tarjetas para aplicar el azul radiactivo ---
            estaciones_actuales = page.session.get("imput_station_files") or {}
            restore_station_ui(estaciones_actuales, actualizar_textos=False)
            
            page.snack_bar = ft.SnackBar(ft.Text(f"Estación {sid} en cola"), bgcolor="blue", open=True)
            page.update()

    def quitar_de_cola(sid):
        cola = page.session.get("cola_imputacion")
        if sid in cola: cola.remove(sid)
        page.session.set("cola_imputacion", cola)
        render_cola_visual()
        
        # --- RE-RENDER REACTIVO: Restauramos la tarjeta a su estado normal al salir de la cola ---
        estaciones_actuales = page.session.get("imput_station_files") or {}
        restore_station_ui(estaciones_actuales, actualizar_textos=False)

    def render_cola_visual():
        cola_espera_ui.controls.clear()
        for sid in (page.session.get("cola_imputacion") or []):
            cola_espera_ui.controls.append(
                ft.Row([
                    ft.Icon(ft.Icons.TIMER, color="orange", size=20),
                    ft.Text(f"{sid} (Espera)", expand=True),
                    ft.IconButton(ft.Icons.DELETE_OUTLINE, icon_color="red", on_click=lambda e, s=sid: quitar_de_cola(s))
                ])
            )
        page.update()

    def render_visor_resultados():
        visor_resultados_grid.controls.clear()
        db = page.session.get("db_series_crudas") or {}
        
        def _trigger_download(sid, df_dict):
            # Lee el Parquet desde el disco duro justo antes de exportarlo
            try:
                df = pd.read_parquet(df_dict["path"]) if isinstance(df_dict, dict) else df_dict
                estacion_a_exportar[0] = df
                picker_exportar_csv.save_file(
                    dialog_title=f"Exportar {sid} a CSV",
                    file_name=f"{sid}_imputado.csv",
                    allowed_extensions=["csv"]
                )
            except Exception as e:
                page.snack_bar = ft.SnackBar(ft.Text(f"Error al leer caché para exportar: {e}"), bgcolor="red", open=True)
                page.update()

        def _trigger_download_log(sid, log_text):
            log_a_exportar[0] = log_text
            picker_exportar_log.save_file(
                dialog_title=f"Descargar Informe QA de {sid}",
                file_name=f"QA_Log_{sid}.txt",
                allowed_extensions=["txt"]
            )

        for sid, data in db.items():
            df_info = data["df"]
            log_info = data.get("log", "")
            visor_resultados_grid.controls.append(
                ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.CHECK_CIRCLE, color="green", size=16), 
                        ft.Text(sid, size=12, weight="bold"),
                        # Botón Inspeccionar
                        ft.IconButton(ft.Icons.REMOVE_RED_EYE, icon_color="white", icon_size=16, tooltip="Inspeccionar Datos", on_click=lambda e, s=sid: abrir_inspeccion(s)),
                        # Botón Exportar CSV
                        ft.IconButton(ft.Icons.DOWNLOAD, icon_color="#00f0ff", icon_size=16, tooltip="Exportar a CSV", on_click=lambda e, s=sid, d=df_info: _trigger_download(s, d)),
                        # Botón Exportar Informe QA (LOG)
                        ft.IconButton(ft.Icons.RECEIPT_LONG, icon_color="orange", icon_size=16, tooltip="Descargar Informe QA", on_click=lambda e, s=sid, l=log_info: _trigger_download_log(s, l))
                    ]),
                    bgcolor="#1e2a1e", padding=5, border_radius=8
                )
            )
        page.update()

    def abrir_inspeccion(sid):
        data = page.session.get("db_series_crudas").get(sid)
        if not data: return
        df_data = data["df"]
        if isinstance(df_data, dict) and df_data.get("type") == "df":
            df = pd.read_parquet(df_data["path"])
        else:
            df = df_data
        dialog = ft.AlertDialog(
            title=ft.Text(f"Inspección: {sid}", color=COLOR_ACENTO),
            content=ft.Column([
                ft.Text("Muestra de datos (Últimos 5):", weight="bold"),
                ft.Container(content=ft.Text(df.tail(5).to_string(), size=9, font_family="monospace"), bgcolor="black", padding=10),
                ft.Divider(),
                ft.Text("Log de Calidad:", weight="bold"),
                # FIX: Se usa Column dentro de Container para habilitar el scroll
                ft.Container(content=ft.Column([ft.Text(data["log"][:800] + "...", size=11, color="grey")], scroll=ft.ScrollMode.AUTO), height=150)
            ], tight=True, width=500),
            # --- ADAPTACIÓN FLET 0.28+: Cierre seguro mediante método atómico de página ---
            actions=[ft.TextButton("Cerrar", on_click=lambda _: page.close(dialog))]
        )
        # --- ADAPTACIÓN FLET 0.28+: Evitamos la asignación obsoleta 'page.dialog = x' que causaba el fallo ---
        page.open(dialog)

    # ==========================================
    # 5. MOTOR DE IMPUTACIÓN Y PROTECCIONES
    # ==========================================
    def on_click_impute(e):
        cola = page.session.get("cola_imputacion")
        if not cola:
            page.snack_bar = ft.SnackBar(ft.Text("❌ Cola vacía"), bgcolor="red", open=True)
            return

        def task():
            state["imputing"] = True
            
            # --- 1. ACTIVACIÓN DEL ESCUDO RADIACTIVO Y BLOQUEO GLOBAL ---
            def _start_visuals():
                bloqueo_radiactivo.visible = True
                terminal_radiactiva.controls.clear()
                page.window.prevent_close = True # Guillotina de SO bloqueada
                
                main_rail = page.session.get("main_rail")
                if main_rail: main_rail.disabled = True
                
                tabs.selected_index = 2 # Salto automático a pestaña "3. Proceso"
                page.update()
            page.run_thread(_start_visuals)
            
            db_crudas = page.session.get("db_series_crudas") or {}
            out_folder = page.session.get("imput_output_folder") or os.getcwd()
            radius_km = int(dd_radio.value)

            for idx, sid in enumerate(cola):
                if not state["imputing"]: break
                
                # Reiniciamos la UI para la nueva estación
                pb_interceptado.value = 0 
                log_radiactivo(f"--- INICIANDO ESTACIÓN {sid} ({idx+1}/{len(cola)}) ---")

                try:
                    # INYECCIÓN DEL INTERCEPTOR: Pasamos pb_interceptado en vez de pb
                    df_res, log_msg = imputacion_logic.impute_target_station(
                        sid, page.session.get("imput_station_files"), page, pb_interceptado, pbl, radius_km, log_callback=log_radiactivo
                    )
                    
                    if df_res is not None:
                        # Guardado Seguro en Parquet
                        cache_dir = page.session.get("project_cache_dir")
                        if not cache_dir:
                            cache_dir = tempfile.mkdtemp(prefix="hidro_cache_")
                            page.session.set("project_cache_dir", cache_dir)
                        
                        df_safe = df_res.copy(deep=True)
                        df_safe.columns = df_safe.columns.astype(str)
                        df_path = os.path.join(cache_dir, f"imput_{sid}.parquet")
                        df_safe.to_parquet(df_path)
                        
                        db_crudas[sid] = {"df": {"type": "df", "path": df_path}, "log": log_msg}
                        imputacion_logic.save_target_csv(df_res, sid, out_folder)
                        log_radiactivo(f"✅ Estación {sid} procesada y guardada con éxito.")
                except Exception as ex:
                    log_radiactivo(f"❌ Error crítico en {sid}: {ex}")
            
            # --- 2. FINALIZACIÓN Y APAGADO DEL ESCUDO ---
            def _finalize_batch():
                page.session.set("db_series_crudas", db_crudas)
                page.session.set("cola_imputacion", [])
                
                out_folder = page.session.get("imput_output_folder")
                if out_folder and os.path.exists(out_folder):
                    page.session.set("txt_backup_imputados", {"__type__": "folder_backup", "path": out_folder})
                
                render_cola_visual()
                render_visor_resultados()
                
                # Apagar Escudo Radiactivo
                bloqueo_radiactivo.visible = False
                page.window.prevent_close = False
                main_rail = page.session.get("main_rail")
                if main_rail: main_rail.disabled = False
                
                state["imputing"] = False
                page.update()
            page.run_thread(_finalize_batch)

        threading.Thread(target=task, daemon=True).start()

    def abort_imputation(e):
        state["imputing"] = False
        terminal_list.controls.append(ft.Text("🛑 Deteniendo proceso...", color="red", weight="bold"))
        page.update()

    def interceptar_cambio_tab(e):
        if state["imputing"]:
            tabs.selected_index = 2 # Fuerza a quedarse en la pestaña de Proceso
            page.snack_bar = ft.SnackBar(ft.Text("⚠️ Imputación en curso. Detenga el proceso para navegar."), bgcolor="orange", open=True)
            page.update()

    # ==========================================
    # 6. COMPONENTES Y DISEÑO
    # ==========================================
    dd_radio = ft.Dropdown(label="Radio (Km)", width=150, options=[
        ft.dropdown.Option("50"), ft.dropdown.Option("150"), 
        ft.dropdown.Option("250"), ft.dropdown.Option("300")
    ], value="150")
    
    indicador_estacion = ft.Text("Ninguna estación en cola.", color="grey", size=12, italic=True)
    
    open_map_button = ft.ElevatedButton("Ver Mapa", icon=ft.Icons.MAP, disabled=True, 
                                       on_click=lambda _: webbrowser.open(page.session.get("imput_map_path").get("path") if isinstance(page.session.get("imput_map_path"), dict) else page.session.get("imput_map_path")))
    
    # --- BUSCADOR Y CONTENEDOR DE TARJETAS ---
    search_station_input = ft.TextField(
        label="Buscar estación por ID o nombre...",
        prefix_icon=ft.Icons.SEARCH,
        visible=False, # Se oculta hasta que haya datos
        on_change=lambda e: filtrar_estaciones(e.control.value),
        border_color=COLOR_ACENTO
    )
    # expand=True es vital para que Flet sepa que este contenedor debe estirarse y permitir scroll
    station_files_container = ft.Column(spacing=10, scroll=ft.ScrollMode.AUTO, expand=True)

    impute_button = ft.ElevatedButton("Iniciar Lote", icon=ft.Icons.PLAY_ARROW, on_click=on_click_impute, bgcolor=COLOR_ACENTO, color="black")
    detener_button = ft.ElevatedButton("Detener", icon=ft.Icons.STOP, visible=False, on_click=abort_imputation, bgcolor="red", color="white")

    # ==========================================
    # 7. ESTRUCTURA DE PESTAÑAS (TABS)
    # ==========================================
    view_fuente = ft.Container(content=ft.Column([
        ft.Text("1. Carpeta de Datos", size=18, weight="bold", color=COLOR_ACENTO),
        ft.Row([ft.ElevatedButton("Seleccionar Fuente", icon=ft.Icons.FOLDER, on_click=lambda _: file_picker_input.get_directory_path()), open_map_button]),
        selected_input_folder, station_count_text,
        ft.Divider(),
        search_station_input,
        station_files_container
    ]), padding=20,
        expand=True)

    view_objetivo = ft.Container(content=ft.Column([
        ft.Text("2. Configuración y Cola", size=18, weight="bold", color=COLOR_ACENTO),
        ft.Row([dd_radio, ft.ElevatedButton("Carpeta Salida", icon=ft.Icons.SAVE, on_click=lambda _: file_picker_output.get_directory_path())]),
        selected_output_folder,
        ft.Divider(),
        ft.Text("Cola de Trabajo:"),
        cola_espera_ui,
        ft.Row([impute_button], alignment=ft.MainAxisAlignment.CENTER)
    ], scroll=ft.ScrollMode.AUTO), padding=20, expand=True)

    view_exec = ft.Container(content=ft.Column([
        ft.Text("3. Ejecución en Memoria", size=18, weight="bold", color=COLOR_ACENTO),
        ft.Row([detener_button], alignment=ft.MainAxisAlignment.END), # <--- FRENO DE EMERGENCIA AQUÍ
        ft.Container(content=terminal_list, bgcolor="black", expand=True, padding=10),
        ft.Text("Base de Datos en Sesión (Estaciones Listas):"),
        visor_resultados_grid
    ], scroll=ft.ScrollMode.AUTO), padding=20, expand=True)

    tabs = ft.Tabs(
        selected_index=0,
        on_change=interceptar_cambio_tab,
        tabs=[
            ft.Tab(text="1. Fuente", icon=ft.Icons.FOLDER, content=view_fuente),
            ft.Tab(text="2. Cola", icon=ft.Icons.LIST, content=view_objetivo),
            ft.Tab(text="3. Proceso", icon=ft.Icons.MEMORY, content=view_exec),
        ], expand=True
    )

    # REHIDRATACIÓN
    def filtrar_estaciones(query):
        estaciones_totales = page.session.get("imput_station_files") or {}
        if not query:
            # Si el buscador está vacío, mostramos todas
            restore_station_ui(estaciones_totales, actualizar_textos=False)
            return
        
        # Filtro en tiempo real ignorando mayúsculas/minúsculas
        filtradas = {k: v for k, v in estaciones_totales.items() if query.lower() in str(k).lower()}
        restore_station_ui(filtradas, actualizar_textos=False)

    def restore_station_ui(estaciones, actualizar_textos=True):
        station_files_container.controls.clear()
        cola_activa = page.session.get("cola_imputacion") or []
        
        # --- ALGORITMO TOP-FLOAT: Clasificación por estado de selección ---
        seleccionadas = []
        no_seleccionadas = []
        
        for station_id, info in estaciones.items():
            if station_id in cola_activa:
                seleccionadas.append((station_id, info))
            else:
                no_seleccionadas.append((station_id, info))
                
        # Consolidamos colocando el enjambre seleccionado al inicio de la lista
        lista_ordenada = seleccionadas + no_seleccionadas
        
        for station_id, info in lista_ordenada:
            en_cola = station_id in cola_activa
            
            # Encapsulamos en Container para evadir las limitaciones estéticas del ft.Card nativo
            card_content = ft.Container(
                content=ft.Column([
                    ft.ListTile(
                        leading=ft.Icon(
                            ft.Icons.CHECK_CIRCLE if en_cola else ft.Icons.GPS_FIXED, 
                            color="#00f0ff" if en_cola else COLOR_ACENTO
                        ),
                        title=ft.Text(station_id, weight="bold", color="#00f0ff" if en_cola else "white"),
                        subtitle=ft.Text(
                            f"Lat: {info['lat']}, Lon: {info['lon']}", 
                            color=add_opacity("#00f0ff", 0.7) if en_cola else "grey"
                        ),
                    ),
                    ft.Row([
                        ft.TextButton(
                            text="✔ En Cola" if en_cola else "➕ Añadir a Cola",
                            disabled=en_cola, # Bloqueo estricto anti-duplicados
                            style=ft.ButtonStyle(color="#00f0ff" if en_cola else COLOR_ACENTO),
                            on_click=lambda e, sid=station_id: agregar_a_cola(sid)
                        ),
                    ], alignment=ft.MainAxisAlignment.END),
                ]),
                padding=10,
                # Identidad visual radical: borde neón y fondo translúcido si está activa
                border=ft.border.all(1, "#00f0ff" if en_cola else "transparent"),
                bgcolor=add_opacity("#00f0ff", 0.05) if en_cola else COLOR_SUPERFICIE,
                border_radius=8
            )
            
            station_files_container.controls.append(ft.Card(content=card_content))
            
        if actualizar_textos:
            station_count_text.value = f"Estaciones encontradas: {len(estaciones)}"
            station_count_text.visible = True
            search_station_input.visible = True # Hacemos visible el buscador
            
        page.update()

    # --- MOTOR DE HIDRATACIÓN AUTOMÁTICA (VFS) ---
    async def trigger_hydration():
        await asyncio.sleep(0.1) # Breve pausa para asegurar que Flet haya renderizado el DOM
        
        ruta_en_memoria = page.session.get("imput_folder_path")
        estaciones_actuales = page.session.get("imput_station_files")
        
        # Escenario 1: Hay una ruta validada en la memoria, pero aún no se leen los archivos
        # Esto ocurre cuando vienes del Módulo 1 o recién abres un proyecto .hds
        if ruta_en_memoria and os.path.exists(ruta_en_memoria):
            if not estaciones_actuales:
                auto_procesar_fuente(ruta_en_memoria)
            else:
                # Escenario 2: Ya estaban leídos (Ej. si regresaste de otra pestaña)
                selected_input_folder.value = f"Fuente Activa: {ruta_en_memoria}"
                open_map_button.disabled = False if page.session.get("imput_map_path") else True
                restore_station_ui(estaciones_actuales)
                
    # Lanzamos el gatillo asíncrono
    page.run_task(trigger_hydration)
    
    render_visor_resultados()

    return tabs