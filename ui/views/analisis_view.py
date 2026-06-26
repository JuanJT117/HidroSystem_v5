import flet as ft
import pandas as pd
import os 
import traceback
import asyncio # <--- 1. CORRECCIÓN: IMPORTACIÓN ASÍNCRONA AÑADIDA

# --- ARQUITECTURA HEXAGONAL: Importación desde el Core ---
from core import Analisis
from core import lluvias_logic as lluvias  
from core import analisis_cuenca 

# Píxel transparente
TRANSPARENT_PIXEL = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

def build_analisis_view(page: ft.Page):
    
    # --- 1. INICIALIZACIÓN DE SESIÓN (Protegida) ---
    keys_temp = [
        "b64_hist", "b64_series", "b64_violin", "b64_max_annual", "b64_acf", 
        "b64_weibull", "b64_dist_comp", "cuenca_plot1_b64", "cuenca_plot2_b64", 
        "cuenca_plot3_b64", "cuenca_plot4_b64"
    ]
    
    # Inyección de 'df_parametros' incluida para la persistencia de la tabla QA
    keys_data = [
        "df_procesado", "df_filtrado", "df_estadisticas", "df_maximos_mensuales", 
        "df_weibull", "best_fit_name", "df_homogeneidad", "df_acf", "df_ajustes", 
        "df_altura", "df_intensidad", "df_parametros"
    ]
    
    # =================================================================
    # SISTEMA DE BLOQUEO UI Y PERSISTENCIA UNIVERSAL (MULTI-ESTACIÓN)
    # =================================================================
    loading_ring = ft.ProgressRing(color="#00f0ff", stroke_width=5)
    loading_text = ft.Text("Calculando...", color="#00f0ff", weight="bold", font_family="Roboto Mono")
    bloqueo_ui = ft.Container(
        content=ft.Column([loading_ring, loading_text], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        bgcolor="#d9050505", # Hex con canal Alpha (Transparencia 85%)
        expand=True, alignment=ft.alignment.center, visible=False
    )
    page.overlay.append(bloqueo_ui)

    def persistir_estacion_actual():
        """Guarda dinámicamente cualquier avance de la estación actual en el caché y RAM"""
        sid = page.session.get("target_station_id")
        if not sid: return
        db_curvas = page.session.get("db_curvas_procesadas") or {}
        cache_dir = page.session.get("project_cache_dir")
        if not cache_dir:
            import tempfile
            cache_dir = tempfile.mkdtemp(prefix="hidro_cache_")
            page.session.set("project_cache_dir", cache_dir)
        
        station_cache = os.path.join(cache_dir, sid)
        os.makedirs(station_cache, exist_ok=True)
        
        refs = {}
        for k in keys_data + keys_temp:
            val = page.session.get(k)
            if val is None: continue
            try:
                if isinstance(val, pd.DataFrame) and not val.empty:
                    df_safe = val.copy(deep=True)
                    df_safe.columns = df_safe.columns.astype(str)
                    if df_safe.index.dtype == object: df_safe.index = df_safe.index.astype(str)
                    for col in df_safe.columns:
                        if df_safe[col].dtype == object:
                            try: df_safe[col] = pd.to_numeric(df_safe[col])
                            except: df_safe[col] = df_safe[col].astype(str)
                    path = os.path.join(station_cache, f"{k}.parquet")
                    df_safe.to_parquet(path, engine="pyarrow")
                    refs[k] = {"type": "df", "path": path}
                elif isinstance(val, str) and len(val) > 100:
                    path = os.path.join(station_cache, f"{k}.txt")
                    with open(path, "w") as f: f.write(val)
                    refs[k] = {"type": "b64", "path": path}
                else: refs[k] = {"type": "raw", "value": val}
            except Exception:
                refs[k] = {"type": "raw", "value": val}
        db_curvas[sid] = refs
        page.session.set("db_curvas_procesadas", db_curvas)
    # =================================================================
    
    # --- [FASE 2] INICIALIZACIÓN Y SELECTOR MAESTRO ---
    if not page.session.get("db_curvas_procesadas"): 
        page.session.set("db_curvas_procesadas", {})

    dd_estacion_maestra = ft.Dropdown(
        label="Estación de Trabajo Activa",
        hint_text="Seleccione una estación...",
        width=350,
        border_color="#00ff41",
        on_change=lambda e: sincronizar_estacion_activa(e.control.value)
    )

    def actualizar_dropdown_maestro():
        db_crudas = page.session.get("db_series_crudas") or {}
        dd_estacion_maestra.options = [ft.dropdown.Option(k) for k in db_crudas.keys()]
        page.update()

    # --- 2. CORRECCIÓN: FUNCIÓN DE LIMPIEZA AÑADIDA ---
    def limpiar_tablas_ui():
        tbl_mx.controls = [ft.Text("Esperando cálculo...", italic=True)]
        tbl_alt.controls = [ft.Text("Esperando cálculo...", italic=True)]
        data_table_filt.controls = [ft.Text("Esperando aplicación de filtros...", italic=True)]
        data_table_stats.controls = [ft.Text("Esperando cálculo...", italic=True)]
        tbl_homo.controls = [ft.Text("Esperando cálculo...", italic=True)]
        tbl_wei.controls = [ft.Text("Esperando cálculo...", italic=True)]
        tbl_ajustes.controls = [ft.Text("Esperando cálculo...", italic=True)]
        tbl_int.controls = [ft.Text("Esperando cálculo...", italic=True)]
        tbl_parametros.controls = [ft.Text("Esperando cálculo...", italic=True)]
        log_cuenca.value = ""

    def sincronizar_estacion_activa(sid):
        db_curvas = page.session.get("db_curvas_procesadas") or {}
        if sid in db_curvas:
            # Rehidratación instantánea (con soporte a caché en disco o modo Legacy)
            data = db_curvas[sid]
            for k in keys_data + keys_temp: 
                val = data.get(k)
                if isinstance(val, dict) and "type" in val:
                    # Es un puntero al disco
                    if val["type"] == "df" and os.path.exists(val["path"]):
                        try: page.session.set(k, pd.read_parquet(val["path"]))
                        except: page.session.set(k, None)
                    elif val["type"] == "b64" and os.path.exists(val["path"]):
                        try:
                            with open(val["path"], "r") as f: page.session.set(k, f.read())
                        except: page.session.set(k, None)
                    else:
                        page.session.set(k, val.get("value"))
                else:
                    # Legacy: el dato estaba en crudo en la sesión
                    page.session.set(k, val)
            rehidratar_interfaz() 
            page.snack_bar = ft.SnackBar(ft.Text(f"✅ {sid}: Análisis recuperado desde disco/memoria."), bgcolor="green", open=True)
        else:
            # --- CARGA DE DATOS CRUDOS DESDE SQLITE ---
            ruta_bd = page.session.get("ruta_bd_activa")
            df_procesado = None
            if ruta_bd:
                if ruta_bd.endswith('.tar.xz') or ruta_bd.endswith('.xz'):
                    ruta_hds = ruta_bd.replace('.tar.xz', '.gpkg').replace('.xz', '.gpkg')
                elif not ruta_bd.endswith('.gpkg') and not ruta_bd.endswith('.sqlite'):
                    ruta_hds = ruta_bd + "_ARF.gpkg"
                else:
                    ruta_hds = ruta_bd
                    
                import sqlite3
                conn = sqlite3.connect(ruta_hds)
                try:
                    query = f"SELECT * FROM serie_imputada_{sid}"
                    df_estacion = pd.read_sql_query(query, conn, parse_dates=['FECHA'])
                    if not df_estacion.empty:
                        df_estacion.set_index('FECHA', inplace=True)
                        mapa_nombres = {}
                        if 'PRECIP' in df_estacion.columns: mapa_nombres['PRECIP'] = 'PRECIP_imputado'
                        if 'PRECIP_ORIGINAL' in df_estacion.columns: mapa_nombres['PRECIP_ORIGINAL'] = 'PRECIP_original'
                        df_estacion.rename(columns=mapa_nombres, inplace=True)
                        if 'PRECIP_original' not in df_estacion.columns and 'PRECIP_imputado' in df_estacion.columns:
                            df_estacion['PRECIP_original'] = df_estacion['PRECIP_imputado']
                        df_estacion.drop(columns=[c for c in ['lat', 'lon', 'station_id'] if c in df_estacion.columns], inplace=True)
                        df_procesado = df_estacion
                except Exception as e:
                    print(f"Error cargando estación {sid} desde SQLite: {e}")
                finally:
                    conn.close()
            
            for k in keys_data + keys_temp: page.session.set(k, None)

            page.session.set("df_procesado", df_procesado)
            page.session.set("target_station_id", sid)
            csv_path_txt.value = f"Datos Crudos: {sid}"
            data_table_raw.controls = [ft.Text(f"Previsualización {sid}:"), dataframe_to_datatable(df_procesado)]
            
            # Reset visual
            for img in [img_h, img_s, img_v, img_max_an, img_acf, img_wei, img_dist, img_c1, img_c2, img_c3, img_c4]:
                img.content.src_base64 = TRANSPARENT_PIXEL
            
            limpiar_tablas_ui()
            btn_nav_filt.disabled = False 
            
            # --- MEJORA UX CRÍTICA: Redirección Atómica de Flujo ---
            # Al iniciar el análisis de una nueva estación, obligamos a la UI a regresar
            # a la Pestaña 1 (Carga) para evitar que el usuario se quede viendo la Pestaña 6 vacía.
            tabs.selected_index = 0
            
            page.snack_bar = ft.SnackBar(ft.Text(f"📥 {sid}: Datos formateados y listos para analizar."), bgcolor="blue", open=True)
            page.update()
    
    for k in keys_temp + keys_data:
        if page.session.get(k) is None: 
            page.session.set(k, None)

    # --- 2. CONTROLES DE INTERFAZ Y HELPERS ---
    csv_path_txt = ft.Column(expand=True, controls=[ft.Text("Esperando filtros...")])
    data_table_raw = ft.Column(expand=True, controls=[ft.Text("Esperando filtros...")])
    data_table_stats = ft.Column(expand=True, controls=[ft.Text("Calculando...")])
    
    def safe_img(session_key):
        val = page.session.get(session_key)
        src = val if (val and len(val) > 100) else TRANSPARENT_PIXEL
        return ft.Container(content=ft.Image(src_base64=src, fit=ft.ImageFit.CONTAIN), expand=True)

    #--------- estilo de tablas

    def dataframe_to_datatable(df, max_filas=90, auto_expand=False): # Subimos a 50 filas de muestra
        if df is None or (hasattr(df, 'empty') and df.empty): return ft.Text("Sin datos")
        try:
            df_view = df.head(max_filas).copy()
            if df_view.index.name is not None or not isinstance(df_view.index, pd.RangeIndex):
                df_view.reset_index(inplace=True)
            df_view.columns = df_view.columns.astype(str)
            cols = [ft.DataColumn(ft.Text(col[:15], weight="bold", color="#00ff41")) for col in df_view.columns]
            rows = [ft.DataRow([ft.DataCell(ft.Text(str(val)[:20], size=11)) for val in row]) for _, row in df_view.iterrows()]
            
            tabla = ft.DataTable(columns=cols, rows=rows, column_spacing=15, border=ft.border.all(1, "#333333"), vertical_lines=ft.border.BorderSide(1, "#333333"))
            
            # --- SCROLL 2D ---
            inner_col = ft.Column(controls=[ft.Row([tabla], scroll=ft.ScrollMode.AUTO)], scroll=ft.ScrollMode.AUTO)
            
            # --- AUTOAJUSTE INTELIGENTE ---
            if auto_expand:
                inner_col.expand = True
                return ft.Container(content=inner_col, expand=True, border=ft.border.all(1, "#222222"), border_radius=5)
            else:
                return ft.Container(content=inner_col, height=370, border=ft.border.all(1, "#222222"), border_radius=5)
        except: return ft.Text("Error tabla", color="red")

    def stats_to_datatable(df, auto_expand=False):
        if df is None: return ft.Text("Sin estadísticas")
        try: return dataframe_to_datatable(df.reset_index().rename(columns={'index': 'Métrica'}), auto_expand=auto_expand)
        except: return ft.Text("Error stats")

    # --- 3. GESTOR DE EXPORTACIÓN ---
    csv_picker = ft.FilePicker()
    export_picker = ft.FilePicker()
    page.overlay.extend([csv_picker, export_picker])

    def on_export_result(e: ft.FilePickerResultEvent):
        if e.path:
            try:
                content = page.session.get("temp_export_content")
                ext = page.session.get("temp_export_ext")
                if content is None: raise ValueError("No hay datos en memoria para guardar.")

                if ext == "png":
                    import base64
                    path = f"{e.path}.png" if not e.path.endswith(".png") else e.path
                    with open(path, "wb") as f: f.write(base64.b64decode(content))
                elif ext == "csv":
                    path = f"{e.path}.csv" if not e.path.endswith(".csv") else e.path
                    if hasattr(content, 'columns') and 'TR (AÑOS)' in content.columns: 
                        content.set_index('TR (AÑOS)').to_csv(path)
                    else: 
                        content.to_csv(path, index=False, encoding='utf-8-sig')
                
                page.snack_bar = ft.SnackBar(ft.Text(f"Archivo guardado exitosamente"), bgcolor="green", open=True)
            except Exception as ex:
                page.snack_bar = ft.SnackBar(ft.Text(f"Error al guardar: {str(ex)}"), bgcolor="red", open=True)
            page.update()

    export_picker.on_result = on_export_result

    def trigger_save_image(session_key, default_name):
        b64_string = page.session.get(session_key)
        if not b64_string or b64_string == TRANSPARENT_PIXEL:
            page.snack_bar = ft.SnackBar(ft.Text("No hay gráfico generado para guardar."), bgcolor="orange", open=True)
            page.update(); return
        page.session.set("temp_export_content", b64_string)
        page.session.set("temp_export_ext", "png")
        export_picker.save_file(dialog_title="Guardar Gráfico", file_name=default_name, allowed_extensions=["png"])

    def trigger_save_csv(session_key, default_name):
        df_to_save = page.session.get(session_key)
        if df_to_save is None or (hasattr(df_to_save, 'empty') and df_to_save.empty):
            page.snack_bar = ft.SnackBar(ft.Text("No hay tabla generada para guardar."), bgcolor="orange", open=True)
            page.update(); return
        page.session.set("temp_export_content", df_to_save)
        page.session.set("temp_export_ext", "csv")
        export_picker.save_file(dialog_title="Guardar Tabla", file_name=default_name, allowed_extensions=["csv"])

    # --- 4. ELEMENTOS VISUALES BASE ---
    data_table_filt, stats_table_container = ft.Column(scroll=ft.ScrollMode.AUTO, height=300), ft.Column()
    chk_c1, chk_c2, chk_c3 = ft.Checkbox(label="Eliminar Nulos", value=True), ft.Checkbox(label="Eliminar < 5 Vecinos", value=True), ft.Checkbox(label="Eliminar Ruido", value=True)
    img_h, img_s, img_v = safe_img("b64_hist"), safe_img("b64_series"), safe_img("b64_violin")
    img_max_an, img_acf, img_wei, img_dist = safe_img("b64_max_annual"), safe_img("b64_acf"), safe_img("b64_weibull"), safe_img("b64_dist_comp")
    img_c1, img_c2, img_c3, img_c4 = safe_img("cuenca_plot1_b64"), safe_img("cuenca_plot2_b64"), safe_img("cuenca_plot3_b64"), safe_img("cuenca_plot4_b64")
    tbl_homo, tbl_acf, tbl_ajustes, tbl_wei, tbl_mx, tbl_alt, tbl_int, tbl_parametros = ft.Column(), ft.Column(), ft.Column(), ft.Column(), ft.Column(), ft.Column(), ft.Column(), ft.Column()
    txt_best_fit, inp_ylim_h, inp_ylim_i, log_cuenca = ft.Text("No detectado", weight="bold", color="#00ff41"), ft.TextField(label="Límite Altura", width=150), ft.TextField(label="Límite Intensidad", width=150), ft.Markdown()

    # --- 5. LÓGICA DE EVENTOS (Adaptada al Core) ---
    def advance_tab(index):
        tabs.selected_index = index
        page.update()

    def on_load(e):
        if e.files:
            csv_path_txt.value = e.files[0].name; csv_path_txt.color="green"
            try:
                df = Analisis.procesar_datos(e.files[0].path)
                if df is not None:
                    page.session.set("df_procesado", df)
                    data_table_raw.controls = [ft.Text("Datos:"), dataframe_to_datatable(df)]
                    btn_nav_filt.disabled = False
            except: pass
            page.update()

    def run_filter(e=None):
        try:
            df_procesado = page.session.get("df_procesado")
            if df_procesado is None: return
            
            # Ejecución de la lógica de tu Core
            df = page.session.get("df_filtrado") if (e is None and page.session.get("df_filtrado") is not None) else Analisis.filtrar_datos(df_procesado, chk_c1.value, chk_c2.value, chk_c3.value)
            
            if df is not None:
                page.session.set("df_filtrado", df)
                data_table_filt.controls = [ft.Text(f"Datos Filtrados: {len(df)} registros"), dataframe_to_datatable(df)]
                btn_nav_stats.disabled = False
                if e: advance_tab(1)
            
            if e: page.update()
            
        except Exception as ex:
            # Si hay un error, lo mostramos en pantalla inmediatamente
            page.snack_bar = ft.SnackBar(ft.Text(f"❌ Error en Filtrado: {str(ex)}"), bgcolor="red", open=True)
            page.update()

    def run_stats(e=None):
        df_filtrado = page.session.get("df_filtrado")
        if df_filtrado is None: return
        st = page.session.get("df_estadisticas") if (e is None and page.session.get("df_estadisticas") is not None) else Analisis.analizar_estadisticas(df_filtrado)
        if st is not None:
            page.session.set("df_estadisticas", st)
            stats_table_container.controls = [ft.Text("Estadísticas:", color="#00ff41"), stats_to_datatable(st)]
            btn_nav_graph.disabled = False
            if e: advance_tab(2)
        if e: page.update()

    def run_graphs(e=None):
        df_filtrado = page.session.get("df_filtrado")
        if df_filtrado is None: return
        sid = page.session.get("target_station_id") or ""
        imgs = Analisis.generar_graficos(df_filtrado, page.session.get("df_estadisticas"), sid)
        if imgs:
            img_h.content.src_base64, img_s.content.src_base64, img_v.content.src_base64 = imgs['hist'], imgs['series'], imgs['violin']
            page.session.set("b64_hist", imgs['hist'])
            page.session.set("b64_series", imgs['series'])
            page.session.set("b64_violin", imgs['violin'])
            btn_nav_lluvias.disabled = False
            if e: advance_tab(3)
        if e: page.update()

    def run_lluvias(e=None):
        df_filtrado = page.session.get("df_filtrado")
        if df_filtrado is None: return
        sid = page.session.get("target_station_id") or ""
        res = lluvias.analizar_eventos_lluvia(df_filtrado, sid)
        if res:
            page.session.set("df_homogeneidad", res["df_homogeneidad"])
            page.session.set("df_maximos_mensuales", res["df_maximos_mensuales"])
            page.session.set("best_fit_name", res["best_fit_name"])
            page.session.set("df_acf", res["df_acf"])
            page.session.set("df_ajustes", res["df_ajustes"])
            page.session.set("df_weibull", res["df_weibull"])
            
            page.session.set("b64_max_annual", res["max_annual_series_b64"])
            page.session.set("b64_acf", res["acf_plot_b64"])
            page.session.set("b64_weibull", res["weibull_plot_b64"])
            page.session.set("b64_dist_comp", res["dist_comparison_b64"])
            
            tbl_homo.controls = [dataframe_to_datatable(res["df_homogeneidad"])]
            tbl_acf.controls = [dataframe_to_datatable(res["df_acf"])]
            tbl_ajustes.controls = [dataframe_to_datatable(res["df_ajustes"])]
            tbl_wei.controls = [dataframe_to_datatable(res["df_weibull"])]
            tbl_mx.controls = [dataframe_to_datatable(res["df_maximos_mensuales"])]
            
            img_max_an.content.src_base64 = res["max_annual_series_b64"]
            img_acf.content.src_base64 = res["acf_plot_b64"]
            img_wei.content.src_base64 = res["weibull_plot_b64"]
            img_dist.content.src_base64 = res["dist_comparison_b64"]
            
            txt_best_fit.value = f"Mejor Ajuste: {res['best_fit_name']}"
            btn_nav_cuenca.disabled = False
            if e: advance_tab(4)
        if e: page.update()
    
    def run_cuenca(e=None):
        def _task():
            sid = page.session.get("target_station_id")
            if not sid: return
            
            # 1. Bloqueo visual estricto (Evita que el usuario rompa el flujo)
            bloqueo_ui.visible = True
            loading_text.value = f"Calculando A-D-TR / I-D-TR ({sid})..."
            page.update()
            
            try:
                df_m = page.session.get("df_maximos_mensuales")
                
                if df_m is None or df_m.empty: 
                    raise ValueError("Faltan Máximos Mensuales. Ejecuta la pestaña 5 (Probabilidad) primero para generarlos.")
                
                # 1. Recuperamos el Mejor Ajuste desde la memoria (calculado en pestaña 5)
                best_fit = page.session.get("best_fit_name")
                if not best_fit:
                    raise ValueError("No se detectó un modelo de Mejor Ajuste. Ejecuta la pestaña 5 primero.")
                
                # 2. Capturamos los límites (Y-Lim) que el usuario puede escribir en los TextFields
                ylim_h = float(inp_ylim_h.value) if inp_ylim_h.value else None
                ylim_i = float(inp_ylim_i.value) if inp_ylim_i.value else None
                
                # --- CONEXIÓN AL NÚCLEO MATEMÁTICO REAL ---
                # Invocamos la función exacta que vive en analisis_cuenca.py
                res = analisis_cuenca.run_cuenca_analysis(
                    best_fit_name=best_fit,
                    df_maximos_mensuales=df_m,
                    ylim_altura=ylim_h,
                    ylim_intensidad=ylim_i,
                    station_id=sid
                )
                
                if res:
                    page.session.set("log_cuenca", res["log_text"])
                    page.session.set("df_altura", res["df_altura"])
                    page.session.set("df_intensidad", res["df_intensidad"])
                    page.session.set("df_parametros", res.get("df_parametros"))
                    
                    page.session.set("cuenca_plot1_b64", res["plot_1_b64"])
                    page.session.set("cuenca_plot2_b64", res["plot_2_b64"])
                    page.session.set("cuenca_plot3_b64", res["plot_3_b64"])
                    page.session.set("cuenca_plot4_b64", res["plot_4_b64"])
                    
                    # Llamada a nuestro nuevo motor persistente universal
                    persistir_estacion_actual()
                
                # --- RENDERIZADO UI ---
                
                # --- CORRECCIÓN: RESTAURACIÓN DEL LOG DETALLADO ---
                # Usamos la variable real de tu interfaz (log_cuenca)
                log_txt = page.session.get("log_cuenca")
                if log_txt:
                    # Lo envolvemos en sintaxis Markdown para que se vea estructurado y limpio
                    log_cuenca.value = f"```text\n{log_txt}\n```"
                else:
                    log_cuenca.value = "Análisis completado."
                           
                p1 = page.session.get("cuenca_plot1_b64")
                p2 = page.session.get("cuenca_plot2_b64")
                p3 = page.session.get("cuenca_plot3_b64")
                p4 = page.session.get("cuenca_plot4_b64")
                
                # --- CORRECCIÓN DE VARIABLES GRÁFICAS ---
                if p1 and p2 and p3 and p4:
                    img_c1.content.src_base64 = p1
                    img_c2.content.src_base64 = p2
                    img_c3.content.src_base64 = p3
                    img_c4.content.src_base64 = p4
                
                # --- CORRECCIÓN CRÍTICA DE RENDERIZADO (BUG FLUTTER DATATABLE) ---
                # Aplicamos .reset_index() OBLIGATORIAMENTE para convertir el índice en columna
                # y evitar que Flutter colapse al dibujar la tabla.
                # --- CORRECCIÓN CRÍTICA DE RENDERIZADO (BUG FLUTTER DATATABLE) ---
                # Aplicamos .reset_index() OBLIGATORIAMENTE para convertir el índice en columna
                
                df_alt = page.session.get("df_altura")
                if df_alt is not None: tbl_alt.controls = [dataframe_to_datatable(df_alt.reset_index(), auto_expand=True)]
                
                df_int = page.session.get("df_intensidad")
                if df_int is not None: tbl_int.controls = [dataframe_to_datatable(df_int.reset_index(), auto_expand=True)]
                
                df_param = page.session.get("df_parametros")
                if df_param is not None: tbl_parametros.controls = [dataframe_to_datatable(df_param.reset_index(), auto_expand=True)]
                
            except Exception as ex:
                import traceback
                print(traceback.format_exc()) # El error técnico se queda en tu consola
                
                # --- ALERTA VISUAL SEGURA ---
                # Le mostramos al usuario exactamente qué le faltó calcular usando la barra roja
                page.snack_bar = ft.SnackBar(
                    ft.Text(f"⚠️ {str(ex)}", color="white", weight="bold"), 
                    bgcolor="#cc0000", open=True
                )
                
            finally:
                bloqueo_ui.visible = False
                page.update()
                
        # Desacoplamos del Hilo de UI para máxima fluidez
        page.run_thread(_task)

    csv_picker.on_result = on_load

    # --- 6. BOTONES DE TRANSICIÓN ---
    btn_nav_filt = ft.ElevatedButton("Siguiente: Filtros", icon=ft.Icons.ARROW_FORWARD, on_click=lambda e: advance_tab(1), disabled=True)
    btn_nav_stats = ft.ElevatedButton("Ejecutar Estadísticas", icon=ft.Icons.ARROW_FORWARD, on_click=run_stats, disabled=True)
    btn_nav_graph = ft.ElevatedButton("Generar Gráficos", icon=ft.Icons.ARROW_FORWARD, on_click=run_graphs, disabled=True)
    btn_nav_lluvias = ft.ElevatedButton("Iniciar Probabilidades", icon=ft.Icons.ARROW_FORWARD, on_click=run_lluvias, disabled=True)
    btn_nav_cuenca = ft.ElevatedButton("Calcular PDR / IDF", icon=ft.Icons.ARROW_FORWARD, on_click=run_cuenca, disabled=True)

    # --- 7. CONSTRUCCIÓN DE LAS PESTAÑAS (TABS) ---
    tab_carga = ft.Tab(text="1. Carga", icon=ft.Icons.INPUT, content=ft.Container(ft.Column([
        ft.Text("Paso 1: Serie Temporal Imputada", color="#00ff41", size=20), 
        ft.Row([ft.ElevatedButton("Cargar CSV Manual", icon=ft.Icons.UPLOAD, on_click=lambda _: csv_picker.pick_files()), csv_path_txt]), 
        data_table_raw, # ESTA TABLA SE ESTIRARÁ SOLA
        btn_nav_filt
    ]), padding=20))

    tab_filtros = ft.Tab(text="2. Filtros", icon=ft.Icons.FILTER_LIST, content=ft.Container(ft.Column([
        ft.Text("Paso 2: Limpieza Analítica", color="#00ff41", size=20), 
        ft.Row([chk_c1, chk_c2, chk_c3]), # <-- Checkboxes en fila para ahorrar espacio
        ft.ElevatedButton("Aplicar Filtros", icon=ft.Icons.FILTER_ALT, on_click=run_filter), 
        data_table_filt, # ESTA TABLA SE ESTIRARÁ SOLA
        btn_nav_stats
    ]), padding=20))

    tab_stats = ft.Tab(text="3. Stats", icon=ft.Icons.ANALYTICS, content=ft.Container(ft.Column([
        ft.Text("Paso 3: Resumen Estadístico", color="#00ff41", size=20), 
        stats_table_container, # ESTA TABLA SE ESTIRARÁ SOLA
        btn_nav_graph
    ]), padding=20))

    tab_graficos = ft.Tab(text="4. Gráficos", icon=ft.Icons.IMAGE, content=ft.Container(ft.Column([
        ft.Text("Paso 4: Comportamiento Visual", color="#00ff41", size=20), 
        ft.Text("Histograma"), img_h, ft.ElevatedButton("Guardar", on_click=lambda _: trigger_save_image("b64_hist", "histograma.png")), 
        ft.Text("Series"), img_s, ft.ElevatedButton("Guardar", on_click=lambda _: trigger_save_image("b64_series", "series.png")), 
        ft.Text("Violin"), img_v, ft.ElevatedButton("Guardar", on_click=lambda _: trigger_save_image("b64_violin", "violin.png")), 
        btn_nav_lluvias
    ], scroll=ft.ScrollMode.AUTO), padding=20))

    tab_lluvias = ft.Tab(text="5. Probabilidad", icon=ft.Icons.WATER_DROP, content=ft.Container(ft.Column([
        ft.Text("Paso 5: Ajuste de Funciones", color="#00ff41", size=20), 
        ft.Text("Homogeneidad:", weight="bold"), tbl_homo, ft.ElevatedButton("CSV", on_click=lambda _: trigger_save_csv("df_homogeneidad", "homogeneidad.csv")), ft.Divider(), 
        ft.Text("Series de Valores Máximos Anuales:", weight="bold"), img_max_an, ft.ElevatedButton("IMG", on_click=lambda _: trigger_save_image("b64_max_annual", "max_anuales.png")), ft.Divider(), 
        ft.Text("Autocorrelación (ACF):", weight="bold"), img_acf, ft.ElevatedButton("IMG", on_click=lambda _: trigger_save_image("b64_acf", "acf.png")), tbl_acf, ft.ElevatedButton("CSV", on_click=lambda _: trigger_save_csv("df_acf", "acf.csv")), ft.Divider(), 
        ft.Text("Weibull:", weight="bold"), img_wei, ft.ElevatedButton("IMG", on_click=lambda _: trigger_save_image("b64_weibull", "weibull.png")), tbl_wei, ft.ElevatedButton("CSV", on_click=lambda _: trigger_save_csv("df_weibull", "weibull.csv")), ft.Divider(), 
        ft.Text("Ajuste de Distribuciones:", weight="bold"), img_dist, ft.ElevatedButton("IMG", on_click=lambda _: trigger_save_image("b64_dist_comp", "distribuciones.png")), tbl_ajustes, ft.ElevatedButton("CSV", on_click=lambda _: trigger_save_csv("df_ajustes", "ajustes.csv")), ft.Divider(), 
        ft.Text("Máximos Mensuales:", weight="bold"), tbl_mx, ft.ElevatedButton("CSV", on_click=lambda _: trigger_save_csv("df_maximos_mensuales", "maximos_mensuales.csv")), 
        btn_nav_cuenca
    ], scroll=ft.ScrollMode.AUTO), padding=20))

    tab_cuenca = ft.Tab(text="6. A-D-TR/I-D-TR", icon=ft.Icons.MAP, content=ft.Container(ft.Column([
        ft.Text("Paso 6: Curvas Hidrológicas", color="#00ff41", size=20), txt_best_fit, 
        ft.Row([inp_ylim_h, inp_ylim_i, ft.ElevatedButton("Recalcular Ejes", on_click=run_cuenca)]), 
        ft.Text("PDR:"), img_c1, ft.ElevatedButton("Guardar", on_click=lambda _: trigger_save_image("cuenca_plot1_b64", "pdr.png")), 
        img_c2, ft.ElevatedButton("Guardar Zoom", on_click=lambda _: trigger_save_image("cuenca_plot2_b64", "pdr_zoom.png")), tbl_alt, ft.ElevatedButton("CSV", on_click=lambda _: trigger_save_csv("df_altura", "alturas.csv")), 
        ft.Text("IDF:"), img_c3, ft.ElevatedButton("Guardar", on_click=lambda _: trigger_save_image("cuenca_plot3_b64", "idf.png")), 
        img_c4, ft.ElevatedButton("Guardar Zoom", on_click=lambda _: trigger_save_image("cuenca_plot4_b64", "idf_zoom.png")), tbl_int, ft.ElevatedButton("CSV", on_click=lambda _: trigger_save_csv("df_intensidad", "I-D-TR.csv")), 
        ft.Divider(),
        ft.Text("Parámetros del Modelo y Ajuste Estadístico:", color="#00ff41", weight="bold"), 
        tbl_parametros, 
        ft.ElevatedButton("Exportar Parámetros (CSV)", icon=ft.Icons.DOWNLOAD, on_click=lambda _: trigger_save_csv("df_parametros", "parametros_modelo.csv")),
        ft.Text("Log Detallado:"), log_cuenca
    ], scroll=ft.ScrollMode.AUTO), padding=20))

    tabs = ft.Tabs(
        selected_index=0,
        animation_duration=300,
        tabs=[tab_carga, tab_filtros, tab_stats, tab_graficos, tab_lluvias, tab_cuenca],
        expand=True
    )

    # --- 8. CORRECCIÓN: REHIDRATACIÓN ASÍNCRONA ---
    def rehidratar_interfaz():
        try:
            df_proc = page.session.get("df_procesado")
            if df_proc is not None:
                csv_path_txt.value = "✅ Datos Activos en Sesión"
                data_table_raw.controls = [ft.Text("Datos Procesados:"), dataframe_to_datatable(df_proc)]
                
                # Restaurar Imágenes seguras
                img_h.content.src_base64 = page.session.get("b64_hist") or TRANSPARENT_PIXEL
                img_s.content.src_base64 = page.session.get("b64_series") or TRANSPARENT_PIXEL
                img_v.content.src_base64 = page.session.get("b64_violin") or TRANSPARENT_PIXEL
                img_max_an.content.src_base64 = page.session.get("b64_max_annual") or TRANSPARENT_PIXEL
                img_acf.content.src_base64 = page.session.get("b64_acf") or TRANSPARENT_PIXEL
                img_wei.content.src_base64 = page.session.get("b64_weibull") or TRANSPARENT_PIXEL
                img_dist.content.src_base64 = page.session.get("b64_dist_comp") or TRANSPARENT_PIXEL
                img_c1.content.src_base64 = page.session.get("cuenca_plot1_b64") or TRANSPARENT_PIXEL
                img_c2.content.src_base64 = page.session.get("cuenca_plot2_b64") or TRANSPARENT_PIXEL
                img_c3.content.src_base64 = page.session.get("cuenca_plot3_b64") or TRANSPARENT_PIXEL
                img_c4.content.src_base64 = page.session.get("cuenca_plot4_b64") or TRANSPARENT_PIXEL
                
                # Restaurar Tablas
                df_f = page.session.get("df_filtrado")
                if df_f is not None: data_table_filt.controls = [ft.Text(f"Filtrados: {len(df_f)}"), dataframe_to_datatable(df_f, auto_expand=True)]
                df_st = page.session.get("df_estadisticas")
                if df_st is not None: stats_table_container.controls = [ft.Text("Estadísticas:", color="#00ff41"), stats_to_datatable(df_st, auto_expand=True)]
                df_homo = page.session.get("df_homogeneidad")
                if df_homo is not None: tbl_homo.controls = [dataframe_to_datatable(df_homo, auto_expand=True)]
                df_ajustes = page.session.get("df_ajustes")
                if df_ajustes is not None: tbl_ajustes.controls = [dataframe_to_datatable(df_ajustes, auto_expand=True)]
                df_mx = page.session.get("df_maximos_mensuales")
                if df_mx is not None: tbl_mx.controls = [dataframe_to_datatable(df_mx, auto_expand=True)]
                # --- CORRECCIÓN CRÍTICA: Reset Index en Rehidratación (Evita crash de Flutter) ---
                df_alt = page.session.get("df_altura")
                if df_alt is not None: tbl_alt.controls = [dataframe_to_datatable(df_alt.reset_index(), auto_expand=True)]
                
                df_int = page.session.get("df_intensidad")
                if df_int is not None: tbl_int.controls = [dataframe_to_datatable(df_int.reset_index(), auto_expand=True)]
                
                df_param = page.session.get("df_parametros")
                if df_param is not None: tbl_parametros.controls = [dataframe_to_datatable(df_param.reset_index(), auto_expand=True)]
                
                # Desbloquear botones de navegación si los datos existen
                btn_nav_filt.disabled = False
                if df_f is not None: btn_nav_stats.disabled = False
                if df_st is not None: btn_nav_graph.disabled = False
                if page.session.get("b64_hist") is not None: btn_nav_lluvias.disabled = False
                if df_homo is not None: btn_nav_cuenca.disabled = False
                
                page.update()
        except: pass

    # --- MOTOR DE ELIMINACIÓN DE ESTACIÓN ---
    def eliminar_estacion_analizada(e):
        sid = dd_estacion_maestra.value
        if not sid:
            page.snack_bar = ft.SnackBar(ft.Text("⚠️ Selecciona una estación para eliminar.", color="white", weight="bold"), bgcolor="#cc0000", open=True)
            page.update()
            return
            
        # 1. Purga de Memoria Persistente (Evita fugas hacia el Módulo 4)
        db_curvas = page.session.get("db_curvas_procesadas") or {}
        db_curvas.pop(sid, None) # EAFP: Elimina si existe, no hace nada si no
        page.session.set("db_curvas_procesadas", db_curvas)
        
        page.session.set("target_station_id", None)
        
        # 2. Limpieza de Interfaz
        dd_estacion_maestra.value = None
        limpiar_tablas_ui()
        actualizar_dropdown_maestro()
        
        # 3. Bloqueo de Pestañas (Regresamos al inicio)
        btn_nav_filt.disabled = True
        btn_nav_stats.disabled = True
        btn_nav_graph.disabled = True
        btn_nav_lluvias.disabled = True
        btn_nav_cuenca.disabled = True
        tabs.selected_index = 0
        
        page.snack_bar = ft.SnackBar(ft.Text(f"🗑️ Análisis de la estación {sid} eliminado.", color="white", weight="bold"), bgcolor="#cc0000", open=True)
        page.update()

    async def trigger_hydration():
        await asyncio.sleep(0.1)
        actualizar_dropdown_maestro()
        rehidratar_interfaz()

    page.run_task(trigger_hydration)

    # --- 3. CORRECCIÓN: RETORNO DE LA VISTA CON LA BARRA SUPERIOR ---
    return ft.Column([
        ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.ANALYTICS, color="#00ff41"),
                ft.Text("MÓDULO 3: ANÁLISIS CLIMÁTICO", weight="bold"),
                ft.VerticalDivider(),
                dd_estacion_maestra,
                ft.IconButton(ft.Icons.REFRESH, on_click=lambda _: actualizar_dropdown_maestro(), tooltip="Actualizar lista"),
                
                # --- NUEVO: BOTÓN DE ELIMINACIÓN ---
                ft.IconButton(
                    icon=ft.Icons.DELETE_FOREVER, 
                    icon_color="#cc0000", 
                    tooltip="Eliminar estación actual de la memoria",
                    on_click=eliminar_estacion_analizada
                )
            ]),
            padding=10, bgcolor="#111111", border=ft.border.all(1, "#333333"), border_radius=5
        ),
        ft.Container(content=tabs, expand=True, padding=10)
    ], expand=True)