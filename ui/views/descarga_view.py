import flet as ft
import flet.canvas as cv
import threading
import os
import asyncio

# --- ARQUITECTURA HEXAGONAL: Importación desde el Core ---
from core import descarga_logic

def build_descarga_view(page: ft.Page):
    
    # --- 1. VARIABLES DE SESIÓN (Protegidas) ---
    modo_actual = ft.Text("POR ESTADO", color="#00ff41", weight="bold", size=16)
    elementos_seleccionados = [] 
    poligonos_cargados = [] 
    estados_auditados = {}
    
    ruta_bd_activa = page.session.get("ruta_bd_activa")
    indice_activo = page.session.get("indice_bd_local")
    
    # --- 2. CONTROLES DE UI Y TERMINAL ---
    terminal_list = ft.ListView(expand=True, spacing=5, auto_scroll=True)
    progreso_bar = ft.ProgressBar(width=None, color="#00ff41", bgcolor="#1a1a1a", value=0, visible=False)
    
    lbl_bd_activa = ft.Text(f"BD: {ruta_bd_activa if ruta_bd_activa else 'NINGUNA (Requiere Vinculación)'}", color="#00ff41" if ruta_bd_activa else "grey", size=10, selectable=True)
    lbl_indice_status = ft.Text(f"Índice HDS: {'Activo en memoria' if indice_activo is not None else 'Vacío'}", color="#00ff41" if indice_activo is not None else "red", size=10)

    def log_terminal(mensaje):
        def _update():
            terminal_list.controls.append(ft.Text(mensaje, color="#00ff41", font_family="Roboto Mono", size=12))
            # --- PROTECCIÓN OOM: Buffer Cíclico ---
            # Mantenemos el árbol de renderizado ligero eliminando registros viejos
            if len(terminal_list.controls) > 150:
                terminal_list.controls.pop(0)
            page.update()
        page.run_thread(_update)

    def update_progreso(valor):
        def _update():
            progreso_bar.value = valor
            page.update()
        page.run_thread(_update)
    
    def saltar_a_imputacion(e):
        nav_func = page.session.get("navigate_to_module")
        if nav_func: nav_func(1)

    # --- NUEVOS CONTROLADORES DE ARCHIVO (In-Memory) ---
    def on_vincular_bd_result(e: ft.FilePickerResultEvent):
        if e.files:
            ruta = e.files[0].path
            page.session.set("ruta_bd_activa", ruta)
            lbl_bd_activa.value = f"BD: {os.path.basename(ruta)}"
            lbl_bd_activa.color = "#00ff41"
            log_terminal(f"> 📦 VINCULANDO BASE DE DATOS: {ruta}")
            
            bloquear_interfaz(True)
            progreso_bar.visible = True; progreso_bar.value = 0
            
            def _task():
                try:
                    descarga_logic.señal_abortar.clear()
                    cuencas = descarga_logic.cargar_poligonos("POR CUENCA")
                    log_terminal("> Escaneando compresión LZMA y reconstruyendo Índice HDS (8 Columnas)...")
                    df_indice = descarga_logic.indexar_base_datos_tar(ruta, cuencas, log_terminal, update_progreso)
                    
                    if df_indice is not None and not df_indice.empty:
                        page.session.set("indice_bd_local", df_indice) # PERSISTENCIA HDS
                        def _success():
                            lbl_indice_status.value = f"Índice HDS: Activo ({len(df_indice)} est.)"
                            lbl_indice_status.color = "#00ff41"
                            log_terminal("> ✅ Índice reconstruido e inyectado en la memoria del proyecto.")
                            page.update()
                        page.run_thread(_success)
                    else: log_terminal("> ❌ Error: No se pudo generar el índice.")
                except Exception as ex: log_terminal(f"> ❌ Error: {ex}")
                finally:
                    def _finalize():
                        bloquear_interfaz(False)
                        btn_detener.visible = False
                        page.update()
                    page.run_thread(_finalize)
            threading.Thread(target=_task, daemon=True).start()

    def on_exportar_csv_result(e: ft.FilePickerResultEvent):
        if e.path:
            df = page.session.get("indice_bd_local")
            if df is not None:
                df.to_csv(e.path, index=False, encoding='utf-8')
                log_terminal(f"> ✅ Índice exportado a CSV en: {e.path}")
            else: log_terminal("> [ERROR] No hay índice en memoria.")

    def on_guardar_bd_masiva(e: ft.FilePickerResultEvent):
        if e.path: run_extraction_thread(ruta_target=e.path)

    picker_vincular_bd = ft.FilePicker(on_result=on_vincular_bd_result)
    picker_exportar_csv = ft.FilePicker(on_result=on_exportar_csv_result)
    picker_guardar_bd = ft.FilePicker(on_result=on_guardar_bd_masiva)
    page.overlay.extend([picker_vincular_bd, picker_exportar_csv, picker_guardar_bd])

    # --- 3. BOTONES DE CONTROL ---
    btn_modo_estado = ft.TextButton("🗺️ POR ESTADO", on_click=lambda e: cambiar_modo(e, "POR ESTADO"), style=ft.ButtonStyle(color="#00ff41"))
    btn_modo_cuenca = ft.TextButton("🌊 POR CUENCA", on_click=lambda e: cambiar_modo(e, "POR CUENCA"), style=ft.ButtonStyle(color="#00ff41"))
    btn_modo_masivo = ft.TextButton("📦 RESPALDO MASIVO", on_click=lambda e: cambiar_modo(e, "RESPALDO MASIVO"), style=ft.ButtonStyle(color="#1c75fa"))
    
    # --- MOTOR DE AUDITORÍA Y PICKERS ---
    def procesar_auditoria_hilo(ruta_salida):
        ruta_bd = page.session.get("ruta_bd_activa")
        if not ruta_bd:
            log_terminal("> [ERROR] Vincula una BD (.tar.xz) primero.")
            return
        # (El resto de la lógica de auditoría se mantiene igual...)
        bloquear_interfaz(True)
        progreso_bar.visible = True; progreso_bar.value = 0
        btn_detener.visible, btn_iniciar.visible = True, False
        def task():
            try:
                # Recibimos el diccionario de estadísticas del Core
                exito, msj, stats_audit = descarga_logic.auditar_base_datos_profunda(ruta_bd, ruta_salida, log_terminal, update_progreso)
                
                def final():
                    bloquear_interfaz(False)
                    btn_detener.visible, btn_iniciar.visible = False, True
                    
                    # --- CONEXIÓN DE ESTADO AL MAPA ---
                    if exito and stats_audit:
                        estados_auditados.clear()
                        for k, v in stats_audit.items():
                            estados_auditados[k] = v["sanas"] > 0
                        dibujar_mapa() # Redibujamos el mapa para que pinte la auditoría real
                    
                    page.snack_bar = ft.SnackBar(ft.Text(msj, weight="bold", color="black"), bgcolor="#00ff41" if exito else "red", open=True)
                    log_terminal(f"> {msj}")
                    page.update()
                page.run_thread(final)
            except Exception as e: log_terminal(f"> [CRÍTICO] Fallo de auditoría: {e}")
        threading.Thread(target=task, daemon=True).start()

    def on_audit_dir_result(e: ft.FilePickerResultEvent):
        if e.path:
            modo_actual.value = "AUDITORÍA PROFUNDA"
            elementos_seleccionados.clear(); dibujar_mapa()
            procesar_auditoria_hilo(e.path)

    picker_auditoria = ft.FilePicker(on_result=on_audit_dir_result)
    page.overlay.append(picker_auditoria)

    btn_auditar = ft.TextButton("✅ AUDITAR BD LOCAL", on_click=lambda _: picker_auditoria.get_directory_path(dialog_title="Guardar Informe"), style=ft.ButtonStyle(color="#ff9900"))
    btn_vincular_bd = ft.OutlinedButton("VINCULAR BD (.xz)", icon=ft.Icons.LINK, on_click=lambda _: picker_vincular_bd.pick_files(allowed_extensions=["xz"]), style=ft.ButtonStyle(color="white"))
    btn_exportar_csv = ft.OutlinedButton("EXPORTAR ÍNDICE (CSV)", icon=ft.Icons.TABLE_CHART, on_click=lambda _: picker_exportar_csv.save_file(file_name="Indice_Tlaloc.csv", allowed_extensions=["csv"]), style=ft.ButtonStyle(color="white"))
    
    def on_iniciar_click(e):
        if modo_actual.value == "RESPALDO MASIVO": picker_guardar_bd.save_file(file_name="Tlaloc_BD_Nacional_Comprimida.tar.xz", allowed_extensions=["xz"])
        else:
            if not page.session.get("ruta_bd_activa"): log_terminal("> [ERROR] Debes VINCULAR una BD (.tar.xz) primero."); return
            run_extraction_thread(ruta_target=page.session.get("ruta_bd_activa"))

    btn_iniciar = ft.ElevatedButton("INICIAR EXTRACCIÓN", color="#050505", bgcolor="#00ff41", icon=ft.Icons.CLOUD_DOWNLOAD, on_click=on_iniciar_click)
    btn_detener = ft.ElevatedButton("DETENER", color="white", bgcolor="#cc0000", icon=ft.Icons.STOP, visible=False, on_click=lambda e: detener_proceso(e))
    btn_puente = ft.ElevatedButton("IR A IMPUTACIÓN", color="white", bgcolor="#1c75fa", icon=ft.Icons.ARROW_FORWARD, visible=False, on_click=saltar_a_imputacion)
    # Botón de Inspección
    btn_inspeccionar = ft.ElevatedButton("🔍 INSPECCIONAR ZONA", color="white", bgcolor="#9900ff", visible=True, on_click=lambda e: abrir_visor_flotante())

    def abrir_visor_flotante():
        if modo_actual.value == "RESPALDO MASIVO":
            log_terminal("> [AVISO] El visor requiere modo 'POR ESTADO' o 'POR CUENCA'.")
            return
            
        if not elementos_seleccionados:
            log_terminal("> [ERROR] Selecciona al menos una zona en el mapa (clic) para inspeccionarla.")
            return
            
        # El catálogo se extrae directamente de la memoria RAM (Persistencia HDS)
        df_cat = page.session.get("indice_bd_local")
        estaciones = descarga_logic.obtener_catalogo_visor(modo_actual.value, elementos_seleccionados, df_cat)
        
        if not estaciones:
            log_terminal("> [ERROR] No hay estaciones locales para la zona. Realiza el Respaldo Masivo primero.")
            return

        dd_estaciones = ft.Dropdown(
            label="Selecciona Estación (Max 150 enlistadas)",
            options=[ft.dropdown.Option(key=str(est['clave']), text=f"[{est['clave']}] {est['nombre']} ({est['estado_origen']})") for est in estaciones[:150]],
            width=500, bgcolor="#1a1a1a", color="white"
        )
        
        txt_local = ft.Text("Esperando consulta...", font_family="Roboto Mono", size=11, color="#888888")
        txt_server = ft.Text("Esperando consulta...", font_family="Roboto Mono", size=11, color="#888888")
        txt_status = ft.Text("ESTADO: EN ESPERA", weight="bold", size=14, color="white")
        
        def on_consultar_click(e):
            if not dd_estaciones.value: return
            clave_sel = dd_estaciones.value
            est_sel = next(s for s in estaciones if str(s['clave']) == clave_sel)
            
            txt_local.value, txt_server.value, txt_status.value, txt_status.color = "Consultando local...", "Conectando Servidor...", "ANALIZANDO...", "white"
            page.update()
            
            # Leemos la ruta vinculada, independientemente de la carpeta donde se encuentre
            ruta_tar = page.session.get("ruta_bd_activa")
            local_data, server_data = descarga_logic.inspeccionar_estacion_aislada(clave_sel, est_sel['estado_origen'], ruta_tar)
            
            txt_local.value, txt_server.value = "\n".join(local_data), "\n".join(server_data)
            
            if local_data == server_data and "Error" not in local_data[0]:
                txt_status.value, txt_status.color = "✅ SISTEMA ACTUALIZADO", "#00ff41"
            elif "Error" in local_data[0] or "Error" in server_data[0]:
                txt_status.value, txt_status.color = "⚠️ ERROR DE LECTURA", "red"
            else:
                txt_status.value, txt_status.color = "⚠️ DATOS NUEVOS DETECTADOS", "#ffdd00"
            page.update()

        dlg_visor = ft.AlertDialog(
            modal=False,
            title=ft.Text("SONDA DE TELEMETRÍA", color="#9900ff", weight="bold"),
            content=ft.Container(
                width=800, height=350,
                content=ft.Column([
                    ft.Row([dd_estaciones, ft.ElevatedButton("COMPARAR", bgcolor="#9900ff", color="white", on_click=on_consultar_click)]),
                    ft.Divider(color="#333333"),
                    ft.Row([
                        ft.Column([ft.Text("📁 BD LOCAL (tar.xz)", weight="bold", color="white"), ft.Container(content=txt_local, bgcolor="#000000", padding=10, border=ft.border.all(1, "#333333"), border_radius=5)], expand=1),
                        ft.Column([ft.Text("🌐 SERVIDOR (En Vivo)", weight="bold", color="white"), ft.Container(content=txt_server, bgcolor="#000000", padding=10, border=ft.border.all(1, "#333333"), border_radius=5)], expand=1)
                    ], expand=True),
                    ft.Divider(color="#333333"),
                    ft.Row([txt_status], alignment=ft.MainAxisAlignment.CENTER)
                ])
            ), bgcolor="#111111"
        )
        page.open(dlg_visor)
    
    def bloquear_interfaz(bloquear: bool):
        """Aislamiento total de la interfaz y menú global durante la descarga."""
        # 1. Controles Locales
        btn_modo_estado.disabled = bloquear
        btn_modo_cuenca.disabled = bloquear
        btn_modo_masivo.disabled = bloquear
        
        # NUEVOS BOTONES IN-MEMORY (Reemplazan a btn_cambiar_dir)
        btn_vincular_bd.disabled = bloquear
        btn_exportar_csv.disabled = bloquear
        btn_auditar.disabled = bloquear
        
        # --- REGLA DE DOMINIO ZERO-TRUST ---
        # Bloqueamos el botón de Auditoría para evitar I/O Race Conditions 
        # (intentar leer con Deep Scan mientras el hilo de Descarga está escribiendo en el disco).
        btn_auditar.disabled = bloquear
        
        # <-- NUEVO: Bloqueamos los botones flotantes de la vista
        btn_inspeccionar.disabled = bloquear 
        btn_puente.disabled = bloquear 
        
        # 2. Lienzo Espacial
        contenedor_mapa.disabled = bloquear 
        
        # 3. Visibilidad de Motores
        btn_iniciar.visible = not bloquear
        btn_detener.visible = bloquear
        btn_detener.disabled = False # El freno de emergencia siempre debe estar libre
        
        # 4. Bloqueo Quirúrgico del NavigationRail Global (main.py)
        main_rail = page.session.get("main_rail")
        if main_rail:
            main_rail.disabled = bloquear # Apaga destinos: Extracción, Imputación, Análisis, Gastos
            
            # Apagamos los botones inferiores (Guardar / Cerrar)
            if hasattr(main_rail, "trailing") and main_rail.trailing:
                for ctrl in main_rail.trailing.controls:
                    ctrl.disabled = bloquear
                    
            main_rail.update() # Forzamos el redibujado del menú
        
        page.update()

    # --- 4. LIENZO ESPACIAL ---
    mapa_canvas = cv.Canvas(expand=True)
    
    # NUEVO: Diccionario mutable para zoom dinámico y preservación estricta de coordenadas
    map_dim = {"w": 800, "h": 450}

    def dibujar_mapa(porcentajes_estados=None):
        if porcentajes_estados is None: porcentajes_estados = {}
        mapa_canvas.shapes.clear()
        try:
            nonlocal poligonos_cargados
            tipo_geometria = "POR ESTADO" if modo_actual.value == "RESPALDO MASIVO" else modo_actual.value
            
            if not poligonos_cargados:
                poligonos_cargados = descarga_logic.cargar_poligonos(tipo_geometria)

            b = descarga_logic.BBOX_MEXICO
            # Aplicación de la matriz dinámica a escala
            scale_x, scale_y = map_dim["w"] / (b["max_x"] - b["min_x"]), map_dim["h"] / (b["max_y"] - b["min_y"])

            for pol in poligonos_cargados:
                geo = pol["geometria"]
                nombre_poligono = pol["nombre"]
                
                if modo_actual.value == "RESPALDO MASIVO":
                    clave_corta = descarga_logic.obtener_clave_estado(nombre_poligono).upper()
                    pct_avance = porcentajes_estados.get(clave_corta, 0.0)
                    alfa_hex = f"{int(pct_avance * 153):02x}"
                    color_fill, color_stroke = (f"#{alfa_hex}00ff41" if pct_avance > 0 else "#000000"), ("#00ff41" if pct_avance >= 1.0 else "#004411")
                
                elif modo_actual.value in ["AUDITORÍA", "AUDITORÍA PROFUNDA"]:
                    clave_corta = descarga_logic.obtener_clave_estado(nombre_poligono).upper()
                    tiene_datos = estados_auditados.get(clave_corta, False)
                    color_fill = "#6600ff41" if tiene_datos else "#66ff0000"
                    color_stroke = "#00ff41" if tiene_datos else "#ff0000"
                
                else:
                    is_selected = nombre_poligono in elementos_seleccionados
                    color_fill, color_stroke = ("#6600ff41" if is_selected else "#000000"), ("#00ff41" if is_selected else "#005522")

                coords_list = [geo["coordinates"]] if geo["type"] == "Polygon" else geo["coordinates"]
                for poligono_anillos in coords_list:
                    path_elements = []
                    for idx, (lon, lat) in enumerate(poligono_anillos[0]):
                        px, py = (lon - b["min_x"]) * scale_x, (b["max_y"] - lat) * scale_y 
                        path_elements.append(cv.Path.MoveTo(px, py) if idx == 0 else cv.Path.LineTo(px, py))
                    path_elements.append(cv.Path.Close())
                    mapa_canvas.shapes.extend([cv.Path(elements=path_elements, paint=ft.Paint(color=color_fill, style=ft.PaintingStyle.FILL)), cv.Path(elements=path_elements, paint=ft.Paint(color=color_stroke, style=ft.PaintingStyle.STROKE, stroke_width=1))])
                
                if modo_actual.value == "RESPALDO MASIVO" and pct_avance > 0:
                    centroide = pol["shapely_obj"].centroid
                    cx, cy = (centroide.x - b["min_x"]) * scale_x, (b["max_y"] - centroide.y) * scale_y 
                    mapa_canvas.shapes.append(cv.Text(x=cx - 10, y=cy + 5, text=f"{int(pct_avance * 100)}%", style=ft.TextStyle(size=10, weight="bold", color="#ffffff" if pct_avance >= 1.0 else "#00ff41", font_family="Roboto Mono")))
            page.update()
        except Exception: pass

    # --- 5. GESTOR DE EVENTOS ---
    def on_mapa_click(e):
        if modo_actual.value in ["RESPALDO MASIVO", "AUDITORÍA"] or btn_detener.visible: return
        b = descarga_logic.BBOX_MEXICO
        # Aplicación de matriz inversa para la detección geoespacial estricta
        scale_x, scale_y = map_dim["w"] / (b["max_x"] - b["min_x"]), map_dim["h"] / (b["max_y"] - b["min_y"])
        lon_clic, lat_clic = (e.local_x / scale_x) + b["min_x"], b["max_y"] - (e.local_y / scale_y)
        pol_detectado = descarga_logic.detectar_clic_poligono(poligonos_cargados, lon_clic, lat_clic)
        
        if pol_detectado:
            nombre = pol_detectado["nombre"] if pol_detectado["nombre"] != "Desconocido" else f"Zona ID: {pol_detectado['id']}"
            if nombre in elementos_seleccionados:
                elementos_seleccionados.remove(nombre)
                log_terminal(f"> [-] Deseleccionado: {nombre}")
            else:
                elementos_seleccionados.append(nombre)
                log_terminal(f"> [+] Seleccionado: {nombre}")
            dibujar_mapa()

    # Contenedor dinámico (inyecta dict en tiempo de renderizado)
    contenedor_mapa = ft.GestureDetector(on_tap_down=on_mapa_click, content=ft.Container(content=mapa_canvas, bgcolor="#0a0a0a", width=map_dim["w"], height=map_dim["h"], border=ft.border.all(1, "#00ff41"), border_radius=5))

    # --- NUEVO: MOTOR DE ZOOM (FIT-TO-SCREEN) ---
    def ajustar_zoom_mapa(e=None):
        if not contenedor_mapa.page: return # Prevención si la vista no está activa
        try:
            pw = page.width if page.width else 1200
            ph = page.height if page.height else 1000
            
            # Cálculo de márgenes dinámicos respetando los Flexboxes (expand=5 vs expand=3)
            espacio_w = max(400, pw - 320) # 250px panel izq + paddings
            espacio_h = max(250, (ph - 150) * 0.6) # Aproximadamente el 60% para el lienzo
            
            b = descarga_logic.BBOX_MEXICO
            aspect_ratio = (b["max_x"] - b["min_x"]) / (b["max_y"] - b["min_y"])
            
            # Restricción 'Fit to Contain' pura
            calc_h = espacio_w / aspect_ratio
            if calc_h > espacio_h:
                calc_h = espacio_h
                calc_w = calc_h * aspect_ratio
            else:
                calc_w = espacio_w
                
            map_dim["w"] = calc_w
            map_dim["h"] = calc_h
            contenedor_mapa.content.width = map_dim["w"]
            contenedor_mapa.content.height = map_dim["h"]
            
            if poligonos_cargados:
                dibujar_mapa()
        except Exception:
            pass

    # Suscripción segura al loop de eventos del Sistema Operativo (On Resize)
    page.on_resized = ajustar_zoom_mapa

    def cambiar_modo(e, nuevo_modo):
        modo_actual.value = nuevo_modo; elementos_seleccionados.clear(); poligonos_cargados.clear()
        log_terminal(f"> ---------------------------------\n> MODO DE EXTRACCIÓN: {nuevo_modo}")
        dibujar_mapa()

    # --- 6. HILO DE EJECUCIÓN ---
    def detener_proceso(e):
        log_terminal("> [SISTEMA] Freno de emergencia activado. Sellando base de datos...")
        btn_detener.disabled = True
        descarga_logic.señal_abortar.set()
        page.update()

    def run_extraction_thread(ruta_target):
        if not elementos_seleccionados and modo_actual.value != "RESPALDO MASIVO":
            log_terminal("> [ERROR] SELECCIONE AL MENOS UNA ZONA."); return
            
        progreso_bar.visible = True
        progreso_bar.value = 0 
        bloquear_interfaz(True)
        
        def task():
            try:
                ruta_activa_sesion = page.session.get("imput_folder_path")
                df_cat = page.session.get("indice_bd_local")
                
                ruta_final, df_generado = descarga_logic.procesar_descarga(
                    modo_actual.value, elementos_seleccionados, ruta_target, df_cat,
                    log_terminal, update_progreso, callback_mapa=dibujar_mapa,
                    carpeta_previa=ruta_activa_sesion 
                )
            except Exception as ex:
                log_terminal(f"> [CRÍTICO] Error en hilo principal: {ex}")
                ruta_final, df_generado = None, None
            finally:
                def _finalize():
                    btn_detener.visible, btn_iniciar.visible = False, True
                    bloquear_interfaz(False)
                    if not descarga_logic.señal_abortar.is_set():
                        
                        # Inyectar Automáticamente en Memoria si fue Respaldo Masivo
                        if modo_actual.value == "RESPALDO MASIVO" and df_generado is not None:
                            page.session.set("ruta_bd_activa", ruta_target)
                            page.session.set("indice_bd_local", df_generado)
                            lbl_bd_activa.value = f"BD: {os.path.basename(ruta_target)}"
                            lbl_bd_activa.color = "#00ff41"
                            lbl_indice_status.value = f"Índice HDS: Activo ({len(df_generado)} est.)"
                            lbl_indice_status.color = "#00ff41"
                        
                        # Habilitar paso a Imputación si fue extracción de datos (Local)
                        if ruta_final and modo_actual.value != "RESPALDO MASIVO":
                            page.session.set("imput_folder_path", ruta_final) 
                            page.session.set("txt_backup", {"__type__": "folder_backup", "path": ruta_final})
                            btn_puente.visible = True
                    page.update()
                page.run_thread(_finalize)
            
        threading.Thread(target=task, daemon=True).start()

    # --- 7. CARGA INICIAL ASÍNCRONA ---
    async def render_mapa_inicial():
        await asyncio.sleep(0.5)
        ajustar_zoom_mapa() # Detonamos el Zoom Dinámico al inicio para que nazca ajustado a la ventana
        dibujar_mapa()
    page.run_task(render_mapa_inicial)

    # --- 8. LAYOUT ---
    return ft.Container(
        content=ft.Row([
            ft.Container(
                content=ft.Column([
                    ft.Text("EXTRACCIÓN", size=18, weight="bold"),
                    ft.Divider(color="#222222"),
                    ft.Text("MÉTODO DE SELECCIÓN", color="white", size=10, weight="bold"),
                    btn_modo_estado, btn_modo_cuenca, btn_modo_masivo,btn_auditar,
                    ft.Divider(color="#222222"),
                    ft.Text("BASE DE DATOS MAESTRA", color="white", size=10, weight="bold"),
                    lbl_bd_activa, lbl_indice_status,
                    btn_vincular_bd, btn_exportar_csv,
                ], spacing=15, scroll=ft.ScrollMode.AUTO),
                width=250, padding=20, border=ft.border.only(right=ft.border.BorderSide(1, "#222222"))
            ),
            ft.Container(
                content=ft.Column([
                    ft.Container(
                        content=ft.Column([
                            ft.Row([
                                ft.Row([ft.Text("MAPA GEOESPACIAL: ", color="white", weight="bold"), modo_actual]),
                                ft.Row([btn_inspeccionar, btn_iniciar, btn_detener, btn_puente])
                            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                            contenedor_mapa,
                            ft.Text("Modo Normal: Clic para seleccionar | Modo Masivo: Monitor de progreso.", color="grey", size=11, italic=True)
                        ]), expand=5, padding=20
                    ),
                    ft.Divider(color="#222222", height=1),
                    ft.Container(
                        content=ft.Column([
                            ft.Text("CONSOLA DE PROCESO", color="white", weight="bold", size=12),
                            progreso_bar,
                            ft.Container(content=terminal_list, bgcolor="#000000", expand=True, padding=10, border=ft.border.all(1, "#222222"))
                        ]), expand=3, padding=20
                    )
                ]), expand=True, padding=0
            )
        ], expand=True, spacing=0),
        bgcolor="#050505", expand=True
    )