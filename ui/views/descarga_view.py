import flet as ft
import flet.canvas as cv
import threading
import os
import asyncio

# --- ARQUITECTURA HEXAGONAL: Importación desde el Core ---
from core import descarga_logic

def build_descarga_view(page: ft.Page):
    
    # --- 1. VARIABLES DE SESIÓN (Protegidas) ---
    if not page.session.get("ruta_descargas_base"):
        # Por defecto, guarda en una carpeta junto al archivo de proyecto activo
        proj_path = page.session.get("current_project_path")
        default_dir = os.path.dirname(proj_path) if proj_path else os.getcwd()
        page.session.set("ruta_descargas_base", os.path.join(default_dir, "Descargas_Tlaloc"))
    
    modo_actual = ft.Text("POR ESTADO", color="#00ff41", weight="bold", size=16)
    elementos_seleccionados = [] 
    poligonos_cargados = [] 
    estados_auditados = {}
    
    # --- 2. CONTROLES DE UI Y TERMINAL ---
    terminal_list = ft.ListView(expand=True, spacing=5, auto_scroll=True)
    progreso_bar = ft.ProgressBar(width=None, color="#00ff41", bgcolor="#1a1a1a", value=0, visible=False)
    ruta_destino_lbl = ft.Text(f"DESTINO: {page.session.get('ruta_descargas_base')}", color="grey", size=10, selectable=True)

    def log_terminal(mensaje):
        def _update():
            terminal_list.controls.append(ft.Text(mensaje, color="#00ff41", font_family="Roboto Mono", size=12))
            page.update()
        page.run_thread(_update)

    def update_progreso(valor):
        def _update():
            progreso_bar.value = valor
            page.update()
        page.run_thread(_update)

    def on_dialog_result(e: ft.FilePickerResultEvent):
        if e.path:
            page.session.set("ruta_descargas_base", e.path)
            ruta_destino_lbl.value = f"DESTINO: {e.path}"
            log_terminal(f"> DIRECTORIO FIJADO: {e.path}")
            page.update()
    
    def saltar_a_imputacion(e):
        nav_func = page.session.get("navigate_to_module")
        if nav_func:
            nav_func(1)

    dir_picker = ft.FilePicker(on_result=on_dialog_result)
    page.overlay.append(dir_picker)

    # --- 3. BOTONES DE CONTROL ---
    btn_modo_estado = ft.TextButton("🗺️ POR ESTADO", on_click=lambda e: cambiar_modo(e, "POR ESTADO"), style=ft.ButtonStyle(color="#00ff41"))
    btn_modo_cuenca = ft.TextButton("🌊 POR CUENCA", on_click=lambda e: cambiar_modo(e, "POR CUENCA"), style=ft.ButtonStyle(color="#00ff41"))
    btn_modo_masivo = ft.TextButton("📦 RESPALDO MASIVO", on_click=lambda e: cambiar_modo(e, "RESPALDO MASIVO"), style=ft.ButtonStyle(color="#1c75fa"))
    # --- NUEVO: BOTÓN DE AUDITORÍA ---
    # --- NUEVO: MOTOR DE AUDITORÍA DEEP SCAN ---
    def procesar_auditoria_hilo(ruta_salida):
        bloquear_interfaz(True) # Congelar UI globalmente
        progreso_bar.visible = True
        progreso_bar.value = 0
        btn_detener.visible = True
        btn_iniciar.visible = False
        
        def task():
            try:
                ruta_base = page.session.get("ruta_descargas_base")
                exito, msj = descarga_logic.auditar_base_datos_profunda(
                    ruta_base, ruta_salida, log_terminal, update_progreso
                )
                
                def final():
                    bloquear_interfaz(False)
                    btn_detener.visible = False
                    btn_iniciar.visible = True
                    color_sb = "#00ff41" if exito else "red"
                    page.snack_bar = ft.SnackBar(ft.Text(msj, weight="bold", color="black"), bgcolor=color_sb, open=True)
                    log_terminal(f"> {msj}")
                    page.update()
                page.run_thread(final)
                
            except Exception as e:
                log_terminal(f"> [CRÍTICO] Fallo en el hilo de auditoría: {e}")
                
        threading.Thread(target=task, daemon=True).start()

    def on_audit_dir_result(e: ft.FilePickerResultEvent):
        if e.path:
            log_terminal(f"> ---------------------------------")
            log_terminal(f"> 🔎 DESTINO DE AUDITORÍA FIJADO: {e.path}")
            modo_actual.value = "AUDITORÍA PROFUNDA"
            elementos_seleccionados.clear()
            dibujar_mapa()
            procesar_auditoria_hilo(e.path)

    picker_auditoria = ft.FilePicker(on_result=on_audit_dir_result)
    page.overlay.append(picker_auditoria)

    btn_auditar = ft.TextButton("✅ AUDITAR BD LOCAL", 
                                on_click=lambda _: picker_auditoria.get_directory_path(dialog_title="Selecciona dónde guardar el Informe LaTeX"), 
                                style=ft.ButtonStyle(color="#ff9900"))
    btn_cambiar_dir = ft.OutlinedButton("CAMBIAR CARPETA", icon=ft.Icons.FOLDER, on_click=lambda _: dir_picker.get_directory_path(), style=ft.ButtonStyle(color="white"))
    
    btn_iniciar = ft.ElevatedButton("INICIAR EXTRACCIÓN", color="#050505", bgcolor="#00ff41", icon=ft.Icons.CLOUD_DOWNLOAD, on_click=lambda e: run_extraction_thread())
    btn_detener = ft.ElevatedButton("DETENER", color="white", bgcolor="#cc0000", icon=ft.Icons.STOP, visible=False, on_click=lambda e: detener_proceso(e))
    
    # Puente desactivado internamente, el usuario usa el NavigationRail de main.py
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
            
        # RESOLUCIÓN SEGURA DE RUTA (Usando el BASE_DIR de la lógica)
        ruta_csv = os.path.join(descarga_logic.BASE_DIR, "assets", "catalogo_tlaloc.csv")
        estaciones = descarga_logic.obtener_catalogo_visor(modo_actual.value, elementos_seleccionados, ruta_csv)
        
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
            
            ruta_tar = os.path.join(page.session.get("ruta_descargas_base"), "Tlaloc_BD_Nacional_Comprimida.tar.xz")
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
        btn_cambiar_dir.disabled = bloquear
        
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
    CANVAS_WIDTH = 800
    CANVAS_HEIGHT = 450

    def dibujar_mapa(porcentajes_estados=None):
        if porcentajes_estados is None: porcentajes_estados = {}
        mapa_canvas.shapes.clear()
        try:
            nonlocal poligonos_cargados
            tipo_geometria = "POR ESTADO" if modo_actual.value == "RESPALDO MASIVO" else modo_actual.value
            
            if not poligonos_cargados:
                poligonos_cargados = descarga_logic.cargar_poligonos(tipo_geometria)

            b = descarga_logic.BBOX_MEXICO
            scale_x, scale_y = CANVAS_WIDTH / (b["max_x"] - b["min_x"]), CANVAS_HEIGHT / (b["max_y"] - b["min_y"])

            for pol in poligonos_cargados:
                geo = pol["geometria"]
                nombre_poligono = pol["nombre"]
                
                if modo_actual.value == "RESPALDO MASIVO":
                    clave_corta = descarga_logic.CATALOGO_ESTADOS_CONAGUA.get(nombre_poligono, "").upper()
                    pct_avance = porcentajes_estados.get(clave_corta, 0.0)
                    alfa_hex = f"{int(pct_avance * 153):02x}"
                    color_fill, color_stroke = (f"#{alfa_hex}00ff41" if pct_avance > 0 else "#000000"), ("#00ff41" if pct_avance >= 1.0 else "#004411")
                
                elif modo_actual.value == "AUDITORÍA":
                    clave_corta = descarga_logic.CATALOGO_ESTADOS_CONAGUA.get(nombre_poligono, "").upper()
                    tiene_datos = estados_auditados.get(clave_corta, False)
                    # Verde semitransparente si tiene datos, Rojo semitransparente si está vacío
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
        scale_x, scale_y = CANVAS_WIDTH / (b["max_x"] - b["min_x"]), CANVAS_HEIGHT / (b["max_y"] - b["min_y"])
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

    contenedor_mapa = ft.GestureDetector(on_tap_down=on_mapa_click, content=ft.Container(content=mapa_canvas, bgcolor="#0a0a0a", width=CANVAS_WIDTH, height=CANVAS_HEIGHT, border=ft.border.all(1, "#00ff41"), border_radius=5))

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

    def run_extraction_thread():
        if not elementos_seleccionados and modo_actual.value != "RESPALDO MASIVO":
            log_terminal("> [ERROR] SELECCIONE AL MENOS UNA ZONA."); return
            
        # 1. Preparar Barra visual
        progreso_bar.visible = True
        progreso_bar.value = 0 
        
        # 2. Bloquear toda la app (esto ya maneja btn_iniciar y btn_detener)
        bloquear_interfaz(True)
        
        def task():
            try:
                # Extraemos de forma segura la ubicación activa del proyecto para enviarla como semilla
                ruta_activa_sesion = page.session.get("imput_folder_path")
                
                ruta_final = descarga_logic.procesar_descarga(
                    modo_actual.value, elementos_seleccionados, page.session.get("ruta_descargas_base"), 
                    log_terminal, update_progreso, callback_mapa=dibujar_mapa,
                    carpeta_previa=ruta_activa_sesion # Enlazamos el historial cargado en RAM/Temp
                )
            except Exception as ex:
                log_terminal(f"> [CRÍTICO] Error en hilo: {ex}")
            finally:
                def _finalize():
                    btn_detener.visible, btn_iniciar.visible = False, True
                    bloquear_interfaz(False)
                    if ruta_final and not descarga_logic.señal_abortar.is_set():
                        page.session.set("imput_folder_path", ruta_final) # Se enlaza automático al Módulo 2
                        
                        # --- NUEVO PAYLOAD: ORDEN DE EMPAQUETADO AL CERRAR ---
                        page.session.set("txt_backup", {"__type__": "folder_backup", "path": ruta_final})
                        
                        btn_puente.visible = True
                    page.update()
                page.run_thread(_finalize)
            
        threading.Thread(target=task, daemon=True).start()

    # --- 7. CARGA INICIAL ASÍNCRONA ---
    async def render_mapa_inicial():
        await asyncio.sleep(0.5)
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
                    ft.Text("DIRECTORIO DE SALIDA", color="white", size=10, weight="bold"),
                    ruta_destino_lbl, btn_cambiar_dir,
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