import flet as ft
import traceback
import os
import threading
import pandas as pd
import sqlite3

from core.arf_metodo_frecuencias import MotorFrecuencias
from core.arf_logic import MotorFiltradoARF, MotorCalculoARF, GestorMatricesARF
from core.arf_metodo_empirico import MotorEmpirico

def build_arf_view(page: ft.Page):
    
    # --- VARIABLES DE ESTADO Y SESIÓN ---
    ruta_imputados = page.session.get("imput_output_folder") 
    
    # --- INDICADORES LED GLOBALES ---
    led_filtro = ft.Icon(ft.Icons.CIRCLE, color="grey", size=12)
    led_grid = ft.Icon(ft.Icons.CIRCLE, color="grey", size=12)
    led_prob = ft.Icon(ft.Icons.CIRCLE, color="grey", size=12)
    
    progreso_arf = ft.ProgressBar(width=None, color="#00ff41", bgcolor="#1a1a1a", value=0, visible=False)
    
    # =========================================================================
    # PESTAÑA 1: MATRIZ DE CONTINUIDAD (ADUANA)
    # =========================================================================
    dt_continuidad = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("Estación", color="#00ff41")),
            ft.DataColumn(ft.Text("Inicio - Fin")),
            ft.DataColumn(ft.Text("Días Efectivos")),
            ft.DataColumn(ft.Text("Max Gap")),
            ft.DataColumn(ft.Text("Integridad"))
        ],
        rows=[], border=ft.border.all(1, "#333333"), heading_row_color="#1a1a1a",
    )

    def abrir_tabla_completa():
        dlg = ft.AlertDialog(
            title=ft.Text("Matriz de Continuidad Completa", color="#00ff41"),
            content=ft.Container(width=page.window.width * 0.9, height=page.window.height * 0.7, content=ft.Column([dt_continuidad], scroll=ft.ScrollMode.ADAPTIVE)),
            bgcolor="#111111"
        )
        page.open(dlg)

    def build_tab_calidad():
        return ft.Container(
            content=ft.Column([
                ft.Row([ft.Text("Matriz de Continuidad Temporal", weight="bold", size=16), ft.IconButton(ft.Icons.FULLSCREEN, icon_color="#00ff41", on_click=lambda e: abrir_tabla_completa())], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Text("Resultado inmutable de la purga C1, C2 y C3.", color="grey", size=12),
                ft.Container(content=ft.Column([dt_continuidad], scroll=ft.ScrollMode.ADAPTIVE), expand=True, border=ft.border.all(1, "#222222"))
            ]), padding=20
        )

    # =========================================================================
    # PESTAÑA 2: DASHBOARD BENTO BOX (LMA)
    # =========================================================================
    # Contenedor dinámico que será inyectado por el hilo al terminar la Fase 2
    bento_container = ft.Container(
        content=ft.Text("Esperando ejecución de Fase 2 (GENERAR LMA)...", color="grey"),
        alignment=ft.alignment.center, expand=True
    )

    def renderizar_bento_box(data):
        def kpi_card(title, value, icon, color):
            return ft.Container(
                content=ft.Column([
                    ft.Row([ft.Icon(icon, color=color, size=16), ft.Text(title, color="grey", size=10, weight="bold")]),
                    ft.Text(value, color="white", size=16, weight="bold")
                ], spacing=2, alignment=ft.MainAxisAlignment.CENTER),
                bgcolor="#111111", padding=15, border_radius=10, border=ft.border.all(1, "#333333"),
                col={"sm": 6, "md": 4, "lg": 4} # Responsive
            )

        # 1. KPIs
        kpis = ft.ResponsiveRow([
            kpi_card("PERIODO HISTÓRICO", f"{data['fecha_inicio'][:4]} - {data['fecha_fin'][:4]}", ft.Icons.CALENDAR_MONTH, "#1c75fa"),
            kpi_card("ESTACIONES (ÉLITE)", str(len(data['claves_activas'])), ft.Icons.SENSORS, "#1c75fa"),
            kpi_card("AUDITORÍA OMM", f"{data['densidad']} km²/est", ft.Icons.POLICY, data['color_omm']),
            kpi_card("DÍAS SECOS", f"{data['pct_secos']:.1f}%", ft.Icons.WB_SUNNY, "#ffaa00"),
            kpi_card("LMA MÁX. (HISTÓRICA)", f"{data['lma_maxima']:.1f} mm", ft.Icons.SHOW_CHART, "#ff0044"),
            kpi_card("ESTATUS DE RED", data['eval_omm'], ft.Icons.FACT_CHECK, data['color_omm']),
        ])

        # 2. Gráfico Cinemático Full Width
        grafico_cinematico = ft.Container(content=ft.Image(src_base64=data['plot_serie'], fit=ft.ImageFit.CONTAIN), bgcolor="#0a0a0a", border=ft.border.all(1, "#333333"), border_radius=10, padding=10)

        # 3. Exportación y AMS
        def exportar_csv(e):
            df_lma = page.session.get("arf_matriz_maestra")
            if df_lma is not None:
                ruta_salida = os.path.join(str(page.session.get("ruta_bd_activa")) + "_LMA_Diaria.csv")
                df_lma.to_csv(ruta_salida)
                page.snack_bar = ft.SnackBar(ft.Text(f"✅ Matriz LMA Exportada a: {ruta_salida}"), bgcolor="#00ff41", open=True)
                page.update()

        btn_exportar = ft.ElevatedButton("📥 DESCARGAR MATRIZ LMA (CSV)", bgcolor="#00ff41", color="black", on_click=exportar_csv)
        
        filas_ams = [ft.DataRow(cells=[ft.DataCell(ft.Text(str(r['anio']))), ft.DataCell(ft.Text(str(r['lma']), color="#00ff41", weight="bold"))]) for r in data['tabla_ams']]
        tabla_ams_ui = ft.Container(
            content=ft.Column([
                ft.Row([ft.Text("Simultaneidad (Máximos Areales)", weight="bold", color="white"), btn_exportar], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Container(content=ft.Column([ft.DataTable(columns=[ft.DataColumn(ft.Text("Año")), ft.DataColumn(ft.Text("LMA Máxima (mm)"))], rows=filas_ams, heading_row_height=30, data_row_max_height=30)], scroll=ft.ScrollMode.ADAPTIVE), height=300)
            ]), bgcolor="#111111", border=ft.border.all(1, "#333333"), border_radius=10, padding=15
        )
        grafico_ams = ft.Container(content=ft.Image(src_base64=data['plot_ams'], fit=ft.ImageFit.CONTAIN), bgcolor="#0a0a0a", border=ft.border.all(1, "#333333"), border_radius=10, padding=10)

        # 4. El Motor Geo-Espacial (Time-Slider de Isoyetas + Envolvente)
        años_mapa = sorted([int(y) for y, v in data.get('mapas_isoyetas', {}).items() if v.get('b64')])
        
        if años_mapa:
            tiene_envolvente = 9999 in años_mapa
            años_reales = [y for y in años_mapa if y != 9999]
            
            min_y, max_y = min(años_reales), max(años_reales)
            
            # Arrancamos mostrando la joya de la corona si existe
            año_inicial = 9999 if tiene_envolvente else max_y
            img_mapa = ft.Image(src_base64=data['mapas_isoyetas'][str(año_inicial)]['b64'], fit=ft.ImageFit.CONTAIN, expand=True)
            
            def get_titulo_epicentro(year):
                info = data['mapas_isoyetas'][str(year)]
                if year == 9999:
                    return f"🌟 MAPA COMPUESTO: Envolvente de Susceptibilidad (LMA Máxima por Epicentro)"
                return f"📅 Fecha: {info['fecha']} | 🎯 Estación Crítica: {info['estacion']} | 💧 Lluvia Máx: {info['valor']:.1f} mm"

            lbl_fecha = ft.Text(get_titulo_epicentro(año_inicial), weight="bold", color="#ffdd00" if tiene_envolvente else "#00ff41", size=15)
            
            def on_slider_change(e):
                y_sel = int(e.control.value)
                
                if tiene_envolvente and y_sel > max_y:
                    img_mapa.src_base64 = data['mapas_isoyetas']["9999"]['b64']
                    lbl_fecha.value = get_titulo_epicentro(9999)
                    lbl_fecha.color = "#ffdd00"
                else:
                    nearest = min(años_reales, key=lambda x: abs(x - y_sel))
                    img_mapa.src_base64 = data['mapas_isoyetas'][str(nearest)]['b64']
                    
                    if nearest != y_sel:
                        lbl_fecha.value = f"{get_titulo_epicentro(nearest)} (Gap en {y_sel})"
                        lbl_fecha.color = "orange"
                    else:
                        lbl_fecha.value = get_titulo_epicentro(nearest)
                        lbl_fecha.color = "#00ff41"
                        
                img_mapa.update()
                lbl_fecha.update()
                
            # Configuración del Slider: Si hay envolvente, le sumamos 1 paso ficticio a la derecha
            val_max_slider = max_y + 1 if tiene_envolvente else max_y
            
            if min_y == max_y and not tiene_envolvente:
                slider_isoyetas = ft.Slider(min=min_y, max=max_y+1, divisions=1, value=min_y, disabled=True)
            else:
                # Usamos un label custom para que si llega al max+1 diga "ENVOLVENTE"
                def formatear_label(valor):
                    if tiene_envolvente and valor > max_y: return "MAPA COMPUESTO"
                    return str(int(valor))
                    
                slider_isoyetas = ft.Slider(
                    min=min_y, max=val_max_slider, 
                    divisions=val_max_slider - min_y, 
                    value=val_max_slider, 
                    label="{value}", 
                    on_change=on_slider_change, 
                    active_color="#ff0044" if tiene_envolvente else "#00ff41", 
                    inactive_color="#333333"
                )
            
            mapa_ui = ft.Container(
                content=ft.Column([
                    ft.Row([ft.Icon(ft.Icons.MAP, color="#1c75fa"), ft.Text("Motor Geo-Espacial de Isoyetas Regionales", weight="bold", size=18, color="white")]),
                    lbl_fecha,
                    slider_isoyetas,
                    ft.Container(content=img_mapa, height=650, alignment=ft.alignment.center)
                ]), bgcolor="#111111", border=ft.border.all(1, "#333333"), border_radius=10, padding=20
            )
        else:
            mapa_ui = ft.Container(content=ft.Text("Geometría de cuenca no disponible para mapeo espacial.", color="grey"), padding=20)

        return ft.Column([
            kpis, 
            grafico_cinematico, 
            ft.ResponsiveRow([
                ft.Column([grafico_ams], col={"md": 12, "lg": 7}), 
                ft.Column([tabla_ams_ui], col={"md": 12, "lg": 5})
            ]),
            mapa_ui
        ], scroll=ft.ScrollMode.ADAPTIVE, expand=True)

    def build_tab_matriz(): return ft.Container(content=bento_container, padding=15)

    # (Placeholders para las siguientes pestañas)
    # =========================================================================
    # PESTAÑA 3: MÉTODO EMPÍRICO (Aislado)
    # =========================================================================
    def build_tab_empirico():
        from core.arf_metodo_empirico import MotorEmpirico
        
        lbl_uswb = ft.Text("0.000", size=24, weight="bold", color="#1c75fa")
        lbl_temez = ft.Text("0.000", size=24, weight="bold", color="#9900ff")
        
        img_plot_area = ft.Image(src_base64="", fit=ft.ImageFit.CONTAIN, expand=True)
        img_plot_tiempo = ft.Image(src_base64="", fit=ft.ImageFit.CONTAIN, expand=True)
        
        dt_temporal = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Minutos", color="#00ff41")),
                ft.DataColumn(ft.Text("Horas", color="grey")),
                ft.DataColumn(ft.Text("USWB (TP-29)")),
                ft.DataColumn(ft.Text("Témez (España)"))
            ],
            rows=[], border=ft.border.all(1, "#333333"), heading_row_color="#1a1a1a"
        )
        
        slider_duracion = ft.Slider(min=1, max=24, divisions=23, value=24, label="{value} horas", active_color="#00ff41")
        
        def _kpi_empirico(titulo, texto_obj, color_borde):
            return ft.Container(
                content=ft.Column([ft.Text(titulo, color="grey", size=11, weight="bold"), texto_obj], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                bgcolor="#111111", border=ft.border.all(1, color_borde), border_radius=10, padding=15, expand=True
            )

        def ejecutar_empirico(e):
            area_cuenca = page.session.get("area_cuenca_km2")
            if not area_cuenca:
                page.snack_bar = ft.SnackBar(ft.Text("Área de la cuenca no encontrada. Ejecute Módulo 1."), bgcolor="#cc0000", open=True); page.update()
                return
                
            duracion = slider_duracion.value
            btn_calc_empirico.disabled = True
            page.update()

            def _task():
                res = MotorEmpirico.ejecutar_analisis(area_cuenca, duracion)
                if res["exito"]:
                    def _success():
                        lbl_uswb.value = f"{res['fra_uswb']:.3f}"
                        lbl_temez.value = f"{res['fra_temez']:.3f}"
                        
                        img_plot_area.src_base64 = res['plot1_b64']
                        img_plot_tiempo.src_base64 = res['plot2_b64']
                        
                        dt_temporal.rows.clear()
                        for fila in res['tabla_datos']:
                            dt_temporal.rows.append(ft.DataRow(cells=[
                                ft.DataCell(ft.Text(f"{fila['minutos']} min", weight="bold")),
                                ft.DataCell(ft.Text(f"{fila['horas']} h", color="grey")),
                                ft.DataCell(ft.Text(f"{fila['uswb']:.3f}", color="#1c75fa")),
                                ft.DataCell(ft.Text(f"{fila['temez']:.3f}", color="#9900ff"))
                            ]))
                            
                        btn_calc_empirico.disabled = False
                        page.update()
                    page.run_thread(_success)
                else:
                    def _fail():
                        page.snack_bar = ft.SnackBar(ft.Text(f"Error: {res['error']}"), bgcolor="#cc0000", open=True)
                        btn_calc_empirico.disabled = False; page.update()
                    page.run_thread(_fail)
                    
            threading.Thread(target=_task, daemon=True).start()

        btn_calc_empirico = ft.ElevatedButton("EVALUAR MÉTODOS EMPÍRICOS", icon=ft.Icons.CALCULATE, bgcolor="#00ff41", color="black", on_click=ejecutar_empirico)

        cuerpo_scroll = ft.Column([
            ft.Row([
                ft.Column([
                    ft.Text("MÉTODO 1: Formulaciones Empíricas (Benchmarking)", weight="bold", size=18, color="#00ff41"),
                    ft.Text("Evaluación estática (geométrica) para acotar físicamente el Factor de Reducción.", color="grey", size=12)
                ], expand=True),
                btn_calc_empirico
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            
            ft.Divider(color="#333333"),
            
            ft.Container(content=ft.Row([ft.Text("Duración (D):", color="white"), ft.Container(content=slider_duracion, expand=True)]), bgcolor="#111111", padding=10, border_radius=10, border=ft.border.all(1, "#333333")),
            
            # Quedan exclusivamente 2 Tarjetas KPI
            ft.Row([
                _kpi_empirico("FRA - USWB (TP-29)", lbl_uswb, "#1c75fa"), 
                _kpi_empirico("FRA - TÉMEZ (España)", lbl_temez, "#9900ff")
            ]),
            
            ft.Container(content=img_plot_area, bgcolor="#0a0a0a", border=ft.border.all(1, "#333333"), border_radius=10, padding=10),
            ft.Container(content=img_plot_tiempo, bgcolor="#0a0a0a", border=ft.border.all(1, "#333333"), border_radius=10, padding=10),
            
            ft.Row([ft.Icon(ft.Icons.TABLE_CHART, color="#00ff41"), ft.Text("Discretización Temporal (Cada 5 min)", weight="bold", color="white", size=14)]),
            ft.Container(content=ft.Column([dt_temporal], scroll=ft.ScrollMode.ADAPTIVE), border=ft.border.all(1, "#333333"), border_radius=10, height=400)
            
        ], spacing=15, scroll=ft.ScrollMode.ADAPTIVE)

        return ft.Container(content=cuerpo_scroll, padding=20, expand=True)
    # =========================================================================
    # FÁBRICA DE PESTAÑAS: ANÁLISIS DE FRECUENCIAS (Área Fija Espacial)
    # =========================================================================
    def crear_pestana_frecuencias(modo_calc: str, titulo_metodo: str, descripcion: str, color_tema: str):
        img_lluvia_tr = ft.Image(src_base64="", fit=ft.ImageFit.CONTAIN, expand=True)
        img_fra_tr = ft.Image(src_base64="", fit=ft.ImageFit.CONTAIN, expand=True)
        # Variables para el visor dinámico
        img_mapa_tr = ft.Image(src_base64="", fit=ft.ImageFit.CONTAIN, expand=True)
        lbl_mapa_tr = ft.Text("Riesgo Espacial: Superficie Kriging", weight="bold", color="white", size=14)
        
        estado_pestana = {"mapas": {}}
        tr_lista = [2, 5, 10, 20, 50, 100, 500, 1000, 10000]

        def on_tr_slider_change(e):
            idx = int(e.control.value)
            tr_sel = tr_lista[idx]
            if str(tr_sel) in estado_pestana["mapas"]:
                img_mapa_tr.src_base64 = estado_pestana["mapas"][str(tr_sel)]
                lbl_mapa_tr.value = f"Riesgo Espacial: Superficie Kriging (Tr {tr_sel} Años)"
                img_mapa_tr.update()
                lbl_mapa_tr.update()

        slider_tr = ft.Slider(
            min=0, max=len(tr_lista)-1, divisions=len(tr_lista)-1, 
            value=len(tr_lista)-1, 
            on_change=on_tr_slider_change, active_color=color_tema, inactive_color="#333333"
        )
        
        lbl_tr100 = ft.Text("---", size=24, weight="bold", color=color_tema)
        lbl_tr1000 = ft.Text("---", size=24, weight="bold", color="#ffdd00")
        lbl_tr10000 = ft.Text("---", size=24, weight="bold", color="#ff0044")
        
        dt_resultados = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Tr (Años)", color="#00ff41")),
                ft.DataColumn(ft.Text("LMA Areal (mm)", color="#1c75fa")),
                ft.DataColumn(ft.Text("Kriging Regional (mm)", color="#ffdd00")),
                ft.DataColumn(ft.Text("FRA Oficial", color="white", weight="bold"))
            ],
            rows=[], border=ft.border.all(1, "#333333"), heading_row_color="#1a1a1a"
        )
        
        dt_distribuciones = ft.DataTable(
            columns=[ft.DataColumn(ft.Text("Estación / Cuenca", color="grey")), ft.DataColumn(ft.Text("Mejor Ajuste (Best-Fit)", color="grey"))],
            rows=[], border=ft.border.all(1, "#222222")
        )

        def _kpi_freq(titulo, texto_obj, color_borde):
            return ft.Container(
                content=ft.Column([ft.Text(titulo, color="grey", size=11, weight="bold"), texto_obj], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                bgcolor="#111111", border=ft.border.all(1, color_borde), border_radius=10, padding=15, expand=True
            )
            
        contenedor_mapa = ft.Container(
            content=ft.Column([
                ft.Row([ft.Icon(ft.Icons.MAP, color=color_tema), lbl_mapa_tr]),
                slider_tr,
                ft.Container(content=img_mapa_tr, height=500, alignment=ft.alignment.center)
            ]), bgcolor="#111111", border=ft.border.all(1, color_tema), border_radius=10, padding=15, visible=False
        )

        def ejecutar_frecuencias_hilo(e):
            df_maestro = page.session.get("arf_matriz_maestra")
            ruta_hds = page.session.get("ruta_bd_activa")
            cuenca_wkt = page.session.get("cuenca_geom_wkt")
            
            if df_maestro is None or not ruta_hds or not cuenca_wkt:
                page.snack_bar = ft.SnackBar(ft.Text("⚠️ Falta Matriz LMA o Geometría. Ejecute la Fase 2."), bgcolor="#cc0000", open=True); page.update()
                return
                
            btn_calc_freq.disabled = True; page.update()

            def _task():
                try:
                    # Reconstruir df_ams (LMA Máxima Anual)
                    df_ams_lma = df_maestro[['LMA_Base']].groupby(df_maestro.index.year).max()
                    df_ams_lma.rename(columns={'LMA_Base': 'LMA_Max_Anual'}, inplace=True)
                    
                    # Recuperar coordenadas directamente de la sesión (sin tocar la BD)
                    claves = [col for col in df_maestro.columns if col != 'LMA_Base']
                    estaciones_dict = page.session.get("imput_station_files") or {}
                    df_indice = page.session.get("indice_bd_local")
                    
                    coords_dict = {}
                    if estaciones_dict:
                        for k, v in estaciones_dict.items():
                            coords_dict[str(k).upper()] = {"LATITUD": v.get("lat", 0), "LONGITUD": v.get("lon", 0)}
                    elif df_indice is not None and not df_indice.empty:
                        for _, row in df_indice.iterrows():
                            coords_dict[str(row.get('clave', '')).upper()] = {"LATITUD": row.get('lat', 0), "LONGITUD": row.get('lon', 0)}
                    
                    datos_coords = []
                    for c in claves:
                        c_upper = str(c).upper()
                        if c_upper in coords_dict:
                            datos_coords.append({"clave_estacion": c, "LATITUD": coords_dict[c_upper]["LATITUD"], "LONGITUD": coords_dict[c_upper]["LONGITUD"]})
                            
                    df_coords = pd.DataFrame(datos_coords)
                    if not df_coords.empty:
                        df_coords.set_index('clave_estacion', inplace=True)

                    # Ejecutar el Motor Espacial
                    res = MotorFrecuencias.ejecutar_analisis_espacial(df_maestro, df_ams_lma, df_coords, cuenca_wkt, modo=modo_calc)
                    
                    if res["exito"]:
                        def _success():
                            # Tablas
                            dt_resultados.rows.clear()
                            for fila in res['tabla']:
                                dt_resultados.rows.append(ft.DataRow(cells=[
                                    ft.DataCell(ft.Text(str(fila['tr']), weight="bold")),
                                    ft.DataCell(ft.Text(str(fila['lma']), color="#1c75fa")),
                                    ft.DataCell(ft.Text(str(fila['kriging']), color="#ffdd00")),
                                    ft.DataCell(ft.Text(str(fila['fra']), color="white", weight="bold"))
                                ]))
                                if fila['tr'] == 100: lbl_tr100.value = str(fila['fra'])
                                if fila['tr'] == 1000: lbl_tr1000.value = str(fila['fra'])
                                if fila['tr'] == 10000: lbl_tr10000.value = str(fila['fra'])
                                
                            dt_distribuciones.rows.clear()
                            for est, nombre_dist in res["distribuciones"].items():
                                col_text = color_tema if est == "LMA Areal (Cuenca)" else "white"
                                dt_distribuciones.rows.append(ft.DataRow(cells=[
                                    ft.DataCell(ft.Text(est, color=col_text, weight="bold")),
                                    ft.DataCell(ft.Text(nombre_dist, color="grey"))
                                ]))

                            # Gráficos
                            img_lluvia_tr.src_base64 = res['plot_lluvia']
                            img_fra_tr.src_base64 = res['plot_arf']
                            
                            if res.get('mapas_tr'):
                                estado_pestana["mapas"] = res['mapas_tr']
                                # Inicializamos la vista en el Tr 10,000 (índice 8)
                                img_mapa_tr.src_base64 = res['mapas_tr']["10000"]
                                lbl_mapa_tr.value = "Riesgo Espacial: Superficie Kriging (Tr 10000 Años)"
                                slider_tr.value = 8
                                contenedor_mapa.visible = True
                            
                            btn_calc_freq.disabled = False
                            page.update()
                        page.run_thread(_success)
                    else:
                        def _fail():
                            page.snack_bar = ft.SnackBar(ft.Text(f"❌ Error Cálculo: {res['error']}"), bgcolor="#cc0000", open=True)
                            btn_calc_freq.disabled = False; page.update()
                        page.run_thread(_fail)
                        
                except Exception as ex:
                    import traceback; traceback.print_exc()
                    def _crash():
                        page.snack_bar = ft.SnackBar(ft.Text(f"❌ Error Crítico: {str(ex)}"), bgcolor="#cc0000", open=True)
                        btn_calc_freq.disabled = False; page.update()
                    page.run_thread(_crash)
                    
            threading.Thread(target=_task, daemon=True).start()

        btn_calc_freq = ft.ElevatedButton(f"EJECUTAR CÁLCULO ({modo_calc.upper()})", icon=ft.Icons.AUTO_GRAPH, bgcolor=color_tema, color="black", on_click=ejecutar_frecuencias_hilo)

        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Column([
                        ft.Text(titulo_metodo, weight="bold", size=18, color=color_tema),
                        ft.Text(descripcion, color="grey", size=12)
                    ], expand=True),
                    btn_calc_freq
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                
                ft.Divider(color="#333333"),
                
                ft.Row([
                    _kpi_freq("FRA Diseño (100 Años)", lbl_tr100, "#333333"),
                    _kpi_freq("FRA Crítico (1,000 Años)", lbl_tr1000, "#ffdd00"),
                    _kpi_freq("FRA Extremo (10,000 Años)", lbl_tr10000, "#ff0044"),
                ]),
                
                ft.ResponsiveRow([
                    ft.Container(content=img_lluvia_tr, bgcolor="#0a0a0a", border=ft.border.all(1, "#333333"), border_radius=10, padding=10, col={"md": 12, "lg": 6}),
                    ft.Container(content=img_fra_tr, bgcolor="#0a0a0a", border=ft.border.all(1, "#333333"), border_radius=10, padding=10, col={"md": 12, "lg": 6})
                ]),
                
                contenedor_mapa,
                
                ft.ResponsiveRow([
                    ft.Container(content=ft.Column([ft.Text("Tabla de Cálculo y Truncamiento", color="white", weight="bold"), dt_resultados], scroll=ft.ScrollMode.ADAPTIVE), border=ft.border.all(1, "#333333"), border_radius=10, col={"md": 12, "lg": 8}, height=350),
                    ft.Container(content=ft.Column([ft.Text("Modelos 'Best-Fit' Seleccionados", color="white", weight="bold"), dt_distribuciones], scroll=ft.ScrollMode.ADAPTIVE), border=ft.border.all(1, "#333333"), border_radius=10, col={"md": 12, "lg": 4}, height=350)
                ])
            ], spacing=15, scroll=ft.ScrollMode.ADAPTIVE), padding=20, expand=True
        )

    # =========================================================================
    # PESTAÑA 6: MÉTODO I.I. UNAM (Tormenta Centrada Topológica)
    # =========================================================================
    def build_tab_unam():
        from core.arf_metodo_unam import MotorUNAM
        import sqlite3
        import pandas as pd

        # Componentes Visuales
        img_mapa_tormenta = ft.Image(src_base64="", fit=ft.ImageFit.CONTAIN, expand=True)
        img_plot_dad = ft.Image(src_base64="", fit=ft.ImageFit.CONTAIN, expand=True)
        
        dt_envolvente = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Área de Cobertura (km²)", color="#00ff41")),
                ft.DataColumn(ft.Text("FRA (Envolvente de Diseño)", color="#ff0044"))
            ],
            rows=[], border=ft.border.all(1, "#333333"), heading_row_color="#1a1a1a"
        )
        
        estado_unam = {"mapas": {}}
        
        def actualizar_mapa(e):
            id_tormenta = e.control.value
            if id_tormenta in estado_unam["mapas"]:
                img_mapa_tormenta.src_base64 = estado_unam["mapas"][id_tormenta]
                img_mapa_tormenta.update()

        dd_tormentas = ft.Dropdown(
            label="Inspeccionar Super-Celda Histórica",
            options=[],
            on_change=actualizar_mapa,
            color="#00ff41", border_color="#333333", expand=True
        )

        def ejecutar_unam(e):
            df_maestro = page.session.get("arf_matriz_maestra")
            ruta_hds = page.session.get("ruta_bd_activa")
            cuenca_wkt = page.session.get("cuenca_geom_wkt")
            
            if df_maestro is None or not ruta_hds or not cuenca_wkt:
                page.snack_bar = ft.SnackBar(ft.Text("⚠️ Faltan datos base. Ejecute la Fase 2 (Matriz LMA) primero."), bgcolor="#cc0000", open=True)
                page.update()
                return
                
            btn_calc_unam.disabled = True
            btn_calc_unam.text = "ESCANEANDO HISTORIAL (Buscando Anomalías...)"
            page.update()

            def _task():
                try:
                    # Extraer coordenadas de la base de datos
                    claves = [col for col in df_maestro.columns if col != 'LMA_Base']
                    conn = sqlite3.connect(ruta_hds)
                    placeholders = ','.join(['?'] * len(claves))
                    q = f"SELECT Clave as clave_estacion, Latitud as LATITUD, Longitud as LONGITUD FROM estaciones_encontradas WHERE Clave IN ({placeholders})"
                    df_coords = pd.read_sql_query(q, conn, params=claves)
                    df_coords.set_index('clave_estacion', inplace=True)
                    conn.close()

                    # Disparar el motor avanzado
                    res = MotorUNAM.ejecutar_analisis_tormenta_centrada(df_maestro, df_coords, cuenca_wkt)
                    
                    if res["exito"]:
                        def _success():
                            # 1. Poblar Dropdown de Mapas
                            dd_tormentas.options.clear()
                            estado_unam["mapas"].clear()
                            
                            for mapa in res["mapas"]:
                                dd_tormentas.options.append(ft.dropdown.Option(mapa["id"]))
                                estado_unam["mapas"][mapa["id"]] = mapa["b64"]
                            
                            if res["mapas"]:
                                primer_id = res["mapas"][0]["id"]
                                dd_tormentas.value = primer_id
                                img_mapa_tormenta.src_base64 = estado_unam["mapas"][primer_id]
                            
                            # 2. Renderizar Gráfico DAD
                            img_plot_dad.src_base64 = res["plot_dad"]
                            
                            # 3. Llenar Tabla de Envolvente
                            dt_envolvente.rows.clear()
                            for fila in res["tabla_envolvente"]:
                                dt_envolvente.rows.append(ft.DataRow(cells=[
                                    ft.DataCell(ft.Text(f"{fila['area']:.2f}")),
                                    ft.DataCell(ft.Text(f"{fila['fra']:.4f}", weight="bold", color="white"))
                                ]))
                            
                            # Mostrar resultados y resetear botón
                            contenedor_resultados.visible = True
                            btn_calc_unam.text = "ESCANEAR SUPER-CELDAS Y GENERAR CURVAS DAD"
                            btn_calc_unam.disabled = False
                            page.snack_bar = ft.SnackBar(ft.Text("✅ Análisis DAD y Kriging completado."), bgcolor="#00ff41", open=True)
                            page.update()
                        page.run_thread(_success)
                    else:
                        def _fail():
                            page.snack_bar = ft.SnackBar(ft.Text(f"❌ Error Cálculo UNAM: {res['error']}"), bgcolor="#cc0000", open=True)
                            btn_calc_unam.text = "ESCANEAR SUPER-CELDAS Y GENERAR CURVAS DAD"
                            btn_calc_unam.disabled = False; page.update()
                        page.run_thread(_fail)
                except Exception as ex:
                    import traceback; traceback.print_exc()
                    def _crash():
                        page.snack_bar = ft.SnackBar(ft.Text(f"❌ Error Crítico: {str(ex)}"), bgcolor="#cc0000", open=True)
                        btn_calc_unam.text = "ESCANEAR SUPER-CELDAS Y GENERAR CURVAS DAD"
                        btn_calc_unam.disabled = False; page.update()
                    page.run_thread(_crash)

            threading.Thread(target=_task, daemon=True).start()

        btn_calc_unam = ft.ElevatedButton(
            "ESCANEAR SUPER-CELDAS Y GENERAR CURVAS DAD", 
            icon=ft.Icons.STORM, bgcolor="#ff0044", color="white", 
            on_click=ejecutar_unam
        )
        
        # Contenedor de Resultados (Oculto al inicio)
        contenedor_resultados = ft.Container(
            content=ft.Column([
                ft.ResponsiveRow([
                    # Panel Izquierdo: Radar de la Tormenta
                    ft.Container(
                        content=ft.Column([
                            ft.Row([ft.Icon(ft.Icons.RADAR, color="#ff0044"), ft.Text("Radar de Isoyetas Históricas", color="white", weight="bold")]),
                            ft.Row([dd_tormentas]),
                            ft.Container(content=img_mapa_tormenta, height=450, alignment=ft.alignment.center)
                        ]),
                        bgcolor="#111111", border=ft.border.all(1, "#ff0044"), border_radius=10, padding=15, col={"md": 12, "lg": 5}
                    ),
                    
                    # Panel Derecho: Gráfico DAD
                    ft.Container(
                        content=ft.Column([
                            ft.Row([ft.Icon(ft.Icons.SHOW_CHART, color="#00ff41"), ft.Text("Curvas Depth-Area (DAD)", color="white", weight="bold")]),
                            ft.Container(content=img_plot_dad, height=480, alignment=ft.alignment.center)
                        ]),
                        bgcolor="#111111", border=ft.border.all(1, "#333333"), border_radius=10, padding=15, col={"md": 12, "lg": 7}
                    )
                ]),
                
                # Matriz Exportable
                ft.Container(
                    content=ft.Column([
                        ft.Text("Matriz de la Envolvente de Diseño (Exportable a HEC-HMS / Paired Data)", weight="bold", color="white"),
                        ft.Container(content=ft.Column([dt_envolvente], scroll=ft.ScrollMode.ADAPTIVE), height=250)
                    ]),
                    bgcolor="#111111", border=ft.border.all(1, "#333333"), border_radius=10, padding=15, margin=ft.margin.only(top=10)
                )
            ]),
            visible=False
        )

        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Column([
                        ft.Text("MÉTODO 3: Inst. Ingeniería UNAM (Tormenta Centrada Topológica)", weight="bold", size=18, color="#ff0044"),
                        ft.Text("Identifica mediante IA las 3 tormentas con mayor masa hídrica en la historia. Mide el área de sus isoyetas y genera la Envolvente DAD de Diseño.", color="grey", size=12)
                    ], expand=True),
                    btn_calc_unam
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Divider(color="#333333"),
                contenedor_resultados
            ], scroll=ft.ScrollMode.ADAPTIVE), padding=20, expand=True
        )
    def build_tab_comparativa(): return ft.Container(alignment=ft.alignment.center, content=ft.Text("Pestaña 6 en desarrollo...", color="grey"))

    # =========================================================================
    # HILOS DE EJECUCIÓN SECUENCIAL DEL PANEL DE CONTROL
    # =========================================================================
    def update_progreso(val):
        progreso_arf.value = val; page.update()

    def ejecutar_fase1_aduana(e):
        ruta_hds = page.session.get("ruta_bd_activa")
        claves_objetivo = page.session.get("claves_objetivo")
        if not ruta_hds or not os.path.exists(ruta_hds) or not claves_objetivo:
            page.snack_bar = ft.SnackBar(ft.Text("⚠️ Faltan datos o BD. Ejecute Módulo 1 y 2."), bgcolor="#cc0000", open=True); page.update()
            return
            
        btn_fase1.disabled = True; progreso_arf.visible = True; progreso_arf.value = 0; page.update()

        def _task():
            try:
                stats = MotorFiltradoARF.procesar_gpkg_arf(ruta_hds, claves_objetivo, update_progreso)
                page.session.set("arf_stats_continuidad", stats)
                
                dt_continuidad.rows.clear()
                for s in stats:
                    color_text = "#00ff41" if s["Integridad_%"] > 80 else ("orange" if s["Integridad_%"] > 50 else "#cc0000")
                    dt_continuidad.rows.append(ft.DataRow(cells=[
                        ft.DataCell(ft.Text(s["Clave"], weight="bold")), ft.DataCell(ft.Text(f"{s['Inicio']} - {s['Fin']}", size=11)),
                        ft.DataCell(ft.Text(str(s["Dias_Efectivos"]))), ft.DataCell(ft.Text(str(s["Max_Gap_Dias"]), color="red" if s["Max_Gap_Dias"] > 30 else "white")),
                        ft.DataCell(ft.Text(f"{s['Integridad_%']}%", color=color_text))
                    ]))
                
                def _success():
                    led_filtro.color = "#00ff41"
                    btn_fase2.disabled = False  # DESBLOQUEA PASO 2
                    btn_fase2.color = "#00ff41" # Ilumina el botón 2
                    page.snack_bar = ft.SnackBar(ft.Text(f"✅ Fase 1 Completada."), bgcolor="#00ff41", open=True)
                    progreso_arf.visible = False; btn_fase1.disabled = False
                    page.update()
                page.run_thread(_success)

            except Exception as ex:
                traceback.print_exc()
                msg = str(ex)
                def _fail():
                    page.snack_bar = ft.SnackBar(ft.Text(f"❌ Error Fase 1: {msg}"), bgcolor="#cc0000", open=True)
                    btn_fase1.disabled = False; progreso_arf.visible = False; page.update()
                page.run_thread(_fail)
        threading.Thread(target=_task, daemon=True).start()

    def ejecutar_fase2_lma(e):
        stats = page.session.get("arf_stats_continuidad")
        ruta_hds = page.session.get("ruta_bd_activa")
        area_cuenca = page.session.get("area_cuenca_km2") or 1.0
        
        # --- PARCHE WKT: Auto-Reconstrucción de Geometría ---
        cuenca_wkt = page.session.get("cuenca_geom_wkt") or ""
        if not cuenca_wkt:
            cuenca_geojson = page.session.get("cuenca_geojson")
            if cuenca_geojson:
                try:
                    import json
                    from shapely.geometry import shape
                    from shapely.ops import unary_union
                    
                    if isinstance(cuenca_geojson, str):
                        cuenca_geojson = json.loads(cuenca_geojson)
                        
                    if "features" in cuenca_geojson:
                        polys = [shape(f["geometry"]) for f in cuenca_geojson["features"]]
                        cuenca_wkt = unary_union(polys).wkt
                    elif "geometry" in cuenca_geojson:
                        cuenca_wkt = shape(cuenca_geojson["geometry"]).wkt
                    else:
                        cuenca_wkt = shape(cuenca_geojson).wkt
                        
                    page.session.set("cuenca_geom_wkt", cuenca_wkt) # Persistir para Fases posteriores
                except Exception as ex:
                    print(f"⚠️ Fallo convirtiendo GeoJSON a WKT: {ex}")

        if not stats or not ruta_hds:
            page.snack_bar = ft.SnackBar(ft.Text("⚠️ Faltan datos de Continuidad. Ejecute Paso 1."), bgcolor="#cc0000", open=True); page.update()
            return
            
        # --- EXTRACCIÓN DE COORDENADAS DESDE MEMORIA RAM ---
        estaciones_dict = page.session.get("imput_station_files") or {}
        df_indice = page.session.get("indice_bd_local")
        
        coords_dict = {}
        if estaciones_dict:
            for k, v in estaciones_dict.items():
                coords_dict[k] = {"LATITUD": v.get("lat", 0), "LONGITUD": v.get("lon", 0)}
        elif df_indice is not None and not df_indice.empty:
            for _, row in df_indice.iterrows():
                coords_dict[str(row.get('clave', '')).upper()] = {"LATITUD": row.get('lat', 0), "LONGITUD": row.get('lon', 0)}

        btn_fase2.disabled = True
        progreso_arf.visible = True
        progreso_arf.value = None
        page.update()

        def _task():
            exito, data = GestorMatricesARF.construir_matriz_dinamica(ruta_hds, stats, area_cuenca, cuenca_wkt, coordenadas_extra=coords_dict, umbral_lluvia=0.5)
            if exito:
                page.session.set("arf_matriz_maestra", data["matriz_maestra"])
                def _success():
                    try:
                        # 1. Reemplazo y Repintado Forzoso
                        bento_container.content = renderizar_bento_box(data)
                        bento_container.alignment = None # Elimina el centrado del texto de espera
                        bento_container.update() # <--- OBLIGA A REFRESCAR LA PANTALLA
                        
                        # 2. Actualización de LEDs y Botones
                        led_grid.color = "#00ff41"
                        led_grid.update()
                        
                        btn_fase3.disabled = False
                        btn_fase3.color = "#00ff41" 
                        btn_fase3.update()
                        
                        tabs_principales.selected_index = 1
                        tabs_principales.update()
                        
                        btn_fase2.disabled = False
                        btn_fase2.update()
                        
                        progreso_arf.visible = False
                        progreso_arf.update()
                        
                        page.snack_bar = ft.SnackBar(ft.Text("✅ Matriz Espacial y Mapas Generados."), bgcolor="#00ff41", open=True)
                        page.update()
                    except Exception as ui_ex:
                        import traceback; traceback.print_exc()
                        page.snack_bar = ft.SnackBar(ft.Text(f"❌ Error interno de UI: {str(ui_ex)}"), bgcolor="#cc0000", open=True)
                        btn_fase2.disabled = False; progreso_arf.visible = False; page.update()
                page.run_thread(_success)
            else:
                def _fail():
                    page.snack_bar = ft.SnackBar(ft.Text(f"❌ Error Fase 2: {data}"), bgcolor="#cc0000", open=True)
                    btn_fase2.disabled = False; progreso_arf.visible = False; page.update()
                page.run_thread(_fail)
        threading.Thread(target=_task, daemon=True).start()

    # =========================================================================
    # PANEL DE CONTROL E INTERFAZ MAESTRA
    # =========================================================================
    # Botones con la misma estética, centralizados e interbloqueados.
    style_btn = ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=5))
    
    btn_fase1 = ft.ElevatedButton("1. FILTRADO MASIVO", icon=ft.Icons.SHIELD, bgcolor="#1a1a1a", color="#00ff41", style=style_btn, on_click=ejecutar_fase1_aduana)
    btn_fase2 = ft.ElevatedButton("2. GENERAR LMA", icon=ft.Icons.GRID_ON, bgcolor="#1a1a1a", color="grey", disabled=True, style=style_btn, on_click=ejecutar_fase2_lma)
    btn_fase3 = ft.ElevatedButton("3. CALCULAR ARF", icon=ft.Icons.ROCKET_LAUNCH, bgcolor="#1a1a1a", color="grey", disabled=True, style=style_btn, on_click=lambda e: (setattr(tabs_principales, 'selected_index', 3), page.update()))
    
    panel_izquierdo = ft.Container(
        width=200, # Ancho reducido a petición tuya
        padding=15,
        bgcolor="#050505",
        border=ft.border.only(right=ft.border.BorderSide(1, "#333333")),
        content=ft.Column([
            ft.Text("PANEL DE CONTROL", weight="bold", color="white", size=13),
            ft.Divider(color="#333333"),
            
            btn_fase1,
            ft.Row([led_filtro, ft.Text("Criba Inmutable", size=11, color="grey")]),
            ft.Container(height=10),
            
            btn_fase2,
            ft.Row([led_grid, ft.Text("Álgebra Vectorial", size=11, color="grey")]),
            ft.Container(height=10),
            
            btn_fase3,
            ft.Row([led_prob, ft.Text("Tr 10,000 & Cópulas", size=11, color="grey")]),
            
            ft.Container(height=15),
            progreso_arf
        ])
    )

    tabs_principales = ft.Tabs(
        selected_index=0,
        animation_duration=300,
        tabs=[
            ft.Tab(text="1. Continuidad", content=build_tab_calidad()),
            ft.Tab(text="2. LLuvia Media Areal", content=build_tab_matriz()),
            ft.Tab(text="3. Método Empírico", content=build_tab_empirico()),
            
            # --- NUEVO: LAS DOS CABEZAS DEL MODELO ESPACIAL ---
            ft.Tab(text="4. Frec. (Homogéneo)", content=crear_pestana_frecuencias(
                modo_calc='homogeneo', 
                titulo_metodo="MÉTODO 2A: Frecuencias Espacial (Forzamiento Regional)", 
                descripcion="Modelo Conservador. Fuerza a todas las estaciones a adoptar la familia estadística de la LMA para generar una interpolación suave sin gradientes anómalos.", 
                color_tema="#1c75fa")
            ),
            ft.Tab(text="5. Frec. (Heterogéneo)", content=crear_pestana_frecuencias(
                modo_calc='heterogeneo', 
                titulo_metodo="MÉTODO 2B: Geo-Estocástico (Microclimas)", 
                descripcion="Modelo de Alta Resolución. Cada estación busca su propio Best-Fit, permitiendo mapear barreras orográficas e islas de calor en extremos probabilísticos.", 
                color_tema="#9900ff")
            ),
            
            ft.Tab(text="6. I.I. UNAM", content=build_tab_unam()),
            ft.Tab(text="7. Comparativa", content=build_tab_comparativa()),
        ],
        indicator_color="#00ff41", label_color="#00ff41", unselected_label_color="grey"
    )

    panel_derecho = ft.Container(expand=True, content=tabs_principales)

    return ft.Row([panel_izquierdo, panel_derecho], expand=True, spacing=0)