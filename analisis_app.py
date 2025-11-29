import flet as ft
import Analisis
import lluvias  
import analisis_cuenca 
import pandas as pd
import base64   
import os 

# Píxel transparente para inicializar imágenes vacías
TRANSPARENT_PIXEL = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

def build_analisis_view(page: ft.Page, on_back_to_menu):
    
    # --- ESTADO DE SESIÓN ---
    if not hasattr(page.session, "df_procesado"): page.session.df_procesado = None
    if not hasattr(page.session, "df_filtrado"): page.session.df_filtrado = None
    if not hasattr(page.session, "df_estadisticas"): page.session.df_estadisticas = None
    
    # Inicializar TODAS las claves para evitar errores de getattr
    # Estas son las claves EXACTAS que usan los botones de guardado
    keys = [
        "b64_hist", "b64_series", "b64_violin",                         # Gráficos Generales
        "b64_max_annual", "b64_acf", "b64_weibull", "b64_dist_comp",    # Lluvias
        "df_maximos_mensuales", "df_weibull", "best_fit_name", 
        "df_homogeneidad", "df_acf", "df_ajustes",
        "cuenca_plot1_b64", "cuenca_plot2_b64", "cuenca_plot3_b64", "cuenca_plot4_b64", # Cuenca
        "df_altura", "df_intensidad"
    ]
    for k in keys:
        if not hasattr(page.session, k): setattr(page.session, k, None)

    # --- HELPERS DE GUARDADO ROBUSTOS ---
    
    def save_img_safe(e, session_key):
        """Busca la imagen en la sesión por su CLAVE y la guarda."""
        b64_data = getattr(page.session, session_key)
        
        if not b64_data:
            page.snack_bar = ft.SnackBar(ft.Text(f"❌ Error: No se encontró la imagen '{session_key}'."), bgcolor="red", open=True)
            page.update()
            return

        if e.path:
            try:
                path = e.path if e.path.endswith(".png") else f"{e.path}.png"
                with open(path, "wb") as f:
                    f.write(base64.b64decode(b64_data))
                page.snack_bar = ft.SnackBar(ft.Text(f"Imagen guardada: {os.path.basename(path)}"), bgcolor="green", open=True)
            except Exception as ex:
                page.snack_bar = ft.SnackBar(ft.Text(f"Error al escribir archivo: {ex}"), bgcolor="red", open=True)
            page.update()

    def save_csv_safe(e, session_key):
        """Busca el DataFrame en la sesión por su CLAVE y lo guarda."""
        df = getattr(page.session, session_key)
        
        if df is None:
            page.snack_bar = ft.SnackBar(ft.Text(f"❌ Error: No se encontró la tabla '{session_key}'."), bgcolor="red", open=True)
            page.update()
            return

        if e.path:
            try:
                path = e.path if e.path.endswith(".csv") else f"{e.path}.csv"
                df.to_csv(path)
                page.snack_bar = ft.SnackBar(ft.Text(f"Tabla guardada: {os.path.basename(path)}"), bgcolor="green", open=True)
            except Exception as ex:
                page.snack_bar = ft.SnackBar(ft.Text(f"Error al escribir CSV: {ex}"), bgcolor="red", open=True)
            page.update()

    def dataframe_to_datatable(df, max_filas=10):
        if df is None or df.empty: return ft.Text("Sin datos")
        return ft.DataTable(
            columns=[ft.DataColumn(ft.Text(str(c))) for c in ["Index"] + list(df.columns)],
            rows=[ft.DataRow([ft.DataCell(ft.Text(str(i)))] + [ft.DataCell(ft.Text(str(r[c]))) for c in df.columns]) for i,r in df.head(max_filas).iterrows()],
            border=ft.border.all(1, "#333333")
        )

    def stats_to_datatable(df):
        if df is None: return ft.Text("Sin estadísticas")
        df_reset = df.reset_index().rename(columns={'index': 'Métrica'})
        return ft.DataTable(
            columns=[ft.DataColumn(ft.Text(c)) for c in df_reset.columns],
            rows=[ft.DataRow([ft.DataCell(ft.Text(str(r[c]))) for c in df_reset.columns]) for _,r in df_reset.iterrows()],
            border=ft.border.all(1, "#333333")
        )

    def safe_img(session_key, h=400):
        val = getattr(page.session, session_key)
        return ft.Image(src_base64=val if val else TRANSPARENT_PIXEL, height=h, fit=ft.ImageFit.CONTAIN)

    # --- PICKERS (Vinculados por NOMBRE DE CLAVE EXACTO) ---
    csv_picker = ft.FilePicker()
    
    # Gráficos Generales (Paso 4)
    pk_hist = ft.FilePicker(on_result=lambda e: save_img_safe(e, "b64_hist"))
    pk_ser = ft.FilePicker(on_result=lambda e: save_img_safe(e, "b64_series"))
    pk_vio = ft.FilePicker(on_result=lambda e: save_img_safe(e, "b64_violin"))
    
    # Lluvias (Paso 5) - CLAVES CORREGIDAS
    pk_max_an = ft.FilePicker(on_result=lambda e: save_img_safe(e, "b64_max_annual"))
    pk_acf_img = ft.FilePicker(on_result=lambda e: save_img_safe(e, "b64_acf"))
    pk_weibull = ft.FilePicker(on_result=lambda e: save_img_safe(e, "b64_weibull"))
    pk_dist = ft.FilePicker(on_result=lambda e: save_img_safe(e, "b64_dist_comp"))
    
    pk_homo_csv = ft.FilePicker(on_result=lambda e: save_csv_safe(e, "df_homogeneidad"))
    pk_acf_csv = ft.FilePicker(on_result=lambda e: save_csv_safe(e, "df_acf"))
    pk_ajustes_csv = ft.FilePicker(on_result=lambda e: save_csv_safe(e, "df_ajustes"))
    pk_wei_csv = ft.FilePicker(on_result=lambda e: save_csv_safe(e, "df_weibull"))
    pk_mx_csv = ft.FilePicker(on_result=lambda e: save_csv_safe(e, "df_maximos_mensuales"))
    
    # Cuenca (Paso 6) - CLAVES CORREGIDAS
    pk_alt_csv = ft.FilePicker(on_result=lambda e: save_csv_safe(e, "df_altura"))
    pk_int_csv = ft.FilePicker(on_result=lambda e: save_csv_safe(e, "df_intensidad"))
    pk_p1 = ft.FilePicker(on_result=lambda e: save_img_safe(e, "cuenca_plot1_b64"))
    pk_p2 = ft.FilePicker(on_result=lambda e: save_img_safe(e, "cuenca_plot2_b64"))
    pk_p3 = ft.FilePicker(on_result=lambda e: save_img_safe(e, "cuenca_plot3_b64"))
    pk_p4 = ft.FilePicker(on_result=lambda e: save_img_safe(e, "cuenca_plot4_b64"))

    page.overlay.extend([csv_picker, pk_hist, pk_ser, pk_vio, pk_max_an, pk_acf_img, pk_weibull, pk_dist, pk_homo_csv, pk_acf_csv, pk_ajustes_csv, pk_wei_csv, pk_mx_csv, pk_alt_csv, pk_int_csv, pk_p1, pk_p2, pk_p3, pk_p4])

    # --- UI ELEMENTS ---
    csv_path_txt = ft.Text("No seleccionado", color="orange")
    
    # 1. Carga
    data_table_raw = ft.Column(scroll=ft.ScrollMode.AUTO, height=300)
    
    # 2. Filtro
    chk_c1 = ft.Checkbox(label="Eliminar Nulos Totales", value=True)
    chk_c2 = ft.Checkbox(label="Eliminar Baja Densidad (< 5 Vecinos)", value=True)
    chk_c3 = ft.Checkbox(label="Eliminar Ruido (Valor > Max Vecinos)", value=True)
    data_table_filt = ft.Column(scroll=ft.ScrollMode.AUTO, height=300)
    
    # 3. Stats
    stats_table_container = ft.Column()
    
    # 4. Graficos
    img_h = safe_img("b64_hist"); img_s = safe_img("b64_series"); img_v = safe_img("b64_violin")
    
    # 5. Lluvias
    tbl_homo, tbl_acf, tbl_ajustes, tbl_wei, tbl_mx = ft.Column(), ft.Column(), ft.Column(), ft.Column(), ft.Column()
    img_max_an = safe_img("b64_max_annual")
    img_acf = safe_img("b64_acf")
    img_wei = safe_img("b64_weibull")
    img_dist = safe_img("b64_dist_comp")
    
    # 6. Cuenca
    txt_best_fit = ft.Text("No detectado", weight="bold", color="#00ff41")
    inp_ylim_h = ft.TextField(label="Límite Y Altura", width=150)
    inp_ylim_i = ft.TextField(label="Límite Y Intensidad", width=150)
    tbl_alt, tbl_int = ft.Column(), ft.Column()
    img_c1 = safe_img("cuenca_plot1_b64")
    img_c2 = safe_img("cuenca_plot2_b64")
    img_c3 = safe_img("cuenca_plot3_b64")
    img_c4 = safe_img("cuenca_plot4_b64")
    log_cuenca = ft.Markdown()

    # --- LOGICA ---
    def on_load(e):
        if e.files:
            csv_path_txt.value = e.files[0].name; csv_path_txt.color="green"
            df = Analisis.procesar_datos(e.files[0].path)
            if df is not None:
                page.session.df_procesado = df
                data_table_raw.controls = [ft.Text("Datos Cargados:"), dataframe_to_datatable(df)]
                btn_nav_filt.disabled = False
            page.update()

    def run_filter(e):
        if page.session.df_procesado is None: return
        df = Analisis.filtrar_datos(page.session.df_procesado, chk_c1.value, chk_c2.value, chk_c3.value)
        if df is not None:
            page.session.df_filtrado = df
            data_table_filt.controls = [ft.Text(f"Filtrados: {len(df)} registros"), dataframe_to_datatable(df)]
            btn_nav_stats.disabled = False
        page.update()

    def run_stats(e):
        if page.session.df_filtrado is None: return
        st = Analisis.analizar_estadisticas(page.session.df_filtrado)
        if st is not None:
            page.session.df_estadisticas = st
            stats_table_container.controls = [ft.Text("Tabla Comparativa:", weight="bold", size=16, color="#00ff41"), stats_to_datatable(st)]
            btn_nav_graph.disabled = False
            rail.selected_index = 2; change_view()
        page.update()

    def run_graphs(e):
        if page.session.df_filtrado is None: return
        imgs = Analisis.generar_graficos(page.session.df_filtrado, page.session.df_estadisticas)
        if imgs:
            # Asignación explícita de claves
            page.session.b64_hist = imgs['hist']
            page.session.b64_series = imgs['series']
            page.session.b64_violin = imgs['violin']
            
            # Actualizar controles visuales
            img_h.src_base64 = imgs['hist']
            img_s.src_base64 = imgs['series']
            img_v.src_base64 = imgs['violin']
            
            btn_nav_lluvias.disabled = False
            rail.selected_index = 3; change_view()
        page.update()

    def run_lluvias(e):
        if page.session.df_filtrado is None: return
        res = lluvias.analizar_eventos_lluvia(page.session.df_filtrado)
        if res:
            # --- MAPPING EXPLÍCITO (Solución del Error) ---
            # Asignamos los resultados de la lógica a las claves exactas que usan los pickers
            page.session.b64_max_annual = res["max_annual_series_b64"]
            page.session.b64_acf = res["acf_plot_b64"]
            page.session.b64_weibull = res["weibull_plot_b64"]
            page.session.b64_dist_comp = res["dist_comparison_b64"]
            
            page.session.df_homogeneidad = res["df_homogeneidad"]
            page.session.df_acf = res["df_acf"]
            page.session.df_ajustes = res["df_ajustes"]
            page.session.df_weibull = res["df_weibull"]
            page.session.df_maximos_mensuales = res["df_maximos_mensuales"]
            page.session.best_fit_name = res["best_fit_name"]

            # UI Updates
            tbl_homo.controls = [dataframe_to_datatable(res["df_homogeneidad"])]
            tbl_acf.controls = [dataframe_to_datatable(res["df_acf"])]
            tbl_ajustes.controls = [dataframe_to_datatable(res["df_ajustes"])]
            tbl_wei.controls = [dataframe_to_datatable(res["df_weibull"])]
            tbl_mx.controls = [dataframe_to_datatable(res["df_maximos_mensuales"])]
            
            img_max_an.src_base64 = res["max_annual_series_b64"]
            img_acf.src_base64 = res["acf_plot_b64"]
            img_wei.src_base64 = res["weibull_plot_b64"]
            img_dist.src_base64 = res["dist_comparison_b64"]
            
            txt_best_fit.value = f"Mejor Ajuste: {res['best_fit_name']}"
            btn_nav_cuenca.disabled = False
            rail.selected_index = 4; change_view()
        page.update()

    def run_cuenca(e):
        if page.session.df_maximos_mensuales is None: return
        try: yh, yi = int(inp_ylim_h.value or 0), int(inp_ylim_i.value or 0)
        except: yh, yi = None, None
        
        res = analisis_cuenca.run_cuenca_analysis(
            page.session.best_fit_name, 
            page.session.df_maximos_mensuales, 
            yh, yi
        )
        
        if res:
            # --- MAPPING EXPLÍCITO (Solución del Error) ---
            page.session.df_altura = res["df_altura"]
            page.session.df_intensidad = res["df_intensidad"]
            page.session.cuenca_plot1_b64 = res["plot_1_b64"]
            page.session.cuenca_plot2_b64 = res["plot_2_b64"]
            page.session.cuenca_plot3_b64 = res["plot_3_b64"]
            page.session.cuenca_plot4_b64 = res["plot_4_b64"]

            # UI Updates
            tbl_alt.controls = [dataframe_to_datatable(res["df_altura"])]
            tbl_int.controls = [dataframe_to_datatable(res["df_intensidad"])]
            img_c1.src_base64 = res["plot_1_b64"]
            img_c2.src_base64 = res["plot_2_b64"]
            img_c3.src_base64 = res["plot_3_b64"]
            img_c4.src_base64 = res["plot_4_b64"]
            log_cuenca.value = res["log_text"]
            rail.selected_index = 5; change_view()
        page.update()

    # --- NAV ---
    btn_nav_filt = ft.ElevatedButton("Ir a Filtrado", on_click=lambda e: setattr(rail, 'selected_index', 1) or change_view(), disabled=True)
    btn_nav_stats = ft.ElevatedButton("Ir a Estadísticas", on_click=lambda e: run_stats(e), disabled=True)
    btn_nav_graph = ft.ElevatedButton("Ir a Gráficos", on_click=lambda e: run_graphs(e), disabled=True)
    btn_nav_lluvias = ft.ElevatedButton("Iniciar Lluvias", on_click=lambda e: run_lluvias(e), disabled=True)
    btn_nav_cuenca = ft.ElevatedButton("Iniciar Cuenca", on_click=lambda e: run_cuenca(e), disabled=True)

    csv_picker.on_result = on_load

    # --- VISTAS ---
    v1 = ft.Container(ft.Column([ft.Text("Paso 1: Carga", color="#00ff41", size=20), ft.Row([ft.ElevatedButton("Cargar CSV", icon=ft.Icons.UPLOAD, on_click=lambda _: csv_picker.pick_files()), csv_path_txt]), data_table_raw, btn_nav_filt]), padding=20, visible=True)
    v2 = ft.Container(ft.Column([ft.Text("Paso 2: Filtros", color="#00ff41", size=20), chk_c1, chk_c2, chk_c3, ft.ElevatedButton("Aplicar", on_click=run_filter), data_table_filt, btn_nav_stats]), padding=20, visible=False)
    v3 = ft.Container(ft.Column([ft.Text("Paso 3: Estadísticas", color="#00ff41", size=20), stats_table_container, btn_nav_graph]), padding=20, visible=False)
    v4 = ft.Container(ft.Column([ft.Text("Paso 4: Gráficos", color="#00ff41", size=20), ft.Text("Histograma"), img_h, ft.ElevatedButton("Guardar", on_click=lambda _: pk_hist.save_file("hist.png")), ft.Text("Series"), img_s, ft.ElevatedButton("Guardar", on_click=lambda _: pk_ser.save_file("series.png")), ft.Text("Violin"), img_v, ft.ElevatedButton("Guardar", on_click=lambda _: pk_vio.save_file("violin.png")), btn_nav_lluvias], scroll=ft.ScrollMode.AUTO), padding=20, visible=False, expand=True)
    v5 = ft.Container(ft.Column([ft.Text("Paso 5: Análisis Lluvias", color="#00ff41", size=20), ft.Text("Homogeneidad:", weight="bold"), tbl_homo, ft.ElevatedButton("CSV", on_click=lambda _: pk_homo_csv.save_file("homo.csv")), ft.Divider(), ft.Text("Series Máximas Anuales:", weight="bold"), img_max_an, ft.ElevatedButton("IMG", on_click=lambda _: pk_max_an.save_file("serie.png")), ft.Divider(), ft.Text("Autocorrelación (ACF):", weight="bold"), img_acf, ft.ElevatedButton("IMG", on_click=lambda _: pk_acf_img.save_file("acf.png")), tbl_acf, ft.ElevatedButton("CSV", on_click=lambda _: pk_acf_csv.save_file("acf.csv")), ft.Divider(), ft.Text("Weibull:", weight="bold"), img_wei, ft.ElevatedButton("IMG", on_click=lambda _: pk_weibull.save_file("weibull.png")), tbl_wei, ft.ElevatedButton("CSV", on_click=lambda _: pk_wei_csv.save_file("weibull.csv")), ft.Divider(), ft.Text("Ajuste de Distribuciones:", weight="bold"), img_dist, ft.ElevatedButton("IMG", on_click=lambda _: pk_dist.save_file("dist.png")), tbl_ajustes, ft.ElevatedButton("CSV", on_click=lambda _: pk_ajustes_csv.save_file("ajustes.csv")), ft.Divider(), ft.Text("Máximos Mensuales:", weight="bold"), tbl_mx, ft.ElevatedButton("CSV", on_click=lambda _: pk_mx_csv.save_file("maximos.csv")), btn_nav_cuenca], scroll=ft.ScrollMode.AUTO), padding=20, visible=False, expand=True)
    v6 = ft.Container(ft.Column([ft.Text("Paso 6: Cuenca", color="#00ff41", size=20), txt_best_fit, ft.Row([inp_ylim_h, inp_ylim_i, ft.ElevatedButton("Recalcular", on_click=run_cuenca)]), ft.Text("PDR:"), img_c1, ft.ElevatedButton("Guardar", on_click=lambda _: pk_p1.save_file("pdr.png")), img_c2, ft.ElevatedButton("Guardar Zoom", on_click=lambda _: pk_p2.save_file("pdr_z.png")), tbl_alt, ft.ElevatedButton("CSV", on_click=lambda _: pk_alt_csv.save_file("alt.csv")), ft.Text("IDF:"), img_c3, ft.ElevatedButton("Guardar", on_click=lambda _: pk_p3.save_file("idf.png")), img_c4, ft.ElevatedButton("Guardar Zoom", on_click=lambda _: pk_p4.save_file("idf_z.png")), tbl_int, ft.ElevatedButton("CSV", on_click=lambda _: pk_int_csv.save_file("I-D-TR.csv")), ft.Text("Log:"), log_cuenca], scroll=ft.ScrollMode.AUTO), padding=20, visible=False, expand=True)

    views = [v1, v2, v3, v4, v5, v6]
    def change_view(idx=None):
        if idx is None: idx = rail.selected_index
        for i,v in enumerate(views): v.visible = (i==idx)
        page.update()

    rail = ft.NavigationRail(selected_index=0, label_type=ft.NavigationRailLabelType.ALL, min_width=100, min_extended_width=400, leading=ft.Column([ft.IconButton(ft.Icons.ARROW_BACK, on_click=on_back_to_menu), ft.Text("Análisis")], spacing=10), destinations=[ft.NavigationRailDestination(icon=i, label=l) for i,l in [(ft.Icons.INPUT,"Carga"), (ft.Icons.FILTER_LIST,"Filtrado"), (ft.Icons.ANALYTICS,"Stats"), (ft.Icons.IMAGE,"Gráficos"), (ft.Icons.WATER_DROP,"Lluvias"), (ft.Icons.MAP,"Cuenca")]], on_change=lambda e: change_view(e.control.selected_index))
    return ft.Row([rail, ft.VerticalDivider(width=1), ft.Column(views, expand=True)], expand=True)