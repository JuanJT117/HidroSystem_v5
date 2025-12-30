import flet as ft
import pandas as pd
import base64
import traceback
import gastos  # Importamos la lógica separada (backend)

# ==========================================
# INTERFAZ DE USUARIO (FLET) - GASTOS
# ==========================================

def build_gastos_view(page: ft.Page, on_back_to_menu):
    
    # --- 1. Inicialización de Sesión ---
    keys = ["datos_cuencas_config", "df_cuencas_base", "df_cotas", "df_hms", 
            "df_intensidad", "df_altura", 
            "res_racional", "res_chow", "df_variables", "temp_graph_b64",
            "estaciones_db", "pesos_estaciones", "station_counter", "target_station_id",
            "res_hms_peak", "res_hidrogramas"] # Nuevas keys para HMS
            
    for k in keys: 
        if not hasattr(page.session, k): 
            if k in ["datos_cuencas_config", "estaciones_db", "pesos_estaciones", "res_hidrogramas"]:
                setattr(page.session, k, {})
            elif k == "station_counter":
                setattr(page.session, k, 1) # Contador inicia en 1
            else:
                setattr(page.session, k, None)

    # --- 2. Helpers Visuales ---
    def create_datatable(df):
        if df is None or df.empty: return ft.Text("Sin datos")
        try:
            df_view = df.head(100).copy()
            df_view.index = df_view.index.astype(str)
            columns = [ft.DataColumn(ft.Text("ID", color="#00ff41"))] + [ft.DataColumn(ft.Text(str(c))) for c in df_view.columns]
            rows = []
            for i, r in df_view.iterrows():
                cells = [ft.DataCell(ft.Text(str(i)))] + [ft.DataCell(ft.Text(str(r[c]))) for c in df_view.columns]
                rows.append(ft.DataRow(cells))
            return ft.DataTable(columns=columns, rows=rows, border=ft.border.all(1, "#333333"))
        except: return ft.Text("Error visualizando tabla", color="red")

    def save_csv_safe(e, df):
        if e.path and df is not None:
            try:
                path = e.path if e.path.endswith(".csv") else f"{e.path}.csv"
                if 'TR (AÑOS)' in df.columns: df.set_index('TR (AÑOS)').to_csv(path)
                elif 'TR' in df.columns: df.set_index('TR').to_csv(path)
                else: df.to_csv(path)
                page.snack_bar = ft.SnackBar(ft.Text("CSV Guardado"), bgcolor="green", open=True)
            except Exception as ex:
                page.snack_bar = ft.SnackBar(ft.Text(f"Error: {ex}"), bgcolor="red", open=True)
            page.update()

    def save_img_safe(e):
        if e.path and hasattr(page.session, "temp_graph_b64"):
            try:
                path = e.path if e.path.endswith(".png") else f"{e.path}.png"
                with open(path, "wb") as f: f.write(base64.b64decode(page.session.temp_graph_b64))
                page.snack_bar = ft.SnackBar(ft.Text("Gráfico Guardado"), bgcolor="green", open=True)
            except Exception as ex:
                page.snack_bar = ft.SnackBar(ft.Text(f"Error: {ex}"), bgcolor="red", open=True)
            page.update()

    def trigger_save_graph(b64, name):
        if b64: page.session.temp_graph_b64 = b64; pk_s_img.save_file(file_name=name)

    # --- 3. Pickers ---
    pk_area = ft.FilePicker()
    pk_int = ft.FilePicker()
    pk_alt = ft.FilePicker()
    pk_cot = ft.FilePicker()
    pk_hms = ft.FilePicker()
    
    # Pickers para carga individual por Estación
    pk_st_int = ft.FilePicker(on_result=lambda e: load_station_file(e, "intensidad"))
    pk_st_alt = ft.FilePicker(on_result=lambda e: load_station_file(e, "altura"))
    
    pk_s_rac = ft.FilePicker(on_result=lambda e: save_csv_safe(e, page.session.res_racional))
    pk_s_chow = ft.FilePicker(on_result=lambda e: save_csv_safe(e, page.session.res_chow))
    pk_s_hms = ft.FilePicker(on_result=lambda e: save_csv_safe(e, page.session.res_hms_peak)) # Nuevo
    pk_s_vars = ft.FilePicker(on_result=lambda e: save_csv_safe(e, page.session.df_variables))
    pk_s_img = ft.FilePicker(on_result=save_img_safe)

    page.overlay.extend([pk_area, pk_int, pk_alt, pk_cot, pk_hms, pk_s_rac, pk_s_chow, pk_s_hms, pk_s_vars, pk_s_img, pk_st_int, pk_st_alt])

    # --- 4. Controles UI Globales ---
    tabla_cuencas = ft.DataTable(columns=[ft.DataColumn(ft.Text(t)) for t in ["ID","Área","LCP","Estado","Editar"]], rows=[])
    st_area = ft.Text("No cargado", color="orange", size=12)
    
    # Estados UI
    st_int = ft.Text("No cargado", color="orange", size=12)
    st_alt = ft.Text("No cargado", color="orange", size=12) 
    st_cota_simple = ft.Text("No cargado", color="orange", size=12)
    st_hms_simple = ft.Text("No cargado (Opcional)", color="grey", size=12)
    st_cota_dist = ft.Text("No cargado", color="orange", size=12)
    st_hms_dist = ft.Text("No cargado (Opcional)", color="grey", size=12)
    
    # --- UI ESTACIONES DINÁMICAS ---
    col_estaciones_list = ft.Column(spacing=10) # Contenedor de "Tarjetas" de estación
    
    tabla_pesos = ft.DataTable(
        columns=[ft.DataColumn(ft.Text("ID Cuenca"))], 
        rows=[],
        border=ft.border.all(1, "grey")
    )
    
    validation_msg = ft.Text("", size=12, weight="bold")
    log_txt = ft.Text("Listo.", color="grey")
    btn_calc = ft.ElevatedButton("CALCULAR", icon=ft.Icons.CALCULATE, disabled=True, style=ft.ButtonStyle(color="#00ff41"))
    tabs_res = ft.Tabs(selected_index=0, visible=False)
    
    rg_modo_calculo = ft.RadioGroup(
        content=ft.Row([
            ft.Radio(value="simple", label="Modo Simple (Global)", fill_color="#00ff41"),
            ft.Radio(value="dist", label="Modo Distribuido (Thiessen)", fill_color="#00ff41")
        ]), value="simple"
    )
    
    # NUEVO: Selector de Método Tc
    rg_metodo_tc = ft.RadioGroup(
        content=ft.Row([
            ft.Radio(value="Kirpich", label="Kirpich (Urbano/Canal)", fill_color="#00ff41"),
            ft.Radio(value="Temez", label="Témez (Rural/SCT)", fill_color="#00ff41")
        ]), value="Kirpich"
    )

    # --- 5. Lógica Interna ---
    
    def upd_tbl():
        if page.session.df_cuencas_base is None: return
        rows, rdy = [], True
        df_view = page.session.df_cuencas_base.head(50)
        for i, r in df_view.iterrows():
            cfg = page.session.datos_cuencas_config.get(str(i))
            if not cfg: rdy = False
            rows.append(ft.DataRow([
                ft.DataCell(ft.Text(str(i))), 
                ft.DataCell(ft.Text(f"{r['area']:.2f}")), 
                ft.DataCell(ft.Text(str(cfg['LCP']) if cfg else "-")),
                ft.DataCell(ft.Icon(ft.Icons.CHECK if cfg else ft.Icons.WARNING, color="green" if cfg else "orange")),
                ft.DataCell(ft.IconButton(ft.Icons.EDIT, on_click=lambda e, c=str(i): open_conf(c)))
            ]))
        tabla_cuencas.rows = rows; btn_calc.disabled = not rdy; page.update()
        if page.session.estaciones_db: upd_tabla_pesos()

    def load(e, k, s, i=0):
        if e.files:
            try: 
                setattr(page.session, k, pd.read_csv(e.files[0].path, header=0, index_col=i))
                s.value = "OK"; s.color = "green"
                upd_tbl()
            except Exception as ex: 
                s.value = "Error"; s.color = "red"
            page.update()

    def load_geometry_hms(e, key):
        if e.files:
            try:
                idx_col = None if key == "df_cotas" else 0
                df = pd.read_csv(e.files[0].path, header=0, index_col=idx_col)
                setattr(page.session, key, df)
                if key == "df_cotas":
                    st_cota_simple.value = "OK"; st_cota_simple.color = "green"
                    st_cota_dist.value = "OK"; st_cota_dist.color = "green"
                elif key == "df_hms":
                    st_hms_simple.value = "OK"; st_hms_simple.color = "green"
                    st_hms_dist.value = "OK"; st_hms_dist.color = "green"
                upd_tbl()
            except Exception as ex:
                if key == "df_cotas": st_cota_simple.value = "Error"; st_cota_simple.color = "red"
            page.update()

    # --- LÓGICA DE ESTACIONES DINÁMICAS (AGREGAR Y BORRAR) ---
    
    def add_new_station(e):
        """Crea una nueva entrada de estación (ST_n)."""
        new_id = f"ST_{page.session.station_counter}"
        default_name = f"Estación {page.session.station_counter}"
        page.session.estaciones_db[new_id] = {
            "nombre": default_name,
            "intensidad": None,
            "altura": None
        }
        page.session.station_counter += 1
        render_stations_list()
        upd_tabla_pesos()
        page.update()

    def delete_station(est_id):
        """Elimina una estación y limpia sus referencias."""
        if est_id in page.session.estaciones_db:
            # 1. Eliminar de la base de datos de estaciones
            del page.session.estaciones_db[est_id]
            
            # 2. Eliminar de la matriz de pesos (limpieza de residuos)
            for cid in page.session.pesos_estaciones:
                if est_id in page.session.pesos_estaciones[cid]:
                    del page.session.pesos_estaciones[cid][est_id]
            
            # 3. Actualizar UI
            render_stations_list()
            upd_tabla_pesos()
            page.update()

    def pick_file_for_station(e, est_id, picker_obj):
        page.session.target_station_id = est_id
        picker_obj.pick_files()

    def load_station_file(e, tipo):
        if not e.files or not page.session.target_station_id: return
        target_id = page.session.target_station_id
        try:
            df = pd.read_csv(e.files[0].path, header=0, index_col=0)
            df.index = df.index.astype(int) 
            if target_id in page.session.estaciones_db:
                page.session.estaciones_db[target_id][tipo] = df
            render_stations_list()
            upd_tabla_pesos()
        except Exception as ex:
            page.snack_bar = ft.SnackBar(ft.Text(f"Error cargando archivo: {ex}"), bgcolor="red", open=True)
        page.update()

    def update_station_name(est_id, new_name):
        if est_id in page.session.estaciones_db:
            page.session.estaciones_db[est_id]["nombre"] = new_name
            upd_tabla_pesos()

    def render_stations_list():
        """Construye las tarjetas visuales de las estaciones."""
        col_estaciones_list.controls.clear()
        
        sorted_keys = sorted(page.session.estaciones_db.keys(), key=lambda x: int(x.split('_')[1]))
        
        for est_id in sorted_keys:
            data = page.session.estaciones_db[est_id]
            has_i = data['intensidad'] is not None
            has_a = data['altura'] is not None
            
            txt_name = ft.TextField(
                value=data['nombre'], 
                label="Nombre Estación", 
                width=200, 
                text_size=12,
                on_blur=lambda e, eid=est_id: update_station_name(eid, e.control.value)
            )
            
            ind_i = ft.Icon(ft.Icons.CHECK_CIRCLE if has_i else ft.Icons.CANCEL, color="green" if has_i else "red", size=20)
            ind_a = ft.Icon(ft.Icons.CHECK_CIRCLE if has_a else ft.Icons.CANCEL, color="green" if has_a else "red", size=20)
            
            btn_i = ft.ElevatedButton("I-D-TR", on_click=lambda e, eid=est_id: pick_file_for_station(e, eid, pk_st_int), height=30, style=ft.ButtonStyle(padding=5))
            btn_a = ft.ElevatedButton("Ap-D-TR", on_click=lambda e, eid=est_id: pick_file_for_station(e, eid, pk_st_alt), height=30, style=ft.ButtonStyle(padding=5))
            
            # --- Botón de Eliminar ---
            btn_del = ft.IconButton(
                icon=ft.Icons.DELETE_OUTLINE, 
                icon_color="red", 
                tooltip="Eliminar Estación",
                on_click=lambda e, eid=est_id: delete_station(eid)
            )

            card = ft.Container(
                content=ft.Row([
                    ft.Text(est_id, color="grey", size=10, width=40),
                    txt_name,
                    ft.VerticalDivider(width=10),
                    btn_i, ind_i,
                    ft.VerticalDivider(width=10),
                    btn_a, ind_a,
                    ft.VerticalDivider(width=20, color="transparent"),
                    btn_del
                ], alignment=ft.MainAxisAlignment.START),
                padding=5,
                bgcolor="#1f1f1f",
                border_radius=5,
                border=ft.border.all(1, "#333333")
            )
            col_estaciones_list.controls.append(card)
    
    def upd_tabla_pesos():
        if page.session.df_cuencas_base is None:
            tabla_pesos.columns = [ft.DataColumn(ft.Text("Cargue Cuencas primero"))]
            tabla_pesos.rows = []
            return
        
        if not page.session.estaciones_db:
            tabla_pesos.columns = [ft.DataColumn(ft.Text("Agregue Estaciones"))]
            tabla_pesos.rows = []
            return

        sorted_ids = sorted(page.session.estaciones_db.keys(), key=lambda x: int(x.split('_')[1]))
        
        cols = [ft.DataColumn(ft.Text("ID Cuenca", color="#00ff41"))]
        cols.extend([
            ft.DataColumn(
                ft.Text(page.session.estaciones_db[eid]['nombre'][:15], tooltip=eid)
            ) for eid in sorted_ids
        ])
        
        rows = []
        cuencas_ids = [str(x) for x in page.session.df_cuencas_base.index]
        
        for cid in cuencas_ids:
            if cid not in page.session.pesos_estaciones:
                page.session.pesos_estaciones[cid] = {eid: 0.0 for eid in sorted_ids}
                if len(sorted_ids) == 1: page.session.pesos_estaciones[cid][sorted_ids[0]] = 1.0

        for cid in cuencas_ids:
            cells = [ft.DataCell(ft.Text(cid, weight="bold"))]
            for eid in sorted_ids:
                current_val = page.session.pesos_estaciones[cid].get(eid, 0.0)
                txt_peso = ft.TextField(
                    value=str(current_val), 
                    width=60, 
                    text_size=12,
                    on_change=lambda e, c=cid, est=eid: update_peso_val(c, est, e.control.value)
                )
                cells.append(ft.DataCell(txt_peso))
            rows.append(ft.DataRow(cells))
            
        tabla_pesos.columns = cols
        tabla_pesos.rows = rows
        page.update()

    def update_peso_val(cid, eid, val_str):
        try:
            val = float(val_str)
            if cid in page.session.pesos_estaciones:
                page.session.pesos_estaciones[cid][eid] = val
        except: pass

    # Callbacks globales
    pk_area.on_result = lambda e: load(e, "df_cuencas_base", st_area, 0)
    pk_int.on_result = lambda e: load(e, "df_intensidad", st_int, 0)
    pk_alt.on_result = lambda e: load(e, "df_altura", st_alt, 0)
    pk_cot.on_result = lambda e: load_geometry_hms(e, "df_cotas")
    pk_hms.on_result = lambda e: load_geometry_hms(e, "df_hms")

    # --- RESET ---
    def on_reset(e):
        page.session.datos_cuencas_config = {}
        page.session.pesos_estaciones = {}
        page.session.station_counter = 1 
        
        for k in ["df_cuencas_base", "df_cotas", "df_hms", "df_altura", "res_racional", "res_chow", "df_variables", "estaciones_db", "res_hms_peak", "res_hidrogramas"]: 
            val = {} if k in ["estaciones_db", "res_hidrogramas"] else None
            setattr(page.session, k, val)
        
        tabla_cuencas.rows = []
        render_stations_list()
        upd_tabla_pesos()
        
        st_area.value = "No cargado"; st_area.color = "orange"
        for s in [st_int, st_alt, st_cota_simple, st_cota_dist]: s.value = "No cargado"; s.color = "orange"
        for s in [st_hms_simple, st_hms_dist]: s.value = "No cargado (Opcional)"; s.color = "grey"
            
        validation_msg.value = ""
        btn_calc.disabled = True; tabs_res.visible = False; log_txt.value = "Reiniciado."
        page.update()

    # --- Config Modal (Cuencas) ---
    c_id = ft.Text(""); inp_lcp = ft.TextField(label="LCP", width=100)
    col_usos = ft.Column(scroll=ft.ScrollMode.ALWAYS, height=250)
    
    def add_row(e, d=None):
        row = ft.Row([
            ft.TextField(value=d['pct'] if d else "", label="%", width=80),
            ft.Dropdown(options=[ft.dropdown.Option(k,v) for k,v in gastos.hidrologia_mx.OPCIONES_C.items()], value=d['c'] if d else None, width=400, label="C"),
            ft.Dropdown(options=[ft.dropdown.Option(k,v) for k,v in gastos.hidrologia_mx.OPCIONES_N.items()], value=d['n'] if d else None, width=380, label="N"),
            ft.Dropdown(options=[ft.dropdown.Option(x) for x in "ABCD"], value=d['g'] if d else None, width=100, label="G"),
            ft.IconButton(ft.Icons.DELETE, icon_color="red", on_click=lambda e: col_usos.controls.remove(row) or page.update())
        ])
        col_usos.controls.append(row); page.update()

    def save_conf(e):
        usos, tot = [], 0
        for r in col_usos.controls:
            try: p = float(r.controls[0].value or 0); tot += p; usos.append({"pct":p, "c":int(r.controls[1].value or 0), "n":int(r.controls[2].value or 0), "g":r.controls[3].value or "A"})
            except: pass
        if tot>100: page.snack_bar = ft.SnackBar(ft.Text("Suma > 100%"), bgcolor="red", open=True); page.update(); return
        page.session.datos_cuencas_config[c_id.value] = {"LCP": float(inp_lcp.value or 0), "usos": usos}
        dlg.open = False; upd_tbl(); page.update()

    dlg = ft.AlertDialog(title=ft.Text("Configuración de Cuenca"), content=ft.Container(ft.Column([c_id, inp_lcp, ft.ElevatedButton("Agregar uso", on_click=lambda e: add_row(e)), col_usos]), height=400, width=1100),
        actions=[ft.TextButton("Cancelar", on_click=lambda e: setattr(dlg, 'open', False) or page.update()), ft.ElevatedButton("Guardar", on_click=save_conf)])
    page.overlay.append(dlg)

    def open_conf(cid):
        c_id.value = str(cid); d = page.session.datos_cuencas_config.get(str(cid), {})
        inp_lcp.value = d.get("LCP", ""); col_usos.controls.clear()
        for u in d.get("usos", []): add_row(None, u)
        if not d.get("usos"): add_row(None)
        dlg.open = True; page.update()

    # --- 6. EJECUCIÓN (AUTO-RUN) ---
    def run(e=None):
        if page.session.res_racional is not None and e is None: pass 
        else:
            modo_dist = (rg_modo_calculo.value == "dist")
            metodo_tc_sel = rg_metodo_tc.value
            
            if not page.session.df_cotas is not None:
                log_txt.value="Falta archivo de Cotas"; log_txt.color="red"; page.update(); return
            
            if modo_dist:
                if not page.session.estaciones_db:
                    log_txt.value="Modo Distribuido: No hay estaciones cargadas."; log_txt.color="red"; page.update(); return
            else:
                if page.session.df_intensidad is None or page.session.df_altura is None:
                    log_txt.value="Modo Simple: Faltan archivos de lluvia."; log_txt.color="red"; page.update(); return

            log_txt.value = "Calculando..."; page.update()
            try:
                df = page.session.df_cuencas_base.copy()
                lcp, pct, c, n, g = [],[],[],[],[]
                for i in df.index:
                    d = page.session.datos_cuencas_config.get(str(i))
                    if not d: continue
                    lcp.append(d['LCP']); pct.append(",".join([str(u['pct']) for u in d['usos']]))
                    c.append(",".join([str(u['c']) for u in d['usos']])); n.append(",".join([str(u['n']) for u in d['usos']])); g.append(",".join([str(u['g']) for u in d['usos']]))
                df['LCP'], df['Porcentaje_terreno'], df['Coeficiente_C_por_Cuenca'], df['indice_Uso_terreno'], df['Grupo_hidrologico'] = lcp, pct, c, n, g
                
                # LLAMADA A LA LÓGICA EXTERNA CON TÉMEZ Y HMS
                res_r, res_c, res_h, df_vars, res_hydros, l = gastos.calcular_coeficientes_y_gastos(
                    df, page.session.df_cotas, metodo_tc=metodo_tc_sel, modo_distribuido=modo_dist,
                    estaciones_db=page.session.estaciones_db, pesos_estaciones=page.session.pesos_estaciones,
                    df_int_global=page.session.df_intensidad, df_alt_global=page.session.df_altura
                )
                
                if res_r is None:
                    log_txt.value = l; log_txt.color = "red"
                else:
                    page.session.res_racional = res_r; page.session.res_chow = res_c; page.session.res_hms_peak = res_h
                    page.session.df_variables = df_vars; page.session.res_hidrogramas = res_hydros
                    log_txt.value = "Cálculo finalizado. Verifique log para advertencias normativas."
            except Exception as ex:
                log_txt.value = f"Error: {ex}\n{traceback.format_exc()}"; page.update(); return

        if page.session.res_racional is not None:
            # LLAMADA A LA LÓGICA EXTERNA PARA GRÁFICOS (Ahora incluye HMS calculado)
            gs = gastos.generar_graficos_comparativos(page.session.res_racional, page.session.res_chow, page.session.res_hms_peak, page.session.df_hms)
            
            t1 = ft.Tab("Racional", ft.Column([ft.ElevatedButton("CSV", icon=ft.Icons.SAVE, on_click=lambda _: pk_s_rac.save_file("Racional.csv")), ft.Row([create_datatable(page.session.res_racional)], scroll=ft.ScrollMode.AUTO)], scroll=ft.ScrollMode.AUTO))
            t2 = ft.Tab("Chow", ft.Column([ft.ElevatedButton("CSV", icon=ft.Icons.SAVE, on_click=lambda _: pk_s_chow.save_file("Chow.csv")), ft.Row([create_datatable(page.session.res_chow)], scroll=ft.ScrollMode.AUTO)], scroll=ft.ScrollMode.AUTO))
            t3 = ft.Tab("HMS (Picos)", ft.Column([ft.ElevatedButton("CSV", icon=ft.Icons.SAVE, on_click=lambda _: pk_s_hms.save_file("HMS_Picos.csv")), ft.Row([create_datatable(page.session.res_hms_peak)], scroll=ft.ScrollMode.AUTO)], scroll=ft.ScrollMode.AUTO))
            t4 = ft.Tab("Variables", ft.Column([ft.ElevatedButton("CSV", icon=ft.Icons.SAVE, on_click=lambda _: pk_s_vars.save_file("Variables.csv")), ft.Row([create_datatable(page.session.df_variables)], scroll=ft.ScrollMode.AUTO)], scroll=ft.ScrollMode.AUTO))
            
            imgs = ft.Column(scroll=ft.ScrollMode.AUTO, spacing=20)
            for tr, b64 in gs:
                btn_save = ft.ElevatedButton("Guardar", icon=ft.Icons.SAVE, on_click=lambda e, b=b64, t=tr: trigger_save_graph(b, f"Graph_TR{t}.png"))
                imgs.controls.append(ft.Column([ft.Text(f"TR {tr}"), ft.Image(src_base64=b64, fit=ft.ImageFit.CONTAIN), btn_save], horizontal_alignment=ft.CrossAxisAlignment.CENTER))
            
            # --- NUEVA PESTAÑA DE HIDROGRAMAS COMPLETOS ---
            t_hydros = ft.Tab("Hidrogramas", ft.Column([ft.Text("Seleccione una cuenca y TR para ver el hidrograma detallado (Próximamente)")], scroll=ft.ScrollMode.AUTO))
            
            tabs_res.tabs = [t1, t2, t3, t4, ft.Tab("Gráficos Barras", imgs), t_hydros]; tabs_res.visible = True
            log_txt.color = "green"
            if e: page.update()

    btn_calc.on_click = run

    # --- 7. VISTAS Y TABS ---
    
    def on_tab_change(e):
        if e.control.selected_index == 0: rg_modo_calculo.value = "simple"
        else: rg_modo_calculo.value = "dist"
        page.update()

    tab_simple = ft.Tab(
        text="Modo Simple (1 Estación)",
        icon=ft.Icons.SQUARE,
        content=ft.Column([
            ft.Container(height=20),
            ft.Text("Datos de Lluvia Global:", weight="bold"),
            ft.Row([ft.ElevatedButton("Cargar I-D-TR (Racional)", on_click=lambda _: pk_int.pick_files()), st_int]),
            ft.Row([ft.ElevatedButton("Cargar Ap-D-TR (Chow)", on_click=lambda _: pk_alt.pick_files()), st_alt]),
            ft.Divider(),
            ft.Text("Geometría y Validación:", weight="bold"),
            ft.Row([ft.ElevatedButton("Cargar Cotas", on_click=lambda _: pk_cot.pick_files()), st_cota_simple]),
            ft.Row([ft.ElevatedButton("Resultados HMS", on_click=lambda _: pk_hms.pick_files()), st_hms_simple]),
        ])
    )
    
    # --- Tab Distribuido ---
    tab_distribuido = ft.Tab(
        text="Modo Distribuido (Thiessen)",
        icon=ft.Icons.GRID_ON,
        content=ft.Column([
            ft.Container(height=10),
            ft.Text("1. Gestión de Estaciones", weight="bold"),
            
            ft.ElevatedButton("Nueva Estación (+)", icon=ft.Icons.ADD, on_click=add_new_station, bgcolor="#222222"),
            
            ft.Container(
                content=ft.Column([col_estaciones_list], scroll=ft.ScrollMode.AUTO), 
                height=200, 
                border=ft.border.all(1, "grey"), 
                padding=10
            ),
            
            ft.Divider(),
            ft.Text("2. Matriz de Pesos (Thiessen)", weight="bold"),
            ft.Text("Los encabezados muestran el nombre asignado arriba.", size=10, italic=True, color="grey"),
            
            ft.Container(content=ft.Column([tabla_pesos], scroll=ft.ScrollMode.ALWAYS), height=200, border=ft.border.all(1, "grey")),
            
            ft.Divider(),
            ft.Text("3. Geometría y Validación:", weight="bold"),
            ft.Row([ft.ElevatedButton("Cargar Cotas", on_click=lambda _: pk_cot.pick_files()), st_cota_dist]),
            ft.Row([ft.ElevatedButton("Resultados HMS", on_click=lambda _: pk_hms.pick_files()), st_hms_dist]),
        ], scroll=ft.ScrollMode.AUTO)
    )
    
    tabs_entradas = ft.Tabs(selected_index=0, tabs=[tab_simple, tab_distribuido], on_change=on_tab_change)

    v_area = ft.Container(ft.Column([
        ft.Text("Paso 1: Cuencas", color="#00ff41", size=20),
        ft.Row([ft.ElevatedButton("Cargar Áreas", icon=ft.Icons.UPLOAD, on_click=lambda _: pk_area.pick_files()), st_area]),
        ft.Text("(Columnas esperadas: ID [Index], area)", size=10, color="grey")
    ]), padding=20)

    v_conf = ft.Container(ft.Column([
        ft.Text("Paso 2: Suelos", color="#00ff41", size=20),
        ft.Text("Área (Km^2) / Longitud Cauce Principal (Km)", color="#00ff41", size=16),
        ft.Container(ft.Column([tabla_cuencas], scroll=ft.ScrollMode.AUTO), height=800, border=ft.border.all(1, "grey"))
    ]), padding=20, visible=False)

    v_data = ft.Container(ft.Column([
        ft.Text("Paso 3: Entradas y Configuración", color="#00ff41", size=20),
        
        ft.Container(
            content=ft.Column([
                ft.Text("Método de Tiempo de Concentración:", weight="bold"),
                rg_metodo_tc
            ]),
            padding=10, border=ft.border.all(1, "grey"), border_radius=5
        ),
        
        ft.Divider(),
        tabs_entradas,
        validation_msg
    ]), padding=20, visible=False)
    
    v_res = ft.Container(ft.Column([
        ft.Text("Paso 4: Resultados", color="#00ff41", size=20), 
        ft.Text("Seleccione el modo de cálculo:", size=12, italic=True),
        rg_modo_calculo,
        ft.Divider(),
        btn_calc, 
        log_txt, 
        ft.Container(tabs_res, expand=True)
    ], scroll=ft.ScrollMode.AUTO), visible=False, padding=20, expand=True)

    views = [v_area, v_conf, v_data, v_res]
    def chg_view(e):
        for i,v in enumerate(views): v.visible = (i == e.control.selected_index)
        page.update()

    # --- RAIL DE NAVEGACIÓN (CON BOTÓN REINICIAR) ---
    rail = ft.NavigationRail(
        selected_index=0, 
        label_type=ft.NavigationRailLabelType.ALL, 
        min_width=100, 
        min_extended_width=400, 
        leading=ft.Column([
            ft.IconButton(ft.Icons.ARROW_BACK, on_click=on_back_to_menu), 
            ft.Text("Gastos")
        ], spacing=10), 
        destinations=[
            ft.NavigationRailDestination(icon=ft.Icons.LANDSCAPE, label="Cuencas"), 
            ft.NavigationRailDestination(icon=ft.Icons.SETTINGS, label="Suelos"), 
            ft.NavigationRailDestination(icon=ft.Icons.DATA_ARRAY, label="Entradas"), 
            ft.NavigationRailDestination(icon=ft.Icons.ANALYTICS, label="Resultados")
        ], 
        on_change=chg_view,
        trailing=ft.Column([
            ft.Divider(),
            ft.IconButton(
                icon=ft.Icons.RESTART_ALT, 
                tooltip="Reiniciar Módulo", 
                on_click=on_reset,
                icon_color="red"
            ),
            ft.Text("Reiniciar", size=10, color="red")
        ], alignment=ft.MainAxisAlignment.END, spacing=5)
    )

    # Restaurar Estado
    if page.session.df_cuencas_base is not None: st_area.value, st_area.color = "Recuperado", "green"
    if page.session.df_intensidad is not None: st_int.value, st_int.color = "Recuperado", "green"
    if page.session.df_altura is not None: st_alt.value, st_alt.color = "Recuperado", "green"
    
    if page.session.df_cotas is not None: 
        st_cota_simple.value = "Recuperado"; st_cota_simple.color = "green"
        st_cota_dist.value = "Recuperado"; st_cota_dist.color = "green"
    if page.session.df_hms is not None: 
        st_hms_simple.value = "Recuperado"; st_hms_simple.color = "green"
        st_hms_dist.value = "Recuperado"; st_hms_dist.color = "green"
        
    if page.session.estaciones_db: 
        render_stations_list()
        upd_tabla_pesos()
    
    upd_tbl()
    if page.session.res_racional is not None: run(None) 

    return ft.Row([rail, ft.VerticalDivider(width=1), ft.Column(views, expand=True)], expand=True)