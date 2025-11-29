import flet as ft
import pandas as pd
import numpy as np
import math
import traceback
import matplotlib
import matplotlib.pyplot as plt
import io
import base64

# Configurar backend de Matplotlib para evitar errores de GUI
matplotlib.use('Agg')

# ==========================================
# 1. DATOS DE REFERENCIA (Cte)
# ==========================================

OPCIONES_C = {
    1: '1. Urbanizada/Superficie asfáltica',
    2: '2. Urbanizada/Concreto-azoteas',
    3: '3. Áreas con pasto (Pobre)/plano (0-2 %)',
    4: '4. Áreas con pasto (Pobre)/promedio (2-7 %)',
    5: '5. Áreas con pasto (Pobre)/pendiente (> 7 %)',
    6: '6. Áreas con pasto (Media)/plano (0-2 %)',
    7: '7. Áreas con pasto (Media)/promedio (2-7 %)',
    8: '8. Áreas con pasto (Media)/pendiente (> 7 %)',
    9: '9. Áreas con pasto (Buena)/plano (0-2 %)',
    10: '10. Áreas con pasto (Buena)/promedio (2-7 %)',
    11: '11. Áreas con pasto (Buena)/pendiente (> 7 %)',
    12: '12. Rural - Cultivos/plano (0-2 %)',
    13: '13. Rural - Cultivos/promedio (2-7 %)',
    14: '14. Rural - Cultivos/pendiente (> 7 %)',
    15: '15. Rural - Pastizales/plano (0-2 %)',
    16: '16. Rural - Pastizales/promedio (2-7 %)',
    17: '17. Rural - Pastizales/pendiente (> 7 %)',
    18: '18. Bosques y montes/plano (0-2 %)',
    19: '19. Bosques y montes/promedio (2-7 %)',
    20: '20. Bosques y montes/pendiente (> 7 %)'
}

C_DATA_VALUES = {
    '2': [0.73,0.75,0.32,0.37,0.40,0.25,0.33,0.37,0.21,0.29,0.34,0.31,0.35,0.39,0.25,0.33,0.37,0.22,0.31,0.35],
    '5': [0.77,0.80,0.34,0.40,0.43,0.28,0.36,0.40,0.23,0.32,0.37,0.34,0.38,0.42,0.28,0.36,0.40,0.25,0.34,0.39],
    '10': [0.81,0.83,0.37,0.43,0.45,0.30,0.38,0.42,0.25,0.35,0.40,0.36,0.41,0.44,0.30,0.38,0.42,0.28,0.36,0.41],
    '20': [0.86,0.88,0.40,0.46,0.49,0.34,0.42,0.46,0.29,0.39,0.44,0.40,0.44,0.48,0.34,0.42,0.46,0.31,0.41,0.45],
    '50': [0.90,0.92,0.44,0.49,0.52,0.37,0.45,0.49,0.32,0.42,0.47,0.43,0.48,0.51,0.37,0.45,0.49,0.35,0.43,0.48],
    '100': [0.95,0.97,0.47,0.53,0.55,0.41,0.49,0.53,0.36,0.46,0.51,0.47,0.51,0.54,0.41,0.49,0.53,0.39,0.47,0.52],
    '500': [1.00,1.00,0.58,0.61,0.62,0.53,0.58,0.60,0.49,0.56,0.58,0.57,0.60,0.61,0.53,0.58,0.60,0.48,0.56,0.58],
    '1000':[1.00,1.00,0.65,0.68,0.70,0.60,0.65,0.68,0.56,0.63,0.65,0.64,0.68,0.70,0.60,0.65,0.68,0.55,0.62,0.65],
    '10000':[1.00,1.00,0.82,0.85,0.86,0.78,0.82,0.84,0.75,0.80,0.82,0.81,0.84,0.85,0.78,0.82,0.84,0.72,0.79,0.81]
}
C_DF = pd.DataFrame(C_DATA_VALUES, index=range(1, 21))

OPCIONES_N = {
    1: '1. Parques, campos abiertos, canchas deportivas - condicion buena (75% pasto)',
    2: '2. Parques, campos abiertos, canchas deportivas - condicion regular (50 - 75% pasto)',
    3: '3. Parques, campos abiertos, canchas deportivas - condicion pobre (<50% pasto)',
    4: '4. Áreas comerciales (85% impermeable)',
    5: '5. Distritos industriales (72% impermeable)',
    6: '6. Zona residencial (<500m2/65% impermeable)',
    7: '7. Zona residencial (1000m2/38% impermeable)',
    8: '8. Zona residencial (<1350m2/30% impermeable)',
    9: '9. Zona residencial (<2000m2/25% impermeable)',
    10: '10. Zona residencial (<4000m2/20% impermeable)',
    11: '11. Zona residencial (<8000m2/12% impermeable)',
    12: '12. Calzadas, tejados, estacionamiento pavimentado, etc.',
    13: '13. Calles Pavimentadas con guarnición y alcantarillado',
    14: '14. Caminos pavimentados, derecho de via y canales',
    15: '15. Caminos engravados - derecho de via',
    16: '16. Caminos de arcilla -derecho de via',
    17: '17. Áreas urbanas en desarrollo (nivelados sin vegetación)'
}
N_DATA = {
    'A': [39,49,68,89,81,77,61,57,54,51,46,98,98,83,76,72,77],
    'B': [61,69,79,92,88,85,75,72,70,68,65,98,98,89,85,82,86],
    'C': [74,79,86,94,91,90,83,81,80,79,77,98,98,92,89,87,91],
    'D': [80,84,89,95,93,92,87,86,85,84,82,98,98,92,91,89,94]
}
N_DF = pd.DataFrame(N_DATA, index=range(1, 18))

# ==========================================
# 2. LÓGICA DE CÁLCULO
# ==========================================

def redondear_tc(tc):
    if tc <= 10: return 10
    lower = (tc // 5) * 5
    return lower if tc - lower < 1 else lower + 5

def obtener_lista_desde_string(cadena, tipo_dato):
    try:
        return [tipo_dato(x.strip()) for x in str(cadena).split(',')]
    except:
        return []

def calcular_coeficientes_y_gastos(df_configurado, df_intensidad, df_altura, df_cotas):
    """
    Calcula Gastos usando:
    - Racional: df_intensidad (I-D-TR) [mm/hr]
    - Chow: df_altura (Ap-D-TR) [mm] -> Se convierte a cm internamente
    """
    log = []
    try:
        tr_cols = ['2', '5', '10', '20', '50', '100', '500', '1000', '10000']
        
        # --- Preparar Tablas de Entrada ---
        if 'TR (AÑOS)' in df_intensidad.columns: df_intensidad.set_index('TR (AÑOS)', inplace=True)
        df_intensidad.index = df_intensidad.index.astype(int)
        
        if 'TR (AÑOS)' in df_altura.columns: df_altura.set_index('TR (AÑOS)', inplace=True)
        df_altura.index = df_altura.index.astype(int)

        # --- 1. Coeficientes Ponderados (C y N) ---
        def _calc_pond(row):
            pcts = obtener_lista_desde_string(row['Porcentaje_terreno'], float)
            idxs_c = obtener_lista_desde_string(row['Coeficiente_C_por_Cuenca'], int)
            c_vals = {}
            for tr in tr_cols:
                c_vals[tr] = sum((p/100)*C_DF.loc[i, tr] for p, i in zip(pcts, idxs_c) if i in C_DF.index)
            
            usos_n = obtener_lista_desde_string(row['indice_Uso_terreno'], int)
            grupos = obtener_lista_desde_string(row['Grupo_hidrologico'], str)
            n_val = sum((p/100)*N_DF.loc[i, g.strip().upper()] for p, i, g in zip(pcts, usos_n, grupos) if i in N_DF.index)
            return pd.Series({**c_vals, 'N_Chow': n_val})

        df_pond = df_configurado.apply(_calc_pond, axis=1)
        df_completo = pd.concat([df_configurado, df_pond], axis=1)
        
        # --- 2. Tc y Pendiente ---
        if 'cuenca' in df_cotas.columns: df_cotas['cuenca'] = df_cotas['cuenca'].astype(str)
        dist_cotas = {key: sub_df for key, sub_df in df_cotas.groupby('cuenca')}
        
        df_completo['Tc_aprox'] = np.nan
        df_completo['Pendiente_S'] = np.nan
        df_completo.index = df_completo.index.astype(str)

        for cid in df_completo.index:
            if cid in dist_cotas:
                sub = dist_cotas[cid]
                lcp = 0
                den = 0
                for _, r in sub.iterrows():
                    d = abs(r['distancia2']-r['distancia1']) or 0.001
                    s = (abs(r['cota mayor']-r['cota menor'])/d) or 0.0001
                    lcp += d
                    den += d/math.sqrt(s)
                S = (lcp/(den or 0.001))**2
                tc = redondear_tc(0.000325*((lcp**0.77)/(S**0.385))*60)
                df_completo.at[cid, 'Tc_aprox'] = tc
                df_completo.at[cid, 'Pendiente_S'] = S
            else:
                log.append(f"Aviso: Cuenca {cid} sin datos de cotas.")

        # --- 3. Cálculos de Gastos ---
        cols_out = [f"TR_{tr}" for tr in tr_cols]
        
        # Inicializar DataFrames con 0.0 explícito
        df_racional = pd.DataFrame(0.0, index=df_completo.index, columns=cols_out)
        df_chow = pd.DataFrame(0.0, index=df_completo.index, columns=cols_out)

        # RACIONAL
        for cid, row in df_completo.iterrows():
            area = row['area']
            tc = row['Tc_aprox']
            if pd.notna(tc) and int(tc) in df_intensidad.index:
                for tr in tr_cols:
                    col_name = int(tr) if int(tr) in df_intensidad.columns else str(tr)
                    if col_name in df_intensidad.columns:
                        I = df_intensidad.loc[int(tc), col_name]
                        C = row[tr]
                        df_racional.at[cid, f"TR_{tr}"] = round(0.278 * C * I * area, 3)

        # CHOW (Unidades corregidas: mm -> cm)
        for cid, row in df_completo.iterrows():
            N = row['N_Chow']
            L = float(row['LCP']) if pd.notna(row['LCP']) else 0
            S = row['Pendiente_S']
            Ac = row['area']
            
            if L > 0 and S > 0 and pd.notna(N) and N > 0:
                Tiempo_retraso = (0.00505 * (L / (S**0.5))**0.64)
                TR_min_series = pd.Series(df_altura.index.values)
                ratio = (TR_min_series / 60.0) / Tiempo_retraso
                
                Z_values = np.select(
                    [ratio > 2, (ratio >= 0.4) & (ratio <= 2), (ratio > 0.005) & (ratio < 0.4)],
                    [1.0, 1.89 * (ratio**0.97) - 1.23, 0.73 * (ratio**0.97)],
                    default=0.0
                )
                
                for tr in tr_cols:
                    col_name = int(tr) if int(tr) in df_altura.columns else str(tr)
                    if col_name in df_altura.columns:
                        # CORRECCIÓN: mm a cm
                        P_col_mm = df_altura[col_name].values
                        P_col_cm = P_col_mm / 10.0 
                        
                        C1 = 508.0 / N; C2 = 5.08
                        C3 = 2032.0 / N; C4 = 20.32
                        C_Ac_Z = 2.78 * Ac * Z_values
                        
                        term_base = (P_col_cm - C1 + C2)**2
                        num = term_base * C_Ac_Z
                        den = P_col_cm + C3 - C4
                        
                        with np.errstate(divide='ignore', invalid='ignore'):
                            res_vector = (num / den) / (TR_min_series / 60.0)
                        
                        res_vector = np.nan_to_num(res_vector, nan=0.0, posinf=0.0, neginf=0.0)
                        max_q = np.max(res_vector)
                        df_chow.at[cid, f"TR_{tr}"] = round(max_q, 3)

        log.append("Cálculos finalizados.")
        return df_racional, df_chow, "\n".join(log)

    except Exception as e:
        return None, None, f"Error: {e}\n{traceback.format_exc()}"

def generar_graficos_comparativos(df_racional, df_chow, df_hms):
    graficos = []
    hms_map = {}
    if df_hms is not None and not df_hms.empty:
        for c in df_hms.columns:
            clean = ''.join(filter(str.isdigit, str(c)))
            if clean: hms_map[f"TR_{clean}"] = c

    df_racional.index = df_racional.index.astype(str)
    df_chow.index = df_chow.index.astype(str)
    if df_hms is not None: df_hms.index = df_hms.index.astype(str)

    trs = ['2', '5', '10', '20', '50', '100', '500', '1000', '10000']
    
    # CONTEXTO DE ESTILO ACADÉMICO (BLANCO)
    with plt.style.context('default'): 
        for tr in trs:
            col_tr = f"TR_{tr}"
            if col_tr in df_racional.columns:
                idx = df_racional.index.tolist()
                val_r = df_racional[col_tr].fillna(0).tolist()
                val_c = df_chow[col_tr].fillna(0).tolist() if col_tr in df_chow.columns else [0]*len(idx)
                val_h = []
                hms_col = hms_map.get(col_tr)
                if df_hms is not None and hms_col and hms_col in df_hms.columns:
                    for c in idx: val_h.append(df_hms.loc[c, hms_col] if c in df_hms.index else 0)
                else:
                    val_h = [0]*len(idx)

                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.arange(len(idx)); w = 0.25
                
                # Colores Académicos
                ax.bar(x - w, val_r, w, label='Método Racional', color='#1f77b4', edgecolor='black', linewidth=0.5)
                ax.bar(x, val_c, w, label='Método Chow', color='#ff7f0e', edgecolor='black', linewidth=0.5)
                ax.bar(x + w, val_h, w, label='Método HMS', color='#2ca02c', edgecolor='black', linewidth=0.5)
                
                ax.set_ylabel('Gasto Máximo ($m^3/s$)', fontsize=11)
                ax.set_title(f'Comparación de Gastos - TR {tr} Años', fontsize=13, fontweight='bold', pad=15)
                ax.set_xticks(x)
                ax.set_xticklabels(idx, rotation=45, ha='right', fontsize=10)
                
                ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
                ax.set_axisbelow(True)
                ax.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=10)
                plt.tight_layout()
                
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
                plt.close(fig)
                b64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
                graficos.append((tr, b64_str))
            
    return graficos

# ==========================================
# 3. INTERFAZ DE USUARIO (FLET)
# ==========================================

def build_gastos_view(page: ft.Page, on_back_to_menu):
    
    keys = ["datos_cuencas_config", "df_cuencas_base", "df_cotas", "df_hms", "df_intensidad", "df_altura", "res_racional", "res_chow", "temp_graph_b64"]
    for k in keys: 
        if not hasattr(page.session, k): setattr(page.session, k, {} if k=="datos_cuencas_config" else None)

    # --- HELPERS DE GUARDADO ROBUSTOS ---
    def save_csv_safe(e, df):
        if e.path and df is not None:
            try:
                path = e.path if e.path.endswith(".csv") else f"{e.path}.csv"
                df.to_csv(path)
                page.snack_bar = ft.SnackBar(ft.Text("CSV Guardado Exitosamente"), bgcolor="green", open=True)
            except Exception as ex:
                page.snack_bar = ft.SnackBar(ft.Text(f"Error guardando CSV: {ex}"), bgcolor="red", open=True)
            page.update()

    def save_img_safe(e):
        if e.path and page.session.temp_graph_b64:
            try:
                path = e.path if e.path.endswith(".png") else f"{e.path}.png"
                with open(path, "wb") as f:
                    f.write(base64.b64decode(page.session.temp_graph_b64))
                page.snack_bar = ft.SnackBar(ft.Text("Gráfico Guardado Exitosamente"), bgcolor="green", open=True)
            except Exception as ex:
                page.snack_bar = ft.SnackBar(ft.Text(f"Error guardando IMG: {ex}"), bgcolor="red", open=True)
            page.update()

    def trigger_save_graph(b64, name):
        page.session.temp_graph_b64 = b64
        pk_s_img.save_file(file_name=name)

    # --- PICKERS ---
    pk_area = ft.FilePicker()
    pk_int = ft.FilePicker()
    pk_alt = ft.FilePicker()
    pk_cot = ft.FilePicker()
    pk_hms = ft.FilePicker()
    
    # Save Pickers (Vinculados a las funciones seguras)
    pk_s_rac = ft.FilePicker(on_result=lambda e: save_csv_safe(e, page.session.res_racional))
    pk_s_chow = ft.FilePicker(on_result=lambda e: save_csv_safe(e, page.session.res_chow))
    pk_s_img = ft.FilePicker(on_result=save_img_safe)

    page.overlay.extend([pk_area, pk_int, pk_alt, pk_cot, pk_hms, pk_s_rac, pk_s_chow, pk_s_img])

    # --- UI CONTROLS ---
    tabla_cuencas = ft.DataTable(columns=[ft.DataColumn(ft.Text(t)) for t in ["ID","Área","LCP","Estado","Editar"]], rows=[])
    
    st_area = ft.Text("No cargado", color="orange", size=12)
    st_cota = ft.Text("No cargado", color="orange", size=12)
    st_hms = ft.Text("No cargado (Opcional)", color="grey", size=12)
    st_int = ft.Text("No cargado", color="orange", size=12)
    st_alt = ft.Text("No cargado", color="orange", size=12) 
    
    log_txt = ft.Text("Listo.", color="grey")
    btn_calc = ft.ElevatedButton("CALCULAR", icon=ft.Icons.CALCULATE, disabled=True, style=ft.ButtonStyle(color="#00ff41"))
    tabs_res = ft.Tabs(selected_index=0, visible=False)

    def upd_tbl():
        if page.session.df_cuencas_base is None: return
        rows, rdy = [], True
        for i, r in page.session.df_cuencas_base.iterrows():
            cfg = page.session.datos_cuencas_config.get(str(i))
            if not cfg: rdy = False
            rows.append(ft.DataRow([ft.DataCell(ft.Text(str(i))), ft.DataCell(ft.Text(f"{r['area']:.2f}")), ft.DataCell(ft.Text(str(cfg['LCP']) if cfg else "-")),
                ft.DataCell(ft.Icon(ft.Icons.CHECK if cfg else ft.Icons.WARNING, color="green" if cfg else "orange")),
                ft.DataCell(ft.IconButton(ft.Icons.EDIT, on_click=lambda e, c=str(i): open_conf(c)))]))
        tabla_cuencas.rows = rows; btn_calc.disabled = not rdy; page.update()

    # Load Logic
    def load(e, k, s, i=0):
        if e.files:
            try: setattr(page.session, k, pd.read_csv(e.files[0].path, header=0, index_col=i)); s.value = "OK"; s.color = "green"; upd_tbl()
            except: s.value = "Error"; s.color = "red"
            page.update()

    # Linking Loaders
    pk_area.on_result = lambda e: load(e, "df_cuencas_base", st_area, 0)
    pk_int.on_result = lambda e: load(e, "df_intensidad", st_int, 0)
    pk_alt.on_result = lambda e: load(e, "df_altura", st_alt, 0)
    pk_cot.on_result = lambda e: load(e, "df_cotas", st_cota, None)
    pk_hms.on_result = lambda e: load(e, "df_hms", st_hms, 0)

    # Reset
    def on_reset(e):
        page.session.datos_cuencas_config = {}
        for k in ["df_cuencas_base", "df_cotas", "df_hms", "df_altura", "res_racional", "res_chow"]: setattr(page.session, k, None)
        tabla_cuencas.rows = []
        for s in [st_area, st_cota, st_int, st_alt]: s.value, s.color = "No cargado", "orange"
        st_hms.value = "No cargado (Opcional)"; st_hms.color = "grey"
        btn_calc.disabled = True; tabs_res.visible = False; log_txt.value = "Reiniciado."
        page.update()

    # Config Modal
    c_id = ft.Text("");inp_lcp = ft.TextField(label="LCP", width=100)
    col_usos = ft.Column(scroll=ft.ScrollMode.ALWAYS, height=250)
    
    def add_row(e, d=None):
        row = ft.Row([
            ft.TextField(value=d['pct'] if d else "", label="%", width=80),
            ft.Dropdown(options=[ft.dropdown.Option(k,v) for k,v in OPCIONES_C.items()], value=d['c'] if d else None, width=400, label="Coeficiente de escurrimiento C"),
            ft.Dropdown(options=[ft.dropdown.Option(k,v) for k,v in OPCIONES_N.items()], value=d['n'] if d else None, width=380, label="Números N de curva de escurrimiento"),
            ft.Dropdown(options=[ft.dropdown.Option(x) for x in "ABCD"], value=d['g'] if d else None, width=180, label="Grupo Hidrológico"),
            ft.IconButton(ft.Icons.DELETE, icon_color="red", on_click=lambda e: col_usos.controls.remove(row) or page.update())
        ])
        col_usos.controls.append(row); page.update()

    def save_conf(e):
        usos, tot = [], 0
        for r in col_usos.controls:
            try: p = float(r.controls[0].value or 0)
            except: p = 0
            if p>100: return page.snack_bar.open(ft.Text("Error > 100%")); page.update()
            tot += p
            usos.append({"pct":p, "c":int(r.controls[1].value or 0), "n":int(r.controls[2].value or 0), "g":r.controls[3].value or "A"})
        
        if tot>100: page.snack_bar = ft.SnackBar(ft.Text("Suma > 100%"), bgcolor="red"); page.snack_bar.open = True; page.update(); return
        page.session.datos_cuencas_config[c_id.value] = {"LCP": float(inp_lcp.value or 0), "usos": usos}
        dlg.open = False; upd_tbl(); page.update()

    dlg = ft.AlertDialog(title=ft.Text("Configuración y coeficientes de la cuenca"), content=ft.Container(ft.Column([c_id, inp_lcp, ft.ElevatedButton("Agregar un área interna", on_click=lambda e: add_row(e)), col_usos]), height=400, width=1100),
        actions=[ft.TextButton("Cancelar", on_click=lambda e: setattr(dlg, 'open', False) or page.update()), ft.ElevatedButton("Guardar", on_click=save_conf)])
    
    page.overlay.append(dlg)

    def open_conf(cid):
        c_id.value = str(cid); d = page.session.datos_cuencas_config.get(str(cid), {})
        inp_lcp.value = d.get("LCP", ""); col_usos.controls.clear()
        for u in d.get("usos", []): add_row(None, u)
        if not d.get("usos"): add_row(None)
        dlg.open = True; page.update()

    # Process
    def run(e):
        if not page.session.df_intensidad is not None: log_txt.value="Falta Intensidad"; log_txt.color="red"; page.update(); return
        if not page.session.df_altura is not None: log_txt.value="Falta Altura (P)"; log_txt.color="red"; page.update(); return
        if not page.session.df_cotas is not None: log_txt.value="Falta Cotas"; log_txt.color="red"; page.update(); return
        
        log_txt.value = "Calculando..."; page.update()
        try:
            df = page.session.df_cuencas_base.copy()
            lcp, pct, c, n, g = [],[],[],[],[]
            for i in df.index:
                d = page.session.datos_cuencas_config[str(i)]
                lcp.append(d['LCP']); pct.append(",".join([str(u['pct']) for u in d['usos']]))
                c.append(",".join([str(u['c']) for u in d['usos']])); n.append(",".join([str(u['n']) for u in d['usos']])); g.append(",".join([str(u['g']) for u in d['usos']]))
            df['LCP'], df['Porcentaje_terreno'], df['Coeficiente_C_por_Cuenca'], df['indice_Uso_terreno'], df['Grupo_hidrologico'] = lcp, pct, c, n, g
            
            res_r, res_c, l = calcular_coeficientes_y_gastos(df, page.session.df_intensidad, page.session.df_altura, page.session.df_cotas)
            if res_r is not None:
                page.session.res_racional = res_r; page.session.res_chow = res_c
                gs = generar_graficos_comparativos(res_r, res_c, page.session.df_hms)
                
                def mk_dt(d): return ft.DataTable(columns=[ft.DataColumn(ft.Text(c)) for c in ["ID"]+list(d.columns)], rows=[ft.DataRow([ft.DataCell(ft.Text(str(i)))]+[ft.DataCell(ft.Text(str(r[c]))) for c in d.columns]) for i,r in d.iterrows()])
                
                t1 = ft.Tab("Racional", ft.Column([ft.ElevatedButton("CSV", icon=ft.Icons.SAVE, on_click=lambda _: pk_s_rac.save_file("Racional.csv")), ft.Row([mk_dt(res_r)], scroll=ft.ScrollMode.AUTO)], scroll=ft.ScrollMode.AUTO))
                t2 = ft.Tab("Chow", ft.Column([ft.ElevatedButton("CSV", icon=ft.Icons.SAVE, on_click=lambda _: pk_s_chow.save_file("Chow.csv")), ft.Row([mk_dt(res_c)], scroll=ft.ScrollMode.AUTO)], scroll=ft.ScrollMode.AUTO))
                
                imgs = ft.Column(scroll=ft.ScrollMode.AUTO, spacing=20)
                for tr, b64 in gs:
                    # FIX LAMBDA CAPTURE: default arguments (b=b64, t=tr)
                    btn_save = ft.ElevatedButton("Guardar", icon=ft.Icons.SAVE, on_click=lambda e, b=b64, t=tr: trigger_save_graph(b, f"Graph_TR{t}.png"))
                    imgs.controls.append(ft.Column([ft.Text(f"TR {tr}"), ft.Image(src_base64=b64, fit=ft.ImageFit.CONTAIN), btn_save], horizontal_alignment=ft.CrossAxisAlignment.CENTER))
                
                tabs_res.tabs = [t1, t2, ft.Tab("Gráficos", imgs)]; tabs_res.visible = True; log_txt.value = "OK"; log_txt.color = "green"
            else: log_txt.value = f"Err: {l}"
        except Exception as ex: log_txt.value = f"Fatal: {ex}"
        page.update()

    btn_calc.on_click = run

    v_area = ft.Container(ft.Column([ft.Text("Paso 1: Cuencas", color="#00ff41", size=20), ft.Row([ft.ElevatedButton("Cargar Áreas", icon=ft.Icons.UPLOAD, on_click=lambda _: pk_area.pick_files()), st_area])]), visible=True, padding=20)
    v_conf = ft.Container(ft.Column([ft.Text("Paso 2: Suelos", color="#00ff41", size=20), ft.Container(ft.Column([tabla_cuencas], scroll=ft.ScrollMode.AUTO), height=800, border=ft.border.all(1, "grey"))]), visible=False, padding=20)
    v_data = ft.Container(ft.Column([
        ft.Text("Paso 3: Entradas", color="#00ff41", size=20), 
        ft.Text("Método Racional (Intensidad mm/hr):"),
        ft.Row([ft.ElevatedButton("Cargar I-D-TR.csv", on_click=lambda _: pk_int.pick_files()), st_int]),
        ft.Text("Método Chow (Altura mm):"),
        ft.Row([ft.ElevatedButton("Cargar Ap-D-TR.csv", on_click=lambda _: pk_alt.pick_files()), st_alt]),
        ft.Divider(),
        ft.Text("Geometría y Comparación:"),
        ft.Row([ft.ElevatedButton("Cargar Cotas", on_click=lambda _: pk_cot.pick_files()), st_cota]), 
        ft.Row([ft.ElevatedButton("HMS (Op)", on_click=lambda _: pk_hms.pick_files()), st_hms])
    ]), visible=False, padding=20)
    v_res = ft.Container(ft.Column([ft.Text("Paso 4: Resultados", color="#00ff41", size=20), ft.Row([btn_calc, ft.IconButton(ft.Icons.REFRESH, on_click=on_reset)]), log_txt, ft.Container(tabs_res, expand=True)], scroll=ft.ScrollMode.AUTO), visible=False, padding=20, expand=True)

    views = [v_area, v_conf, v_data, v_res]
    def chg_view(e):
        for i,v in enumerate(views): v.visible = (i == e.control.selected_index)
        page.update()

    rail = ft.NavigationRail(selected_index=0, label_type=ft.NavigationRailLabelType.ALL, min_width=100, min_extended_width=400,
        leading=ft.Column([ft.IconButton(ft.Icons.ARROW_BACK, on_click=on_back_to_menu), ft.Text("Gastos")], spacing=10),
        destinations=[ft.NavigationRailDestination(icon=i, label=l) for i,l in [(ft.Icons.LANDSCAPE,"Cuencas"), (ft.Icons.SETTINGS,"Suelos"), (ft.Icons.DATA_ARRAY,"Entradas"), (ft.Icons.ANALYTICS,"Resultados")]],
        on_change=chg_view)

    return ft.Row([rail, ft.VerticalDivider(width=1), ft.Column(views, expand=True)], expand=True)