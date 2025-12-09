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
# 1. DATOS DE REFERENCIA (Constantes)
# ==========================================

OPCIONES_C = {
    1: '1. Urbanizada/Superficie asfáltica',
    2: '2. Urbanizada/Concreto-azotea',
    3: '3. Área con pasto (Pobre)/plano (0-2%)',
    4: '4. Área con pasto (Pobre)/promedio (2-7%)',
    5: '5. Área con pasto (Pobre)/pendiente (> 7%)',
    6: '6. Área con pasto (Media)/plano (0-2%)',
    7: '7. Área con pasto (Media)/promedio (2-7%)',
    8: '8. Área con pasto (Media)/pendiente (> 7%)',
    9: '9. Áreascon pasto (Buena)/plano (0-2 %)',
    10: '10. Áreas con pasto (Buena)/promedio (2-7%)',
    11: '11. Áreas con pasto (Buena)/pendiente (> 7%)',
    12: '12. Rural - Cultivo/plano (0-2%)',
    13: '13. Rural - Cultivo/promedio (2-7%)',
    14: '14. Rural - Cultivo/pendiente (> 7%)',
    15: '15. Rural - Pastizal/plano (0-2%)',
    16: '16. Rural - Pastizal/promedio (2-7%)',
    17: '17. Rural - Pastizal/pendiente (> 7%)',
    18: '18. Bosque y monte/plano (0-2%)',
    19: '19. Bosque y monte/promedio (2-7%)',
    20: '20. Bosque y monte/pendiente (> 7%)'
}

C_DATA_VALUES = {
    '2': [0.73, 0.75, 0.32, 0.37, 0.40, 0.25, 0.33, 0.37, 0.21, 0.29, 0.34, 0.31, 0.35, 0.39, 0.25, 0.33, 0.37, 0.22, 0.31, 0.35],
    '5': [0.77, 0.80, 0.34, 0.40, 0.43, 0.28, 0.36, 0.40, 0.23, 0.32, 0.37, 0.34, 0.38, 0.42, 0.28, 0.36, 0.40, 0.25, 0.34, 0.39],
    '10': [0.81, 0.83, 0.37, 0.43, 0.45, 0.30, 0.38, 0.42, 0.25, 0.35, 0.40, 0.36, 0.41, 0.44, 0.30, 0.38, 0.42, 0.28, 0.36, 0.41],
    '20': [0.86, 0.88, 0.40, 0.46, 0.49, 0.34, 0.42, 0.46, 0.29, 0.39, 0.44, 0.40, 0.44, 0.48, 0.34, 0.42, 0.46, 0.31, 0.41, 0.45],
    '50': [0.90, 0.92, 0.44, 0.49, 0.52, 0.37, 0.45, 0.49, 0.32, 0.42, 0.47, 0.43, 0.48, 0.51, 0.37, 0.45, 0.49, 0.35, 0.43, 0.48],
    '100': [0.95, 0.97, 0.47, 0.53, 0.55, 0.41, 0.49, 0.53, 0.36, 0.46, 0.51, 0.47, 0.51, 0.54, 0.41, 0.49, 0.53, 0.39, 0.47, 0.52],
    '500': [1.00, 1.00, 0.58, 0.61, 0.62, 0.53, 0.58, 0.60, 0.49, 0.56, 0.58, 0.57, 0.60, 0.61, 0.53, 0.58, 0.60, 0.48, 0.56, 0.58],
    '1000': [1.00, 1.00, 0.65, 0.68, 0.70, 0.60, 0.65, 0.68, 0.56, 0.63, 0.65, 0.64, 0.68, 0.70, 0.60, 0.65, 0.68, 0.55, 0.62, 0.65],
    '10000': [1.00, 1.00, 0.82, 0.85, 0.86, 0.78, 0.82, 0.84, 0.75, 0.80, 0.82, 0.81, 0.84, 0.85, 0.78, 0.82, 0.84, 0.72, 0.79, 0.81]
}
C_DF = pd.DataFrame(C_DATA_VALUES, index=range(1, 21))

OPCIONES_N = {
    1: '1. Parque, campo abierto, Cancha deportiva-Condicion buena (75% pasto)',
    2: '2. Parque, campo abierto, Cancha deportiva-Condicion regular (50-75% pasto)',
    3: '3. Parque, campo abierto, Cancha deportiva-Condicion pobre (<50% pasto)',
    4: '4. Área comercial (85% impermeable)',
    5: '5. Distrito industrial (72% impermeable)',
    6: '6. Zona residencial (<500m2/65% impermeable)',
    7: '7. Zona residencial (1000m2/38% impermeable)',
    8: '8. Zona residencial (<1350m2/30% impermeable)',
    9: '9. Zona residencial (<2000m2/25% impermeable)',
    10: '10. Zona residencial (<4000m2/20% impermeable)',
    11: '11. Zona residencial (<8000m2/12% impermeable)',
    12: '12. Calzada, tejado, estacionamiento pavimentado, etc.',
    13: '13. Calle Pavimentada con guarnición y alcantarillado',
    14: '14. Camino pavimentado, derecho de via y canales',
    15: '15. Camino engravado - derecho de via',
    16: '16. Camino de arcilla -derecho de via',
    17: '17. Área urbana en desarrollo (nivelado sin vegetación)'
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
    """Calcula Gastos usando Racional y Chow."""
    log = []
    try:
        tr_cols = ['2', '5', '10', '20', '50', '100', '500', '1000', '10000']
        
        # Preparar índices
        if 'TR (AÑOS)' in df_intensidad.columns: df_intensidad.set_index('TR (AÑOS)', inplace=True)
        df_intensidad.index = df_intensidad.index.astype(int)
        
        if 'TR (AÑOS)' in df_altura.columns: df_altura.set_index('TR (AÑOS)', inplace=True)
        df_altura.index = df_altura.index.astype(int)

        # 1. Coeficientes Ponderados
        def _calc_pond(row):
            pcts = obtener_lista_desde_string(row['Porcentaje_terreno'], float)
            idxs_c = obtener_lista_desde_string(row['Coeficiente_C_por_Cuenca'], int)
            c_vals = {}
            for tr in tr_cols:
                val_c = sum((p/100)*C_DF.loc[i, tr] for p, i in zip(pcts, idxs_c) if i in C_DF.index)
                c_vals[f"C_pond_TR{tr}"] = val_c
            
            usos_n = obtener_lista_desde_string(row['indice_Uso_terreno'], int)
            grupos = obtener_lista_desde_string(row['Grupo_hidrologico'], str)
            n_val = sum((p/100)*N_DF.loc[i, g.strip().upper()] for p, i, g in zip(pcts, usos_n, grupos) if i in N_DF.index)
            return pd.Series({**c_vals, 'N_pond': n_val})
       
        df_pond = df_configurado.apply(_calc_pond, axis=1)
        df_completo = pd.concat([df_configurado, df_pond], axis=1)
        
        # 2. Tc y Pendiente
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

        cols_out = [f"TR_{tr}" for tr in tr_cols]
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
                        C = row[f"C_pond_TR{tr}"] 
                        df_racional.at[cid, f"TR_{tr}"] = round(0.278 * C * I * area, 3)

        # CHOW
        for cid, row in df_completo.iterrows():
            N = row['N_pond']
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
        return df_racional, df_chow, df_completo, "\n".join(log)

    except Exception as e:
        return None, None, None, f"Error: {e}\n{traceback.format_exc()}"

def generar_graficos_comparativos(df_racional, df_chow, df_hms):
    graficos = []
    
    def normalize_index(idx):
        return str(idx).strip().replace('.0', '')
    
    df_racional.index = df_racional.index.map(normalize_index)
    df_chow.index = df_chow.index.map(normalize_index)
    
    hms_map = {}
    if df_hms is not None and not df_hms.empty:
        df_hms.index = df_hms.index.map(normalize_index)
        for c in df_hms.columns:
            clean = ''.join(filter(str.isdigit, str(c)))
            if clean: hms_map[f"TR_{clean}"] = c

    trs = ['2', '5', '10', '20', '50', '100', '500', '1000', '10000']
    
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
                    for c in idx:
                        if c in df_hms.index:
                            val = df_hms.loc[c, hms_col]
                            val_h.append(float(val) if pd.notna(val) else 0.0)
                        else:
                            val_h.append(0.0)
                else:
                    val_h = [0.0]*len(idx)

                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.arange(len(idx)); w = 0.25
                
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
    
    # --- 1. Inicialización de Sesión ---
    keys = ["datos_cuencas_config", "df_cuencas_base", "df_cotas", "df_hms", "df_intensidad", "df_altura", "res_racional", "res_chow", "df_variables", "temp_graph_b64"]
    for k in keys: 
        if not hasattr(page.session, k): setattr(page.session, k, {} if k=="datos_cuencas_config" else None)

    # --- 2. Helpers Visuales ---
    def create_datatable(df):
        if df is None or df.empty: return ft.Text("Sin datos")
        try:
            df_view = df.head(100).copy()
            # Asegurar strings para evitar error de tipos
            df_view.index = df_view.index.astype(str)
            
            columns = [ft.DataColumn(ft.Text("ID", color="#00ff41"))] + [ft.DataColumn(ft.Text(str(c))) for c in df_view.columns]
            rows = []
            for i, r in df_view.iterrows():
                cells = [ft.DataCell(ft.Text(str(i)))] + [ft.DataCell(ft.Text(str(r[c]))) for c in df_view.columns]
                rows.append(ft.DataRow(cells))
            
            # Tabla sin altura fija para evitar error
            return ft.DataTable(columns=columns, rows=rows, border=ft.border.all(1, "#333333"))
        except: return ft.Text("Error visualizando tabla", color="red")

    def save_csv_safe(e, df):
        if e.path and df is not None:
            try:
                path = e.path if e.path.endswith(".csv") else f"{e.path}.csv"
                if 'TR (AÑOS)' in df.columns: df.set_index('TR (AÑOS)').to_csv(path)
                elif 'TR' in df.columns: df.set_index('TR').to_csv(path)
                else: df.to_csv(path)
                page.snack_bar = ft.SnackBar(ft.Text("CSV Guardado Exitosamente"), bgcolor="green", open=True)
            except Exception as ex:
                page.snack_bar = ft.SnackBar(ft.Text(f"Error guardando CSV: {ex}"), bgcolor="red", open=True)
            page.update()

    def trigger_save_graph(b64, name):
        if b64:
            page.session.temp_graph_b64 = b64
            pk_s_img.save_file(file_name=name)

    def save_img_safe(e):
        if e.path and hasattr(page.session, "temp_graph_b64"):
            try:
                path = e.path if e.path.endswith(".png") else f"{e.path}.png"
                with open(path, "wb") as f: f.write(base64.b64decode(page.session.temp_graph_b64))
                page.snack_bar = ft.SnackBar(ft.Text("Gráfico Guardado"), bgcolor="green", open=True)
            except Exception as ex:
                page.snack_bar = ft.SnackBar(ft.Text(f"Error guardando IMG: {ex}"), bgcolor="red", open=True)
            page.update()

    # --- 3. Pickers ---
    pk_area = ft.FilePicker()
    pk_int = ft.FilePicker()
    pk_alt = ft.FilePicker()
    pk_cot = ft.FilePicker()
    pk_hms = ft.FilePicker()
    
    pk_s_rac = ft.FilePicker(on_result=lambda e: save_csv_safe(e, page.session.res_racional))
    pk_s_chow = ft.FilePicker(on_result=lambda e: save_csv_safe(e, page.session.res_chow))
    pk_s_vars = ft.FilePicker(on_result=lambda e: save_csv_safe(e, page.session.df_variables))
    pk_s_img = ft.FilePicker(on_result=save_img_safe)

    page.overlay.extend([pk_area, pk_int, pk_alt, pk_cot, pk_hms, pk_s_rac, pk_s_chow, pk_s_vars, pk_s_img])

    # --- 4. Controles UI ---
    tabla_cuencas = ft.DataTable(columns=[ft.DataColumn(ft.Text(t)) for t in ["ID","Área","LCP","Estado","Editar"]], rows=[])
    
    st_area = ft.Text("No cargado", color="orange", size=12)
    st_cota = ft.Text("No cargado", color="orange", size=12)
    st_hms = ft.Text("No cargado (Opcional)", color="grey", size=12)
    st_int = ft.Text("No cargado", color="orange", size=12)
    st_alt = ft.Text("No cargado", color="orange", size=12) 
    
    # Etiquetas de Ayuda Visual (RESTAURADAS)
    help_area = ft.Text("(Columnas esperadas: ID [Index], area)", size=10, color="grey")
    help_int = ft.Text("(Columnas: TR (AÑOS) [Index], duraciones...)", size=10, color="grey")
    help_alt = ft.Text("(Columnas: TR (AÑOS) [Index], duraciones...)", size=10, color="grey")
    help_cot = ft.Text("(Columnas: cuenca, distancia1, distancia2, cota mayor, cota menor)", size=10, color="grey")
    help_hms = ft.Text("(Columnas: Index [ID], TR_X...)", size=10, color="grey")
    
    validation_msg = ft.Text("", size=12, weight="bold")
    log_txt = ft.Text("Listo.", color="grey")
    btn_calc = ft.ElevatedButton("CALCULAR", icon=ft.Icons.CALCULATE, disabled=True, style=ft.ButtonStyle(color="#00ff41"))
    tabs_res = ft.Tabs(selected_index=0, visible=False)

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

    def validar_nombres_cuencas():
        df_area = page.session.df_cuencas_base
        df_cotas = page.session.df_cotas
        
        if df_area is not None and df_cotas is not None:
            ids_area = set(str(x).strip().replace('.0','') for x in df_area.index)
            if 'cuenca' not in df_cotas.columns:
                 validation_msg.value = "⚠️ Error: El archivo de Cotas no tiene columna 'cuenca'."
                 validation_msg.color = "red"
                 return
            ids_cotas = set(str(x).strip().replace('.0','') for x in df_cotas['cuenca'].unique())
            
            diff_a_c = ids_area - ids_cotas 
            diff_c_a = ids_cotas - ids_area 
            
            if diff_a_c or diff_c_a:
                msg = "⚠️ Discrepancia en IDs:\n"
                if diff_a_c: msg += f"Faltan en Cotas: {list(diff_a_c)[:3]}...\n"
                if diff_c_a: msg += f"Sobran en Cotas: {list(diff_c_a)[:3]}..."
                validation_msg.value = msg
                validation_msg.color = "orange"
            else:
                validation_msg.value = "✅ Coincidencia Correcta."
                validation_msg.color = "#00ff41"
        else:
            validation_msg.value = ""

    def load(e, k, s, i=0):
        if e.files:
            try: 
                setattr(page.session, k, pd.read_csv(e.files[0].path, header=0, index_col=i))
                s.value = "OK"; s.color = "green"
                upd_tbl()
                if k in ["df_cuencas_base", "df_cotas"]: validar_nombres_cuencas()
            except Exception as ex: 
                s.value = "Error"; s.color = "red"
                print(f"Error carga: {ex}")
            page.update()

    pk_area.on_result = lambda e: load(e, "df_cuencas_base", st_area, 0)
    pk_int.on_result = lambda e: load(e, "df_intensidad", st_int, 0)
    pk_alt.on_result = lambda e: load(e, "df_altura", st_alt, 0)
    pk_cot.on_result = lambda e: load(e, "df_cotas", st_cota, None) 
    pk_hms.on_result = lambda e: load(e, "df_hms", st_hms, 0)

    # --- RESET ---
    def on_reset(e):
        page.session.datos_cuencas_config = {}
        for k in ["df_cuencas_base", "df_cotas", "df_hms", "df_altura", "res_racional", "res_chow", "df_variables"]: 
            setattr(page.session, k, None)
        
        tabla_cuencas.rows = []
        for s in [st_area, st_cota, st_int, st_alt]: s.value = "No cargado"; s.color = "orange"
        st_hms.value = "No cargado (Opcional)"; st_hms.color = "grey"
        validation_msg.value = ""
        btn_calc.disabled = True; tabs_res.visible = False; log_txt.value = "Reiniciado."
        page.update()

    # Config Modal
    c_id = ft.Text(""); inp_lcp = ft.TextField(label="LCP", width=100)
    col_usos = ft.Column(scroll=ft.ScrollMode.ALWAYS, height=250)
    
    def add_row(e, d=None):
        row = ft.Row([
            ft.TextField(value=d['pct'] if d else "", label="%", width=80),
            ft.Dropdown(options=[ft.dropdown.Option(k,v) for k,v in OPCIONES_C.items()], value=d['c'] if d else None, width=400, label="C"),
            ft.Dropdown(options=[ft.dropdown.Option(k,v) for k,v in OPCIONES_N.items()], value=d['n'] if d else None, width=380, label="N"),
            ft.Dropdown(options=[ft.dropdown.Option(x) for x in "ABCD"], value=d['g'] if d else None, width=100, label="G"),
            ft.IconButton(ft.Icons.DELETE, icon_color="red", on_click=lambda e: col_usos.controls.remove(row) or page.update())
        ])
        col_usos.controls.append(row); page.update()

    def save_conf(e):
        usos, tot = [], 0
        for r in col_usos.controls:
            try: p = float(r.controls[0].value or 0)
            except: p = 0
            tot += p
            usos.append({"pct":p, "c":int(r.controls[1].value or 0), "n":int(r.controls[2].value or 0), "g":r.controls[3].value or "A"})
        
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
        # LÓGICA AUTO-RUN: Si ya hay resultados guardados en memoria, saltamos el cálculo.
        if page.session.res_racional is not None and e is None:
            pass # Usamos datos de memoria
        else:
            if not all([page.session.df_intensidad is not None, page.session.df_altura is not None, page.session.df_cotas is not None]):
                log_txt.value="Faltan datos de entrada"; log_txt.color="red"; page.update(); return
            
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
                
                res_r, res_c, df_vars, l = calcular_coeficientes_y_gastos(df, page.session.df_intensidad, page.session.df_altura, page.session.df_cotas)
                page.session.res_racional = res_r
                page.session.res_chow = res_c
                page.session.df_variables = df_vars
                log_txt.value = "Cálculo finalizado."
            except Exception as ex:
                log_txt.value = f"Error: {ex}"; page.update(); return

        # --- FASE DE GRAFICADO (Siempre se ejecuta) ---
        if page.session.res_racional is not None and page.session.res_chow is not None:
            gs = generar_graficos_comparativos(page.session.res_racional, page.session.res_chow, page.session.df_hms)
            
            t1 = ft.Tab("Racional", ft.Column([ft.ElevatedButton("CSV", icon=ft.Icons.SAVE, on_click=lambda _: pk_s_rac.save_file("Racional.csv")), ft.Row([create_datatable(page.session.res_racional)], scroll=ft.ScrollMode.AUTO)], scroll=ft.ScrollMode.AUTO))
            t2 = ft.Tab("Chow", ft.Column([ft.ElevatedButton("CSV", icon=ft.Icons.SAVE, on_click=lambda _: pk_s_chow.save_file("Chow.csv")), ft.Row([create_datatable(page.session.res_chow)], scroll=ft.ScrollMode.AUTO)], scroll=ft.ScrollMode.AUTO))
            t3 = ft.Tab("Variables", ft.Column([ft.ElevatedButton("CSV", icon=ft.Icons.SAVE, on_click=lambda _: pk_s_vars.save_file("Variables.csv")), ft.Row([create_datatable(page.session.df_variables)], scroll=ft.ScrollMode.AUTO)], scroll=ft.ScrollMode.AUTO))
            
            imgs = ft.Column(scroll=ft.ScrollMode.AUTO, spacing=20)
            for tr, b64 in gs:
                btn_save = ft.ElevatedButton("Guardar", icon=ft.Icons.SAVE, on_click=lambda e, b=b64, t=tr: trigger_save_graph(b, f"Graph_TR{t}.png"))
                imgs.controls.append(ft.Column([ft.Text(f"TR {tr}"), ft.Image(src_base64=b64, fit=ft.ImageFit.CONTAIN), btn_save], horizontal_alignment=ft.CrossAxisAlignment.CENTER))
            
            tabs_res.tabs = [t1, t2, t3, ft.Tab("Gráficos", imgs)]; tabs_res.visible = True
            log_txt.color = "green"
            if e: page.update()

    btn_calc.on_click = run

    # --- 7. VISTAS ---
    
    v_area = ft.Container(ft.Column([
        ft.Text("Paso 1: Cuencas", color="#00ff41", size=20),
        ft.Row([ft.ElevatedButton("Cargar Áreas", icon=ft.Icons.UPLOAD, on_click=lambda _: pk_area.pick_files()), st_area]),
        help_area # <--- RESTAURADO
    ]), padding=20)

    v_conf = ft.Container(ft.Column([
        ft.Text("Paso 2: Suelos", color="#00ff41", size=20),
        ft.Text("Área (Km^2) / Longitud Cauce Principal (Km)", color="#00ff41", size=16),
        ft.Container(ft.Column([tabla_cuencas], scroll=ft.ScrollMode.AUTO), height=800, border=ft.border.all(1, "grey"))
    ]), padding=20, visible=False)
    
    v_data = ft.Container(ft.Column([
        ft.Text("Paso 3: Entradas", color="#00ff41", size=20),
        ft.Text("Método Racional (Intensidad mm/hr):"),
        ft.Row([ft.ElevatedButton("Cargar I-D-TR", on_click=lambda _: pk_int.pick_files()), st_int]),
        help_int, # <--- RESTAURADO
        ft.Text("Método Chow (Altura mm):"),
        ft.Row([ft.ElevatedButton("Cargar Ap-D-TR", on_click=lambda _: pk_alt.pick_files()), st_alt]),
        help_alt, # <--- RESTAURADO
        ft.Divider(),
        ft.Text("Geometría y Comparación:"),
        ft.Row([ft.ElevatedButton("Cargar Cotas", on_click=lambda _: pk_cot.pick_files()), st_cota]), 
        help_cot, # <--- RESTAURADO
        validation_msg, 
        ft.Row([ft.ElevatedButton("HMS (Op)", on_click=lambda _: pk_hms.pick_files()), st_hms]),
        help_hms # <--- RESTAURADO
    ]), padding=20, visible=False)
    
    v_res = ft.Container(ft.Column([
        ft.Text("Paso 4: Resultados", color="#00ff41", size=20), 
        ft.Row([btn_calc, ft.IconButton(ft.Icons.REFRESH, on_click=lambda e: on_reset(e))]), 
        log_txt, 
        ft.Container(tabs_res, expand=True)
    ], scroll=ft.ScrollMode.AUTO), visible=False, padding=20, expand=True)

    views = [v_area, v_conf, v_data, v_res]
    def chg_view(e):
        for i,v in enumerate(views): v.visible = (i == e.control.selected_index)
        page.update()

    rail = ft.NavigationRail(selected_index=0, label_type=ft.NavigationRailLabelType.ALL, min_width=100, min_extended_width=400, leading=ft.Column([ft.IconButton(ft.Icons.ARROW_BACK, on_click=on_back_to_menu), ft.Text("Gastos")], spacing=10), destinations=[ft.NavigationRailDestination(icon=i, label=l) for i,l in [(ft.Icons.LANDSCAPE,"Cuencas"), (ft.Icons.SETTINGS,"Suelos"), (ft.Icons.DATA_ARRAY,"Entradas"), (ft.Icons.ANALYTICS,"Resultados")]], on_change=chg_view)

    # --- 8. RESTAURACIÓN DE ESTADO ---
    if page.session.df_cuencas_base is not None: st_area.value, st_area.color = "Recuperado", "green"
    if page.session.df_intensidad is not None: st_int.value, st_int.color = "Recuperado", "green"
    if page.session.df_altura is not None: st_alt.value, st_alt.color = "Recuperado", "green"
    if page.session.df_cotas is not None: st_cota.value, st_cota.color = "Recuperado", "green"
    if page.session.df_hms is not None: st_hms.value, st_hms.color = "Recuperado", "green"
    
    upd_tbl()
    
    # Auto-ejecutar si hay datos
    if page.session.res_racional is not None:
        run(None) 

    return ft.Row([rail, ft.VerticalDivider(width=1), ft.Column(views, expand=True)], expand=True)