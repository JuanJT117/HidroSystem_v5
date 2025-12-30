import pandas as pd
import numpy as np
import math
import traceback
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
from scipy import interpolate
import hidrologia_mx  # Asegúrate de que este archivo existe y tiene las tablas

# Configurar backend de Matplotlib para evitar errores de GUI
matplotlib.use('Agg')

# ==========================================
# 1. FUNCIONES AUXILIARES
# ==========================================

def redondear_tc(tc):
    """Redondeo de seguridad para visualización, no para cálculo."""
    if tc <= 10: return 10
    lower = (tc // 5) * 5
    return lower if tc - lower < 1 else lower + 5

def obtener_lista_desde_string(cadena, tipo_dato):
    try:
        return [tipo_dato(x.strip()) for x in str(cadena).split(',')]
    except:
        return []

def interpolar_valor_curva(target_x, df_data, col_name):
    """
    Interpola (linealmente) un valor Y para un X dado en un DataFrame.
    Vital para obtener la Intensidad exacta cuando Tc no coincide con los intervalos del CSV.
    """
    try:
        # Asegurar que el índice es numérico y está ordenado
        x_vals = df_data.index.astype(float).values
        y_vals = df_data[col_name].astype(float).values
        
        # Si el valor está fuera de rango, extrapolar con cuidado o usar límites
        if target_x < x_vals[0]: return y_vals[0] # No bajar del mínimo
        
        # Función de interpolación
        f = interpolate.interp1d(x_vals, y_vals, kind='linear', fill_value="extrapolate")
        val = float(f(target_x))
        return max(val, 0) # No permitir negativos
    except Exception as e:
        return 0.0

# ==========================================
# 2. MOTORES HIDROLÓGICOS
# ==========================================

def calcular_tc_temez(L_km, S_m_m):
    """Tc = 0.3 * L^0.76 * S^-0.19 (Resultado en minutos)"""
    if L_km <= 0 or S_m_m <= 0: return 10.0
    return 0.3 * (L_km**0.76) * (S_m_m**-0.19) * 60 

def generar_hietograma_bloques_alternos(duracion_total_min, dt_min, df_intensidad, tr_col):
    """Construye la tormenta de diseño (Hietograma) desde la curva IDF."""
    try:
        num_bloques = int(duracion_total_min / dt_min)
        if num_bloques % 2 == 0: num_bloques += 1 
        
        # Eje de tiempo para la curva de masa
        tiempos = np.arange(dt_min, (num_bloques + 1) * dt_min, dt_min)
        profundidades = []
        
        # Preparar interpolador de la curva IDF seleccionada
        x_idf = df_intensidad.index.astype(float).values
        y_idf = df_intensidad[tr_col].astype(float).values
        f_int = interpolate.interp1d(x_idf, y_idf, kind='linear', fill_value="extrapolate")
        
        for t in tiempos:
            # P = I * t (Intensidad * Duración)
            i_t = float(f_int(t))
            p_accum = (i_t * t) / 60.0 
            profundidades.append(p_accum)
            
        # Bloques incrementales
        bloques = np.diff(profundidades, prepend=0)
        
        # Ordenamiento Alterno (Pico al centro)
        bloques_ord = np.sort(bloques)[::-1] 
        hietograma = np.zeros(len(bloques_ord))
        centro = len(bloques_ord) // 2
        hietograma[centro] = bloques_ord[0] 
        
        for i in range(1, len(bloques_ord)):
            val = bloques_ord[i]
            pos = centro + (i + 1) // 2 if i % 2 != 0 else centro - i // 2
            if 0 <= pos < len(hietograma):
                hietograma[pos] = val
                
        return hietograma 
    except Exception as e:
        # print(f"Error Hietograma: {e}") # Debug only
        return None

def calcular_hidrograma_unitario_hms(area_km2, tc_min, dt_min):
    """Genera el HU adimensional del SCS escalado."""
    try:
        tlag_hr = 0.6 * (tc_min / 60.0)
        dt_hr = dt_min / 60.0
        tp_hr = (dt_hr / 2.0) + tlag_hr
        
        # Qp = (2.08 * A) / Tp (Sistema Métrico, para 1 cm de lluvia)
        qp_m3s = (2.08 * area_km2) / tp_hr
        
        t_ratios = np.array(hidrologia_mx.HU_SCS_ADIMENSIONAL["t_ratio"])
        q_ratios = np.array(hidrologia_mx.HU_SCS_ADIMENSIONAL["q_ratio"])
        
        times_hr = t_ratios * tp_hr
        flows_m3s = q_ratios * qp_m3s
        
        # Interpolar al paso de tiempo del proyecto
        max_time = times_hr[-1]
        t_interp = np.arange(0, max_time, dt_hr)
        
        f_hu = interpolate.interp1d(times_hr, flows_m3s, kind='cubic', fill_value="extrapolate")
        hu_vals = np.maximum(f_hu(t_interp), 0)
        
        # Normalización de Volumen (Balance de Masa) a 10 mm (1 cm)
        vol_teorico = (area_km2 * 1e6) * 0.01 
        vol_calc = np.sum(hu_vals) * dt_hr * 3600
        if vol_calc > 0:
            hu_vals = hu_vals * (vol_teorico / vol_calc)
            
        return hu_vals 
    except: return None

def convolucion_hms(hietograma_total_mm, cn, hu_scs):
    """Aplica pérdidas (CN) y convolución."""
    try:
        # Modelo de Pérdidas SCS
        S_mm = (25400.0 / cn) - 254.0
        Ia_mm = 0.2 * S_mm
        
        P_acum = np.cumsum(hietograma_total_mm)
        Pe_acum = np.zeros_like(P_acum)
        
        mask = P_acum > Ia_mm
        Pe_acum[mask] = ((P_acum[mask] - Ia_mm)**2) / (P_acum[mask] - Ia_mm + S_mm)
        
        Pe_inc = np.diff(Pe_acum, prepend=0) # Lluvia neta por intervalo
        
        # Convolución
        # El HU es para 10 mm. Ajustamos proporcionalmente.
        pe_ratio = Pe_inc / 10.0 
        q_directo = np.convolve(pe_ratio, hu_scs)
        
        return q_directo, Pe_inc
    except: return None, None

# ==========================================
# 3. LÓGICA PRINCIPAL (INTEGRADA)
# ==========================================

def calcular_coeficientes_y_gastos(df_configurado, df_cotas, metodo_tc="Kirpich", modo_distribuido=False, estaciones_db=None, pesos_estaciones=None, df_int_global=None, df_alt_global=None):
    log = []
    res_hidrogramas = {}
    
    try:
        tr_cols = ['2', '5', '10', '20', '50', '100', '500', '1000', '10000']
        
        # --- BLINDAJE DE ÍNDICES ---
        def _preparar_df_lluvia(df_in):
            df = df_in.copy()
            # 1. Buscar columna de duración si no es el índice
            candidates = ['TR (AÑOS)', 'TR', 'Duracion', 'Duración', 'Minutos', 'Time']
            for col in candidates:
                if col in df.columns: 
                    df.set_index(col, inplace=True)
                    break
            # 2. Forzar índice a numérico
            try: df.index = df.index.astype(float).astype(int)
            except: pass # Si falla, esperar que ya sea numérico
            # 3. Limpiar columnas (TRs)
            df.columns = df.columns.astype(str).str.strip()
            df.sort_index(inplace=True) # Importante para interpolar
            return df

        # --- PRE-PROCESAMIENTO ---
        # 1. Calcular Coeficientes Ponderados (C y N)
        def _calc_pond(row):
            pcts = obtener_lista_desde_string(row['Porcentaje_terreno'], float)
            idxs_c = obtener_lista_desde_string(row['Coeficiente_C_por_Cuenca'], int)
            c_vals = {}
            c_df = pd.DataFrame(hidrologia_mx.MATRIZ_C_VALORES, index=range(1, 21))
            n_df = pd.DataFrame(hidrologia_mx.MATRIZ_N_VALORES, index=range(1, 18))
            
            for tr in tr_cols:
                val_c = sum((p/100)*c_df.loc[i, tr] for p, i in zip(pcts, idxs_c) if i in c_df.index)
                c_vals[f"C_pond_TR{tr}"] = val_c
            
            usos_n = obtener_lista_desde_string(row['indice_Uso_terreno'], int)
            grupos = obtener_lista_desde_string(row['Grupo_hidrologico'], str)
            n_val = sum((p/100)*n_df.loc[i, g.strip().upper()] for p, i, g in zip(pcts, usos_n, grupos) if i in n_df.index)
            return pd.Series({**c_vals, 'N_pond': n_val})
       
        df_pond = df_configurado.apply(_calc_pond, axis=1)
        df_completo = pd.concat([df_configurado, df_pond], axis=1)
        
        # 2. Geometría y Tiempo de Concentración (Tc)
        if 'cuenca' in df_cotas.columns: df_cotas['cuenca'] = df_cotas['cuenca'].astype(str)
        dist_cotas = {key: sub_df for key, sub_df in df_cotas.groupby('cuenca')}
        df_completo['Tc_min'] = np.nan
        df_completo['Pendiente_S'] = np.nan
        df_completo.index = df_completo.index.astype(str)
        
        for cid in df_completo.index:
            if cid in dist_cotas:
                sub = dist_cotas[cid]
                lcp = 0; den = 0
                for _, r in sub.iterrows():
                    d = abs(r['distancia2']-r['distancia1']) or 0.001
                    s = (abs(r['cota mayor']-r['cota menor'])/d) or 0.0001
                    lcp += d; den += d/math.sqrt(s)
                
                S_avg = (lcp/(den or 0.001))**2
                
                # SELECCIÓN DEL MÉTODO
                if metodo_tc == "Temez": tc_val = calcular_tc_temez(lcp, S_avg)
                else: tc_val = 0.000325 * ((lcp * 1000)**0.77) / (S_avg**0.385) * 60
                
                df_completo.at[cid, 'Tc_min'] = tc_val # Guardar valor exacto para cálculo
                df_completo.at[cid, 'Pendiente_S'] = S_avg
            else:
                log.append(f"Aviso: Cuenca {cid} sin cotas.")

        # 3. Inicializar Resultados
        cols_out = [f"TR_{tr}" for tr in tr_cols]
        df_racional = pd.DataFrame(0.0, index=df_completo.index, columns=cols_out)
        df_chow = pd.DataFrame(0.0, index=df_completo.index, columns=cols_out)
        df_hms_peak = pd.DataFrame(0.0, index=df_completo.index, columns=cols_out)

        # 4. Preparar Lluvias Globales
        if not modo_distribuido:
            if df_int_global is None or df_alt_global is None:
                return None, None, None, None, None, "Faltan lluvias globales."
            try:
                df_int_global = _preparar_df_lluvia(df_int_global)
                df_alt_global = _preparar_df_lluvia(df_alt_global)
            except Exception as ex:
                return None, None, None, None, None, f"Error formato lluvia: {ex}"

        # --- BUCLE DE CÁLCULO ---
        for cid, row in df_completo.iterrows():
            area = row['area']; tc = row['Tc_min']; n_pond = row['N_pond']
            
            # Selección de Fuente de Lluvia
            df_i_curr = None; df_a_curr = None
            if modo_distribuido:
                # Lógica distribuida omitida por brevedad, asumimos global para debugging
                # (Aquí iría la llamada a _calcular_lluvia_ponderada si la tienes implementada)
                pass 
            else:
                df_i_curr, df_a_curr = df_int_global, df_alt_global
            
            if df_i_curr is None: continue

            # >>> MÉTODO RACIONAL (CORREGIDO: INTERPOLACIÓN) <<<
            if pd.notna(tc):
                for tr in tr_cols:
                    col = str(tr)
                    # Intentar matchear columna (string o int)
                    if col not in df_i_curr.columns and int(tr) in df_i_curr.columns: 
                        col = int(tr)
                    
                    if col in df_i_curr.columns:
                        # AQUÍ EL CAMBIO: Usamos interpolación, no .loc directo
                        I = interpolar_valor_curva(tc, df_i_curr, col)
                        C = row[f"C_pond_TR{tr}"]
                        df_racional.at[cid, f"TR_{tr}"] = round(0.278 * C * I * area, 3)

            # >>> MÉTODO CHOW (SCT) <<<
            if n_pond and area > 0:
                L = float(row['LCP']) if pd.notna(row['LCP']) else 0
                S = row['Pendiente_S']
                if L > 0 and S > 0:
                    t_lag = 0.00505 * (L / (S**0.5))**0.64
                    tr_series = pd.Series(df_a_curr.index.values)
                    ratio = (tr_series / 60.0) / t_lag
                    z_vals = np.select([ratio > 2, (ratio >= 0.4) & (ratio <= 2)], [1.0, 1.89 * (ratio**0.97) - 1.23], default=0.73 * (ratio**0.97))
                    
                    for tr in tr_cols:
                        col = str(tr)
                        if col not in df_a_curr.columns and int(tr) in df_a_curr.columns: col = int(tr)
                        if col in df_a_curr.columns:
                            P_cm = df_a_curr[col].values / 10.0
                            C1 = 508.0/n_pond; C2 = 5.08; C3 = 2032.0/n_pond; C4 = 20.32
                            num = ((P_cm - C1 + C2)**2) * (2.78 * area * z_vals)
                            den = P_cm + C3 - C4
                            with np.errstate(divide='ignore', invalid='ignore'):
                                q_vec = (num / den) / (tr_series / 60.0)
                            df_chow.at[cid, f"TR_{tr}"] = round(np.nan_to_num(np.max(q_vec)), 3)

            # >>> MÉTODO HMS (HIDROGRAMA UNITARIO) <<<
            if pd.notna(tc) and pd.notna(n_pond) and area > 0:
                dt_calc = 10 # Paso de cálculo en minutos
                hu = calcular_hidrograma_unitario_hms(area, tc, dt_calc)
                
                if hu is not None:
                    for tr in tr_cols:
                        col = str(tr)
                        if col not in df_i_curr.columns and int(tr) in df_i_curr.columns: col = int(tr)
                        
                        if col in df_i_curr.columns:
                            # Duración de tormenta suficiente para cubrir 2*Tc
                            dur = max(tc * 2.5, 360) 
                            
                            # Generar Hietograma desde Curva IDF (Interpolada)
                            hieto = generar_hietograma_bloques_alternos(dur, dt_calc, df_i_curr, col)
                            
                            if hieto is not None:
                                Q, _ = convolucion_hms(hieto, n_pond, hu)
                                if Q is not None:
                                    df_hms_peak.at[cid, f"TR_{tr}"] = round(np.max(Q), 3)
                                    # Guardar datos para gráfico
                                    t_axis = np.arange(len(Q)) * dt_calc / 60.0
                                    res_hidrogramas[f"{cid}_TR{tr}"] = (t_axis, Q)

        return df_racional, df_chow, df_hms_peak, df_completo, res_hidrogramas, "\n".join(log)

    except Exception as e:
        return None, None, None, None, None, f"Error Crítico: {traceback.format_exc()}"

def generar_graficos_comparativos(df_racional, df_chow, df_hms_calc, df_hms_ext):
    """
    Genera gráficos de barras comparando los 3 métodos + HMS externo.
    """
    graficos = []
    
    def norm_idx(idx): return str(idx).strip().replace('.0', '')
    
    df_racional.index = df_racional.index.map(norm_idx)
    df_chow.index = df_chow.index.map(norm_idx)
    df_hms_calc.index = df_hms_calc.index.map(norm_idx)
    
    hms_ext_map = {}
    if df_hms_ext is not None:
        df_hms_ext.index = df_hms_ext.index.map(norm_idx)
        for c in df_hms_ext.columns:
            clean = ''.join(filter(str.isdigit, str(c)))
            if clean: hms_ext_map[f"TR_{clean}"] = c

    trs = ['2', '5', '10', '20', '50', '100', '500', '1000', '10000']
    
    with plt.style.context('default'):
        for tr in trs:
            col = f"TR_{tr}"
            if col in df_racional.columns:
                idx = df_racional.index.tolist()
                
                v_r = df_racional[col].fillna(0).tolist()
                v_c = df_chow[col].fillna(0).tolist() if col in df_chow.columns else [0]*len(idx)
                v_h = df_hms_calc[col].fillna(0).tolist() if col in df_hms_calc.columns else [0]*len(idx)
                
                v_ext = []
                if df_hms_ext is not None:
                    ext_c = hms_ext_map.get(col)
                    if ext_c and ext_c in df_hms_ext.columns:
                        for i in idx:
                            v = df_hms_ext.loc[i, ext_c] if i in df_hms_ext.index else 0
                            v_ext.append(float(v) if pd.notna(v) else 0)
                    else: v_ext = [0]*len(idx)
                else: v_ext = [0]*len(idx)

                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.arange(len(idx)); w = 0.2
                
                ax.bar(x - 1.5*w, v_r, w, label='Racional', color='#1f77b4', edgecolor='black')
                ax.bar(x - 0.5*w, v_c, w, label='Chow (SCT)', color='#ff7f0e', edgecolor='black')
                ax.bar(x + 0.5*w, v_h, w, label='HMS (Calc)', color='#9467bd', edgecolor='black')
                if any(v_ext):
                    ax.bar(x + 1.5*w, v_ext, w, label='HMS (Ext)', color='#2ca02c', edgecolor='black')
                
                ax.set_title(f'Comparación Gasto Pico - TR {tr} Años', fontweight='bold')
                ax.set_ylabel('Gasto Máximo ($m^3/s$)')
                ax.set_xticks(x); ax.set_xticklabels(idx, rotation=45)
                ax.legend(); ax.grid(True, axis='y', linestyle='--')
                plt.tight_layout()
                
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100)
                plt.close(fig)
                graficos.append((tr, base64.b64encode(buf.getvalue()).decode('utf-8')))
    
    return graficos

def generar_grafico_hidrograma(t, Q, titulo):
    with plt.style.context('default'):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t, Q, color='blue', linewidth=2)
        ax.set_title(titulo, fontweight="bold")
        ax.set_xlabel("Tiempo (hr)"); ax.set_ylabel("Q (m³/s)")
        ax.grid(True, linestyle="--")
        ax.fill_between(t, Q, color='blue', alpha=0.1)
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode('utf-8')