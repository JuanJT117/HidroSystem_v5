import pandas as pd
import traceback
import io
import base64
import numpy as np
import scipy.stats as stats
from scipy.stats import gamma, lognorm, norm, genextreme, gumbel_r 
import matplotlib
import matplotlib.pyplot as plt
import math 
import warnings

# --- OPTIMIZACIÓN 1: Forzar Backend No-Interactivo ---
matplotlib.use('Agg')

# --- Funciones de Ayuda Gráfica ---

def _figure_to_base64(fig):
    """Convierte una figura de Matplotlib a un string base64 optimizado."""
    buf = io.BytesIO()
    # DPI 100 es suficiente y más rápido
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig) 
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# --- Lógica Común Refactorizada ---

def _calculate_regional_params(cociente_lluvia_duracion, log_list):
    p_r_a = (7353.3606920242*cociente_lluvia_duracion**6) - (12897.462451992*cociente_lluvia_duracion**5) + (8199.0167276778*cociente_lluvia_duracion**4) - (2317.3712794565*cociente_lluvia_duracion**3) + (366.9142815419*cociente_lluvia_duracion**2) + 8.0208384464*cociente_lluvia_duracion + 1.3199316614
    p_r_b = (-6625.0538375378*cociente_lluvia_duracion**6) + (14651.1423180723*cociente_lluvia_duracion**5) - (12769.031780099*cociente_lluvia_duracion**4) + (5526.6701616568*cociente_lluvia_duracion**3) - (1252.848252752*cociente_lluvia_duracion**2) + 176.629005407*cociente_lluvia_duracion - 12.4169934459
    p_r_c = (-143.1687369496*cociente_lluvia_duracion**6) + (366.5291775021*cociente_lluvia_duracion**5) - (375.9641589724*cociente_lluvia_duracion**4) + (196.920213889*cociente_lluvia_duracion**3) - (56.4672253684*cociente_lluvia_duracion**2) + 9.7154529619*cociente_lluvia_duracion - 0.2608153057
    
    log_list.append('PARÁMETRO REGIONAL (a): ' + str(p_r_a))
    log_list.append('PARÁMETRO REGIONAL (b): ' + str(p_r_b))
    log_list.append('PARÁMETRO REGIONAL (c): ' + str(p_r_c))
    
    return p_r_a, p_r_b, p_r_c

def _calculate_curvas_hidrologicas(p_r_a, p_r_b, p_r_c, p_r_f, AP_60_2, AP_60_10):
    TR=[2, 5, 10, 20, 50, 100, 500, 1000, 10000, 20000]
    tr_valores = np.arange(5, 1441, 5) 
    
    # --- Curvas de Altura ---
    df_altura = pd.DataFrame({
        'TR (AÑOS)': tr_valores,
        2: np.nan, 5: np.nan, 10: np.nan, 20: np.nan, 50: np.nan,
        100: np.nan, 500: np.nan, 1000: np.nan, 10000: np.nan
    })

    df_altura[2] =  (0.35 * math.log(2)  + 0.76) * (((0.54 * (df_altura['TR (AÑOS)'] ** 0.25)) - 0.5) * AP_60_2)
    df_altura[5] =  (0.35 * math.log(5)  + 0.76) * (((0.54 * (df_altura['TR (AÑOS)'] ** 0.25)) - 0.5) * AP_60_2)
    df_altura[10] = (0.35 * math.log(10) + 0.76) * (((0.54 * (df_altura['TR (AÑOS)'] ** 0.25)) - 0.5) * AP_60_2)
    df_altura[20] =  (p_r_a * AP_60_10 * math.log10((10 ** (2-p_r_f)) * TR[3] **(p_r_f - 1)) * df_altura['TR (AÑOS)'])/(60 * ((df_altura['TR (AÑOS)'] + p_r_b)** p_r_c))
    df_altura[50] =  (p_r_a * AP_60_10 * math.log10((10 ** (2-p_r_f)) * TR[4] **(p_r_f - 1)) * df_altura['TR (AÑOS)'])/(60 * ((df_altura['TR (AÑOS)'] + p_r_b)** p_r_c))
    df_altura[100] = (p_r_a * AP_60_10 * math.log10((10 ** (2-p_r_f)) * TR[5] **(p_r_f - 1)) * df_altura['TR (AÑOS)'])/(60 * ((df_altura['TR (AÑOS)'] + p_r_b)** p_r_c))
    df_altura[500] = (p_r_a * AP_60_10 * math.log10((10 ** (2-p_r_f)) * TR[6] **(p_r_f - 1)) * df_altura['TR (AÑOS)'])/(60 * ((df_altura['TR (AÑOS)'] + p_r_b)** p_r_c))
    df_altura[1000] = (p_r_a * AP_60_10 * math.log10((10 ** (2-p_r_f)) * TR[7] **(p_r_f - 1)) * df_altura['TR (AÑOS)'])/(60 * ((df_altura['TR (AÑOS)'] + p_r_b)** p_r_c))
    df_altura[10000] = (p_r_a * AP_60_10 * math.log10((10 ** (2-p_r_f)) * TR[8] **(p_r_f - 1)) * df_altura['TR (AÑOS)'])/(60 * ((df_altura['TR (AÑOS)'] + p_r_b)** p_r_c))

    # --- Curvas de Intensidad ---
    df_intensidad = pd.DataFrame({
        'TR (AÑOS)': tr_valores,
        2: np.nan, 5: np.nan, 10: np.nan, 20: np.nan, 50: np.nan,
        100: np.nan, 500: np.nan, 1000: np.nan, 10000: np.nan
    })
    
    for col in [2, 5, 10, 20, 50, 100, 500, 1000, 10000]:
        df_intensidad[col] = df_altura[col] * 60 / df_intensidad['TR (AÑOS)']
        
    return df_altura, df_intensidad

def _generate_cuenca_plots(df_altura_in, df_intensidad_in, ylim_altura, ylim_intensidad):
    
    # --- SANEAMIENTO DE DATOS (FIX KEYERROR) ---
    # Creamos copias para no afectar los datos globales
    df_altura = df_altura_in.copy()
    df_intensidad = df_intensidad_in.copy()

    # Si 'TR (AÑOS)' no está en columnas, reseteamos el índice para sacarlo
    if 'TR (AÑOS)' not in df_altura.columns:
        df_altura.reset_index(inplace=True)
        # Si al resetear se llama 'index' o 'level_0', lo renombramos por seguridad
        if 'TR (AÑOS)' not in df_altura.columns and 'index' in df_altura.columns:
             df_altura.rename(columns={'index': 'TR (AÑOS)'}, inplace=True)

    if 'TR (AÑOS)' not in df_intensidad.columns:
        df_intensidad.reset_index(inplace=True)
        if 'TR (AÑOS)' not in df_intensidad.columns and 'index' in df_intensidad.columns:
             df_intensidad.rename(columns={'index': 'TR (AÑOS)'}, inplace=True)

    # Validacion final para evitar crash si aun asi falla
    if 'TR (AÑOS)' not in df_altura.columns:
        print("ERROR CRÍTICO: No se encontró columna de tiempo 'TR (AÑOS)'")
        return None, None, None, None

    # --- Configuración de Columnas a Plotear ---
    columnas_precipitacion = [2, 5, 10, 20, 50, 100, 500, 1000, 10000]
    columnas_precipitacion_2 = [100, 500, 1000, 10000]

    # --- Cálculo de límites Y (excluyendo columna de tiempo) ---
    cols_data = [c for c in df_altura.columns if c != 'TR (AÑOS)']
    max_h_data = df_altura[cols_data].max().max() if cols_data else 100
    
    cols_data_int = [c for c in df_intensidad.columns if c != 'TR (AÑOS)']
    max_i_data = df_intensidad[cols_data_int].max().max() if cols_data_int else 100

    if ylim_altura is not None and ylim_altura > 0:
        final_ylim_h = ylim_altura
    else:
        final_ylim_h = max_h_data * 1.1 if max_h_data > 0 else 100

    if ylim_intensidad is not None and ylim_intensidad > 0:
        final_ylim_i = ylim_intensidad
    else:
        final_ylim_i = max_i_data * 1.1 if max_i_data > 0 else 100

    # --- Gráfico Altura 1 ---
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    for columna in columnas_precipitacion:
        if columna in df_altura.columns:
            ax1.plot(df_altura['TR (AÑOS)'], df_altura[columna], label=f'TR = {columna} años', alpha=0.8)
    ax1.set_xlabel('DURACIÓN (MIN)')
    ax1.set_ylabel('ALTURA (mm)')
    ax1.set_title('CURVAS DE ALTURA')
    ax1.set_ylim(0, final_ylim_h)
    ax1.legend(loc='upper right', fontsize='small')
    ax1.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_1_b64 = _figure_to_base64(fig1)

    # --- Gráfico Altura 2 (Zoom) ---
    df_filtrado_2 = df_altura[(df_altura['TR (AÑOS)'] >= 0) & (df_altura['TR (AÑOS)'] <= 300)]
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for columna in columnas_precipitacion_2:
        if columna in df_filtrado_2.columns:
            ax2.plot(df_filtrado_2['TR (AÑOS)'], df_filtrado_2[columna], label=f'TR = {columna} años', alpha=0.8)
    ax2.set_xlabel('DURACIÓN (MIN)')
    ax2.set_ylabel('ALTURA (mm)')
    ax2.set_title('CURVAS DE ALTURA (0-300min)')
    ax2.set_ylim(0, final_ylim_h) 
    ax2.set_xlim(0, 300)
    ax2.legend(loc='upper right', fontsize='small')
    ax2.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_2_b64 = _figure_to_base64(fig2)

    # --- Gráfico Intensidad 1 ---
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    for columna in columnas_precipitacion:
        if columna in df_intensidad.columns:
            ax3.plot(df_intensidad['TR (AÑOS)'], df_intensidad[columna], label=f'TR = {columna} años', alpha=0.8)
    ax3.set_xlabel('DURACIÓN (MIN)')
    ax3.set_ylabel('INTENSIDAD (mm/hr)')
    ax3.set_title('CURVAS DE INTENSIDAD')
    ax3.set_ylim(0, final_ylim_i)
    ax3.legend(loc='upper right', fontsize='small')
    ax3.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_3_b64 = _figure_to_base64(fig3)

    # --- Gráfico Intensidad 2 (Zoom) ---
    df_filtrado_3 = df_intensidad[(df_intensidad['TR (AÑOS)'] >= 0) & (df_intensidad['TR (AÑOS)'] <= 300)]
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    for columna in columnas_precipitacion_2:
        if columna in df_filtrado_3.columns:
            ax4.plot(df_filtrado_3['TR (AÑOS)'], df_filtrado_3[columna], label=f'TR = {columna} años', alpha=0.8)
    ax4.set_xlabel('DURACIÓN (MIN)')
    ax4.set_ylabel('INTENSIDAD (mm/hr)')
    ax4.set_title('CURVAS DE INTENSIDAD (0-300min)')
    ax4.set_ylim(0, final_ylim_i) 
    ax4.set_xlim(0, 300)
    ax4.legend(loc='upper right', fontsize='small')
    ax4.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_4_b64 = _figure_to_base64(fig4)
    
    return plot_1_b64, plot_2_b64, plot_3_b64, plot_4_b64

# --- Lógica Específica de Distribución ---

def _run_log_pearson(df_maximos_mensuales, log_list):
    log_list.append("### Ejecutando análisis para Log-Pearson III...")
    df_maximos_mensuales['Ln_Max_Anual'] = np.log(df_maximos_mensuales['Max_Anual'])
    
    Promedio_Ln_Max_Anual = df_maximos_mensuales['Ln_Max_Anual'].mean()
    Desviacion_estandar_Ln_Max_Anual = df_maximos_mensuales['Ln_Max_Anual'].std()
    KLP3TR2 = -0.017
    KLP3TR10 = 1.3010
    KLP3TR100 = 2.4720
    Cociente_lluvia_duracion = 0.3882
    
    p_r_f = (np.exp(Promedio_Ln_Max_Anual + (KLP3TR100 * Desviacion_estandar_Ln_Max_Anual))) / (np.exp(Promedio_Ln_Max_Anual + (KLP3TR10 * Desviacion_estandar_Ln_Max_Anual)))
    AP_60_2 = (np.exp(Promedio_Ln_Max_Anual + (KLP3TR2 * Desviacion_estandar_Ln_Max_Anual))) * Cociente_lluvia_duracion
    AP_60_10 = AP_60_2 * (0.35 * np.log(10) + 0.76) * (0.54 * (60 ** 0.25) - 0.5)

    log_list.append(f'Media (X): {Promedio_Ln_Max_Anual:.4f}')
    log_list.append(f'Relación f: {p_r_f:.4f}')
    
    return p_r_f, AP_60_2, AP_60_10, Cociente_lluvia_duracion

def _run_pearson_iii(df_maximos_mensuales, log_list):
    log_list.append("### Ejecutando análisis para Pearson III...")
    Promedio_Ln_Max_Anual = df_maximos_mensuales['Max_Anual'].mean()
    Desviacion_estandar_Ln_Max_Anual = df_maximos_mensuales['Max_Anual'].std()
    Coeficiente_asimetria_Ln_Max_Anual = df_maximos_mensuales['Max_Anual'].skew()
    Cociente_lluvia_duracion = 0.3882
    
    if abs(Coeficiente_asimetria_Ln_Max_Anual) < 0.001: Coeficiente_asimetria_Ln_Max_Anual = 0.001
        
    Beta = (2/Coeficiente_asimetria_Ln_Max_Anual)**2  
    Alfa = (Desviacion_estandar_Ln_Max_Anual)/((Beta)**0.5)   
    Gamma = Promedio_Ln_Max_Anual-(Alfa * Beta)   

    p_r_f = ((gamma.ppf(0.99, a=Beta, loc=Gamma, scale=Alfa))) / ((gamma.ppf(0.9, a=Beta, loc=Gamma, scale=Alfa)))
    AP_60_2 = Cociente_lluvia_duracion * ((gamma.ppf(0.5, a=Beta, loc=Gamma, scale=Alfa)))
    AP_60_10 = AP_60_2 * (0.35 * np.log(10) + 0.76) * (0.54 * (60 ** 0.25) - 0.5)

    log_list.append(f'Media (X): {Promedio_Ln_Max_Anual:.4f}')
    log_list.append(f'Relación f: {p_r_f:.4f}')
    
    return p_r_f, AP_60_2, AP_60_10, Cociente_lluvia_duracion

def _run_exponencial(df_maximos_mensuales, log_list):
    log_list.append("### Ejecutando análisis para Exponencial...")
    Promedio_Max_Anual = df_maximos_mensuales['Max_Anual'].mean()
    Desviacion_estandar_Max_Anual = df_maximos_mensuales['Max_Anual'].std()
    Cociente_lluvia_duracion = 0.3882
    A = Promedio_Max_Anual - Desviacion_estandar_Max_Anual
    B = Desviacion_estandar_Max_Anual
    p_r_f = (A - (B * np.log(0.01)))/(A - (B * np.log(0.1)))
    AP_60_2 = Cociente_lluvia_duracion * (A -(B * np.log(0.5)))
    AP_60_10 = AP_60_2 * (0.35 * np.log(10) + 0.76) * (0.54 * (60 ** 0.25) - 0.5)
    log_list.append(f'Media (X): {Promedio_Max_Anual:.4f}')
    return p_r_f, AP_60_2, AP_60_10, Cociente_lluvia_duracion

def _run_gamma(df_maximos_mensuales, log_list):
    log_list.append("### Ejecutando análisis para Gamma...")
    Promedio_Max_Anual = df_maximos_mensuales['Max_Anual'].mean()
    Desviacion_estandar_Max_Anual = df_maximos_mensuales['Max_Anual'].std()
    Cociente_lluvia_duracion = 0.3882
    A = (Desviacion_estandar_Max_Anual**2)/Promedio_Max_Anual
    B = (Promedio_Max_Anual/Desviacion_estandar_Max_Anual)**2
    p_r_f = (gamma.ppf(0.99, a=B, scale=A))/(gamma.ppf(0.9, a=B, scale=A))
    AP_60_2 = Cociente_lluvia_duracion*(gamma.ppf(0.5, a=B, scale=A))
    AP_60_10 = AP_60_2 * (0.35 * np.log(10) + 0.76) * (0.54 * (60 ** 0.25) - 0.5)
    log_list.append(f'Media (X): {Promedio_Max_Anual:.4f}')
    return p_r_f, AP_60_2, AP_60_10, Cociente_lluvia_duracion

def _run_log_normal(df_maximos_mensuales, log_list):
    log_list.append("### Ejecutando análisis para Log-Normal...")
    df_maximos_mensuales['Ln_Max_Anual'] = np.log(df_maximos_mensuales['Max_Anual'])
    Promedio_Max_Anual = df_maximos_mensuales['Ln_Max_Anual'].mean()
    Desviacion_estandar_Max_Anual = df_maximos_mensuales['Ln_Max_Anual'].std()
    Cociente_lluvia_duracion = 0.3882
    p_r_f = lognorm.ppf(0.99, s=Desviacion_estandar_Max_Anual, loc=0, scale=np.exp(Promedio_Max_Anual)) / lognorm.ppf(0.9, s=Desviacion_estandar_Max_Anual, loc=0, scale=np.exp(Promedio_Max_Anual))
    AP_60_2 = ((lognorm.ppf(0.5, s=Desviacion_estandar_Max_Anual, loc=0, scale=np.exp(Promedio_Max_Anual)))*Cociente_lluvia_duracion)
    AP_60_10 = AP_60_2 * (0.35 * np.log(10) + 0.76) * (0.54 * (60 ** 0.25) - 0.5)
    return p_r_f, AP_60_2, AP_60_10, Cociente_lluvia_duracion

def _run_normal(df_maximos_mensuales, log_list):
    log_list.append("### Ejecutando análisis para Normal...")
    Promedio_Max_Anual = df_maximos_mensuales['Max_Anual'].mean()
    Desviacion_estandar_Max_Anual = df_maximos_mensuales['Max_Anual'].std()
    Cociente_lluvia_duracion = 0.3882
    p_r_f = norm.ppf(0.99, loc=Promedio_Max_Anual, scale=Desviacion_estandar_Max_Anual) / norm.ppf(0.9, loc=Promedio_Max_Anual, scale=Desviacion_estandar_Max_Anual)
    AP_60_2 = Cociente_lluvia_duracion * (norm.ppf(0.5, loc=Promedio_Max_Anual, scale=Desviacion_estandar_Max_Anual))
    AP_60_10 = AP_60_2 * (0.35 * np.log(10) + 0.76) * (0.54 * (60 ** 0.25) - 0.5)
    return p_r_f, AP_60_2, AP_60_10, Cociente_lluvia_duracion

def _run_gumbel(df_maximos_mensuales, log_list):
    log_list.append("### Ejecutando análisis para Gumbel...")
    Count_YE = df_maximos_mensuales['Max_Anual'].size
    Promedio_Max_Anual = df_maximos_mensuales['Max_Anual'].mean()
    Desviacion_estandar_Max_Anual = df_maximos_mensuales['Max_Anual'].std()
    Cociente_lluvia_duracion = 0.3882
    My = (-1.6483*(10**-12)*(Count_YE**6)) + (6.2811*(10**-10)*(Count_YE**5)) - (9.7156*(10**-8)*(Count_YE**4)) + (7.8564*(10**-6)*(Count_YE**3)) - (3.5782*(10**-4)*(Count_YE**2)) + (9.3362*(10**-3)*Count_YE+0.4308)
    Sy = (-7.1424*(10**-12)*(Count_YE**6)) + (2.7029*(10**-9)*(Count_YE**5)) - (4.1453*(10**-7)*(Count_YE**4)) + (3.3168*(10**-5)*(Count_YE**3)) - (1.4904*(10**-3)*(Count_YE**2)) + (3.8198*(10**-2)*Count_YE+0.6881)
    A1 = Sy/Desviacion_estandar_Max_Anual
    B1 = Promedio_Max_Anual-(My/A1)
    p_r_f =  (B1-(1/A1*(np.log(np.log(100/99)))))/(B1-(1/A1*(np.log(np.log(10/9)))))
    AP_60_2 = Cociente_lluvia_duracion*(B1-((1/A1)*(np.log(np.log(2/1)))))
    AP_60_10 = AP_60_2 * (0.35 * np.log(10) + 0.76) * (0.54 * (60 ** 0.25) - 0.5)
    return p_r_f, AP_60_2, AP_60_10, Cociente_lluvia_duracion

def _run_gumbel_2_poblaciones(df_maximos_mensuales, log_list):
    log_list.append("### Ejecutando análisis para Gumbel 2 Poblaciones...")
    max_annual_data = df_maximos_mensuales['Max_Anual'].dropna()
    mid = len(max_annual_data) // 2
    g1 = max_annual_data[:mid]
    g2 = max_annual_data[mid:]
    params1 = stats.gumbel_r.fit(g1) 
    params2 = stats.gumbel_r.fit(g2) 
    Cociente_lluvia_duracion = 0.3882
    ppf_99 = 0.5 * stats.gumbel_r.ppf(0.99, *params1) + 0.5 * stats.gumbel_r.ppf(0.99, *params2)
    ppf_90 = 0.5 * stats.gumbel_r.ppf(0.90, *params1) + 0.5 * stats.gumbel_r.ppf(0.90, *params2)
    ppf_50 = 0.5 * stats.gumbel_r.ppf(0.50, *params1) + 0.5 * stats.gumbel_r.ppf(0.50, *params2)
    p_r_f = ppf_99 / ppf_90
    AP_60_2 = Cociente_lluvia_duracion * ppf_50
    AP_60_10 = AP_60_2 * (0.35 * np.log(10) + 0.76) * (0.54 * (60 ** 0.25) - 0.5)
    return p_r_f, AP_60_2, AP_60_10, Cociente_lluvia_duracion

def _run_gev(df_maximos_mensuales, log_list):
    log_list.append("### Ejecutando análisis para GEV...")
    max_annual_data = df_maximos_mensuales['Max_Anual'].dropna()
    Cociente_lluvia_duracion = 0.3882
    params_gev = stats.genextreme.fit(max_annual_data) 
    p_r_f = stats.genextreme.ppf(0.99, *params_gev) / stats.genextreme.ppf(0.90, *params_gev)
    AP_60_2 = Cociente_lluvia_duracion * (stats.genextreme.ppf(0.5, *params_gev))
    AP_60_10 = AP_60_2 * (0.35 * np.log(10) + 0.76) * (0.54 * (60 ** 0.25) - 0.5)
    return p_r_f, AP_60_2, AP_60_10, Cociente_lluvia_duracion

# --- Función Principal (Orquestador) ---

def run_cuenca_analysis(best_fit_name, df_maximos_mensuales, ylim_altura=None, ylim_intensidad=None):
    results = {
        "log_text": "", "df_altura": pd.DataFrame(), "df_intensidad": pd.DataFrame(),
        "plot_1_b64": None, "plot_2_b64": None, "plot_3_b64": None, "plot_4_b64": None
    }
    log_list = []
    
    try:
        print(f"Iniciando cálculo para: {best_fit_name}")
        func_map = {
            'Log Pearson': _run_log_pearson, 'Pearson III': _run_pearson_iii,
            'Exponencial': _run_exponencial, 'Gamma': _run_gamma,
            'Log Normal': _run_log_normal, 'Normal': _run_normal,
            'Gumbel': _run_gumbel, 'Gumbel 2 Poblaciones': _run_gumbel_2_poblaciones,
            'General Valores Extremos': _run_gev
        }
        
        func = func_map.get(best_fit_name)
        if not func:
            log_list.append(f"Método {best_fit_name} no reconocido.")
            results["log_text"] = "\n".join(log_list)
            return results

        p_r_f, AP_60_2, AP_60_10, Cociente_lluvia_duracion = func(df_maximos_mensuales, log_list)
        p_r_a, p_r_b, p_r_c = _calculate_regional_params(Cociente_lluvia_duracion, log_list)
        df_altura, df_intensidad = _calculate_curvas_hidrologicas(p_r_a, p_r_b, p_r_c, p_r_f, AP_60_2, AP_60_10)
        
        plot_1_b64, plot_2_b64, plot_3_b64, plot_4_b64 = _generate_cuenca_plots(df_altura, df_intensidad, ylim_altura, ylim_intensidad)

        results = {
            "log_text": "\n".join(log_list),
            "df_altura": df_altura,
            "df_intensidad": df_intensidad,
            "plot_1_b64": plot_1_b64,
            "plot_2_b64": plot_2_b64,
            "plot_3_b64": plot_3_b64,
            "plot_4_b64": plot_4_b64
        }
        print("Análisis completado.")
        return results

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        results["log_text"] = f"Error: {e}"
        return results