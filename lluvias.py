import pandas as pd
import traceback
import io
import base64
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import warnings

# --- Diccionario de Nombres de Parámetros ---
PARAM_NAMES = {
    'Normal': ['loc', 'scale'],
    'Log Normal': ['s', 'loc', 'scale'],
    'Exponencial': ['loc', 'scale'],
    'Gamma': ['a', 'loc', 'scale'],
    'Pearson III': ['skew', 'loc', 'scale'],
    'General Valores Extremos': ['c', 'loc', 'scale'],
    'Gumbel': ['loc', 'scale'],
    'Log Pearson': ['skew', 'loc', 'scale'],
}

# --- Funciones de Ayuda ---

def _figure_to_base64(fig):
    buf = io.BytesIO()
    # Estilo académico: Fondo blanco explícito
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def _test_homogeneity_data(g1, g2, name):
    """Calcula pruebas y retorna lista de diccionarios."""
    data = []
    try:
        # Cramer-von Mises
        cvm = stats.cramervonmises_2samp(g1, g2)
        data.append({'Prueba': f'{name} (CvM)', 'Estadístico': round(cvm.statistic, 4), 'P-Valor': round(cvm.pvalue, 4), 'Resultado': 'Homogénea' if cvm.pvalue > 0.05 else 'No Homogénea'})
        
        # T-Student
        t = stats.ttest_ind(g1, g2)
        data.append({'Prueba': f'{name} (T-Student)', 'Estadístico': round(t.statistic, 4), 'P-Valor': round(t.pvalue, 4), 'Resultado': 'Medias Iguales' if t.pvalue > 0.05 else 'Diferente Media'})
        
        # Levene
        lev = stats.levene(g1, g2)
        data.append({'Prueba': f'{name} (Levene)', 'Estadístico': round(lev.statistic, 4), 'P-Valor': round(lev.pvalue, 4), 'Resultado': 'Varianzas Iguales' if lev.pvalue > 0.05 else 'Diferente Varianza'})
        
        return data
    except Exception as e:
        return [{'Prueba': name, 'Estadístico': 0, 'P-Valor': 0, 'Resultado': f'Error: {str(e)}'}]

# --- Función Principal de Análisis ---

def analizar_eventos_lluvia(df_filtrado):
    
    warnings.filterwarnings('ignore')
    
    # Inicializar DataFrames de retorno
    results = {
        "df_homogeneidad": pd.DataFrame(),
        "max_annual_series_b64": None,
        "df_acf": pd.DataFrame(), 
        "acf_plot_b64": None,
        "df_weibull": pd.DataFrame(), 
        "weibull_plot_b64": None,
        "dist_comparison_b64": None,
        "df_ajustes": pd.DataFrame(), 
        "df_maximos_mensuales": pd.DataFrame(), 
        "best_fit_name": None
    }
    
    try:
        if df_filtrado is None or df_filtrado.empty: return None
        if 'PRECIP_imputado' not in df_filtrado.columns: return None
        
        max_annual = df_filtrado['PRECIP_imputado'].resample('YE').max().dropna()
        if max_annual.empty or len(max_annual) < 4: return None

        # --- 1. Pruebas Estadísticas (Homogeneidad + Anderson-Darling) ---
        tests_data = []
        
        # Anderson-Darling (Integrado aquí como pediste)
        try:
            ad = stats.anderson(max_annual, dist='norm')
            # Usamos el nivel de significancia del 5% (índice 2)
            crit_val = ad.critical_values[2]
            is_normal = ad.statistic < crit_val
            tests_data.append({
                'Prueba': 'Anderson-Darling (Indep)',
                'Estadístico': round(ad.statistic, 4),
                'P-Valor': f"Crit(5%)={crit_val:.2f}", # AD no da p-valor exacto
                'Resultado': 'Normal/Indep.' if is_normal else 'No Normal'
            })
        except: pass

        # Homogeneidad Temporal
        mid = len(max_annual) // 2
        tests_data.extend(_test_homogeneity_data(max_annual.values[:mid], max_annual.values[mid:], "Temporal"))

        # Homogeneidad Estratificada
        try:
            median_val = max_annual.median()
            labels = (max_annual > median_val).astype(int)
            if len(np.unique(labels)) > 1:
                g1_s, g2_s = train_test_split(max_annual.values, test_size=0.5, stratify=labels, random_state=42)
                tests_data.extend(_test_homogeneity_data(g1_s, g2_s, "Estratificada"))
        except: pass
        
        results["df_homogeneidad"] = pd.DataFrame(tests_data)
        
        # --- 2. Gráfico Series ---
        with plt.style.context('default'):
            fig_series, ax_series = plt.subplots(figsize=(12, 6)) 
            ax_series.plot(max_annual.index, max_annual.values, marker='o', color='#4682B4', linestyle='-', linewidth=1, markersize=4, markeredgecolor='black')
            ax_series.set_title('Serie de Valores Máximos Anuales', fontweight='bold')
            ax_series.set_xlabel('Año')
            ax_series.set_ylabel('Precipitación (mm)')
            ax_series.grid(True, linestyle='--', alpha=0.5)
            results["max_annual_series_b64"] = _figure_to_base64(fig_series)

        # --- 3. ACF ---
        acf_vals, confint = acf(max_annual, nlags=10, alpha=0.05, fft=False)
        df_acf = pd.DataFrame({
            'Lag': range(1, len(acf_vals)),
            'Autocorrelación': np.round(acf_vals[1:], 4),
            'Límite Inf': np.round(confint[1:, 0] - acf_vals[1:], 4),
            'Límite Sup': np.round(confint[1:, 1] - acf_vals[1:], 4)
        })
        limit = 1.96 / np.sqrt(len(max_annual))
        df_acf['Interpretación'] = np.where(np.abs(df_acf['Autocorrelación']) > limit, 'Significativa', 'No Signif.')
        results["df_acf"] = df_acf
        
        with plt.style.context('default'):
            fig_acf, ax_acf = plt.subplots(figsize=(12, 6)) 
            plot_acf(max_annual, lags=10, ax=ax_acf, fft=False, color='black', vlines_kwargs={'colors': 'black'}, alpha=0.05)
            ax_acf.set_title('Correlograma (ACF)', fontweight='bold')
            ax_acf.grid(True, linestyle='--', alpha=0.5)
            results["acf_plot_b64"] = _figure_to_base64(fig_acf)

        # --- 4. Weibull ---
        n = len(max_annual)
        sorted_max = np.sort(max_annual)[::-1] 
        ranks = np.arange(1, n + 1)
        t_r_weibull = (n + 1) / ranks
        
        results["df_weibull"] = pd.DataFrame({
            'Orden': ranks, 'P_Max': np.round(sorted_max, 2), 'TR': np.round(t_r_weibull, 2)
        })
        
        with plt.style.context('default'):
            fig_wei, ax_wei = plt.subplots(figsize=(12, 6)) 
            ax_wei.plot(t_r_weibull, sorted_max, 'o', color='#4682B4', markeredgecolor='black')
            ax_wei.set_xscale('log')
            ax_wei.set_xticks([1, 2, 5, 10, 20, 50, 100])
            ax_wei.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax_wei.set_xlabel('Periodo de Retorno (años)')
            ax_wei.set_ylabel('Precipitación (mm)')
            ax_wei.set_title('Posición de Graficación (Weibull)', fontweight='bold')
            ax_wei.grid(True, which='both', linestyle='--', alpha=0.5)
            results["weibull_plot_b64"] = _figure_to_base64(fig_wei)

        # --- 5. Ajuste Distribuciones (Restaurado Completo) ---
        distributions = {
            'Normal': stats.norm, 'Log Normal': stats.lognorm, 'Exponencial': stats.expon,
            'Gamma': stats.gamma, 'Pearson III': stats.pearson3, 'Gumbel': stats.gumbel_r,
            'Log Pearson': lambda d: stats.pearson3.fit(np.log(d + 1e-10)),
            'Gumbel 2 Pob': lambda d: (stats.gumbel_r.fit(d[:len(d)//2]), stats.gumbel_r.fit(d[len(d)//2:])),
        }
        
        fit_results = []
        x_range = np.linspace(max(0.1, max_annual.min()*0.5), max_annual.max()*1.5, 200)
        
        for name, dist in distributions.items():
            try:
                if name == 'Log Pearson':
                    params = dist(max_annual)
                    pdf_func = lambda x: np.where(x>0, stats.pearson3.pdf(np.log(x), *params)/x, 0)
                    cdf_func = lambda x: np.where(x>0, stats.pearson3.cdf(np.log(x), *params), 0)
                elif name == 'Gumbel 2 Pob':
                    p1, p2 = dist(max_annual)
                    params = (p1, p2)
                    pdf_func = lambda x: 0.5*stats.gumbel_r.pdf(x,*p1) + 0.5*stats.gumbel_r.pdf(x,*p2)
                    cdf_func = lambda x: 0.5*stats.gumbel_r.cdf(x,*p1) + 0.5*stats.gumbel_r.cdf(x,*p2)
                else:
                    params = dist.fit(max_annual)
                    pdf_func = lambda x: dist.pdf(x, *params)
                    cdf_func = lambda x: dist.cdf(x, *params)

                # Métricas Estadísticas
                ks_stat, ks_p = stats.kstest(max_annual, cdf_func)
                
                # MSE y SE
                sorted_dat = np.sort(max_annual)
                emp_prob = 1 - (1/t_r_weibull)[::-1] 
                theo_prob = cdf_func(sorted_dat)
                mse = np.mean((emp_prob - theo_prob) ** 2)
                se = np.sqrt(mse)
                
                # Chi Cuadrada
                num_bins = max(5, int(len(max_annual) / 5))
                obs_freq, bin_edges = np.histogram(max_annual, bins=num_bins)
                exp_probs = np.diff(cdf_func(bin_edges))
                exp_freq = len(max_annual) * exp_probs
                # Ajuste para chi2
                exp_freq = np.maximum(exp_freq, 0.01)
                exp_freq = exp_freq * (obs_freq.sum() / exp_freq.sum())
                chi2_stat, chi2_p = stats.chisquare(f_obs=obs_freq, f_exp=exp_freq)

                fit_results.append({
                    'Name': name, 'Params': params, 'P_Val': ks_p, 
                    'KS': ks_stat, 'MSE': mse, 'SE': se, 
                    'Chi2': chi2_stat, 'Chi2_P': chi2_p,
                    'Y_Vals': pdf_func(x_range)
                })
            except: pass

        fit_results.sort(key=lambda x: x['P_Val'], reverse=True)
        
        # Tabla Resumen (Columnas Solicitadas)
        results["df_ajustes"] = pd.DataFrame([{
            'Distribución': r['Name'], 
            'Ajuste': 'Aceptado' if r['P_Val']>0.05 else 'Rechazado',
            'KS Stat': round(r['KS'], 4), 
            'P-Valor': round(r['P_Val'], 4),
            'MSE': round(r['MSE'], 5),
            'Error Std (SE)': round(r['SE'], 5),
            'Chi2 Stat': round(r['Chi2'], 4)
        } for r in fit_results])
        
        if fit_results:
            results["best_fit_name"] = fit_results[0]['Name']
            # Mapeo de nombres
            if 'Gumbel 2' in results["best_fit_name"]: results["best_fit_name"] = 'Gumbel 2 Poblaciones'
            if 'General' in results["best_fit_name"]: results["best_fit_name"] = 'General Valores Extremos'

        # --- Gráfico Comparativo (Estilo Académico + Todas las curvas) ---
        with plt.style.context('default'):
            fig_dist, ax_dist = plt.subplots(figsize=(12, 6))
            
            # Histograma Base
            counts, bins, _ = ax_dist.hist(max_annual, bins='auto', density=True, alpha=0.3, color='lightgray', edgecolor='black', label='Datos Observados')
            max_density = counts.max() if len(counts) > 0 else 0.1
            
            # Colores para diferenciar todas las curvas
            colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#a65628','#f781bf','#999999']
            
            # Graficar TODAS las curvas
            for i, res in enumerate(fit_results):
                col = colors[i % len(colors)]
                # Línea más gruesa para el ganador (índice 0)
                lw = 2.5 if i == 0 else 1.2
                alpha = 1.0 if i == 0 else 0.7
                label = f"{res['Name']} (p={res['P_Val']:.2f})"
                
                # Solo graficar si tiene sentido visual
                if np.max(res['Y_Vals']) > 0.0001:
                    ax_dist.plot(x_range, res['Y_Vals'], color=col, linewidth=lw, alpha=alpha, label=label)

            # Smart Scaling: Cortar el infinito
            ax_dist.set_ylim(0, max_density * 1.4) 
            
            ax_dist.set_title('Comparación de Distribuciones de Probabilidad', fontweight='bold')
            ax_dist.set_xlabel('Precipitación (mm)')
            ax_dist.set_ylabel('Densidad de Probabilidad')
            ax_dist.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=9)
            ax_dist.grid(True, linestyle='--', alpha=0.5)
            
            results["dist_comparison_b64"] = _figure_to_base64(fig_dist)

        # --- 7. Máximos Mensuales ---
        df_proc = df_filtrado.copy()
        df_proc['Año'], df_proc['Mes'] = df_proc.index.year, df_proc.index.month
        df_mx = df_proc.pivot_table(index='Año', columns='Mes', values='PRECIP_imputado', aggfunc='max')
        for m in range(1, 13): 
            if m not in df_mx.columns: df_mx[m] = 0.0
        df_mx = df_mx[sorted(df_mx.columns)]
        df_mx.columns = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
        df_mx['Max_Anual'] = df_mx.max(axis=1)
        results["df_maximos_mensuales"] = df_mx
        
        return results

    except Exception as e:
        print(f"Error lluvias: {e}")
        traceback.print_exc()
        return None