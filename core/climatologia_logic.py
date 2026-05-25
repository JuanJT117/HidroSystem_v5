import pandas as pd
import numpy as np
import io
import traceback
import matplotlib
import matplotlib.pyplot as plt
import base64

matplotlib.use('Agg') # Backend sin UI para Flet

# ==========================================
# 1. PARSER CRUDO AISLADO (ETL INDEPENDIENTE)
# ==========================================
def leer_estacion_climatologica(ruta_archivo):
    """
    Lee un archivo .txt crudo de CONAGUA (ej. 19052.txt) de forma 100% aislada.
    Ignora las cabeceras de texto y convierte 'NULO' a NaN matemáticos.
    """
    try:
        with open(ruta_archivo, 'r', encoding='utf-8', errors='ignore') as f:
            lineas = f.readlines()
            
        # 1. Buscar dinámicamente dónde empiezan los datos
        start_idx = 0
        for i, linea in enumerate(lineas):
            if linea.startswith('FECHA'):
                start_idx = i
                break
                
        if start_idx == 0:
            return False, None, "No se encontró la cabecera 'FECHA' en el archivo."

        # 2. Leer con Pandas saltando la cabecera dinámica
        # El separador en los txt de CONAGUA suele ser tabulación (\t)
        df = pd.read_csv(
            ruta_archivo, 
            sep=r'\t+', # Expresión regular para 1 o más tabulaciones
            skiprows=start_idx, 
            engine='python',
            na_values=['NULO', 'NULO ', ' NULO']
        )
        
        # 3. Limpieza Estructural
        # Limpiar espacios en los nombres de las columnas
        df.columns = [str(c).strip() for c in df.columns]
        
        # Eliminar la fila de unidades (ej. "(mm)", "(°C)")
        df = df[~df['FECHA'].astype(str).str.contains(r'\(', na=False)]
        df = df.dropna(subset=['FECHA'])
        
        # 4. Tipado de Datos Seguro (Casting)
        df['FECHA'] = pd.to_datetime(df['FECHA'], errors='coerce')
        df = df.dropna(subset=['FECHA']) # Tirar filas que no tengan fecha válida
        
        columnas_numericas = ['PRECIP', 'EVAP', 'TMAX', 'TMIN']
        for col in columnas_numericas:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # 5. Agregar columnas de tiempo para el análisis
        df['Anio'] = df['FECHA'].dt.year
        df['Mes'] = df['FECHA'].dt.month
        
        return True, df, "Datos cargados correctamente."
        
    except Exception as e:
        traceback.print_exc()
        return False, None, f"Error aisaldo en parser climatológico: {str(e)}"

# ==========================================
# 2. MOTOR ESTADÍSTICO (PROBABILIDAD Y NORMALES)
# ==========================================
def calcular_normales_climatologicas(df):
    """
    Calcula las medias térmicas y replica el cálculo exacto de MÁXIMOS pluviales 
    usado en el Módulo de Probabilidad (Analisis Lluvias).
    """
    try:
        df_proc = df.copy()
        # Asegurar casteo a numérico
        columnas_num = ['PRECIP', 'TMAX', 'TMIN', 'EVAP']
        for col in columnas_num:
            if col in df_proc.columns:
                df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce')
                
        if 'Anio' not in df_proc.columns: df_proc['Anio'] = df_proc['FECHA'].dt.year
        if 'Mes' not in df_proc.columns: df_proc['Mes'] = df_proc['FECHA'].dt.month

        # === 1. EXTRACCIÓN DE PROBABILIDAD (CLON MÓDULO 3) ===
        # Lluvia Máxima en 24h usando pivot_table idéntico a lluvias_logic.py
        df_mx = df_proc.pivot_table(index='Anio', columns='Mes', values='PRECIP', aggfunc='max')
        
        # Garantizar columnas de meses (1-12) aunque haya vacíos
        for m in range(1, 13): 
            if m not in df_mx.columns: df_mx[m] = 0.0
        df_mx = df_mx[sorted(df_mx.columns)]
        
        # Promedio Máximo Mensual por Año
        precip_promedio = df_mx.mean()
        # Máximo Mensual Absoluto Histórico
        precip_absoluta = df_mx.max()

        # === 2. CLIMATOLOGÍA TÉRMICA (Normales) ===
        temp_normales = df_proc.groupby('Mes').agg({'TMAX': 'mean', 'TMIN': 'mean'})
        
        # Evaporación: Suma mensual por año, luego el promedio de esas sumas
        evap_anio = df_proc.groupby(['Anio', 'Mes'])['EVAP'].sum().reset_index()
        evap_normal = evap_anio.groupby('Mes')['EVAP'].mean()

        # === 3. CONSOLIDACIÓN ===
        normales = pd.DataFrame({
            'Mes': range(1, 13),
            'TMAX': temp_normales['TMAX'],
            'TMIN': temp_normales['TMIN'],
            'PRECIP_Max_Promedio': precip_promedio.values,
            'PRECIP_Max_Absoluta': precip_absoluta.values,
            'EVAP': evap_normal
        })
        
        return normales.round(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None

def obtener_extremos_historicos(df):
    """ Escanea la serie para encontrar los días de calor y frío récord. """
    extremos = {}
    try:
        if not df['TMAX'].dropna().empty:
            idx_max = df['TMAX'].idxmax()
            extremos['Temp_Max_Absoluta'] = df.loc[idx_max, 'TMAX']
            extremos['Fecha_Temp_Max'] = df.loc[idx_max, 'FECHA'].strftime('%Y-%m-%d')
        if not df['TMIN'].dropna().empty:
            idx_min = df['TMIN'].idxmin()
            extremos['Temp_Min_Absoluta'] = df.loc[idx_min, 'TMIN']
            extremos['Fecha_Temp_Min'] = df.loc[idx_min, 'FECHA'].strftime('%Y-%m-%d')
    except Exception as e:
        pass
    return extremos

# ==========================================
# 3. GENERADOR DE GRÁFICOS (CLIMOGRAMAS CON EVAPORACIÓN)
# ==========================================
def generar_climograma_b64(df_normales, titulo_estacion="Estación Climatológica"):
    """
    Genera el climograma mostrando la Lluvia Máxima (24h) y la Evaporación en el mismo eje (mm).
    """
    if df_normales is None or df_normales.empty: return None
    
    meses_nombres = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    meses_num = df_normales['Mes'].astype(int).values
    etiquetas = [meses_nombres[m-1] for m in meses_num]
    
    x = np.arange(len(etiquetas))
    width = 0.3  
    
    with plt.style.context('default'):
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        color_promedio = '#1c75fa'
        color_absoluto = '#003366'
        color_evap = '#ff7f00' # Naranja para la curva de evaporación
        
        # 1. Dibujar barras agrupadas de Lluvia Extrema (mm)
        ax1.bar(x - width/2, df_normales['PRECIP_Max_Promedio'], width, color=color_promedio, alpha=0.8, label='Lluvia Máx. Promedio (24h)')
        ax1.bar(x + width/2, df_normales['PRECIP_Max_Absoluta'], width, color=color_absoluto, alpha=0.9, label='Lluvia Máx. Absoluta Histórica')
        
        # 2. Dibujar curva de Evaporación Media Mensual (mm) - Comparte el mismo eje Y
        ax1.plot(x, df_normales['EVAP'], color=color_evap, marker='^', linestyle='--', linewidth=2, label='Evaporación Media Mensual (mm)')
        
        ax1.set_xlabel('Meses')
        ax1.set_ylabel('Precipitación y Evaporación (mm)', color='black', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(etiquetas)
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, linestyle='--', alpha=0.4, axis='y')
        
        # 3. Eje Y Secundario (Temperaturas en °C)
        ax2 = ax1.twinx()  
        ax2.plot(x, df_normales['TMAX'], color='#cc0000', marker='o', linewidth=2, label='T. Máxima Media (°C)')
        ax2.plot(x, df_normales['TMIN'], color='#00aaff', marker='o', linewidth=2, label='T. Mínima Media (°C)')
        
        ax2.set_ylabel('Temperatura (°C)', color='black', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='black')
        
        plt.title(f'Climograma de Extremos y Evaporación Mensual - {titulo_estacion}', fontweight='bold', fontsize=14)
        
        # Unir todas las leyendas y posicionarlas limpiamente arriba
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5, fontsize=8)
        
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    
# ==========================================
# 4. MOTOR DE FUSIÓN (DATOS FILTRADOS/LIMPIOS)
# ==========================================
def fusionar_precipitacion_filtrada(df_clima_raw, df_filtrada):
    """
    Toma el DataFrame crudo y reemplaza PRECIP con la serie limpia.
    BLINDAJE: Evita seleccionar columnas temporales como 'Año' o 'Semana'.
    """
    try:
        df_res = df_clima_raw.copy(deep=True)
        df_filt = df_filtrada.copy(deep=True)
        
        # 1. Anclar Fechas
        col_fecha = next((c for c in df_filt.columns if 'fech' in str(c).lower() or 'date' in str(c).lower()), None)
        if col_fecha:
            df_filt[col_fecha] = pd.to_datetime(df_filt[col_fecha], errors='coerce')
            df_filt = df_filt.dropna(subset=[col_fecha])
            df_filt.set_index(col_fecha, inplace=True)
        else:
            df_filt.index = pd.to_datetime(df_filt.index, errors='coerce')
            
        # 2. Caza del Dato Correcto (Evitar el Bug del Año)
        if 'PRECIP_imputado' in df_filt.columns:
            col_lluvia = 'PRECIP_imputado'
        elif 'PRECIP_original' in df_filt.columns:
            col_lluvia = 'PRECIP_original'
        else:
            precips = [c for c in df_filt.columns if 'PRECIP' in str(c).upper()]
            col_lluvia = precips[0] if precips else df_filt.select_dtypes(include=[np.number]).columns[0]
        
        # 3. Fusión
        df_res.set_index('FECHA', inplace=True)
        df_res.update(df_filt[[col_lluvia]].rename(columns={col_lluvia: 'PRECIP'}))
        df_res.reset_index(inplace=True)
        
        return True, df_res, "Precipitación filtrada fusionada con éxito."
    except Exception as e:
        return False, df_clima_raw, f"Error en fusión: {str(e)}"
    
# ==========================================
# 5. AUTO-EXTRACTOR DE MEMORIA (ZERO-TRUST)
# ==========================================
def obtener_lluvia_limpia_automatica(est_id, page_session):
    """
    Busca la serie de lluvia filtrada en memoria o la recalcula al vuelo 
    usando el motor del Módulo 3 para garantizar datos impecables.
    """
    import os
    from core import Analisis
    
    # 1. Intentar obtener el CSV imputado desde la carpeta del proyecto
    ruta_imputados = page_session.get("imput_folder_path")
    if ruta_imputados:
        csv_path = os.path.join(ruta_imputados, f"{est_id}.csv")
        if os.path.exists(csv_path):
            try:
                # Usamos el motor de Analisis para procesar y limpiar al vuelo
                df_proc = Analisis.procesar_datos(csv_path)
                if df_proc is not None:
                    # Aplicamos el filtro de calidad del Módulo 3
                    res_filtro = Analisis.filtrar_datos(df_proc)
                    # Manejo defensivo por si filtrar_datos devuelve tupla o solo DF
                    df_limpio = res_filtro[0] if isinstance(res_filtro, tuple) else res_filtro
                    return df_limpio
            except Exception as e:
                print(f"Error al recalcular lluvia limpia para {est_id}: {e}")

    # 2. Fallback: Buscar en la base de datos de estaciones activa
    est_db = page_session.get("estaciones_db") or {}
    if est_id in est_db and "df_filtrado" in est_db[est_id]:
        return est_db[est_id]["df_filtrado"]

    # 3. Fallback final: Buscar en el DF activo de la sesión
    df_activo = page_session.get("df_filtrado")
    if df_activo is not None:
        return df_activo

    return None
# ==========================================
# 6. MOTOR DE EXPORTACIÓN CLIMÁTICA LATEX (MINIMALISTA)
# ==========================================
def exportar_reporte_climatico_latex(ruta_completa_tex, df_normales, extremos, nombre_estacion):
    """
    Genera un archivo .tex independiente, limpio y ultra-compatible.
    No requiere paquetes exóticos, solo 'graphicx' para la imagen.
    """
    try:
        import os
        from jinja2 import Template
        
        # Preparar estructura de carpetas de salida
        directorio_base = os.path.dirname(ruta_completa_tex)
        os.makedirs(directorio_base, exist_ok=True)
        
        dir_graficos = os.path.join(directorio_base, "Graficos")
        os.makedirs(dir_graficos, exist_ok=True)
        
        # Renderizar gráfico en alta definición (200 DPI) para la impresión final
        img_name = f"Climograma_{nombre_estacion}.png"
        ruta_imagen = os.path.join(dir_graficos, img_name)
        
        meses_nombres = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        etiquetas = [meses_nombres[int(m)-1] for m in df_normales['Mes'].values]
        x = np.arange(len(etiquetas))
        width = 0.3
        
        with plt.style.context('default'):
            fig, ax1 = plt.subplots(figsize=(11, 5.5))
            ax1.bar(x - width/2, df_normales['PRECIP_Max_Promedio'], width, color='#1c75fa', alpha=0.8, label='Lluvia Máx. Promedio')
            ax1.bar(x + width/2, df_normales['PRECIP_Max_Absoluta'], width, color='#003366', alpha=0.9, label='Lluvia Máx. Absoluta')
            ax1.plot(x, df_normales['EVAP'], color='#ff7f00', marker='^', linestyle='--', linewidth=2, label='Evaporación Media')
            ax1.set_xlabel('Meses')
            ax1.set_ylabel('Precipitación y Evaporación (mm)', color='black', fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(etiquetas)
            ax1.grid(True, linestyle='--', alpha=0.4, axis='y')
            
            ax2 = ax1.twinx()
            ax2.plot(x, df_normales['TMAX'], color='#cc0000', marker='o', linewidth=2, label='T. Máxima')
            ax2.plot(x, df_normales['TMIN'], color='#00aaff', marker='o', linewidth=2, label='T. Mínima')
            ax2.set_ylabel('Temperatura (°C)', color='black', fontweight='bold')
            
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5, fontsize=8)
            
            plt.tight_layout()
            fig.savefig(ruta_imagen, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close(fig)

        # Mapeo de meses en español para la tabla
        meses_map = {1:'Enero', 2:'Febrero', 3:'Marzo', 4:'Abril', 5:'Mayo', 6:'Junio', 
                     7:'Julio', 8:'Agosto', 9:'Septiembre', 10:'Octubre', 11:'Noviembre', 12:'Diciembre'}
        
        filas_tabla = []
        for _, row in df_normales.iterrows():
            filas_tabla.append({
                'mes': meses_map[int(row['Mes'])],
                'tmax': f"{row['TMAX']:.1f}",
                'tmin': f"{row['TMIN']:.1f}",
                'p_prom': f"{row['PRECIP_Max_Promedio']:.1f}",
                'p_abs': f"{row['PRECIP_Max_Absoluta']:.1f}",
                'evap': f"{row['EVAP']:.1f}"
            })

        # Plantilla LaTeX Minimalista Formal (Garantiza compilación sin librerías pesadas)
        template_latex = """
% --- REPORTE CLIMATOLÓGICO LOCAL INDEPENDIENTE ---
\\section{Análisis Climatológico - Estación {{ nombre_estacion }}}

Este documento resume la caracterización climática y el comportamiento termopluviométrico extremo procesado para la estación de trabajo. Las series de precipitación presentadas corresponden a los datos limpios y filtrados empleados en los modelos estadísticos de diseño.

\\subsection{Extremos Térmicos Históricos}

A continuación se presentan las temperaturas absolutas extremas registradas históricamente en la estación:

\\begin{table}[h!]
\\centering
\\renewcommand{\\arraystretch}{1.3}
\\begin{tabular}{|l|c|c|}
\\hline
\\textbf{Parámetro Térmico} & \\textbf{Temperatura (°C)} & \\textbf{Fecha de Registro} \\\\ \\hline
Temperatura Máxima Absoluta & {{ ext_tmax }} °C & {{ fecha_tmax }} \\\\ \\hline
Temperatura Mínima Absoluta & {{ ext_tmin }} °C & {{ fecha_tmin }} \\\\ \\hline
\\end{tabular}
\\caption{Extremos térmicos absolutos históricos.}
\\end{table}

\\subsection{Normales Climatológicas y Balance Evaporativo}

La siguiente tabla consolida los valores medios mensuales de temperaturas, pérdidas por evaporación y eventos de precipitación máxima diaria (24 horas) calculados para el proyecto:

\\begin{table}[h!]
\\centering
\\renewcommand{\\arraystretch}{1.3}
\\resizebox{\\textwidth}{!}{
\\begin{tabular}{|l|c|c|c|c|c|}
\\hline
\\textbf{Mes} & \\textbf{T. Máx (°C)} & \\textbf{T. Mín (°C)} & \\textbf{P. Máx Prom (mm)} & \\textbf{P. Máx Abs (mm)} & \\textbf{Evaporación (mm)} \\\\ \\hline
{% for f in filas %}
{{ f.mes }} & {{ f.tmax }} & {{ f.tmin }} & {{ f.p_prom }} & {{ f.p_abs }} & {{ f.evap }} \\\\ \\hline
{% endfor %}
\\end{tabular}
}
\\caption{Resumen mensual de normales climatológicas y extremos hídricos.}
\\end{table}

\\newpage
\\subsection{Comportamiento Gráfico (Climograma Local)}

El climograma de la Figura ilustra la correlación entre las fluctuaciones de temperatura y el balance hídrico entre entradas extremas de tormenta y pérdidas por evaporación mensual.

\\begin{figure}[h!]
    \\centering
    \\includegraphics[width=0.95\\textwidth]{Graficos/{{ img_name }}}
    \\caption{Climograma local de extremos y evaporación - Estación {{ nombre_estacion }}.}
\\end{figure}
"""
        
        tex_content = Template(template_latex).render(
            nombre_estacion=nombre_estacion,
            ext_tmax=f"{extremos.get('Temp_Max_Absoluta', 0.0):.1f}",
            fecha_tmax=extremos.get('Fecha_Temp_Max', 'N/D'),
            ext_tmin=f"{extremos.get('Temp_Min_Absoluta', 0.0):.1f}",
            fecha_tmin=extremos.get('Fecha_Temp_Min', 'N/D'),
            filas=filas_tabla,
            img_name=img_name
        )
        
        with open(ruta_completa_tex, 'w', encoding='utf-8') as f:
            f.write(tex_content)
            
        return True, f"✅ Reporte LaTeX (.tex) guardado con éxito: {os.path.basename(ruta_completa_tex)}"
    except Exception as e:
        return False, f"❌ Error en LaTeX: {str(e)}"