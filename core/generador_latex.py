import os
import base64
from jinja2 import Template

def exportar_informe_latex(res_hydros, df_vars, res_r, res_c, res_h, graficos_metodos, graficos_tr, ruta_salida, nombre_proyecto="Proyecto Hidrológico"):
    """
    Motor Jinja2 Modular: Genera archivos .tex independientes listos para hacer \include{} en un archivo maestro.
    """
    if df_vars is None or df_vars.empty: 
        return False, "No hay datos calculados para generar el reporte."
    
    # 1. Preparar directorios
    os.makedirs(ruta_salida, exist_ok=True)
    dir_graficos = os.path.join(ruta_salida, "Graficos")
    os.makedirs(dir_graficos, exist_ok=True)

    archivos_generados = []

    # ==========================================
    # MÓDULO 1: VARIABLES FÍSICAS
    # ==========================================
    if df_vars is not None and not df_vars.empty:
        cols_vars = df_vars.columns.tolist()
        filas_vars = [{"idx": str(idx), "vals": [f"{float(v):.3f}" if isinstance(v, (int, float)) else str(v) for v in row]} for idx, row in df_vars.iterrows()]
        
        tpl_vars = """
\\chapter{Análisis de Parámetros Físicos}
\\section{Morfometría y Cinemática}

La siguiente tabla resume las propiedades físicas, coeficientes de escurrimiento y tiempos de concentración procesados para cada cuenca del modelo.

\\begin{table}[h!]
\\centering
\\resizebox{\\textwidth}{!}{
\\renewcommand{\\arraystretch}{1.5}
\\begin{tabular}{|l|{% for col in cols %}c|{% endfor %}}
\\hline
\\rowcolor[HTML]{00FF41}
\\textbf{Cuenca} {% for col in cols %} & \\textbf{ {{ col|replace('_', '\\_') }} } {% endfor %} \\\\ \\hline
{% for row in filas %}
\\textbf{ {{ row.idx|replace('_', '\\_') }} } {% for val in row.vals %} & {{ val }} {% endfor %} \\\\ \\hline
{% endfor %}
\\end{tabular}
}
\\caption{Parámetros morfométricos y cinemáticos de las cuencas.}
\\end{table}
        """
        with open(os.path.join(ruta_salida, "1_Variables_Fisicas.tex"), "w", encoding="utf-8") as f:
            f.write(Template(tpl_vars).render(cols=cols_vars, filas=filas_vars))
        archivos_generados.append("1_Variables")

    # ==========================================
    # MÓDULO 2: GASTOS PICO
    # ==========================================
    def preparar_tabla_gastos(df):
        if df is None or df.empty: return None
        return {
            "cols": df.columns.tolist(),
            "filas": [{"idx": str(idx), "vals": [f"{float(v):.3f}" for v in row]} for idx, row in df.iterrows()]
        }

    tpl_gastos = """
\\chapter{Caudales Máximos de Diseño}
\\section{Resumen de Gastos Pico}

En esta sección se presentan los caudales pico ($m^3/s$) calculados para distintos Periodos de Retorno (TR).

{% macro render_tabla(titulo, data) %}
\\subsection{ {{ titulo }} }
\\begin{table}[h!]
\\centering
\\resizebox{\\textwidth}{!}{
\\renewcommand{\\arraystretch}{1.5}
\\begin{tabular}{|l|{% for col in data.cols %}c|{% endfor %}}
\\hline
\\rowcolor[HTML]{00FF41}
\\textbf{Cuenca} {% for col in data.cols %} & \\textbf{ {{ col|replace('_', '\\_') }} } {% endfor %} \\\\ \\hline
{% for row in data.filas %}
\\textbf{ {{ row.idx|replace('_', '\\_') }} } {% for val in row.vals %} & {{ val }} {% endfor %} \\\\ \\hline
{% endfor %}
\\end{tabular}
}
\\caption{Caudales máximos ($m^3/s$) - {{ titulo }}.}
\\end{table}
{% endmacro %}

{% if d_r %} {{ render_tabla("Método Racional", d_r) }} {% endif %}
{% if d_c %} {{ render_tabla("Método de Ven Te Chow", d_c) }} {% endif %}
{% if d_h %} {{ render_tabla("Método HEC-HMS (SCS)", d_h) }} {% endif %}
    """
    d_r, d_c, d_h = preparar_tabla_gastos(res_r), preparar_tabla_gastos(res_c), preparar_tabla_gastos(res_h)
    with open(os.path.join(ruta_salida, "2_Gastos_Pico.tex"), "w", encoding="utf-8") as f:
        f.write(Template(tpl_gastos).render(d_r=d_r, d_c=d_c, d_h=d_h))
    archivos_generados.append("2_Gastos")

    # ==========================================
    # MÓDULO 3: GRÁFICOS GLOBALES
    # ==========================================
    if graficos_metodos or graficos_tr:
        lista_graficos = []
        
        # Decodificamos y guardamos archivos físicos
        for lst, prefijo, titulo in [(graficos_metodos, "Metodo", "Evolución Multicuenca"), (graficos_tr, "TR", "Comparativa")]:
            if not lst: continue
            for key_val, b64 in lst:
                fname = f"Global_{prefijo}_{str(key_val).replace(' ', '_')}.png"
                with open(os.path.join(dir_graficos, fname), "wb") as f_img:
                    f_img.write(base64.b64decode(b64))
                lista_graficos.append({"file": fname, "cap": f"{titulo} - {key_val}"})

        tpl_graficos = """
\\chapter{Análisis Visual Comparativo}

A continuación se presentan las comparativas gráficas multicuenca.

{% for g in graficos %}
\\begin{figure}[h!]
    \\centering
    \\includegraphics[width=0.95\\textwidth]{Graficos/{{ g.file }}}
    \\caption{ {{ g.cap }} }
\\end{figure}
{% endfor %}
        """
        with open(os.path.join(ruta_salida, "3_Graficos_Globales.tex"), "w", encoding="utf-8") as f:
            f.write(Template(tpl_graficos).render(graficos=lista_graficos))
        archivos_generados.append("3_Graficos")

    # ==========================================
    # MÓDULO 4: HIDROGRAMAS DE TRÁNSITO
    # ==========================================
    if res_hydros:
        tpl_hydros = """
\\chapter{Hidrogramas de Diseño}
\\section{Curvas de Tránsito}

{% for key, datos in cuencas.items() %}
\\subsection{Resultados: {{ key|replace('_', '\\_') }}}

\\begin{table}[h!]
\\centering
\\renewcommand{\\arraystretch}{1.5}
\\begin{tabular}{|l|c|}
\\hline
\\rowcolor[HTML]{00FF41} \\textbf{Parámetro} & \\textbf{Valor} \\\\ \\hline
{% for k, v in datos.resumen.items() %} {{ k|replace('_', '\\_') }} & {{ v }} \\\\ \\hline {% endfor %}
\\end{tabular}
\\caption{Balance hídrico para {{ key|replace('_', '\\_') }}.}
\\end{table}

\\begin{figure}[h!]
    \\centering
    \\includegraphics[width=0.85\\textwidth]{Graficos/{{ key }}_hidrograma.png}
    \\caption{Hidrograma de diseño {{ key|replace('_', '\\_') }}.}
\\end{figure}
\\newpage
{% endfor %}
        """
        contexto_cuencas = {}
        try:
            for key, pack in res_hydros.items():
                df_det = pack["detalle"]
                
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 5))
                cols = df_det.columns.tolist()
                t_col, q_col = cols[0], cols[-1]
                t_vals, q_vals = df_det[t_col].values / 60.0, df_det[q_col].values
                
                ax.plot(t_vals, q_vals, color='blue', linewidth=2)
                ax.fill_between(t_vals, q_vals, color='blue', alpha=0.1)
                ax.set_title(f"Curva de Tránsito: {key}", fontweight='bold')
                ax.set_xlabel("Tiempo (Horas)"); ax.set_ylabel("Caudal Directo ($m^3/s$)")
                ax.grid(True, linestyle="--", alpha=0.6)
                
                fig.savefig(os.path.join(dir_graficos, f"{key}_hidrograma.png"), dpi=200, bbox_inches='tight')
                plt.close(fig)

                contexto_cuencas[key] = {"resumen": pack.get("resumen", {})}

            with open(os.path.join(ruta_salida, "4_Hidrogramas.tex"), "w", encoding="utf-8") as f:
                f.write(Template(tpl_hydros).render(cuencas=contexto_cuencas))
            archivos_generados.append("4_Hidrogramas")
        except Exception as e:
            import traceback
            traceback.print_exc()

    return True, f"✅ Módulos exportados: {', '.join(archivos_generados)}"