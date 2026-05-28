import os
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from datetime import datetime

# Definición de la Paleta de Marca (Académica pero moderna)
COLOR_BG_CARD = (245, 247, 250)
COLOR_TEXT_MAIN = (30, 30, 30)
COLOR_VERDE_TLALOC = (0, 168, 45) # Verde legible en fondo blanco
COLOR_ROJO_ALERTA = (204, 0, 0)
COLOR_GRIS = (120, 120, 120)

class DQA_Dashboard(FPDF):
    def header(self):
        self.set_fill_color(10, 10, 10)
        self.rect(0, 0, 210, 25, 'F')
        
        self.set_font('helvetica', 'B', 16)
        self.set_text_color(0, 255, 65) # Verde Cyberpunk para el título
        self.cell(0, 10, 'T L Á L O C   -   H I D R O S I S T E M', 0, 1, 'C')
        
        self.set_font('helvetica', 'I', 10)
        self.set_text_color(200, 200, 200)
        self.cell(0, 5, 'Data Quality Assessment (DQA) - Auditoría Profunda de Base de Datos', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.cell(0, 10, f'Reporte autogenerado por HidroSistem | Engine V10.9 | {timestamp} | Pág {self.page_no()}', 0, 0, 'C')

    def draw_kpi_card(self, x, y, title, value, status_color):
        # Fondo de la tarjeta
        self.set_fill_color(*COLOR_BG_CARD)
        self.rect(x, y, 60, 25, 'F')
        
        # Línea indicadora izquierda
        self.set_fill_color(*status_color)
        self.rect(x, y, 2, 25, 'F')
        
        # Textos
        self.set_xy(x + 5, y + 3)
        self.set_font('helvetica', 'B', 9)
        self.set_text_color(*COLOR_GRIS)
        self.cell(50, 5, title, 0, 1, 'L')
        
        self.set_xy(x + 5, y + 10)
        self.set_font('helvetica', 'B', 14)
        self.set_text_color(*COLOR_TEXT_MAIN)
        self.cell(50, 10, str(value), 0, 1, 'L')

def generar_graficos_temporales(stats):
    """Genera gráficas de Matplotlib/Seaborn y las retorna como buffers en memoria (BytesIO)"""
    # Preparar datos
    datos_calor = []
    for k, v in stats.items():
        if v["sanas"] > 0:
            prom_inicio = int(sum(v["años_inicio"])/len(v["años_inicio"]))
            prom_fin = int(sum(v["años_fin"])/len(v["años_fin"]))
            datos_calor.append({"Estado": v["nom"], "Inicio": prom_inicio, "Fin": prom_fin, "Gaps": v["max_gap"]})
            
    df_graf = pd.DataFrame(datos_calor).sort_values("Inicio")
    
    # Gráfico 1: Rango Operativo
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    for i, row in df_graf.iterrows():
        ax1.plot([row["Inicio"], row["Fin"]], [row["Estado"], row["Estado"]], color='#1c75fa', linewidth=3)
    ax1.set_title("Línea de Tiempo Operativa Promedio por Estado", fontweight="bold", fontsize=10)
    ax1.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', dpi=200)
    buf1.seek(0)
    plt.close(fig1)

    return buf1

def generar_dashboard_dqa(stats, ruta_salida, nacional_sanas, nacional_corruptas):
    """Orquesta la construcción vectorial del PDF."""
    pdf = DQA_Dashboard()
    pdf.add_page()
    
    # --- FILA 1: TARJETAS KPI ---
    total = nacional_sanas + nacional_corruptas
    ratio_sanas = (nacional_sanas / total * 100) if total > 0 else 0
    total_registros = sum(v["total_dias"] for v in stats.values())
    total_nulos = sum(v["nulos"] for v in stats.values())
    sparsity = (total_nulos / total_registros * 100) if total_registros > 0 else 0
    total_outliers = sum(v["outliers"] for v in stats.values())
    
    # Determinar color del semáforo
    color_salud = COLOR_VERDE_TLALOC if ratio_sanas > 75 else (255, 165, 0)
    
    pdf.draw_kpi_card(15, 30, "INTEGRIDAD GLOBAL", f"{ratio_sanas:.1f}% Sanas", color_salud)
    pdf.draw_kpi_card(80, 30, "VOLUMEN DE DATOS", f"{total_registros/1_000_000:.1f} M Reg.", (28, 117, 250))
    pdf.draw_kpi_card(145, 30, "SPARSITY / NULOS", f"{sparsity:.2f}% Vacíos", COLOR_ROJO_ALERTA if sparsity > 10 else COLOR_VERDE_TLALOC)
    
    pdf.draw_kpi_card(15, 60, "ARCHIVOS CORRUPTOS", f"{nacional_corruptas} Estaciones", COLOR_ROJO_ALERTA if nacional_corruptas > 50 else COLOR_VERDE_TLALOC)
    pdf.draw_kpi_card(80, 60, "OUTLIERS DETECTADOS", f"{total_outliers} Días", (153, 0, 255))
    pdf.draw_kpi_card(145, 60, "MÁX. BRECHA TEMPORAL", f"{max((v['max_gap'] for v in stats.values()), default=0)} Años Sin Data", (255, 165, 0))

    # --- FILA 2: GRÁFICOS MATPLOTLIB EN MEMORIA ---
    buf_timeline = generar_graficos_temporales(stats)
    pdf.image(buf_timeline, x=15, y=90, w=180)
    
    # --- FILA 3: MATRIZ DE DETALLE VECTORIAL (Tabla de Riesgo) ---
    pdf.add_page()
    pdf.set_font('helvetica', 'B', 14)
    pdf.set_text_color(*COLOR_TEXT_MAIN)
    pdf.cell(0, 10, 'Matriz de Inspección Hídrica por Entidad', 0, 1, 'L')
    pdf.ln(2)
    
    # Cabeceras de tabla
    pdf.set_font('helvetica', 'B', 10)
    pdf.set_fill_color(220, 220, 220)
    columnas = [("Estado", 40), ("Operativas", 30), ("Corruptas", 30), ("Sparsity %", 30), ("Max Gap", 25), ("Integridad (Visual)", 40)]
    for col, width in columnas:
        pdf.cell(width, 8, col, 1, 0, 'C', True)
    pdf.ln()

    # Filas de tabla
    pdf.set_font('helvetica', '', 9)
    estados_ordenados = sorted(stats.items(), key=lambda x: x[1]["sanas"], reverse=True)
    
    for estado_clave, data in estados_ordenados:
        if data["sanas"] == 0 and data["corruptas"] == 0: continue
        
        tot_est = data["sanas"] + data["corruptas"]
        ratio_int = (data["sanas"] / tot_est) if tot_est > 0 else 0
        ratio_null = (data["nulos"] / data["total_dias"] * 100) if data["total_dias"] > 0 else 0
        
        pdf.cell(40, 8, data["nom"][:18], 1, 0, 'L')
        pdf.cell(30, 8, str(data["sanas"]), 1, 0, 'C')
        pdf.cell(30, 8, str(data["corruptas"]), 1, 0, 'C')
        
        # Destacar sparsity alto en rojo
        pdf.set_text_color(204,0,0) if ratio_null > 15 else pdf.set_text_color(*COLOR_TEXT_MAIN)
        pdf.cell(30, 8, f"{ratio_null:.1f}%", 1, 0, 'C')
        pdf.set_text_color(*COLOR_TEXT_MAIN)
        
        pdf.cell(25, 8, f"{data['max_gap']} Años", 1, 0, 'C')
        
        # --- SPARKLINE VECTORIAL (Mini-Barra de progreso dentro de la tabla) ---
        x_start = pdf.get_x()
        y_start = pdf.get_y()
        pdf.cell(40, 8, "", 1, 0) # Celda vacía como marco
        
        # Pintar el progreso
        pdf.set_fill_color(*COLOR_ROJO_ALERTA)
        pdf.rect(x_start + 2, y_start + 2, 36, 4, 'F') # Fondo rojo (Corruptas)
        if ratio_int > 0:
            pdf.set_fill_color(*COLOR_VERDE_TLALOC)
            pdf.rect(x_start + 2, y_start + 2, 36 * ratio_int, 4, 'F') # Frente verde (Sanas)
            
        pdf.ln()

    ruta_pdf = os.path.join(ruta_salida, "Dashboard_DQA_Tlaloc.pdf")
    pdf.output(ruta_pdf)
    
    # Limpieza de memoria RAM
    buf_timeline.close()
    return ruta_pdf