import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import traceback

class MotorEmpirico:
    """
    [MÓDULO AISLADO: ARF EMPÍRICO]
    Calcula el Factor de Reducción Areal utilizando modelos matemáticos 
    estáticos (geométricos) avalados internacionalmente.
    Rango Físico Estricto: 0.40 <= ARF <= 1.00
    """

    @staticmethod
    def _calcular_uswb(area_km2: float, duracion_hr: float) -> float:
        """Modelo 1: U.S. Weather Bureau (Aprox. de Leclerc & Schaake)"""
        try:
            area_mi2 = area_km2 / 2.58999
            d_025 = duracion_hr ** 0.25
            arf = 1 - np.exp(-1.1 * d_025) + np.exp(-1.1 * d_025 - 0.015 * np.sqrt(area_mi2))
            return float(np.clip(arf, 0.4, 1.0))
        except: return 1.0

    @staticmethod
    def _calcular_temez(area_km2: float) -> float:
        """Modelo 2: Norma de Drenaje Española (Fórmula de Témez)"""
        try:
            if area_km2 < 1.0: return 1.0
            arf = 1 - (np.log10(area_km2) / 15.0)
            return float(np.clip(arf, 0.4, 1.0))
        except: return 1.0

    @staticmethod
    def ejecutar_analisis(area_objetivo_km2: float, duracion_hr: float) -> dict:
        try:
            if area_objetivo_km2 <= 0:
                raise ValueError("El área de la cuenca debe ser mayor a 0.")
                
            # 1. Cálculos Puntuales Actuales
            val_uswb = MotorEmpirico._calcular_uswb(area_objetivo_km2, duracion_hr)
            val_temez = MotorEmpirico._calcular_temez(area_objetivo_km2)

            # 2. GRÁFICO 1: FRA vs ÁREA (Logarítmico)
            areas_plot = np.logspace(0, 4, 100)
            y_uswb_a = [MotorEmpirico._calcular_uswb(a, duracion_hr) for a in areas_plot]
            y_temez_a = [MotorEmpirico._calcular_temez(a) for a in areas_plot]

            fig1, ax1 = plt.subplots(figsize=(9, 4))
            fig1.patch.set_facecolor('#0a0a0a'); ax1.set_facecolor('#0a0a0a')
            ax1.plot(areas_plot, y_uswb_a, label='USWB (TP-29)', color='#1c75fa', linewidth=2)
            ax1.plot(areas_plot, y_temez_a, label='Témez (España)', color='#9900ff', linewidth=2)
            
            ax1.scatter([area_objetivo_km2]*2, [val_uswb, val_temez], color='#00ff41', s=100, zorder=5, marker='X')
            ax1.axvline(x=area_objetivo_km2, color='#00ff41', linestyle='--', alpha=0.3)
            
            ax1.set_xscale('log')
            ax1.set_ylim(0.35, 1.05)
            ax1.set_title(f"Decaimiento Areal (Duración: {duracion_hr} h)", color='white', pad=10)
            ax1.set_xlabel("Área de la Cuenca (km²)", color='grey'); ax1.set_ylabel("Factor (FRA)", color='grey')
            ax1.tick_params(colors='grey')
            for spine in ax1.spines.values(): spine.set_color('#333333')
            ax1.legend(facecolor='#111111', edgecolor='#333333', labelcolor='white')
            ax1.grid(True, linestyle=':', color='#333333', alpha=0.6)
            
            plt.tight_layout(); buf1 = io.BytesIO(); plt.savefig(buf1, format='png', dpi=120); plt.close(fig1)

            # 3. GRÁFICO 2: FRA vs TIEMPO (Minutos hasta 24h)
            duraciones_min = np.linspace(5, 1440, 200)
            y_uswb_t = [MotorEmpirico._calcular_uswb(area_objetivo_km2, d/60.0) for d in duraciones_min]
            y_temez_t = [MotorEmpirico._calcular_temez(area_objetivo_km2) for d in duraciones_min] 

            fig2, ax2 = plt.subplots(figsize=(10, 4))
            fig2.patch.set_facecolor('#0a0a0a'); ax2.set_facecolor('#0a0a0a')
            ax2.plot(duraciones_min, y_uswb_t, label='USWB (TP-29)', color='#1c75fa', linewidth=2)
            ax2.plot(duraciones_min, y_temez_t, label='Témez (España)', color='#9900ff', linewidth=2, linestyle='-.')
            
            minutos_actuales = duracion_hr * 60
            ax2.scatter([minutos_actuales]*2, [val_uswb, val_temez], color='#00ff41', s=100, zorder=5, marker='X')
            ax2.axvline(x=minutos_actuales, color='#00ff41', linestyle='--', alpha=0.3)
            
            ax2.set_xlim(0, 1440) 
            ax2.set_ylim(0.35, 1.05)
            ax2.set_title(f"Evolución Temporal del FRA (Área fija: {area_objetivo_km2:.1f} km²)", color='white', pad=10)
            ax2.set_xlabel("Duración de la Tormenta (Minutos)", color='grey'); ax2.set_ylabel("Factor (FRA)", color='grey')
            ax2.tick_params(colors='grey')
            for spine in ax2.spines.values(): spine.set_color('#333333')
            ax2.legend(facecolor='#111111', edgecolor='#333333', labelcolor='white')
            ax2.grid(True, linestyle=':', color='#333333', alpha=0.6)
            
            plt.tight_layout(); buf2 = io.BytesIO(); plt.savefig(buf2, format='png', dpi=120); plt.close(fig2)

            # 4. MATRIZ DE TABLA DE DATOS
            duraciones_clave = list(range(5, 1445, 5))
            tabla_datos = []
            for d_min in duraciones_clave:
                d_h = d_min / 60.0
                tabla_datos.append({
                    "minutos": d_min, 
                    "horas": round(d_h, 2),
                    "uswb": MotorEmpirico._calcular_uswb(area_objetivo_km2, d_h),
                    "temez": MotorEmpirico._calcular_temez(area_objetivo_km2)
                })

            return {
                "exito": True, "area": area_objetivo_km2, "duracion": duracion_hr,
                "fra_uswb": val_uswb, "fra_temez": val_temez,
                "plot1_b64": base64.b64encode(buf1.getvalue()).decode('utf-8'),
                "plot2_b64": base64.b64encode(buf2.getvalue()).decode('utf-8'),
                "tabla_datos": tabla_datos
            }

        except Exception as e:
            traceback.print_exc()
            return {"exito": False, "error": str(e)}