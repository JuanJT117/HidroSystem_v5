import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely import wkt
from scipy.interpolate import Rbf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import traceback
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

class MotorUNAM:
    """
    [MÓDULO AISLADO: ARF UNAM - TORMENTA CENTRADA (TOPOLÓGICO)]
    Supera el método de "círculos fijos" midiendo el área y volumen real 
    de las isoyetas (huella de la tormenta) usando Kriging Anisotrópico.
    """

    @staticmethod
    def _calcular_area_pixel(x, y):
        """Calcula el área en km² de un pixel del grid basado en lat/lon."""
        mean_lat = np.mean(y)
        # 1 grado de latitud = ~111.32 km. Longitud varía por el coseno de la latitud.
        dx_km = (x[1] - x[0]) * 111.32 * np.cos(np.radians(mean_lat))
        dy_km = (y[1] - y[0]) * 111.32
        return abs(dx_km * dy_km)

    @staticmethod
    def ejecutar_analisis_tormenta_centrada(df_maestro: pd.DataFrame, df_coords: pd.DataFrame, cuenca_wkt: str) -> dict:
        try:
            print("🌪️ [TLÁLOC] Iniciando Fase 5: Tormenta Centrada (Método UNAM Topológico)")
            
            estaciones = [col for col in df_maestro.columns if col != 'LMA_Base']
            
            # --- 1. PROXY VOLUMÉTRICO: Encontrar a los 3 "Godzillas" ---
            # En lugar de Kriging a 40 años, buscamos los días con mayor masa de agua regional
            df_proxy = df_maestro[estaciones].mean(axis=1)
            top_3_fechas = df_proxy.nlargest(3).index
            
            # --- 2. CONFIGURACIÓN DEL GRID REGIONAL ---
            x_coords, y_coords, claves_validas = [], [], []
            for est in estaciones:
                if est in df_coords.index:
                    x_coords.append(df_coords.loc[est, 'LONGITUD'])
                    y_coords.append(df_coords.loc[est, 'LATITUD'])
                    claves_validas.append(est)
                    
            x, y = np.array(x_coords), np.array(y_coords)
            
            # Grid extendido (+20% margen) para ver cómo muere la tormenta fuera de la cuenca
            minx, maxx = x.min(), x.max()
            miny, maxy = y.min(), y.max()
            margen_x = (maxx - minx) * 0.20
            margen_y = (maxy - miny) * 0.20
            
            xi = np.linspace(minx - margen_x, maxx + margen_x, 150)
            yi = np.linspace(miny - margen_y, maxy + margen_y, 150)
            XI, YI = np.meshgrid(xi, yi)
            
            pixel_area_km2 = MotorUNAM._calcular_area_pixel(xi, yi)
            
            curvas_tormentas = []
            mapas_tormentas = []
            
            # --- 3. RECONSTRUCCIÓN FORENSE DE LAS 3 SUPER-TORMENTAS ---
            for rank, fecha in enumerate(top_3_fechas):
                z = np.array([df_maestro.loc[fecha, est] for est in claves_validas])
                
                # Interpolación Radial (Simulación de Isoyetas)
                rbf = Rbf(x, y, z, function='linear')
                ZI = rbf(XI, YI)
                ZI = np.clip(ZI, 0, None)
                
                z_max_absoluto = ZI.max()
                if z_max_absoluto <= 0: continue
                
                # Descenso Topológico: Medir área y volumen capa por capa
                # Bajamos desde el 99% del epicentro hasta el 5%
                umbrales_pct = np.linspace(0.99, 0.05, 40)
                areas_km2 = []
                fra_valores = []
                
                for pct in umbrales_pct:
                    umbral_lluvia = z_max_absoluto * pct
                    # Máscara booleana: Pixeles que pertenecen a esta isoyeta
                    mask_isoyeta = ZI >= umbral_lluvia
                    
                    area_actual = mask_isoyeta.sum() * pixel_area_km2
                    lluvia_media_areal = ZI[mask_isoyeta].mean()
                    
                    fra_actual = lluvia_media_areal / z_max_absoluto
                    
                    if area_actual > 0:
                        areas_km2.append(area_actual)
                        fra_valores.append(fra_actual)
                        
                curvas_tormentas.append({
                    "fecha": fecha.strftime('%Y-%m-%d'),
                    "rank": rank + 1,
                    "p_max": z_max_absoluto,
                    "areas": np.array(areas_km2),
                    "fras": np.array(fra_valores)
                })
                
                # RENDERIZADO DEL MAPA DE LA TORMENTA (Modo Radar)
                fig_m, ax_m = plt.subplots(figsize=(6, 4))
                fig_m.patch.set_facecolor('#0a0a0a'); ax_m.set_facecolor('#0a0a0a')
                
                contour = ax_m.contourf(XI, YI, ZI, levels=20, cmap='turbo', alpha=0.85)
                # Trazar las curvas de nivel (isoyetas)
                ax_m.contour(XI, YI, ZI, levels=10, colors='black', linewidths=0.5, alpha=0.5)
                
                cbar = fig_m.colorbar(contour, ax=ax_m, fraction=0.046, pad=0.04)
                cbar.set_label('Precipitación (mm)', color='white')
                cbar.ax.yaxis.set_tick_params(color='white')
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
                
                # Cuenca para referencia
                if cuenca_wkt:
                    gdf_cuenca = gpd.GeoDataFrame(index=[0], geometry=[wkt.loads(cuenca_wkt)], crs="EPSG:4326")
                    gdf_cuenca.plot(ax=ax_m, facecolor='none', edgecolor='#00ff41', linewidth=2)
                    
                ax_m.scatter(x, y, c='white', edgecolor='black', s=15, alpha=0.5)
                
                # Marcar el epicentro
                idx_y, idx_x = np.unravel_index(ZI.argmax(), ZI.shape)
                ax_m.scatter(XI[idx_y, idx_x], YI[idx_y, idx_x], c='#ff0044', marker='*', s=200, edgecolor='white', zorder=10)
                
                ax_m.set_title(f"Anatomía de Super-Celda #{rank+1} | {fecha.strftime('%Y-%m-%d')}\n$P_{{max}}$: {z_max_absoluto:.1f} mm", color='white', fontsize=10)
                ax_m.axis('off')
                plt.tight_layout(); buf_m = io.BytesIO(); plt.savefig(buf_m, format='png', dpi=120); plt.close(fig_m)
                
                mapas_tormentas.append({
                    "id": f"Tormenta {rank+1} ({fecha.strftime('%Y-%m-%d')})",
                    "b64": base64.b64encode(buf_m.getvalue()).decode('utf-8')
                })

            # --- 4. CONSTRUCCIÓN DE LA CURVA ENVOLVENTE DE DISEÑO ---
            # Definimos un vector maestro de Áreas para interpolar todas las curvas a una misma escala X
            area_maxima_global = max([c['areas'].max() for c in curvas_tormentas])
            areas_maestras = np.linspace(1, area_maxima_global, 100)
            
            matriz_fras_interpolados = []
            for curva in curvas_tormentas:
                # np.interp requiere X creciente
                idx_sort = np.argsort(curva['areas'])
                fra_interp = np.interp(areas_maestras, curva['areas'][idx_sort], curva['fras'][idx_sort])
                matriz_fras_interpolados.append(fra_interp)
                
            matriz_fras_np = np.array(matriz_fras_interpolados)
            # La envolvente de diseño es el FRA MÍNIMO para cada tamaño de área (El escenario más crítico)
            fra_envolvente = np.min(matriz_fras_np, axis=0)

            # RENDERIZADO DEL GRÁFICO DAD (Depth-Area)
            fig_d, ax_d = plt.subplots(figsize=(9, 5))
            fig_d.patch.set_facecolor('#0a0a0a'); ax_d.set_facecolor('#0a0a0a')
            
            colores_tormentas = ['#1c75fa', '#9900ff', '#ffaa00']
            for i, curva in enumerate(curvas_tormentas):
                ax_d.plot(curva['areas'], curva['fras'], color=colores_tormentas[i], alpha=0.6, linewidth=1.5, label=f"T{i+1}: {curva['fecha']}")
                
            ax_d.plot(areas_maestras, fra_envolvente, color='#ff0044', linewidth=3, label='ENVOLVENTE DE DISEÑO')
            
            ax_d.set_xscale('log') # Escala logarítmica para ver bien el inicio de la cuenca
            ax_d.set_ylim(0.4, 1.05)
            ax_d.set_title("Curvas Área-Profundidad (Tormenta Centrada) - Método UNAM", color='white', pad=10)
            ax_d.set_xlabel("Área de Cobertura de la Tormenta (km²) - Escala Log", color='grey')
            ax_d.set_ylabel("Factor de Reducción Areal (FRA)", color='grey')
            ax_d.tick_params(colors='grey')
            for spine in ax_d.spines.values(): spine.set_color('#333333')
            ax_d.legend(facecolor='#111111', edgecolor='#333333', labelcolor='white')
            ax_d.grid(True, which='both', linestyle=':', color='#333333', alpha=0.5)
            plt.tight_layout(); buf_d = io.BytesIO(); plt.savefig(buf_d, format='png', dpi=120); plt.close(fig_d)

            # Generar Tabla de la Envolvente para exportar
            tabla_envolvente = [{"area": round(a, 1), "fra": round(f, 4)} for a, f in zip(areas_maestras, fra_envolvente)]

            return {
                "exito": True,
                "mapas": mapas_tormentas,
                "plot_dad": base64.b64encode(buf_d.getvalue()).decode('utf-8'),
                "tabla_envolvente": tabla_envolvente
            }

        except Exception as e:
            traceback.print_exc()
            return {"exito": False, "error": str(e)}
