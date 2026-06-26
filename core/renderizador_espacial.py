import io
import base64
import matplotlib
matplotlib.use('Agg') # Directriz 3 y 4: Evita fugas de memoria y colisiones de hilos UI
import matplotlib.pyplot as plt
import geopandas as gpd
import traceback

class VisorEspacial:
    @staticmethod
    def renderizar_mapa_base64(capas: dict, kwargs_visuales: dict) -> str:
        """
        Renderiza las geometrías en un mapa estático académico.
        capas = {'cuenca': gdf_cuenca, 'cauces': gdf_cauces, 'suelo': gdf_suelo, 'thiessen': gdf_thiessen}
        kwargs_visuales = {'mostrar_cuenca': True, 'mostrar_cauces': True...}
        """
        try:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
            fig.patch.set_facecolor('#ffffff') # Fondo blanco (Académico)
            ax.set_facecolor('#f4f4f9')
            
            # Control de bounding box global
            minx, miny, maxx, maxy = float('inf'), float('inf'), float('-inf'), float('-inf')
            hay_datos = False

            def update_bounds(gdf):
                nonlocal minx, miny, maxx, maxy, hay_datos
                if gdf is not None and not gdf.empty:
                    bounds = gdf.total_bounds
                    minx, miny = min(minx, bounds[0]), min(miny, bounds[1])
                    maxx, maxy = max(maxx, bounds[2]), max(maxy, bounds[3])
                    hay_datos = True

            # 1. Uso de Suelo (Base inferior)
            if kwargs_visuales.get('mostrar_suelo') and capas.get('suelo') is not None:
                gdf = capas['suelo']
                # Colorear por Grupo Hidrológico
                gdf.plot(ax=ax, column='Grupo_Hidro', cmap='Pastel1', legend=True, alpha=0.6, edgecolor='gray', linewidth=0.5)
                update_bounds(gdf)

            # 2. Polígonos de Thiessen
            if kwargs_visuales.get('mostrar_thiessen') and capas.get('thiessen') is not None:
                gdf = capas['thiessen']
                gdf.plot(ax=ax, facecolor='none', edgecolor='#cc0000', linewidth=1.5, linestyle='--')
                update_bounds(gdf)

            import numpy as np
            from matplotlib.collections import LineCollection

            # Función de simplificación topológica
            def simplificar_seguro(gdf_in):
                try:
                    gdf_out = gdf_in.copy()
                    bounds = gdf_out.total_bounds
                    if bounds is not None and len(bounds) == 4:
                        tol = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / 1000.0
                        if tol > 0:
                            gdf_out['geometry'] = gdf_out['geometry'].simplify(tolerance=tol, preserve_topology=False)
                    return gdf_out
                except Exception:
                    return gdf_in

            # 3. Límite de Cuenca (Polígono superior hueco)
            if kwargs_visuales.get('mostrar_cuenca') and capas.get('cuenca') is not None:
                gdf = simplificar_seguro(capas['cuenca'])
                # FIX CRÍTICO: Geopandas plot crashea en C++. Trazamos manualmente.
                for geom in gdf['geometry']:
                    if geom is None or geom.is_empty: continue
                    if geom.geom_type == 'Polygon':
                        x, y = geom.exterior.xy
                        ax.plot(x, y, color='#000000', linewidth=2.5)
                    elif geom.geom_type == 'MultiPolygon':
                        for poly in geom.geoms:
                            x, y = poly.exterior.xy
                            ax.plot(x, y, color='#000000', linewidth=2.5)
                update_bounds(gdf)

            # 4. Red Hídrica (Líneas)
            if kwargs_visuales.get('mostrar_cauces') and capas.get('cauces') is not None:
                gdf = simplificar_seguro(capas['cauces'])
                col_princ = 'Es_Princip' if 'Es_Princip' in gdf.columns else ('Es_Principal' if 'Es_Principal' in gdf.columns else None)
                
                # FIX CRÍTICO: Trazado ultrarrápido sin Geopandas para evadir crasheos
                lineas = []
                colores = []
                grosores = []
                
                for idx, row in gdf.iterrows():
                    geom = row['geometry']
                    if geom is None or geom.is_empty: continue
                    
                    is_principal = False
                    if col_princ is not None:
                        is_principal = (row[col_princ] == 1)
                        
                    c = 'blue' if is_principal else 'deepskyblue'
                    lw = 2.5 if is_principal else 1.0
                    
                    if geom.geom_type == 'LineString':
                        lineas.append(np.column_stack((geom.xy[0], geom.xy[1])))
                        colores.append(c)
                        grosores.append(lw)
                    elif geom.geom_type == 'MultiLineString':
                        for line in geom.geoms:
                            lineas.append(np.column_stack((line.xy[0], line.xy[1])))
                            colores.append(c)
                            grosores.append(lw)
                
                if lineas:
                    lc = LineCollection(lineas, colors=colores, linewidths=grosores)
                    ax.add_collection(lc)
                    
                update_bounds(gdf)

            if not hay_datos:
                ax.text(0.5, 0.5, 'Sin Datos Espaciales Cargados\nGenere o importe capas.', 
                        horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='gray')
                ax.set_xticks([]); ax.set_yticks([])
            else:
                # Ajustar zoom con margen del 5%
                margen_x = (maxx - minx) * 0.05
                margen_y = (maxy - miny) * 0.05
                ax.set_xlim(minx - margen_x, maxx + margen_x)
                ax.set_ylim(miny - margen_y, maxy + margen_y)
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.set_xlabel('Coordenadas X (UTM)')
                ax.set_ylabel('Coordenadas Y (UTM)')

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
            
        except Exception as e:
            traceback.print_exc()
            plt.close('all')
            return None
