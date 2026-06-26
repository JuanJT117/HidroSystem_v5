import json
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.ops import polygonize
from scipy.spatial import Voronoi
import fiona
import os
import traceback
import gc

# --- [PARCHE DEVSECOPS: FIJAR RUTAS PROJ PARA FIONA] ---
# Corrige "PROJ: Cannot find proj.db" en entornos Conda / PyInstaller
_fiona_proj = os.path.join(os.path.dirname(fiona.__file__), "proj_data")
if os.path.exists(_fiona_proj):
    os.environ["PROJ_LIB"] = _fiona_proj
    os.environ["PROJ_DATA"] = _fiona_proj
# --------------------------------------------------------

# Definición del CRS oficial del sistema (UTM Zona 14N por defecto, parametrizable)
DEFAULT_CRS = "EPSG:32614"

class MotorEspacial:
    
    @staticmethod
    def generar_plantilla(schema_class, tipo_geometria: str, crs=DEFAULT_CRS) -> gpd.GeoDataFrame:
        """Genera un GeoDataFrame vacío con el tipado estricto del esquema Pydantic."""
        def get_dtype(annotation):
            if annotation is int: return 'Int64'
            if annotation is float: return 'float64'
            if annotation is str: return 'object'
            return 'object'
            
        esquema = schema_class.model_fields
        columnas = {nombre: pd.Series(dtype=get_dtype(field.annotation)) 
                    for nombre, field in esquema.items()}
        
        df = pd.DataFrame(columnas)
        # Inyección segura de geometría vacía
        gdf = gpd.GeoDataFrame(df, geometry=pd.Series(dtype='geometry'), crs=crs)
        return gdf

    @staticmethod
    def procesar_cuenca_objetivo(ruta_shp: str, buffer_km: int = 100) -> dict:
        """
        Ingesta, valida y proyecta la Cuenca Objetivo (Zero-Trust).
        Aplica una 'Caja de Arena Métrica' para cálculos de área y buffers,
        y retorna geometrías en WGS84 para UI y Base de Datos.
        """
        try:
            # 1. Ingesta Defensiva (Aduana Espacial)
            cuenca_gdf = gpd.read_file(ruta_shp)
            if cuenca_gdf.empty:
                raise ValueError("El Shapefile está vacío o carece de geometría.")

            # 2. Diagnóstico y Auto-Curación del CRS original
            if cuenca_gdf.crs is None:
                # Si el creador no incluyó archivo .prj, evaluamos por heurística
                minx, miny, maxx, maxy = cuenca_gdf.total_bounds
                if maxx > 180 or minx < -180:
                    cuenca_gdf.set_crs("EPSG:32614", inplace=True) # Asumimos UTM Zona 14N
                else:
                    cuenca_gdf.set_crs("EPSG:4326", inplace=True)  # Asumimos Geográficas
            
            # 3. CAJA DE ARENA MÉTRICA (Proyección temporal)
            # EPSG:3857 preserva el cálculo matemático en Metros.
            cuenca_metrica = cuenca_gdf.to_crs("EPSG:3857")

            # 4. Cálculo Matemático de Área (Estricto en km²)
            area_m2 = cuenca_metrica.geometry.area.sum()
            area_km2 = area_m2 / 1_000_000.0

            # 5. Generación de Buffer en metros (100 km = 100,000 m)
            buffer_metrico = cuenca_metrica.copy()
            buffer_metrico.geometry = cuenca_metrica.geometry.buffer(buffer_km * 1000)

            # 6. Reproyección de Salida a WGS84 (EPSG:4326) para Motor Gráfico y Relacional
            cuenca_geo = cuenca_metrica.to_crs("EPSG:4326")
            buffer_geo = buffer_metrico.to_crs("EPSG:4326")

            # 7. Conversión a GeoJSON (Geometría pura Polygon/MultiPolygon para el Canvas de Flet)
            cuenca_geojson = cuenca_geo.geometry.unary_union.__geo_interface__
            buffer_geojson = buffer_geo.geometry.unary_union.__geo_interface__
            bbox = list(buffer_geo.total_bounds) # [minx, miny, maxx, maxy]

            # 8. Recolección de basura explícita (Protección de RAM)
            del cuenca_gdf, cuenca_metrica, buffer_metrico
            gc.collect()

            return {
                "area_km2": round(area_km2, 2),
                "buffer_geom": buffer_geo.geometry.unary_union, # Unifica polígonos divididos si los hay
                "cuenca_geojson": cuenca_geojson,
                "buffer_geojson": buffer_geojson,
                "bbox": bbox,
                "arf_activado": area_km2 > 25.0
            }

        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Rechazo en Aduana Espacial: {str(e)}")

    @staticmethod
    def exportar_plantilla_shp(schema_class, tipo_geometria: str, ruta_salida: str, crs=DEFAULT_CRS):
        """
        [PARCHE DEVSECOPS] Exporta un Shapefile vacío forzando estrictamente 
        el tipo de geometría en el Header binario del archivo .shp.
        Evita el fallback a LineString de GDAL.
        """
        esquema_pydantic = schema_class.model_fields
        
        # 1. Mapeo de tipos de Python a tipos compatibles con C/Fiona
        propiedades_fiona = {}
        for nombre, field in esquema_pydantic.items():
            tipo_str = str(field.annotation).lower()
            if 'int' in tipo_str: 
                fiona_type = 'int'
            elif 'float' in tipo_str: 
                fiona_type = 'float'
            else: 
                fiona_type = 'str'
            propiedades_fiona[nombre] = fiona_type

        # 2. Definición inquebrantable del Esquema
        schema_fiona = {
            'geometry': tipo_geometria, # 'Polygon' o 'LineString' forzado
            'properties': propiedades_fiona
        }
        
        # 3. Creación atómica del archivo usando Fiona en modo escritura ('w')
        # Al no escribir ningún registro (pass), se crea un archivo perfecto y vacío.
        try:
            with fiona.open(ruta_salida, 'w', driver='ESRI Shapefile', crs=crs, schema=schema_fiona) as c:
                pass 
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Fallo al forzar el header del Shapefile en Fiona: {str(e)}")

    @staticmethod
    def generar_thiessen(df_estaciones: pd.DataFrame, gdf_cuenca: gpd.GeoDataFrame, crs=DEFAULT_CRS) -> gpd.GeoDataFrame:
        """
        Calcula Polígonos de Thiessen (Voronoi) vectorizados y los recorta con la cuenca.
        df_estaciones DEBE contener ['Clave', 'X_UTM', 'Y_UTM', 'Prec_Base']
        """
        try:
            # 1. Extraer coordenadas
            coords = df_estaciones[['X_UTM', 'Y_UTM']].values
            
            # 2. Truco SciPy: Añadir "Puntos Fantasma" muy lejanos para cerrar los polígonos periféricos
            radio_expansion = 200000 # 200km
            center = coords.mean(axis=0)
            puntos_fantasma = np.array([
                center + [radio_expansion, 0], center + [-radio_expansion, 0],
                center + [0, radio_expansion], center + [0, -radio_expansion]
            ])
            puntos_totales = np.vstack([coords, puntos_fantasma])
            
            # 3. Álgebra de Voronoi (C++ Backend)
            vor = Voronoi(puntos_totales)
            
            # 4. Reconstrucción Topológica Shapely
            lines = [LineString(vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line]
            poligonos = list(polygonize(lines))
            
            # 5. Mapeo Espacial Punto a Polígono
            gdf_voronoi = gpd.GeoDataFrame(geometry=poligonos, crs=crs)
            gdf_puntos = gpd.GeoDataFrame(df_estaciones, geometry=gpd.points_from_xy(df_estaciones.X_UTM, df_estaciones.Y_UTM), crs=crs)
            
            # Spatial Join: ¿Qué estación quedó dentro de qué polígono?
            gdf_unido = gpd.sjoin(gdf_voronoi, gdf_puntos, how="inner", predicate="intersects")
            
            # 6. Intersección con la Cuenca Limitante (Recorte)
            gdf_final = gpd.overlay(gdf_unido, gdf_cuenca, how='intersection')
            
            # 7. Cálculo de Áreas Automático (En Metros Cuadrados -> km2)
            gdf_final['Area_km2'] = gdf_final.geometry.area / 1e6
            area_total = gdf_final['Area_km2'].sum()
            gdf_final['Peso_Relativo'] = gdf_final['Area_km2'] / area_total
            
            return gdf_final
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Fallo en álgebra espacial Voronoi: {str(e)}")

    @staticmethod
    def ingestar_capa_externa(ruta_archivo: str, schema_class, capa_nombre: str = None) -> gpd.GeoDataFrame:
        """
        DevSecOps: Aduana de validación. Verifica que el SHP/GPKG de QGIS cumpla el contrato.
        """
        try:
            # EAFP: Intentar cargar (GDAL maneja el binario)
            gdf = gpd.read_file(ruta_archivo, layer=capa_nombre)
        except Exception as e:
            raise RuntimeError(f"Archivo corrupto o ilegible por GDAL: {str(e)}")
            
        # Validación de Columnas
        columnas_requeridas = set(schema_class.model_fields.keys())
        columnas_actuales = set(gdf.columns)
        
        faltantes = columnas_requeridas - columnas_actuales
        if faltantes:
            raise ValueError(f"Fallo de Ingesta: Faltan las siguientes columnas obligatorias: {faltantes}")
            
        # Forzar CRS a UTM 14N si viene en WGS84 (Tolerancia Cero a EPSG:4326)
        if gdf.crs != DEFAULT_CRS:
            gdf = gdf.to_crs(DEFAULT_CRS)
            
        return gdf
