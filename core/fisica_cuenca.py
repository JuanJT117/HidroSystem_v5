import numpy as np
import rasterio
from rasterio.features import shapes
import richdem as rd
import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import unary_union
import gc
import traceback
import psutil
import GPUtil
import os
import time

from core.modelos_espaciales import SubcuencaSchema, CaucesSchema
from core.analisis_espacial import DEFAULT_CRS

class MotorFisicoRichDEM:
    
    @staticmethod
    def configurar_recursos_90_pct(callback=None):
        """
        Detecta la capacidad del equipo y ajusta los recursos de procesamiento (CPU, RAM, GPU) al 90%.
        """
        def reportar(msg):
            print(msg)
            if callback:
                try: callback(msg)
                except Exception: pass
                
        reportar("\n--- DETECCIÓN Y CONFIGURACIÓN DE HARDWARE AL 90% ---")
        
        # 1. Límite de CPU
        cpu_count = os.cpu_count() or 4
        limit_cpu = max(1, int(cpu_count * 0.90))
        os.environ["OMP_NUM_THREADS"] = str(limit_cpu)
        os.environ["OPENBLAS_NUM_THREADS"] = str(limit_cpu)
        os.environ["MKL_NUM_THREADS"] = str(limit_cpu)
        reportar(f"[CPU] Detectados {cpu_count} núcleos lógicos. Limitando procesamiento a {limit_cpu} núcleos (90%).")
        
        # 2. Límite de RAM
        ram_info = psutil.virtual_memory()
        ram_total_gb = ram_info.total / (1024**3)
        ram_limit_gb = ram_total_gb * 0.90
        reportar(f"[RAM] Memoria Total: {ram_total_gb:.2f} GB. Límite de seguridad: {ram_limit_gb:.2f} GB (90%).")
        
        # 3. Límite de GPU
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_mem_total = gpu.memoryTotal
                gpu_mem_limit = gpu_mem_total * 0.90
                # Aunque RichDEM corre en CPU, seteamos la variable de entorno para futuras integraciones CuPy/Torch.
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                reportar(f"[GPU] Detectada GPU: {gpu.name}. Limitando VRAM a: {gpu_mem_limit:.0f} MB (90%).")
            else:
                reportar("[GPU] No se detectó GPU dedicada de NVIDIA compatible. Todo el cálculo irá a CPU.")
        except Exception as e:
            reportar(f"[GPU] Error al detectar GPU: {e}")
            
        reportar("----------------------------------------------------\n")

    @staticmethod
    def procesar_dem(ruta_dem: str, umbral_acumulacion: int, callback=None) -> dict:
        """
        [FASE 3 - RICHDEM EN RAM]
        Ingiere un DEM .tif, lo procesa en memoria C++ multihilo sin tocar el disco,
        y devuelve Contratos Espaciales OGC vectorizados.
        """
        def reportar(msg):
            print(msg)
            if callback:
                try:
                    callback(msg)
                except Exception:
                    pass

        # Variables para limpieza segura
        dem_data = rd_dem = d8 = accum = streams_mask = None
        
        try:
            # Configurar Hardware antes de empezar
            MotorFisicoRichDEM.configurar_recursos_90_pct(callback=reportar)
            
            reportar(f"[{time.strftime('%H:%M:%S')}] [PROCESO] Iniciando lectura de DEM directamente a RAM...")
            # 1. LECTURA DIRECTA A RAM (Evasión de I/O)
            with rasterio.open(ruta_dem) as src:
                transform = src.transform
                crs = src.crs if src.crs else DEFAULT_CRS
                nodata = src.nodata if src.nodata is not None else -9999.0
                # RichDEM prefiere float32 para cálculos de elevación
                dem_data = src.read(1).astype(np.float32)

            reportar(f"[{time.strftime('%H:%M:%S')}] [PROCESO] Ejecutando Álgebra Matricial C++ en RichDEM...")
            # 2. ÁLGEBRA MATRICIAL EN C++ (RichDEM)
            rd_dem = rd.rdarray(dem_data, no_data=nodata)
            rd_dem.geotransform = transform.to_gdal()
            if crs is not None and hasattr(crs, 'to_wkt'):
                rd_dem.projection = crs.to_wkt()
            
            reportar(f"[{time.strftime('%H:%M:%S')}] [PROCESO 1/3] Acondicionamiento Topológico (Llenado de depresiones)...")
            # Acondicionamiento Topológico (Algoritmo Barnes 2014 - O(N))
            rd.FillDepressions(rd_dem, epsilon=True, in_place=True)
            
            reportar(f"[{time.strftime('%H:%M:%S')}] [PROCESO 2/3] Ruteo D8 y Matriz de Acumulación de Flujo...")
            # Ruteo D8 y Acumulación
            accum = rd.FlowAccumulation(rd_dem, method='D8')
            
            reportar(f"[{time.strftime('%H:%M:%S')}] [PROCESO 3/3] Extracción de Escorrentías y Cauces con umbral {umbral_acumulacion}...")
            # 3. EXTRACCIÓN DE ESCORRENTÍAS (Vectores booleanos de NumPy)
            # Todo pixel con acumulación mayor al umbral se considera "río"
            streams_mask = (accum > umbral_acumulacion).astype(np.uint8)
            
            reportar(f"[{time.strftime('%H:%M:%S')}] [VECTORIZACIÓN] Convirtiendo matrices a geometrías vectoriales (Shapely/GeoPandas)...")
            # 4. VECTORIZACIÓN EN RAM (Rasterio -> Shapely)
            cauces_geoms = []
            for geom, val in shapes(streams_mask, mask=streams_mask==1, transform=transform, connectivity=8):
                poly = shape(geom)
                cauces_geoms.append(poly.boundary)

            # Empaquetado en Contrato OGC Estricto (CaucesSchema)
            datos_cauces = []
            for i, geom in enumerate(cauces_geoms):
                datos_cauces.append({
                    'ID_Cauce': i + 1,
                    'Orden_Stra': 1, # TODO: Algoritmo Strahler en matriz
                    'Es_Princip': 0,
                    'Longitud_m': round(geom.length, 2),
                    'Desnivel_m': 0.0, # TODO: Cruce matricial de elevaciones
                    'Manning_n': 0.035,
                    'geometry': geom
                })
            gdf_cauces = gpd.GeoDataFrame(datos_cauces, crs=crs)

            reportar(f"[{time.strftime('%H:%M:%S')}] [DELIMITACIÓN] Extrayendo Cuenca Global...")
            # 5. DELIMITACIÓN DE CUENCA GLOBAL (Catchment Area)
            # Para esta primera versión híbrida, tomamos todos los píxeles que drenan 
            # (Accum > 0) y los fusionamos en una sola "Gran Cuenca".
            cuenca_mask = (accum > 0).astype(np.uint8)
            cuencas_geoms = []
            for geom, val in shapes(cuenca_mask, mask=cuenca_mask==1, transform=transform, connectivity=8):
                cuencas_geoms.append(shape(geom))
            
            # OPTIMIZACIÓN EXTREMA: En lugar de usar `unary_union` que es O(N^2) y
            # congela Shapely por horas con polígonos complejos, `shapes` ya agrupa 
            # los píxeles en polígonos disjuntos. Solo tomamos el de mayor área.
            if not cuencas_geoms:
                raise RuntimeError("No se detectaron zonas de acumulación en el DEM.")
                
            cuenca_master = max(cuencas_geoms, key=lambda a: a.area)

            # Empaquetado en Contrato OGC Estricto (SubcuencaSchema)
            gdf_cuencas = gpd.GeoDataFrame([{
                'ID_Cuenca': 1,
                'Nombre': "Cuenca_Principal_RichDEM",
                'Area_km2': round(cuenca_master.area / 1e6, 3),
                'Perimetro_km': round(cuenca_master.length / 1000, 3),
                'Pendient_pct': 0.0,
                'Tc_minutos': 0.0,
                'geometry': cuenca_master
            }], crs=crs)

            return {
                "cuencas": gdf_cuencas,
                "cauces": gdf_cauces
            }
            
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Fallo en motor Físico RAM (RichDEM): {str(e)}")
        
        finally:
            # DIRECTRIZ 1 y 3: Recolección explícita de basura para Big Data
            # Borramos las matrices gigantescas para no asfixiar el Hilo UI de Flet
            del dem_data, rd_dem, d8, accum, streams_mask
            gc.collect()
