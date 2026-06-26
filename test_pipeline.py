import sys
import os
import traceback

sys.path.append(os.getcwd())

from core.descarga_logic import indexar_base_datos_tar, cargar_poligonos, GestorExtraccion
from core.analisis_espacial import MotorEspacial

def test_pipeline():
    try:
        print("1. Cargando poligonos de Cuencas...")
        cuencas = cargar_poligonos("POR CUENCA")
        
        print("\n2. Indexando BD...")
        ruta_bd = r"C:\Users\USER\Documents\proyectos\conda\CONAGUA-Completando_Datos\Version11\assets\TEST\Tlaloc_BD_Nacional_Comprimida4.tar.xz"
        
        def mock_log(msg): pass
        def mock_prog(prog): pass
        
        df_indice = indexar_base_datos_tar(ruta_bd, cuencas, mock_log, mock_prog)
        print(f"BD Indexada exitosamente. Total estaciones en la BD: {len(df_indice)}")
        
        print("\n3. Procesando SHP Cuenca Objetivo...")
        ruta_shp = r"C:\Users\USER\Documents\proyectos\conda\CONAGUA-Completando_Datos\Version11\assets\TEST\Cuenca-Test-15-06-2026.shp"
        resultado_espacial = MotorEspacial.procesar_cuenca_objetivo(ruta_shp)
        
        print(f"Area Cuenca: {resultado_espacial['area_km2']} km2")
        print(f"BBOX Buffer: {resultado_espacial['bbox']}")
        
        # Check GeoJSON integrity for UI
        cuenca_geojson = resultado_espacial['cuenca_geojson']
        buffer_geojson = resultado_espacial['buffer_geojson']
        print(f"Tipo GeoJSON Cuenca: {cuenca_geojson['type']}")
        print(f"Tipo GeoJSON Buffer: {buffer_geojson['type']}")
        
        print("\n4. Cruzando Estaciones con el Buffer...")
        buffer_geom = resultado_espacial['buffer_geom']
        
        # --- DEBUG PUNTOS ---
        print(f"Muestra de BD coords: \n{df_indice[['clave', 'lat', 'lon']].head()}")
        # --------------------
        
        claves_obj = GestorExtraccion.filtrar_estaciones_por_buffer(buffer_geom, df_indice)
        print(f"Total estaciones interceptadas en la cuenca objetivo: {len(claves_obj)}")
        print(f"Estaciones extraidas: {claves_obj[:10]}...")
        
        print("\nTEST FINALIZADO CON EXITO.")
        
    except Exception as e:
        print("\nERROR EN EL TEST:")
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
