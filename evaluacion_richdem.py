import os
import sys
import numpy as np
import rasterio
from rasterio.transform import from_origin
import time
import multiprocessing

# --- FIX PARA PROJ EN ENTORNOS CONDA ---
_proj_win = os.path.join(sys.prefix, "Library", "share", "proj")
_proj_unix = os.path.join(sys.prefix, "share", "proj")
if os.path.exists(os.path.join(_proj_win, "proj.db")):
    os.environ["PROJ_LIB"] = _proj_win
    os.environ["PROJ_DATA"] = _proj_win
elif os.path.exists(os.path.join(_proj_unix, "proj.db")):
    os.environ["PROJ_LIB"] = _proj_unix
    os.environ["PROJ_DATA"] = _proj_unix

# Forzamos que se cargue desde el core
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
try:
    from core.fisica_cuenca import MotorFisicoRichDEM
except ImportError as e:
    print(f"❌ Error importando MotorFisicoRichDEM: {e}")
    sys.exit(1)


def main():
    multiprocessing.freeze_support()
    print("=========================================================")
    print("🔬 TEST INTENSIVO DE ESTABILIDAD: MotorFisicoRichDEM en RAM")
    print("=========================================================")
    
    ruta_test = "test_large.tif"
    
    if not os.path.exists(ruta_test):
        print(f"❌ Error: No se encontró el archivo '{ruta_test}' para la evaluación.")
        sys.exit(1)
        
    try:
        from tqdm import tqdm
        print("\n🚀 Ejecutando MotorFisicoRichDEM.procesar_dem()...")
        umbral_acumulacion = 50 # Umbral bajo para forzar la extracción de muchos cauces en un raster pequeño
        
        # Barra de progreso simulada para la consola superior (el log real está dentro de fisica_cuenca)
        with tqdm(total=100, desc="Procesando DEM Completo", bar_format="{l_bar}{bar} [ tiempo restante: {eta} ]") as pbar:
            pbar.update(10)
            resultados = MotorFisicoRichDEM.procesar_dem(ruta_test, umbral_acumulacion)
            pbar.update(90)
        
        cuencas = resultados.get("cuencas")
        cauces = resultados.get("cauces")
        
        print("\n📊 RESULTADOS DE LA EVALUACIÓN:")
        print(f"   -> Cuencas detectadas: {len(cuencas)} polígonos.")
        print(f"   -> Tramos de Cauces extraídos: {len(cauces)} segmentos.")
        
        if len(cuencas) > 0 and len(cauces) > 0:
            print("\n✅ ESTABILIDAD CONFIRMADA: El motor RichDEM operó exitosamente en RAM y retornó geometrías.")
        else:
            print("\n⚠️ ALERTA: El motor ejecutó pero no encontró resultados. Revisar parámetros físicos.")
            
    except Exception as e:
        print("\n❌ FALLO CRÍTICO DURANTE LA EJECUCIÓN:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
