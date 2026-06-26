import os
import sys
import pandas as pd
import numpy as np
import tempfile
import unittest

# Añadir el path al PYTHONPATH para simular la estructura
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from core import imputacion_logic

class TestImputacionHeadless(unittest.TestCase):

    def setUp(self):
        # Crear un directorio temporal para las estaciones falsas
        self.temp_dir = tempfile.TemporaryDirectory()
        self.station_files = {}

        fechas = pd.date_range("2020-01-01", "2020-01-31", freq='D')
        
        # 1. Estación Objetivo (Con huecos)
        df_target = pd.DataFrame({'FECHA': fechas, 'PRECIP': np.random.uniform(0, 10, len(fechas))})
        # Forzar NAs
        df_target.loc[5:10, 'PRECIP'] = np.nan 
        
        target_path = os.path.join(self.temp_dir.name, "TARGET_999.txt")
        # El parser busca líneas con "FECHA,PRECIP", así que simulamos ese formato sucio
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write("ESTACION: 999\nFECHA,PRECIP,EVAP\n")
            for _, r in df_target.iterrows():
                val = f"{r['PRECIP']:.1f}" if pd.notna(r['PRECIP']) else "Nulo"
                f.write(f"{r['FECHA'].strftime('%Y-%m-%d')},{val},0\n")

        self.station_files['999'] = {'path': target_path, 'lat': 20.0, 'lon': -100.0}

        # 2. Estaciones Vecinas (Datos completos)
        for i in range(1, 6):
            df_nb = pd.DataFrame({'FECHA': fechas, 'PRECIP': np.random.uniform(0, 10, len(fechas))})
            nb_path = os.path.join(self.temp_dir.name, f"NB_{i}.txt")
            with open(nb_path, 'w', encoding='utf-8') as f:
                f.write(f"ESTACION: {i}\nFECHA,PRECIP,EVAP\n")
                for _, r in df_nb.iterrows():
                    f.write(f"{r['FECHA'].strftime('%Y-%m-%d')},{r['PRECIP']:.1f},0\n")
            
            # Ubicación cercana
            self.station_files[str(i)] = {'path': nb_path, 'lat': 20.01 + (i*0.01), 'lon': -100.01 + (i*0.01)}

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_imputation_survival(self):
        print("\n--- INICIANDO TEST HEADLESS (SIN FLET) ---")
        
        # 1. Precalcular rango (Orquestador UI simulado)
        global_range = imputacion_logic.obtener_rango_global_fechas(self.station_files)
        self.assertIsNotNone(global_range, "Fallo al calcular el rango global")

        # 2. Callbacks Mocks (Simulando hilo seguro de Flet)
        def mock_progress(pct, msg):
            p = f"[{int(pct*100)}%]" if pct else "[---]"
            m = msg if msg else ""
            print(f"UI PROGRESS: {p} {m}".strip())

        def mock_log(msg):
            print(f"UI LOG: {msg}")

        # 3. Disparar el motor matemático sin GIL lock
        print("Llamando a impute_target_station...")
        df_res, logs = imputacion_logic.impute_target_station(
            target_id='999',
            station_files=self.station_files,
            radius_km=150,
            global_range=global_range,
            progress_callback=mock_progress,
            log_callback=mock_log
        )

        print("\n--- RESULTADO OBTENIDO ---")
        print(f"Total filas: {len(df_res)}")
        print(f"NAs restantes: {df_res['PRECIP'].isna().sum()}")
        print("Muestra del DF resultante:")
        print(df_res.head(15).to_string())

        self.assertIsNotNone(df_res, "El motor devolvió None")
        self.assertEqual(df_res['PRECIP'].isna().sum(), 0, "No se llenaron todos los huecos")

if __name__ == '__main__':
    unittest.main()
