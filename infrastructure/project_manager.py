# infrastructure/project_manager.py
import os
import sys
import traceback
import sqlite3
import json
import io
import pandas as pd
import numpy as np
import uuid
import tempfile
import zipfile
import shutil
import atexit
import geopandas as gpd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session
from core.models import Base, Proyecto, SessionState

# --- CODIFICADOR CUSTOM PARA NUMPY/PANDAS E IGNORAR TIPOS NO SERIALIZABLES ---
class NumpyEncoder(json.JSONEncoder):
    """Garantiza que los tipos numéricos de C/Numpy se serialicen a JSON nativo sin romper el script."""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        try:
            return super(NumpyEncoder, self).default(obj)
        except TypeError:
            # Si el objeto no es serializable (ej. threading.Event, function), guardarlo como string para evitar crash
            return f"<NonSerializable: {type(obj).__name__}>"

class GeoProjectManager:
    """
    Adaptador de Persistencia basado en GeoPackage/SQLite.
    Patrón Singleton por Proyecto cargado.
    """
    EXTENSION = ".hds"

    def __init__(self):
        self.db_path = None
        self.engine = None
        self.SessionLocal = None
        self._cache_dir = None
        
        # --- SHUTDOWN HOOK ---
        atexit.register(self.close_project)

    def _enforce_extension(self, path: str) -> str:
        if not path.endswith(self.EXTENSION):
            return path + self.EXTENSION
        return path

    def _get_absolute_path(self, path: str) -> str:
        """DevSecOps: Resuelve rutas seguras para PyInstaller y Runtime normal."""
        try:
            base_path = sys._MEIPASS
        except AttributeError:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, path)

    def inicializar_proyecto(self, ruta_archivo: str, nombre_proyecto: str = "Proyecto Nuevo", force_recreate_if_invalid: bool = False):
        """
        Filosofía EAFP: Intenta crear y conectar. Si falla, propaga el error con auditoría.
        El archivo de salida usará extensión .hidro (o .hds), pero formato .gpkg.
        """
        ruta_archivo = self._enforce_extension(ruta_archivo)
        self.db_path = self._get_absolute_path(ruta_archivo)
        
        try:
            # 0. Mitigación contra Flet Picker y Legacy files
            if os.path.exists(self.db_path):
                # Si es un archivo de 0 bytes creado por Flet, borrarlo inmediatamente
                if os.path.getsize(self.db_path) == 0:
                    os.remove(self.db_path)
                else:
                    # Validar la integridad del SQLite antes de inicializar SQLAlchemy
                    conn = None
                    try:
                        conn = sqlite3.connect(self.db_path)
                        conn.execute("PRAGMA journal_mode;")
                    except sqlite3.DatabaseError as e:
                        if conn:
                            conn.close()
                            conn = None
                        if force_recreate_if_invalid:
                            os.remove(self.db_path)
                        else:
                            raise RuntimeError("❌ Proyecto Incompatible: Este archivo es de una versión anterior de HyDaS (v10 o menor) o el archivo está corrupto. La nueva versión requiere bases de datos GeoPackage.")
                    finally:
                        if conn:
                            conn.close()

            # 1. Bootstrap OGC GeoPackage
            if not os.path.exists(self.db_path):
                gdf_empty = gpd.GeoDataFrame({'geometry': []}, crs="EPSG:32614")
                gdf_empty.to_file(self.db_path, driver="GPKG", layer="init_layer")

            # 2. Configuración SQLAlchemy con mitigación de Thread-Locking
            sqlite_url = f"sqlite:///{self.db_path}"
            self.engine = create_engine(
                sqlite_url, 
                connect_args={"check_same_thread": False}, # Crucial para Flet (UI Thread vs Worker Thread)
                pool_pre_ping=True
            )
            
            # 3. Forzar Modo WAL (Write-Ahead Logging) para concurrencia masiva
            with self.engine.connect() as conn:
                conn.execute(text("PRAGMA journal_mode=WAL;"))
                conn.execute(text("PRAGMA synchronous=NORMAL;"))

            # 4. Construir esquema relacional de HidroSistem
            Base.metadata.create_all(bind=self.engine)
            
            # 5. Fábrica de sesiones
            self.SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=self.engine))
            
            # 6. Registrar Metadatos Iniciales si es nuevo
            db = self.SessionLocal()
            try:
                if db.query(Proyecto).count() == 0:
                    nuevo_proyecto = Proyecto(nombre=nombre_proyecto)
                    db.add(nuevo_proyecto)
                    db.commit()
            except Exception:
                db.rollback()
                raise
            finally:
                db.close()

        except Exception as e:
            print("[CRÍTICO] Fallo al inicializar GeoPackage.", file=sys.stderr)
            traceback.print_exc()
            raise RuntimeError(f"Corrupción en inicialización I/O: {str(e)}")

    def get_session(self):
        """Generador para Inyección de Dependencias. Manejo estricto de memoria."""
        if not self.SessionLocal:
            raise ValueError("Proyecto no inicializado. Llama a inicializar_proyecto primero.")
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()
            import gc
            gc.collect()

    def conectar_proyecto(self, ruta_archivo: str, is_new: bool = False):
        """Conecta a un proyecto existente sin sobreescribir los metadatos base."""
        self.inicializar_proyecto(ruta_archivo, force_recreate_if_invalid=is_new)

    def close_project(self):
        """Cierre seguro del motor. Fuerza la limpieza atómica del -wal y -shm."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
        self.SessionLocal = None
        self.db_path = None
        if self._cache_dir:
            import shutil
            shutil.rmtree(self._cache_dir, ignore_errors=True)
            self._cache_dir = None

    # --- COMPATIBILIDAD CON ARQUITECTURA DE SESIÓN ANTERIOR ---
    
    def save_project(self, path: str, session_data: dict) -> bool:
        """Guarda todo el session_data en la tabla SessionState de SQLite."""
        path = self._enforce_extension(path)
        if not self.engine:
            self.conectar_proyecto(path, is_new=True)
            
        db = self.SessionLocal()
        try:
            # Primero, vaciar el estado de sesión actual para evitar duplicados residuales
            db.query(SessionState).delete()
            
            def process_node(k, node):
                # Caso 1: Backup de Carpeta
                if isinstance(node, dict) and node.get("__type__") == "folder_backup":
                    folder_path = node.get("path")
                    if os.path.exists(folder_path):
                        zip_io = io.BytesIO()
                        with zipfile.ZipFile(zip_io, 'w', zipfile.ZIP_DEFLATED) as zf:
                            for root, _, files in os.walk(folder_path):
                                for f in files:
                                    if f.endswith(".txt"):
                                        zf.write(os.path.join(root, f), arcname=f)
                        zip_io.seek(0)
                        
                        db.add(SessionState(
                            key=k,
                            data_type="folder_backup",
                            blob_data=zip_io.read()
                        ))
                    return

                # Caso 2: DataFrame
                if isinstance(node, (pd.DataFrame, pd.Series)):
                    df_safe = pd.DataFrame(node) if isinstance(node, pd.Series) else node.copy(deep=True)
                    df_safe.columns = df_safe.columns.astype(str)
                    if df_safe.index.dtype == object:
                        df_safe.index = df_safe.index.astype(str)
                    for col in df_safe.columns:
                        if df_safe[col].dtype == object:
                            try:
                                df_safe[col] = pd.to_numeric(df_safe[col])
                            except Exception:
                                df_safe[col] = df_safe[col].astype(str)
                                
                    f_io = io.BytesIO()
                    df_safe.to_parquet(f_io, engine="pyarrow", index=True)
                    f_io.seek(0)
                    
                    db.add(SessionState(
                        key=k,
                        data_type="parquet",
                        blob_data=f_io.read()
                    ))
                    return

                # Caso 3: Punteros de caché a archivos en disco
                if isinstance(node, dict) and node.get("type") in ["df", "b64", "file", "parquet"] and "path" in node:
                    disk_file_path = node["path"]
                    if os.path.exists(disk_file_path):
                        with open(disk_file_path, "rb") as f:
                            db.add(SessionState(
                                key=k,
                                data_type=f"disk_cache_{node['type']}",
                                blob_data=f.read()
                            ))
                    return

                # Caso 4: Jsonizable por defecto
                json_str = json.dumps(node, cls=NumpyEncoder).encode("utf-8")
                db.add(SessionState(
                    key=k,
                    data_type="json",
                    blob_data=json_str
                ))

            # Iterar y guardar la primera capa del diccionario (keys raíz)
            for k, v in session_data.items():
                if v is None: continue
                process_node(str(k), v)
                
            db.commit()
            return True
            
        except Exception as e:
            print(f"Error crítico al guardar en GeoPackage: {e}")
            traceback.print_exc()
            db.rollback()
            return False
        finally:
            db.close()

    def guardar_capa_espacial(self, nombre_capa: str, gdf: gpd.GeoDataFrame):
        """
        [FASE 2] Persiste un GeoDataFrame como capa nativa OGC dentro del GeoPackage (.hds).
        Esto permite su lectura nativa e inmediata en QGIS.
        """
        print(f"DEBUG: Entrando a guardar_capa_espacial para la capa '{nombre_capa}'. db_path={self.db_path}")
        if not self.db_path:
            raise RuntimeError("Proyecto no inicializado. Carga o crea un proyecto primero.")
        try:
            # FIX CRÍTICO DEVSECOPS: En Windows, SQLAlchemy en modo WAL bloquea el archivo SQLite.
            # Cuando Fiona/GDAL intenta escribir en el mismo GeoPackage con modo 'a', 
            # colisionan los locks de C y Python crashea de forma silenciosa (Segmentation Fault).
            # Para evitar esto, liberamos el pool de SQLAlchemy ANTES de escribir.
            if self.engine:
                print("DEBUG: Liberando self.engine (dispose)")
                self.engine.dispose()
                import time
                time.sleep(0.1) # Breve respiro para que el SO suelte los handles del archivo
                
            print(f"DEBUG: Llamando a gdf.to_file driver='GPKG' layer='{nombre_capa}' mode='a'")
            # El driver GPKG de GeoPandas interactúa de forma segura con la DB SQLite subyacente.
            # mode='a' es crucial para añadir tablas sin borrar las relacionales de SQLAlchemy.
            gdf.to_file(self.db_path, driver="GPKG", layer=nombre_capa, mode='a')
            print(f"DEBUG: to_file finalizado exitosamente para '{nombre_capa}'")
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error bloqueante al escribir capa '{nombre_capa}' en GeoPackage: {str(e)}")

    def load_project(self, file_path: str, progress_callback=None, log_callback=None) -> dict:
        """Hidrata el diccionario de sesión desde el SQLite GeoPackage."""
        file_path = self._enforce_extension(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo de proyecto no existe: {file_path}")
            
        self.conectar_proyecto(file_path)
        db = self.SessionLocal()
        recovered_data = {}
        
        try:
            states = db.query(SessionState).all()
            total_files = len(states)
            processed = 0
            
            for state in states:
                processed += 1
                if log_callback: log_callback(f"Descomprimiendo tensor DB: {state.key}")
                if progress_callback and total_files > 0:
                    progress_callback(processed / total_files)
                    
                if state.data_type == "json":
                    recovered_data[state.key] = json.loads(state.blob_data.decode("utf-8"))
                    
                elif state.data_type == "parquet":
                    f_io = io.BytesIO(state.blob_data)
                    recovered_data[state.key] = pd.read_parquet(f_io, engine="pyarrow")
                    
                elif state.data_type == "folder_backup":
                    if not self._cache_dir:
                        self._cache_dir = tempfile.mkdtemp(prefix="hidro_cache_load_")
                        # Usar default argument para fijar el valor de d en el closure
                        atexit.register(lambda d=self._cache_dir: shutil.rmtree(d, ignore_errors=True))
                        
                    zip_io = io.BytesIO(state.blob_data)
                    out_folder = os.path.join(self._cache_dir, f"{state.key}_unzipped")
                    os.makedirs(out_folder, exist_ok=True)
                    with zipfile.ZipFile(zip_io, 'r') as zf:
                        zf.extractall(out_folder)
                    recovered_data[state.key] = {"__type__": "folder_backup", "path": out_folder}
                    
                elif state.data_type.startswith("disk_cache_"):
                    orig_type = state.data_type.replace("disk_cache_", "")
                    if not self._cache_dir:
                        self._cache_dir = tempfile.mkdtemp(prefix="hidro_cache_load_")
                        atexit.register(lambda d=self._cache_dir: shutil.rmtree(d, ignore_errors=True))
                        
                    ext = ".parquet" if orig_type in ["df", "parquet"] else ".txt" if orig_type == "b64" else ".tmp"
                    out_path = os.path.join(self._cache_dir, f"{state.key}{ext}")
                    with open(out_path, "wb") as f:
                        f.write(state.blob_data)
                    recovered_data[state.key] = {"type": orig_type, "path": out_path}

            if progress_callback: progress_callback(1.0)
            return recovered_data
            
        except Exception as e:
            print(f"❌ [CRÍTICO] Error al cargar el proyecto {file_path}:")
            traceback.print_exc()
            raise RuntimeError(f"Falla de descompresión DB: {str(e)}")
        finally:
            db.close()

# Singleton global para su uso en main.py y otras vistas
project_manager_instance = GeoProjectManager()