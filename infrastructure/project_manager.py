import tarfile
import json
import os
import io
import traceback
import numpy as np
import pandas as pd
import uuid
import zipfile
import tempfile
import shutil
import atexit


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

class ProjectManager:
    """
    Motor de Base de Datos y Persistencia para HidroSistem v10.9.
    Utiliza compresión LZMA (.xz) mediante el formato tarfile.
    SEGURO: No usa pickle. Guarda DataFrames en formato Parquet.
    """
    
    EXTENSION = ".hds"

    @staticmethod
    def save_project(file_path: str, session_data: dict) -> bool:
        if not file_path.endswith(ProjectManager.EXTENSION):
            file_path += ProjectManager.EXTENSION
            
        temp_file_path = file_path + ".tmp"

        try:
            with tarfile.open(temp_file_path, "w:xz") as tar:
                
                # Función recursiva para separar Parquet de JSON
                def process_node(node, current_path=""):
                    # --- REGLA ZERO-I/O: EMPAQUETADO ZIP EN MEMORIA HACIA EL .HDS ---
                    if isinstance(node, dict) and node.get("__type__") == "folder_backup":
                        folder_path = node.get("path")
                        if os.path.exists(folder_path):
                            zip_filename = f"{current_path}_{uuid.uuid4().hex[:8]}.zip"
                            
                            zip_io = io.BytesIO()
                            with zipfile.ZipFile(zip_io, 'w', zipfile.ZIP_DEFLATED) as zf:
                                for root, _, files in os.walk(folder_path):
                                    for f in files:
                                        if f.endswith(".txt"): # Blindaje de seguridad
                                            zf.write(os.path.join(root, f), arcname=f)
                            zip_io.seek(0)
                            
                            tarinfo = tarfile.TarInfo(name=zip_filename)
                            tarinfo.size = zip_io.getbuffer().nbytes
                            tar.addfile(tarinfo, zip_io)
                            
                            return {"__type__": "folder_backup", "file": zip_filename}
                        return None
                    # ---------------------------------------------------------
                    
                    if isinstance(node, (pd.DataFrame, pd.Series)):
                        # --- CORRECCIÓN CRÍTICA: Unificación y sanitización PyArrow ---
                        df_safe = pd.DataFrame(node) if isinstance(node, pd.Series) else node.copy(deep=True)
                        df_safe.columns = df_safe.columns.astype(str)
                        
                        # =================================================================
                        # ESCUDO GLOBAL DE SANITIZACIÓN PARA PARQUET (Anti-ArrowTypeError)
                        # =================================================================
                        # 1. Blindar el Índice
                        if df_safe.index.dtype == object:
                            df_safe.index = df_safe.index.astype(str)
                            
                        # 2. Blindar el interior de las columnas
                        for col in df_safe.columns:
                            if df_safe[col].dtype == object:
                                try:
                                    df_safe[col] = pd.to_numeric(df_safe[col])
                                except Exception:
                                    df_safe[col] = df_safe[col].astype(str)
                        # =================================================================
                        
                        filename = f"{current_path}_{uuid.uuid4().hex[:8]}.parquet"
                        
                        # Usamos I/O en RAM pura
                        f_io = io.BytesIO()
                        df_safe.to_parquet(f_io, engine="pyarrow", index=True)
                        f_io.seek(0)
                        
                        tarinfo = tarfile.TarInfo(name=filename)
                        tarinfo.size = f_io.getbuffer().nbytes
                        tar.addfile(tarinfo, f_io)
                        
                        return {"__type__": "parquet", "file": filename}
                        
                    elif isinstance(node, dict):
                        if node.get("type") in ["df", "b64", "file", "parquet"] and "path" in node:
                            disk_file_path = node["path"]
                            if os.path.exists(disk_file_path):
                                # Evaluamos tanto los punteros de Módulo 2/3 ("df") como los del Módulo 5 ("parquet")
                                ext = ".parquet" if node["type"] in ["df", "parquet"] else ".txt" if node["type"] == "b64" else os.path.splitext(disk_file_path)[1]
                                filename = f"{current_path}_{uuid.uuid4().hex[:8]}{ext}"
                                tarinfo = tarfile.TarInfo(name=filename)
                                tarinfo.size = os.path.getsize(disk_file_path)
                                with open(disk_file_path, "rb") as f:
                                    tar.addfile(tarinfo, f)
                                
                                # Convertimos a formato nativo de hds
                                return {"__type__": "disk_cache", "file": filename, "orig_type": node["type"]}
                            else:
                                return None
                        # -------------------------------------------------------------
                        
                        new_dict = {}
                        for k, v in node.items():
                            if v is None:
                                continue
                            new_path = f"{current_path}_{k}" if current_path else str(k)
                            # Limpieza de path
                            new_path = "".join([c if c.isalnum() else "_" for c in new_path])
                            new_dict[k] = process_node(v, new_path)
                        return new_dict
                        
                    elif isinstance(node, list):
                        return [process_node(v, current_path) for v in node]
                        
                    elif isinstance(node, (pd.DataFrame, pd.Series)):
                        df = pd.DataFrame(node) if isinstance(node, pd.Series) else node
                        df.columns = df.columns.astype(str)
                        
                        filename = f"{current_path}_{uuid.uuid4().hex[:8]}.parquet"
                        
                        f_io = io.BytesIO()
                        df.to_parquet(f_io, engine="pyarrow", index=True)
                        f_io.seek(0)
                        
                        tarinfo = tarfile.TarInfo(name=filename)
                        tarinfo.size = f_io.getbuffer().nbytes
                        tar.addfile(tarinfo, f_io)
                        
                        return {"__type__": "parquet", "file": filename}
                    else:
                        return node

                # Procesar todo el session_data recursivamente
                safe_session_data = process_node(session_data)
                
                # Guardar la estructura de árbol como JSON maestro
                json_str = json.dumps(safe_session_data, cls=NumpyEncoder, indent=4).encode("utf-8")
                tarinfo_m = tarfile.TarInfo(name="session_tree.json")
                tarinfo_m.size = len(json_str)
                tar.addfile(tarinfo_m, io.BytesIO(json_str))
                
            # --- VERIFICACIÓN DE INTEGRIDAD LZMA (DOBLE VALIDACIÓN) ---
            # Abrimos el archivo .tmp recién creado en modo lectura estricta
            # para comprobar que la estructura tar no esté dañada y evitar
            # sobreescribir el proyecto original con datos corruptos.
            try:
                with tarfile.open(temp_file_path, "r:xz") as check_tar:
                    members = check_tar.getmembers()
                    if not members:
                        raise ValueError("El archivo .tmp generado está vacío estructuralmente.")
            except Exception as check_ex:
                print(f"Error Crítico de Integridad: El archivo temporal generado falló la validación LZMA: {check_ex}")
                if os.path.exists(temp_file_path):
                    try: os.remove(temp_file_path)
                    except: pass
                return False

            # --- CORRECCIÓN CRÍTICA: Transacción Atómica (Zero-Trust I/O) ---
            # os.replace delega la sobrescritura al Kernel del Sistema Operativo.
            # Es a prueba de fallos eléctricos: si falla, el archivo original intacto permanece.
            os.replace(temp_file_path, file_path)
            return True
            
            
        except Exception as e:
            print(f"Error crítico al guardar: {e}")
            traceback.print_exc()
            if os.path.exists(temp_file_path):
                try: os.remove(temp_file_path)
                except: pass
            return False

    @staticmethod
    def load_project(file_path: str, progress_callback=None, log_callback=None) -> dict:
        """
        Lee un archivo .hds comprimido y devuelve un diccionario con los datos hidratados.
        Emite eventos de progreso para la UI.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo de proyecto no existe: {file_path}")

        try:
            with tarfile.open(file_path, "r:xz") as tar:
                if log_callback: log_callback("Extrayendo árbol de sesión maestro...")
                tree_file = tar.extractfile("session_tree.json")
                
                if tree_file is None:
                    manifest_file = tar.extractfile("manifest.json")
                    if manifest_file:
                        raise RuntimeError("Formato obsoleto. HidroSistem v10.9 no soporta proyectos antiguos por seguridad.")
                    else:
                        raise ValueError("Archivo corrupto: No se encontró session_tree.json")
                    
                safe_session_data = json.loads(tree_file.read().decode("utf-8"))

                # --- TELEMETRÍA: Contar archivos totales para el porcentaje ---
                def count_files(node):
                    c = 0
                    if isinstance(node, dict):
                        if node.get("__type__") in ["disk_cache", "parquet", "folder_backup"] and "file" in node:
                            return 1
                        for v in node.values(): c += count_files(v)
                    elif isinstance(node, list):
                        for v in node: c += count_files(v)
                    return c
                
                total_files = count_files(safe_session_data)
                processed = [0]

                # Función recursiva para rehidratar
                def hydrate_node(node):
                    if isinstance(node, dict):
                        # Detectamos si es un archivo para actualizar la terminal y la barra
                        is_file_node = node.get("__type__") in ["disk_cache", "parquet", "folder_backup"] and "file" in node
                        if is_file_node:
                            filename = node["file"]
                            processed[0] += 1
                            if log_callback: log_callback(f"Descomprimiendo tensor: {filename}")
                            if progress_callback and total_files > 0:
                                progress_callback(processed[0] / total_files)

                        if node.get("__type__") == "disk_cache" and "file" in node:
                            filename = node["file"]
                            extracted = tar.extractfile(filename)
                            if extracted:
                                global_cache = getattr(hydrate_node, 'cache_dir', None)
                                if not global_cache:
                                    global_cache = tempfile.mkdtemp(prefix="hidro_cache_load_")
                                    atexit.register(lambda: shutil.rmtree(global_cache, ignore_errors=True))
                                    hydrate_node.cache_dir = global_cache
                                
                                out_path = os.path.join(global_cache, filename)
                                with open(out_path, "wb") as f: f.write(extracted.read())
                                return {"type": node["orig_type"], "path": out_path}
                            return None
                        
                        elif node.get("__type__") == "parquet" and "file" in node:
                            filename = node["file"]
                            extracted = tar.extractfile(filename)
                            if extracted is None: return None
                            return pd.read_parquet(extracted, engine="pyarrow")
                        
                        elif node.get("__type__") == "folder_backup" and "file" in node:
                            filename = node["file"]
                            extracted_zip = tar.extractfile(filename)
                            if extracted_zip:
                                temp_dir = tempfile.mkdtemp(prefix="HidroSistem_Txt_")
                                atexit.register(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
                                with zipfile.ZipFile(extracted_zip, 'r') as zf:
                                    zf.extractall(temp_dir)
                                return {"__type__": "folder_backup", "path": temp_dir}
                            return None
                        else:
                            return {k: hydrate_node(v) for k, v in node.items()}
                    elif isinstance(node, list):
                        return [hydrate_node(v) for v in node]
                    else:
                        return node

                recovered_data = hydrate_node(safe_session_data)
                if progress_callback: progress_callback(1.0)
            return recovered_data

        except Exception as e:
            print(f"❌ [CRÍTICO] Error al cargar el proyecto {file_path}:")
            traceback.print_exc()
            raise RuntimeError(f"Falla de descompresión LZMA: {str(e)}")