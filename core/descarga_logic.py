import sys
import os
import time
import random
import pandas as pd
import requests
import urllib3
import re
import tarfile
import io
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import shapefile 
from shapely.geometry import shape, Point 

# --- 1. RESOLUCIÓN ABSOLUTA DE RUTAS (DevSecOps & PyInstaller) ---
# Si se ejecuta como .exe (PyInstaller), sys.frozen es True y los assets están en sys._MEIPASS
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    # Como este archivo está en core/, subimos un nivel para llegar a la raíz del proyecto
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# --- 2. CATÁLOGOS Y AGRUPACIONES ---
CATALOGO_ESTADOS_CONAGUA = {
    "Aguascalientes": "ags", "Baja California": "bc", "Baja California Sur": "bcs",
    "Campeche": "camp", "Coahuila": "coah", "Colima": "col", "Chiapas": "chis",
    "Chihuahua": "chih", "Ciudad de México": "df", "Durango": "dgo",
    "Guanajuato": "gto", "Guerrero": "gro", "Hidalgo": "hgo", "Jalisco": "jal",
    "Estado de México": "mex", "Michoacán": "mich", "Morelos": "mor",
    "Nayarit": "nay", "Nuevo León": "nl", "Oaxaca": "oax", "Puebla": "pue",
    "Querétaro": "qro", "Quintana Roo": "qroo", "San Luis Potosí": "slp",
    "Sinaloa": "sin", "Sonora": "son", "Tabasco": "tab", "Tamaulipas": "tamps",
    "Tlaxcala": "tlax", "Veracruz": "ver", "Yucatán": "yuc", "Zacatecas": "zac"
}

MAPA_IDS_CONAGUA = {
    "ags": 1, "bc": 2, "bcs": 3, "camp": 4, "coah": 5, "col": 6, "chis": 7,
    "chih": 8, "df": 9, "dgo": 10, "gto": 11, "gro": 12, "hgo": 13, "jal": 14,
    "mex": 15, "mich": 16, "mor": 17, "nay": 18, "nl": 19, "oax": 20, "pue": 21,
    "qro": 22, "qroo": 23, "slp": 24, "sin": 25, "son": 26, "tab": 27, "tamps": 28,
    "tlax": 29, "ver": 30, "yuc": 31, "zac": 32
}

# --- 3. VARIABLES GEOESPACIALES GLOBALES ---
BBOX_MEXICO = {"min_x": -118.5, "max_x": -86.5, "min_y": 14.5, "max_y": 33.0}

# Candado para evitar corrupción al escribir en el archivo comprimido .tar.xz
escritura_lock = threading.Lock()
# Señal de interrupción de emergencia
señal_abortar = threading.Event()

# --- 3. MOTOR ESPACIAL (Pyshp y Shapely) ---
def cargar_poligonos(modo):
    """Lee los Shapefiles desde la carpeta assets según el modo seleccionado."""
    if modo == "RESPALDO MASIVO":
        return []
        
    nombre_archivo = "estados.shp" if modo == "POR ESTADO" else "cuencas.shp"
    archivo_shp = os.path.join(BASE_DIR, "assets", nombre_archivo)
    
    if not os.path.exists(archivo_shp):
        raise FileNotFoundError(f"No se encontró el archivo. Ruta buscada: {archivo_shp}")
        
    sf = shapefile.Reader(archivo_shp)
    poligonos = []
    
    for sr in sf.shapeRecords():
        atr = sr.record.as_dict()
        nombre = atr.get('ESTADO') or atr.get('NOM_CUENCA') or "Desconocido"
        id_pol = atr.get('ID_ESTADO') or atr.get('ID_CUENCA') or "0"
        
        poligonos.append({
            "id": str(id_pol),
            "nombre": nombre,
            "geometria": sr.shape.__geo_interface__, 
            "shapely_obj": shape(sr.shape.__geo_interface__) 
        })
        
    return poligonos

def detectar_clic_poligono(poligonos, lon, lat):
    """Verifica si la coordenada cae dentro de algún polígono con Shapely."""
    punto_clic = Point(lon, lat)
    for pol in poligonos:
        if pol["shapely_obj"].contains(punto_clic):
            return pol
    return None

# --- 4. MOTOR DE WEB SCRAPING Y ETL ---
def obtener_archivos_directorio(url_estado):
    """Hace scraping del índice web de CONAGUA para obtener los nombres de los .txt."""
    try:
        response = requests.get(url_estado, timeout=30, verify=False) 
        response.raise_for_status()
        archivos = re.findall(r'href="(\d+\.txt)"', response.text)
        return list(set(archivos)) 
    except Exception as e:
        return []

def extraer_metadata(texto_crudo):
    """Extrae coordenadas y datos base usando búsqueda insensible a mayúsculas."""
    metadata = {"lat": None, "lon": None, "estado": "Desconocido", "nombre": "Desconocido"}
    
    # Escaneamos un poco más de líneas (20) por si acaso
    cabecera = "\n".join(texto_crudo.splitlines()[:20])
    
    # CORRECCIÓN: re.IGNORECASE permite encontrar 'Latitud', 'LATITUD' o 'latitud'
    match_lat = re.search(r'LATITUD\s*:\s*(-?\d+\.?\d*)', cabecera, re.IGNORECASE)
    match_lon = re.search(r'LONGITUD\s*:\s*(-?\d+\.?\d*)', cabecera, re.IGNORECASE)
    match_est = re.search(r'ESTADO\s*:\s*(.*)', cabecera, re.IGNORECASE)
    match_nom = re.search(r'NOMBRE\s*:\s*(.*)', cabecera, re.IGNORECASE)
    
    if match_lat: metadata["lat"] = float(match_lat.group(1))
    if match_lon: metadata["lon"] = float(match_lon.group(1))
    if match_est: metadata["estado"] = match_est.group(1).strip().replace('°', '')
    if match_nom: metadata["nombre"] = match_nom.group(1).strip()
    
    return metadata

def descargar_y_procesar_estacion(url_archivo, clave_estado, poligonos_cuencas):
    """Hilo individual de descarga, limpieza y geocruce."""
    if señal_abortar.is_set():
        return False, None, None
        
    # Agregamos reintentos para evitar pérdida de datos por rate limit de CONAGUA
    for intento in range(4):
        try:
            # REGLA ZERO-TRUST: Simular latencia humana y usar Headers legítimos
            time.sleep(random.uniform(0.6, 1.5))
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) HidroSistem/10.5'}
            
            r = requests.get(url_archivo, headers=headers, timeout=30, verify=False)
            
            # Filtro 404 (Si no existe, ignorar silenciosamente y salir rápido)
            if r.status_code == 404:
                return False, None, None
            
            # Si hay error temporal (500, 502, 503, 504), forzar reintento
            if r.status_code >= 500:
                time.sleep(1 + intento)
                continue
                
            r.raise_for_status()
            texto = r.text
            
            # Filtro de archivos basura
            if len(texto) < 500 or "<html" in texto.lower():
                return False, None, None
                
            nombre_archivo = url_archivo.split('/')[-1]
            
            # CORRECCIÓN: Limpiamos el prefijo 'dia' para que nuestra BD tenga el ID puro
            clave_estacion = nombre_archivo.replace('.txt', '').replace('dia', '')
            
            meta = extraer_metadata(texto)
            meta["clave"] = clave_estacion
            meta["estado_origen"] = clave_estado.upper()
            meta["cuenca_id"] = "No Asignada"
            meta["cuenca_nombre"] = "No Asignada"
            
            # Geocruce
            if meta["lat"] and meta["lon"] and poligonos_cuencas:
                lon_real = meta["lon"] if meta["lon"] < 0 else -meta["lon"]
                pol_detectado = detectar_clic_poligono(poligonos_cuencas, lon_real, meta["lat"])
                if pol_detectado:
                    meta["cuenca_id"] = pol_detectado["id"]
                    meta["cuenca_nombre"] = pol_detectado["nombre"]
            
            contenido_bytes = texto.encode('utf-8')
            return True, meta, contenido_bytes
        except requests.exceptions.RequestException:
            # Errores de Timeout o conexión se reintentan
            time.sleep(1 + intento * 1.5)
            
    # Si falla 4 veces, lo ignoramos para no frenar el enjambre
    return False, None, None

# --- 5. ORQUESTADOR PRINCIPAL ---
def procesar_descarga(modo, elementos, ruta_base, callback_log, callback_progreso, callback_mapa=None, carpeta_previa=None):
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    señal_abortar.clear() 
    
    # CORRECCIÓN DE ERROR CRÍTICO: Definimos las rutas aquí afuera para que todos los modos las vean
    ruta_tar = os.path.join(ruta_base, "Tlaloc_BD_Nacional_Comprimida.tar.xz")
    ruta_csv = os.path.join(BASE_DIR, "assets", "catalogo_tlaloc.csv")

    try:
        if modo == "RESPALDO MASIVO":
            callback_log(f"> ---------------------------------------")
            callback_log(f"> INICIANDO PROTOCOLO DE RESPALDO MASIVO (LZMA)")
            
            os.makedirs(ruta_base, exist_ok=True)
            
            # Cargar Cuencas para el Cruce
            poligonos_cuencas = []
            try: 
                poligonos_cuencas = cargar_poligonos("POR CUENCA")
                callback_log(f"> Capa de Cuencas cargada para cruce espacial.")
            except: 
                callback_log(f"> [AVISO] No se pudo cargar cuencas.shp. El catálogo no tendrá cuencas.")
            
            todas_las_urls = []
            callback_log(f"> 🔍 Cargando Lista estaciones publicas de CONAGUA...")
            
            # Ubicar el archivo maestro de CONAGUA en assets/
            ruta_txt_estaciones = os.path.join(BASE_DIR, "assets", "Claves-Estaciones-CONAGUA.txt")
            if not os.path.exists(ruta_txt_estaciones):
                callback_log(f"> [CRÍTICO] No se encontró el catálogo {ruta_txt_estaciones}.")
                return None
                
            try:
                # REGLA ZERO-TRUST: Tolerancia a codificaciones (UTF-8 vs Windows-1252/ANSI)
                # Aplicamos EAFP para esquivar crasheos por acentos (ej. la "ó" en 0xf3)
                try:
                    df_estaciones = pd.read_csv(ruta_txt_estaciones, sep='\t', encoding='utf-8')
                except UnicodeDecodeError:
                    df_estaciones = pd.read_csv(ruta_txt_estaciones, sep='\t', encoding='latin1')
                
                # Normalizamos nombres de columnas por si el TXT tiene espacios ocultos
                df_estaciones.columns = [str(c).strip() for c in df_estaciones.columns]
                
                for _, row in df_estaciones.iterrows():
                    clave_estado = str(row['Clave estado']).strip().lower()
                    clave_estacion = str(row['Clave']).strip()
                    
                    if clave_estado in CATALOGO_ESTADOS_CONAGUA.values():
                        url = f"https://smn.conagua.gob.mx/tools/RESOURCES/Normales_Climatologicas/Diarios/{clave_estado}/dia{clave_estacion}.txt"
                        todas_las_urls.append((url, clave_estado))
                        
            except Exception as e:
                callback_log(f"> [ERROR CRÍTICO] Fallo al parsear la Lista de estaciones CONAGUA: {e}")
                return None
            
            total_urls = len(todas_las_urls)
            callback_log(f"> 📍 Lista validada de estaciones publicas de CONAGUA: {total_urls} estaciones a descargar.")
            
            # Calcular el total real por estado para la barra de progreso
            progreso_estados = {}
            for clave in CATALOGO_ESTADOS_CONAGUA.values():
                total_est = sum(1 for u, c in todas_las_urls if c == clave)
                progreso_estados[clave.upper()] = {"procesados": 0, "total": total_est if total_est > 0 else 1}
            
            estaciones_procesadas = 0
            exitosos = 0
            registros_catalogo = []
            
            with tarfile.open(ruta_tar, "w:xz") as tar:
                # REGLA DE SCRAPING ÉTICO: Reducimos concurrencia para no saturar al servidor de CONAGUA
                with ThreadPoolExecutor(max_workers=15) as executor:
                    futuros_map = {executor.submit(descargar_y_procesar_estacion, url, clave, poligonos_cuencas): clave for url, clave in todas_las_urls}
                    
                    for futuro in as_completed(futuros_map):
                        if señal_abortar.is_set():
                            callback_log("> [🛑] ABORTANDO EJECUCIÓN...")
                            break 
                        
                        clave_estado = futuros_map[futuro]
                        estaciones_procesadas += 1
                        progreso_estados[clave_estado.upper()]["procesados"] += 1
                        
                        exito, meta, datos_bytes = futuro.result()
                        
                        if exito:
                            exitosos += 1
                            registros_catalogo.append(meta)
                            with escritura_lock:
                                info = tarfile.TarInfo(name=f"{meta['estado_origen']}/{meta['clave']}.txt")
                                info.size = len(datos_bytes)
                                tar.addfile(info, io.BytesIO(datos_bytes))
                        
                        if estaciones_procesadas % 150 == 0 or estaciones_procesadas == total_urls:
                            callback_progreso(estaciones_procesadas / total_urls)
                            if callback_mapa:
                                porcentajes = {k: min(v["procesados"]/v["total"], 1.0) for k, v in progreso_estados.items()}
                                callback_mapa(porcentajes)
            
            if not señal_abortar.is_set():
                callback_log(f"> Generando catálogo relacional...")
                df_catalogo = pd.DataFrame(registros_catalogo)
                df_catalogo.to_csv(ruta_csv, index=False, encoding='utf-8')
                callback_log(f"> ✅ RESPALDO COMPLETADO. Estaciones: {exitosos}")
                
            return ruta_base

        # =======================================================
        # MOTOR DE CONSULTA LOCAL (POR ESTADO Y POR CUENCA)
        # =======================================================
        elif modo in ["POR ESTADO", "POR CUENCA"]:
            callback_log(f"> ---------------------------------------")
            callback_log(f"> INICIANDO MOTOR DE EXTRACCIÓN LOCAL ({modo})")
            
            if not os.path.exists(ruta_tar) or not os.path.exists(ruta_csv):
                callback_log("> [ERROR] Falta Base de Datos Local (.tar.xz o .csv).")
                return None
                
            df_catalogo = pd.read_csv(ruta_csv)
            
            # --- 1. Filtrado Inteligente y Resumen ---
            if modo == "POR ESTADO":
                # Limpiador de texto para evitar que falten estados por culpa de los acentos del Shapefile
                def limpiar(t): 
                    return str(t).lower().replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u').strip()
                
                claves_seleccionadas = []
                for e in elementos:
                    # Buscamos la clave ignorando mayúsculas y acentos
                    match = next((v for k, v in CATALOGO_ESTADOS_CONAGUA.items() if limpiar(k) == limpiar(e)), None)
                    if match:
                        claves_seleccionadas.append(match.upper())
                    else:
                        callback_log(f"> [AVISO] El nombre '{e}' del mapa no coincide con el catálogo.")
                
                df_filtro = df_catalogo[df_catalogo['estado_origen'].isin(claves_seleccionadas)]
                
                # Imprimir el Desglose en la Consola
                callback_log("> --- RESUMEN DE SELECCIÓN ---")
                conteo = df_filtro['estado_origen'].value_counts()
                for est_clave, cant in conteo.items():
                    # Buscar el nombre original para imprimirlo
                    nom_est = next((k for k, v in CATALOGO_ESTADOS_CONAGUA.items() if v.upper() == est_clave), est_clave)
                    callback_log(f"> 📍 {nom_est}: {cant} estaciones localizadas")
                    
            else: # POR CUENCA
                df_filtro = df_catalogo[df_catalogo['cuenca_nombre'].isin(elementos)]
                
                # Imprimir el Desglose en la Consola
                callback_log("> --- RESUMEN DE SELECCIÓN ---")
                conteo = df_filtro['cuenca_nombre'].value_counts()
                for cue_nom, cant in conteo.items():
                    callback_log(f"> 🌊 Cuenca [{cue_nom}]: {cant} estaciones localizadas")
                
            total_archivos = len(df_filtro)
            if total_archivos == 0:
                callback_log("> [ERROR] No se encontraron estaciones en la base de datos para la zona seleccionada.")
                return None
                
            callback_log(f"> Total a extraer de la BD: {total_archivos} archivos.")
            
            # --- 2. Preparar Entorno Seguro ---
            carpeta_salida = os.path.join(ruta_base, "Tlaloc_Extraccion_Activa")
            import shutil
            
            # REGLA DE PERSISTENCIA: Se elimina rmtree para permitir crecimiento acumulativo en disco duro
            os.makedirs(carpeta_salida, exist_ok=True)
            
            # --- PROTECCIÓN VFS: Recuperar y fusionar estaciones históricas de la sesión ---
            if carpeta_previa and os.path.exists(carpeta_previa):
                # Evitamos bucles redundantes si las rutas físicas llegan a coincidir
                if os.path.abspath(carpeta_previa) != os.path.abspath(carpeta_salida):
                    callback_log("> 📚 Sincronizando historial pluvial de la sesión actual...")
                    try:
                        for f in os.listdir(carpeta_previa):
                            if f.endswith(".txt"):
                                shutil.copy(os.path.join(carpeta_previa, f), os.path.join(carpeta_salida, f))
                    except Exception as ex:
                        # Estilo EAFP: Continuamos si algún archivo está bloqueado para no congelar la app
                        callback_log(f"> [AVISO] Ocurrió un conflicto menor al fusionar archivos previos: {ex}")
            
            # --- 3. Generar la lista de archivos (BÚSQUEDA POR RAÍZ) ---
            # Como los nombres (ej. 05001.txt) son únicos a nivel nacional, 
            # buscaremos solo por nombre de archivo ignorando las carpetas.
            nombres_a_extraer = [f"{row['clave']}.txt" for _, row in df_filtro.iterrows()]
            
            extraidos = 0
            faltantes = 0
            
            # --- 4. Extracción del Comprimido ---
            with tarfile.open(ruta_tar, "r:xz") as tar:
                # Mapeamos los archivos del TAR ignorando en qué carpeta están
                miembros_tar = {os.path.basename(m.name): m for m in tar.getmembers() if m.isfile()}
                
                for i, nombre_archivo in enumerate(nombres_a_extraer):
                    if señal_abortar.is_set():
                        callback_log("> [🛑] ABORTANDO EXTRACCIÓN...")
                        break
                        
                    if nombre_archivo in miembros_tar:
                        # Extraemos el archivo directamente a la RAM
                        f_in = tar.extractfile(miembros_tar[nombre_archivo])
                        if f_in:
                            ruta_destino_txt = os.path.join(carpeta_salida, nombre_archivo)
                            with open(ruta_destino_txt, 'wb') as f_out:
                                f_out.write(f_in.read())
                            extraidos += 1
                    else:
                        faltantes += 1 # Si por alguna razón no está en el zip, lo contamos
                    
                    # Actualizar barra de progreso
                    if i % 20 == 0 or i == total_archivos - 1:
                        callback_progreso((i + 1) / total_archivos)
            
            callback_log(f"> ---------------------------------------")
            callback_log(f"> ✅ EXTRACCIÓN COMPLETADA.")
            callback_log(f"> Archivos desempacados y listos: {extraidos}")
            
            if faltantes > 0:
                callback_log(f"> [AVISO] {faltantes} estaciones de la BD no estaban en el .tar.xz (Posible descarga parcial)")
            
            # Retornamos esta carpeta "plana" para el Módulo de Imputación
            return carpeta_salida
            
        return None
        
    except Exception as e:
        callback_log(f"> [ERROR CRÍTICO]: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
# =======================================================
# 6. MÓDULO DE INSPECCIÓN (SONDA EN VIVO)
# =======================================================

def obtener_catalogo_visor(modo, elementos, ruta_csv):
    """Filtra la BD local y devuelve una lista de estaciones para el visor."""
    if not os.path.exists(ruta_csv) or not elementos: 
        return []
        
    df = pd.read_csv(ruta_csv)
    
    if modo == "POR ESTADO":
        def limpiar(t): return str(t).lower().replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u').strip()
        claves = []
        for e in elementos:
            match = next((v for k, v in CATALOGO_ESTADOS_CONAGUA.items() if limpiar(k) == limpiar(e)), None)
            if match: claves.append(match.upper())
        df_filtro = df[df['estado_origen'].isin(claves)]
    else:
        df_filtro = df[df['cuenca_nombre'].isin(elementos)]
        
    # Devolvemos una lista de diccionarios con lo básico
    return df_filtro[['nombre', 'clave', 'estado_origen']].to_dict('records')

def inspeccionar_estacion_aislada(clave, estado_origen, ruta_tar):
    """Lee 5 líneas del disco local y 5 líneas de CONAGUA y las compara."""
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # 1. LECTURA LOCAL (Disco Duro)
    local_lines = []
    try:
        with tarfile.open(ruta_tar, "r:xz") as tar:
            nombre_archivo = f"{clave}.txt"
            miembro = next((m for m in tar.getmembers() if os.path.basename(m.name) == nombre_archivo), None)
            if miembro:
                f = tar.extractfile(miembro)
                if f:
                    # Leemos y nos quedamos solo con las últimas 5 líneas que no estén vacías
                    lineas = [l.decode('utf-8').strip() for l in f.readlines() if l.strip()]
                    local_lines = lineas[-5:] if lineas else ["Archivo vacío."]
            else:
                local_lines = ["No encontrada en BD Local."]
    except Exception as e:
        local_lines = [f"Error al leer local: {e}"]

    # 2. LECTURA SERVIDOR (Scraping Vivo)
    server_lines = []
    try:
        url = f"https://smn.conagua.gob.mx/tools/RESOURCES/Normales_Climatologicas/Diarios/{estado_origen.lower()}/dia{clave}.txt"
        r = requests.get(url, timeout=5, verify=False)
        
        if r.status_code == 200:
            lineas_srv = [l.strip() for l in r.text.splitlines() if l.strip()]
            server_lines = lineas_srv[-5:] if lineas_srv else ["Archivo vacío en servidor."]
        else:
            server_lines = [f"Error 404: No existe en servidor."]
    except Exception as e:
        server_lines = [f"Error de conexión: {e}"]

    return local_lines, server_lines

# =======================================================
# 7. MÓDULO DE AUDITORÍA PROFUNDA Y REPORTE (DEEP SCAN)
# =======================================================

def auditar_base_datos_profunda(ruta_base, ruta_salida, callback_log, callback_progreso):
    """
    Escaneo físico riguroso (Zero-Trust) del archivo .tar.xz.
    Genera gráficas y compila un reporte .tex automatizado.
    """
    import datetime, re, io
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    
    ruta_tar = os.path.join(ruta_base, "Tlaloc_BD_Nacional_Comprimida.tar.xz")
    if not os.path.exists(ruta_tar):
        return False, "Error: No se encontró el archivo de base de datos comprimida (.tar.xz)."

    # Estructura del Data Warehouse temporal
    stats = {clave.upper(): {"sanas": 0, "corruptas": 0, "años_inicio": [], "años_fin": [], "nom": nom} 
             for nom, clave in CATALOGO_ESTADOS_CONAGUA.items()}
    
    nacional_sanas = 0
    nacional_corruptas = 0

    callback_log("> 🔎 INICIANDO MOTOR DE ESCANEO DE BAJO NIVEL (I/O)...")
    señal_abortar.clear()

    try:
        # FASE 1: EXTRACCIÓN Y CONTEO FÍSICO
        with tarfile.open(ruta_tar, "r:xz") as tar:
            miembros = tar.getmembers()
            total_archivos = len(miembros)
            
            for i, miembro in enumerate(miembros):
                if señal_abortar.is_set():
                    return False, "🛑 AUDITORÍA ABORTADA POR EL USUARIO."
                
                # Actualizar UI cada 50 archivos para no asfixiar el GIL de Flet
                if i % 50 == 0:
                    callback_progreso(i / total_archivos)
                    
                if not miembro.isfile() or not miembro.name.endswith(".txt"):
                    continue
                    
                partes = miembro.name.split('/')
                if len(partes) != 2: continue
                estado_clave = partes[0].upper()
                
                if estado_clave not in stats: continue
                
                # REGLA ZERO-TRUST: Si el archivo pesa menos de 300 bytes, es inservible
                if miembro.size < 300:
                    stats[estado_clave]["corruptas"] += 1
                    nacional_corruptas += 1
                    continue
                    
                # Leer en crudo
                f = tar.extractfile(miembro)
                if not f:
                    stats[estado_clave]["corruptas"] += 1
                    nacional_corruptas += 1
                    continue
                
                try:
                    # Aplicamos EAFP para esquivar caracteres extraños en los metadatos de CONAGUA
                    contenido = f.read().decode('utf-8', errors='ignore').splitlines()
                    
                    # Búsqueda rápida de líneas de datos válidas mediante Regex YYYY-MM-DD
                    lineas_validas = [l for l in contenido if re.match(r'^\d{4}-\d{2}-\d{2}', l)]
                    
                    if not lineas_validas or len(lineas_validas) < 30:
                        stats[estado_clave]["corruptas"] += 1
                        nacional_corruptas += 1
                        continue
                        
                    # Extracción de línea de tiempo
                    fecha_inicio = lineas_validas[0].split()[0]
                    fecha_fin = lineas_validas[-1].split()[0]
                    
                    stats[estado_clave]["años_inicio"].append(int(fecha_inicio.split('-')[0]))
                    stats[estado_clave]["años_fin"].append(int(fecha_fin.split('-')[0]))
                    
                    stats[estado_clave]["sanas"] += 1
                    nacional_sanas += 1
                    
                except Exception:
                    stats[estado_clave]["corruptas"] += 1
                    nacional_corruptas += 1

        callback_progreso(1.0)
        callback_log(f"> ✅ Escaneo finalizado. Sanas: {nacional_sanas} | Corruptas: {nacional_corruptas}")
        
        # FASE 2: GENERACIÓN DEL MOTOR GRÁFICO (Matplotlib Agg)
        callback_log("> 📊 Generando gráficos estadísticos de alta resolución...")
        dir_graficos = os.path.join(ruta_salida, "Graficos_Auditoria")
        os.makedirs(dir_graficos, exist_ok=True)
        
        # 2.1 Gráfico de Pastel: Integridad Nacional
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        ax1.pie([nacional_sanas, nacional_corruptas], labels=['Sanas/Activas', 'Corruptas/Vacías'], 
                autopct='%1.1f%%', colors=['#00a82d', '#cc0000'], startangle=90, textprops={'color':"black"})
        ax1.set_title("Integridad de la Base de Datos Nacional", fontweight="bold")
        ruta_pie = os.path.join(dir_graficos, "integridad_nacional.png")
        fig1.savefig(ruta_pie, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig1)
        
        # 2.2 Gráfico de Barras: Densidad por Estado
        estados_ordenados = sorted(stats.items(), key=lambda x: x[1]["sanas"], reverse=True)
        nombres = [v["nom"][:12] for k, v in estados_ordenados if (v["sanas"] > 0 or v["corruptas"] > 0)]
        sanas = [v["sanas"] for k, v in estados_ordenados if (v["sanas"] > 0 or v["corruptas"] > 0)]
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.barh(nombres[::-1], sanas[::-1], color='#1c75fa')
        ax2.set_xlabel("Número de Estaciones Sanas")
        ax2.set_title("Densidad Pluviométrica Fáctica por Estado", fontweight="bold")
        ax2.grid(axis='x', linestyle='--', alpha=0.7)
        ruta_barras = os.path.join(dir_graficos, "densidad_estados.png")
        fig2.savefig(ruta_barras, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig2)

        # FASE 3: COMPILACIÓN LATEX
        callback_log("> 📑 Compilando código fuente LaTeX (.tex)...")
        
        filas_tabla = []
        for estado_clave, data in estados_ordenados:
            if data["sanas"] == 0 and data["corruptas"] == 0: continue
            
            prom_inicio = int(sum(data["años_inicio"])/len(data["años_inicio"])) if data["años_inicio"] else "N/A"
            prom_fin = int(sum(data["años_fin"])/len(data["años_fin"])) if data["años_fin"] else "N/A"
            
            filas_tabla.append(f"{data['nom']} & {data['sanas']} & {data['corruptas']} & {prom_inicio} & {prom_fin} \\\\ \\hline")
            
        tabla_latex = "\n".join(filas_tabla)
        
        plantilla_tex = fr"""\\documentclass[12pt,a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[spanish]{{babel}}
\\usepackage{{graphicx}}
\\usepackage{{geometry}}
\\geometry{{a4paper, margin=1in}}
\\usepackage{{array}}

\\title{{Auditoría Física de Base de Datos - HidroSistem Tláloc}}
\\author{{Reporte Generado Automáticamente}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle

\\section{{Resumen Ejecutivo}}
El presente informe documenta el estado físico real de la base de datos comprimida (LZMA), evadiendo metadatos e inspeccionando el núcleo de los archivos. 
Se han detectado \\textbf{{{nacional_sanas}}} estaciones operativas y \\textbf{{{nacional_corruptas}}} archivos corruptos/vacíos.

\\begin{{figure}}[h!]
    \\centering
    \\includegraphics[width=0.48\\textwidth]{{Graficos_Auditoria/integridad_nacional.png}}
    \\hfill
    \\includegraphics[width=0.48\\textwidth]{{Graficos_Auditoria/densidad_estados.png}}
    \\caption{{Integridad nacional (Izquierda) y Densidad operativa por Estado (Derecha).}}
\\end{{figure}}

\\newpage
\\section{{Desglose Paramétrico por Entidad Federativa}}
\\begin{{table}}[h!]
\\centering
\\renewcommand{{\\arraystretch}}{{1.3}}
\\begin{{tabular}}{{|l|c|c|c|c|}}
\\hline
\\textbf{{Estado}} & \\textbf{{E. Sanas}} & \\textbf{{E. Corruptas}} & \\textbf{{Año Prom. Inicio}} & \\textbf{{Año Prom. Fin}} \\\\ \\hline
{tabla_latex}
\\end{{tabular}}
\\caption{{Consolidado de auditoría física profunda por estado.}}
\\end{{table}}

\\end{{document}}
"""
        ruta_tex = os.path.join(ruta_salida, "Reporte_Auditoria_Tlaloc.tex")
        with open(ruta_tex, 'w', encoding='utf-8') as f:
            f.write(plantilla_tex)
            
        return True, f"Auditoría rigurosa completada. Reporte y gráficas guardados en: {ruta_salida}"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Error Crítico: {str(e)}"

