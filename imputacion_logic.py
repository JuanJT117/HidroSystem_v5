import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from geopy.distance import great_circle 
import pmdarima as pm
import folium 
import traceback
import warnings

# Ignorar advertencias de modelos para limpieza de consola
warnings.filterwarnings("ignore")

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# 1. GESTIÓN DE ARCHIVOS Y MAPA
#-------------------------------------------------------------------------------------------------------------------------------------------------------

def leer_estaciones(folder_path):
    print(f"--- Escaneando carpeta: {folder_path} ---")
    local_station_files = {}
    try:
        archivos = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    except Exception as e:
        print(f"Error al listar archivos: {e}")
        return {}

    for nombre_archivo in archivos:
        try:
            path = os.path.join(folder_path, nombre_archivo)
            lat, lon = None, None
            # Leemos solo el encabezado para velocidad
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for _ in range(20):
                    linea = f.readline()
                    if not linea: break
                    if 'LATITUD' in linea:
                        lat = float(linea.split(':')[1].strip().split(' ')[0])
                    elif 'LONGITUD' in linea:
                        lon = float(linea.split(':')[1].strip().split(' ')[0])
                    if lat is not None and lon is not None: break
            
            if lat is not None and lon is not None:
                station_id = nombre_archivo.split('.')[0]
                local_station_files[station_id] = {'file': nombre_archivo, 'lat': lat, 'lon': lon, 'path': path}
        except Exception as e:
            print(f"Error leyendo cabecera de {nombre_archivo}: {e}")
    return local_station_files

def generar_mapa_html(station_files, output_dir="."):
    if not station_files: return None
    try:
        lats = [v['lat'] for v in station_files.values()]
        lons = [v['lon'] for v in station_files.values()]
        center = [np.mean(lats), np.mean(lons)]
        m = folium.Map(location=center, zoom_start=8, tiles='CartoDB dark_matter')
        for sid, info in station_files.items():
            folium.Marker([info['lat'], info['lon']], popup=f"Est: {sid}", tooltip=sid, icon=folium.Icon(color="green", icon="info-sign")).add_to(m)
        path = os.path.join(output_dir, "mapa_estaciones.html")
        m.save(path)
        return os.path.abspath(path)
    except Exception as e:
        print(f"Error mapa: {e}")
        return None

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# 2. PARSEO Y RANGO GLOBAL
#-------------------------------------------------------------------------------------------------------------------------------------------------------

def parse_station_data(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            parsing = False
            for line in lines:
                if 'FECHA' in line and 'PRECIP' in line: parsing = True; continue
                if parsing:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            fecha = parts[0]
                            val = parts[1]
                            precip = float(val) if val != 'NULO' else np.nan
                            data.append([fecha, precip])
                        except: continue
        df = pd.DataFrame(data, columns=['FECHA', 'PRECIP'])
        df['FECHA'] = pd.to_datetime(df['FECHA'], format='%Y-%m-%d', errors='coerce')
        df = df.dropna(subset=['FECHA']).set_index('FECHA').sort_index()
        df = df[~df.index.duplicated(keep='first')]
        return df
    except Exception as e:
        print(f"Error parseando {file_path}: {e}")
        return pd.DataFrame(columns=['FECHA', 'PRECIP'])

def obtener_rango_global_fechas(station_files):
    """Escanea rápidamente todas las estaciones para encontrar Min y Max absolutos."""
    fechas_min = []
    fechas_max = []
    
    for info in station_files.values():
        df = parse_station_data(info['path'])
        if not df.empty:
            fechas_min.append(df.index.min())
            fechas_max.append(df.index.max())
            
    if not fechas_min: return None
    return pd.date_range(min(fechas_min), max(fechas_max), freq='D')

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# 3. UTILERÍAS DE CÁLCULO
#-------------------------------------------------------------------------------------------------------------------------------------------------------

def calculate_distance(lat1, lon1, lat2, lon2):
    return great_circle((lat1, lon1), (lat2, lon2)).km

def _filtrar_ruido_intermedio(df_target, df_neighbors, original_mask):
    """Filtra valores imputados que exceden por mucho el máximo de los vecinos."""
    try:
        temp_df = df_neighbors.copy()
        temp_df['Año'] = temp_df.index.year
        temp_df['Semana'] = temp_df.index.isocalendar().week
        cols_vecinos = [c for c in temp_df.columns if c not in ['Año', 'Semana']]
        
        if not cols_vecinos: return df_target, 0 
            
        maximos_semanales = temp_df.groupby(['Año', 'Semana'])[cols_vecinos].max().max(axis=1)
        idx_to_group = pd.DataFrame({'Año': df_target.index.year, 'Semana': df_target.index.isocalendar().week}, index=df_target.index)
        map_dict = maximos_semanales.to_dict()
        
        def get_limit(r): return map_dict.get((r['Año'], r['Semana']), 9999) 
        limites = idx_to_group.apply(get_limit, axis=1)
        
        imputed_mask = ~original_mask & df_target['PRECIP'].notna()
        suspicious = imputed_mask & (df_target['PRECIP'] > (limites * 1.2 + 5)) 
        count_removed = suspicious.sum()
        
        if count_removed > 0:
            df_target.loc[suspicious, 'PRECIP'] = np.nan
            
        return df_target, count_removed
    except Exception as e:
        return df_target, 0

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# 4. NÚCLEO DE IMPUTACIÓN (SINGLE TARGET INTELIGENTE + ROBUSTO)
#-------------------------------------------------------------------------------------------------------------------------------------------------------

def impute_target_station(target_id, station_files, page, pb, pbl, radius_km):
    log = []
    try:
        # --- 1. DEFINIR RANGO GLOBAL ---
        pbl.value = "Calculando rango de fechas global..."; page.update()
        global_range = obtener_rango_global_fechas(station_files)
        if global_range is None: return None, "Error: No se pudieron determinar fechas."
        log.append(f"Rango Global: {global_range.min().date()} a {global_range.max().date()}")
        log.append(f"Radio de Búsqueda seleccionado: {radius_km} km")

        # --- 2. CARGAR TARGET ---
        info_target = station_files[target_id]
        pbl.value = f"Cargando objetivo: {target_id}..."; page.update()
        
        df_target = parse_station_data(info_target['path'])
        df_target = df_target.reindex(global_range)
        df_target.index.name = 'FECHA'
        
        df_target['PRECIP_ORIGINAL'] = df_target['PRECIP'].copy()
        original_data_mask = df_target['PRECIP'].notna()
        
        initial_nans = df_target['PRECIP'].isna().sum()
        log.append(f"Huecos totales a rellenar: {initial_nans}")

        # --- 3. CARGAR VECINOS (Radio Dinámico) ---
        pbl.value = f"Cargando vecinos (Radio {radius_km}km)..."; page.update()
        
        lat_t, lon_t = info_target['lat'], info_target['lon']
        neighbors_all = []
        
        for sid, info in station_files.items():
            if sid == target_id: continue
            dist = calculate_distance(lat_t, lon_t, info['lat'], info['lon'])
            
            if dist <= radius_km: # Uso del parámetro dinámico
                df_nb = parse_station_data(info['path'])
                df_nb = df_nb.reindex(global_range)
                
                if not df_nb['PRECIP'].dropna().empty:
                    col_name = f"PRECIP_{sid}"
                    neighbors_all.append({
                        'id': sid,
                        'dist': dist,
                        'col_name': col_name,
                        'data': df_nb['PRECIP'].rename(col_name),
                        'weight': 1/(dist**2 if dist>0 else 0.001)
                    })

        log.append(f"Vecinos detectados (<{radius_km}km): {len(neighbors_all)}")
        
        if neighbors_all:
            df_neighbors = pd.concat([n['data'] for n in neighbors_all], axis=1)
        else:
            df_neighbors = pd.DataFrame(index=global_range)

        pb.value = 0.2; page.update()

        # ==============================================================================
        # FASE 1: IDW (Espacial) - CON CANDADO >= 5 VECINOS
        # ==============================================================================
        missing_indices = df_target[df_target['PRECIP'].isna()].index
        
        if not df_neighbors.empty and len(missing_indices) > 0:
            pbl.value = "Fase 1: IDW (Req. >= 5 vecinos)..."
            page.update()
            count_idw = 0
            skipped_idw = 0
            
            for date in missing_indices:
                row_vals = df_neighbors.loc[date]
                valid_neighbors_for_date = row_vals.count()
                
                if valid_neighbors_for_date < 5:
                    skipped_idw += 1
                    continue 
                
                num, den = 0, 0
                for nb in neighbors_all:
                    val = row_vals[nb['col_name']]
                    if pd.notna(val):
                        num += val * nb['weight']
                        den += nb['weight']
                
                if den > 0:
                    df_target.at[date, 'PRECIP'] = round(num / den, 2)
                    count_idw += 1
            
            log.append(f"Rellenados IDW: {count_idw} (Omitidos por falta de vecinos: {skipped_idw})")
            
            df_target, rm = _filtrar_ruido_intermedio(df_target, df_neighbors, original_data_mask)
            if rm > 0: log.append(f"-> Filtro IDW: Se eliminaron {rm} datos ruidosos.")

        pb.value = 0.4; page.update()

        # ==============================================================================
        # SELECCIÓN DE "ÉLITES" (Filtro Híbrido Dinámico)
        # Criterio: Distancia <= Radio  Y  Correlación > 0.7
        # ==============================================================================
        pbl.value = "Seleccionando estaciones Élite..."
        page.update()
        
        elite_neighbors = []
        
        if not df_neighbors.empty:
            correlations = df_neighbors.corrwith(df_target['PRECIP'])
            
            # INTENTO 1: Criterio Estricto (r > 0.7)   --> cambie de 0.7 a 0.6
            for nb in neighbors_all:
                r_val = correlations.get(nb['col_name'], 0)
                # Usa el radio dinámico también para las élites
                if nb['dist'] <= radius_km and r_val >= 0.6:
                    nb['corr'] = r_val
                    elite_neighbors.append(nb)
            
            # INTENTO 2: Criterio Relajado (r > 0.5) si no hay élites estrictas  --> cambie de 0.5 a 0.4
            if not elite_neighbors:
                log.append("⚠️ No se encontraron vecinas con r>0.7. Relajando criterio a r>0.5...")
                for nb in neighbors_all:
                    r_val = correlations.get(nb['col_name'], 0)
                    if nb['dist'] <= radius_km and r_val >= 0.4:
                        nb['corr'] = r_val
                        elite_neighbors.append(nb)

            elite_neighbors.sort(key=lambda x: x['corr'], reverse=True)
            
        elite_cols = [n['col_name'] for n in elite_neighbors]
        log.append(f"Estaciones Élite finales: {len(elite_neighbors)}")

        # ==============================================================================
        # FASE 2: MLR (Regresión) - UMBRALES AJUSTADOS
        # ==============================================================================
        missing_indices = df_target[df_target['PRECIP'].isna()].index
        
        if len(missing_indices) > 0 and len(elite_neighbors) > 0:
            pbl.value = "Fase 2: MLR (Req. >= 5 élites, Max 7)..."
            page.update()
            
            df_train = df_neighbors[elite_cols].copy()
            df_train['TARGET'] = df_target['PRECIP']
            df_train = df_train.dropna()
            
            if len(df_train) >= 14: 
                X = df_train.drop(columns=['TARGET'])
                y = df_train['TARGET']
                
                try:
                    model = LinearRegression().fit(X, y)
                    
                    count_mlr = 0
                    skipped_mlr_data = 0
                    
                    for date in missing_indices:
                        if count_mlr >= 7: break
                        
                        row_elite = df_neighbors.loc[date, elite_cols]
                        if row_elite.count() < 5:
                            skipped_mlr_data += 1
                            continue
                        
                        X_input = row_elite.fillna(0).to_frame().T 
                        pred = max(model.predict(X_input)[0], 0)
                        
                        df_target.at[date, 'PRECIP'] = round(pred, 2)
                        count_mlr += 1
                        
                    log.append(f"Rellenados MLR: {count_mlr}")
                    if skipped_mlr_data > 0: log.append(f"-> MLR omitidos por falta datos élite: {skipped_mlr_data}")
                    
                    df_target, rm = _filtrar_ruido_intermedio(df_target, df_neighbors, original_data_mask)
                    if rm > 0: log.append(f"-> Filtro MLR: {rm} eliminados.")
                    
                except Exception as ex:
                    log.append(f"Fallo en MLR: {ex}")
            else:
                log.append(f"MLR omitido: Solo {len(df_train)} días coinciden con élites (Req. >= 14).")

        pb.value = 0.7; page.update()

        # ==============================================================================
        # FASE 3: SARIMAX (ARIMA) - LÍMITE AMPLIADO
        # ==============================================================================
        missing_indices = df_target[df_target['PRECIP'].isna()].index
        
        if len(missing_indices) > 0:
            pbl.value = "Fase 3: SARIMAX/Interpolación..."
            page.update()
            
            try:
                exog_data = None
                if len(elite_neighbors) > 0:
                    exog_data = df_neighbors[elite_cols].fillna(0)
                
                y_train_temp = df_target['PRECIP'].interpolate(method='linear', limit_direction='both').fillna(0)
                
                if len(missing_indices) < 50500: # ---> 3650   se amplia a 50500 para incementar el tango de indices que completa
                    model = pm.auto_arima(
                        y_train_temp,
                        X=exog_data,
                        start_p=1, start_q=1, max_p=2, max_q=2,
                        seasonal=False, 
                        stepwise=True,
                        error_action='ignore',
                        suppress_warnings=True
                    )
                    
                    fitted_vals = model.predict_in_sample(X=exog_data)
                    
                    if not isinstance(fitted_vals, pd.Series):
                        fitted_vals = pd.Series(fitted_vals, index=global_range)
                        
                    for date in missing_indices:
                        val = fitted_vals.loc[date]
                        df_target.at[date, 'PRECIP'] = round(max(val, 0), 2)
                    
                    log.append(f"Rellenados SARIMAX: {len(missing_indices)}")
                else:
                    raise Exception(f"Demasiados huecos ({len(missing_indices)}) para ARIMA eficiente.")
                    
            except Exception as ex_arima:
                log.append(f"Fallback a Interpolación ({str(ex_arima)[:50]}...)")
                before_int = df_target['PRECIP'].isna().sum()
                df_target['PRECIP'] = df_target['PRECIP'].interpolate(method='time', limit_direction='both')
                filled_int = before_int - df_target['PRECIP'].isna().sum()
                log.append(f"Rellenados Interpolación: {filled_int}")

        # --- CIERRE ---
        df_target['PRECIP'] = df_target['PRECIP'].round(2)
        final_nans = df_target['PRECIP'].isna().sum()
        
        if final_nans > 0:
            log.append(f"⚠️ ADVERTENCIA: Quedaron {final_nans} datos vacíos (NaN).")
        else:
            log.append("✅ Serie completada al 100%.")

        pbl.value = "Consolidando..."
        page.update()
        
        df_final = df_target.join(df_neighbors, how='left')
        cols_base = ['PRECIP', 'PRECIP_ORIGINAL']
        cols_final = cols_base + elite_cols + [c for c in df_final.columns if c not in cols_base and c not in elite_cols]
        df_final = df_final[cols_final]
        
        pb.value = 1.0; pbl.value = "Finalizado."
        page.update()
        return df_final.reset_index(), "\n".join(log)

    except Exception as e:
        return None, f"Error crítico: {e}\n{traceback.format_exc()}"

def save_target_csv(df, target_id, output_folder):
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    filename = f"{target_id}_imputado.csv"
    path = os.path.join(output_folder, filename)
    df.to_csv(path, index=False)
    return path