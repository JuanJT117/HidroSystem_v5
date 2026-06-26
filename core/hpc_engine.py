import numpy as np
import scipy.sparse as sp
import warnings
import traceback

# Patrón de Degradación Elegante para Hardware
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class MotorTensorial:
    """
    Capa de abstracción de hardware.
    Enruta las multiplicaciones de matrices a la GPU si está disponible,
    o utiliza Matrices Dispersas en CPU (Multihilo) para ahorrar RAM.
    """

    @staticmethod
    def filtrar_y_comprimir_matriz(matriz_lluvia: np.ndarray, umbral: float = 0.5):
        """
        Aplica el filtro de llovizna (Thresholding) propuesto por el usuario
        y convierte la matriz a un formato disperso para salvar la RAM.
        """
        try:
            # 1. Filtro de Llovizna (Todo lo menor al umbral se vuelve 0)
            matriz_lluvia[matriz_lluvia <= umbral] = 0.0

            # 2. Compresión a Matriz Dispersa (CSC - Compressed Sparse Column)
            # Ideal para multiplicaciones rápidas de columnas.
            matriz_dispersa = sp.csc_matrix(matriz_lluvia)
            
            # Log de optimización (opcional)
            sparsity = 1.0 - (matriz_dispersa.nnz / matriz_lluvia.size)
            print(f"⚡ [HPC] Matriz comprimida. Espacio ahorrado: {sparsity * 100:.2f}%")
            
            return matriz_dispersa
            
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error en compresión tensorial: {str(e)}")

    @staticmethod
    def calcular_lluvia_areal_dia(matriz_estaciones_dispersa, pesos_idw_grid):
        """
        Multiplicación matricial (Estaciones x Pesos del Grid).
        LMA = Suma(Precip_Estacion_i * Peso_Areal_i)
        """
        try:
            if GPU_AVAILABLE:
                # --- RUTA 1: ACELERACIÓN GPU (CuPy) ---
                # Enviamos los datos a la VRAM de la tarjeta de video
                estaciones_gpu = cp.sparse.csc_matrix(matriz_estaciones_dispersa)
                pesos_gpu = cp.asarray(pesos_idw_grid)
                
                # Multiplicación tensorial paralela masiva
                lma_gpu = estaciones_gpu.dot(pesos_gpu)
                
                # Retornamos a la RAM de la CPU
                return cp.asnumpy(lma_gpu)
                
            else:
                # --- RUTA 2: ACELERACIÓN CPU (SciPy Sparse) ---
                # Multiplicación optimizada en CPU usando C/C++ backend de SciPy
                lma_cpu = matriz_estaciones_dispersa.dot(pesos_idw_grid)
                return lma_cpu
                
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Fallo en álgebra de LMA: {str(e)}")