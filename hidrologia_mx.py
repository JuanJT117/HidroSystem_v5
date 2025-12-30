# hidrologia_mx.py
# Base de Datos de Parámetros Hidrológicos para México
# Fuentes: SCT, CONAGUA (MAPAS), NRCS (TR-55), INEGI

# ==========================================
# 1. MÉTODO RACIONAL (Coeficiente C)
# ==========================================
OPCIONES_C = {
    1: '1. Urbanizada/Superficie asfáltica',
    2: '2. Urbanizada/Concreto-azotea',
    3: '3. Área con pasto (Pobre)/plano (0-2%)',
    4: '4. Área con pasto (Pobre)/promedio (2-7%)',
    5: '5. Área con pasto (Pobre)/pendiente (> 7%)',
    6: '6. Área con pasto (Media)/plano (0-2%)',
    7: '7. Área con pasto (Media)/promedio (2-7%)',
    8: '8. Área con pasto (Media)/pendiente (> 7%)',
    9: '9. Áreascon pasto (Buena)/plano (0-2 %)',
    10: '10. Áreas con pasto (Buena)/promedio (2-7%)',
    11: '11. Áreas con pasto (Buena)/pendiente (> 7%)',
    12: '12. Rural - Cultivo/plano (0-2%)',
    13: '13. Rural - Cultivo/promedio (2-7%)',
    14: '14. Rural - Cultivo/pendiente (> 7%)',
    15: '15. Rural - Pastizal/plano (0-2%)',
    16: '16. Rural - Pastizal/promedio (2-7%)',
    17: '17. Rural - Pastizal/pendiente (> 7%)',
    18: '18. Bosque y monte/plano (0-2%)',
    19: '19. Bosque y monte/promedio (2-7%)',
    20: '20. Bosque y monte/pendiente (> 7%)'
}

MATRIZ_C_VALORES = {
    '2': [0.73, 0.75, 0.32, 0.37, 0.40, 0.25, 0.33, 0.37, 0.21, 0.29, 0.34, 0.31, 0.35, 0.39, 0.25, 0.33, 0.37, 0.22, 0.31, 0.35],
    '5': [0.77, 0.80, 0.34, 0.40, 0.43, 0.28, 0.36, 0.40, 0.23, 0.32, 0.37, 0.34, 0.38, 0.42, 0.28, 0.36, 0.40, 0.25, 0.34, 0.39],
    '10': [0.81, 0.83, 0.37, 0.43, 0.45, 0.30, 0.38, 0.42, 0.25, 0.35, 0.40, 0.36, 0.41, 0.44, 0.30, 0.38, 0.42, 0.28, 0.36, 0.41],
    '20': [0.86, 0.88, 0.40, 0.46, 0.49, 0.34, 0.42, 0.46, 0.29, 0.39, 0.44, 0.40, 0.44, 0.48, 0.34, 0.42, 0.46, 0.31, 0.41, 0.45],
    '50': [0.90, 0.92, 0.44, 0.49, 0.52, 0.37, 0.45, 0.49, 0.32, 0.42, 0.47, 0.43, 0.48, 0.51, 0.37, 0.45, 0.49, 0.35, 0.43, 0.48],
    '100': [0.95, 0.97, 0.47, 0.53, 0.55, 0.41, 0.49, 0.53, 0.36, 0.46, 0.51, 0.47, 0.51, 0.54, 0.41, 0.49, 0.53, 0.39, 0.47, 0.52],
    '500': [1.00, 1.00, 0.58, 0.61, 0.62, 0.53, 0.58, 0.60, 0.49, 0.56, 0.58, 0.57, 0.60, 0.61, 0.53, 0.58, 0.60, 0.48, 0.56, 0.58],
    '1000': [1.00, 1.00, 0.65, 0.68, 0.70, 0.60, 0.65, 0.68, 0.56, 0.63, 0.65, 0.64, 0.68, 0.70, 0.60, 0.65, 0.68, 0.55, 0.62, 0.65],
    '10000': [1.00, 1.00, 0.82, 0.85, 0.86, 0.78, 0.82, 0.84, 0.75, 0.80, 0.82, 0.81, 0.84, 0.85, 0.78, 0.82, 0.84, 0.72, 0.79, 0.81]
}

# ==========================================
# 2. MÉTODO CHOW / HMS (Número de Curva N)
# ==========================================
OPCIONES_N = {
    1: '1. Parque, campo abierto, Cancha deportiva-Condicion buena (75% pasto)',
    2: '2. Parque, campo abierto, Cancha deportiva-Condicion regular (50-75% pasto)',
    3: '3. Parque, campo abierto, Cancha deportiva-Condicion pobre (<50% pasto)',
    4: '4. Área comercial (85% impermeable)',
    5: '5. Distrito industrial (72% impermeable)',
    6: '6. Zona residencial (<500m2/65% impermeable)',
    7: '7. Zona residencial (1000m2/38% impermeable)',
    8: '8. Zona residencial (<1350m2/30% impermeable)',
    9: '9. Zona residencial (<2000m2/25% impermeable)',
    10: '10. Zona residencial (<4000m2/20% impermeable)',
    11: '11. Zona residencial (<8000m2/12% impermeable)',
    12: '12. Calzada, tejado, estacionamiento pavimentado, etc.',
    13: '13. Calle Pavimentada con guarnición y alcantarillado',
    14: '14. Camino pavimentado, derecho de via y canales',
    15: '15. Camino engravado - derecho de via',
    16: '16. Camino de arcilla -derecho de via',
    17: '17. Área urbana en desarrollo (nivelado sin vegetación)'
}

MATRIZ_N_VALORES = {
    'A': [39,49,68,89,81,77,61,57,54,51,46,98,98,83,76,72,77],
    'B': [61,69,79,92,88,85,75,72,70,68,65,98,98,89,85,82,86],
    'C': [74,79,86,94,91,90,83,81,80,79,77,98,98,92,89,87,91],
    'D': [80,84,89,95,93,92,87,86,85,84,82,98,98,92,91,89,94]
}

# ==========================================
# 3. HIDROGRAMA UNITARIO SCS (Adimensional)
# ==========================================
# Coordenadas de la curva Gamma adimensional del SCS
# t/tp, q/qp
HU_SCS_ADIMENSIONAL = {
    "t_ratio": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 
                1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 
                2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.5, 5.0],
    "q_ratio": [0.0, 0.03, 0.1, 0.19, 0.31, 0.47, 0.66, 0.82, 0.93, 0.99, 1.0, 
                0.99, 0.93, 0.86, 0.78, 0.68, 0.56, 0.46, 0.39, 0.33, 0.28, 0.207, 
                0.147, 0.107, 0.077, 0.055, 0.04, 0.029, 0.021, 0.015, 0.011, 0.005, 0.0]
}