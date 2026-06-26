from pydantic import BaseModel, Field
from typing import Literal, Optional

class SubcuencaSchema(BaseModel):
    ID_Cuenca: int = Field(default=1, description="Identificador único")
    Nombre: str = Field(default="Cuenca_Principal")
    Area_km2: float = Field(default=0.0)
    Perimetro_km: float = Field(default=0.0)
    Pendiente_pct: float = Field(default=0.0)
    Tc_minutos: float = Field(default=0.0)

class UsoSueloSchema(BaseModel):
    ID_Poligono: int = Field(default=1)
    Grupo_Hidro: Literal["A", "B", "C", "D"] = Field(default="B")
    Uso_Cobertura: str = Field(default="Sin Asignar")
    Condicion: Literal["Pobre", "Regular", "Buena"] = Field(default="Regular")
    CN_Asignado: float = Field(default=75.0, ge=0.0, le=100.0)

class CaucesSchema(BaseModel):
    ID_Cauce: int = Field(default=1)
    Orden_Strahler: int = Field(default=1)
    Es_Principal: int = Field(default=0, description="1 para Cauce Principal, 0 para afluentes")
    Longitud_m: float = Field(default=0.0)
    Desnivel_m: float = Field(default=0.0)
    Manning_n: float = Field(default=0.035)

class ThiessenSchema(BaseModel):
    ID_Poligono: int = Field(default=1)
    Clave_Estacion: str
    Prec_Base: float = Field(default=0.0)
    Area_km2: float = Field(default=0.0)
    Peso_Relativo: float = Field(default=0.0)
