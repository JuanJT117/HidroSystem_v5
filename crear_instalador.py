import os
import subprocess

# Este es el código nativo del instalador NSIS
nsi_script = """
!define APPNAME "HidroSistem"
!define APPVERSION "10.9"
!define EXECUTABLE "HidroSistem_v10.9.exe"

Name "${APPNAME} v${APPVERSION}"
OutFile "Instalador_HidroSistem_10.9.exe"
InstallDir "$PROGRAMFILES64\\${APPNAME}"

RequestExecutionLevel admin

Section "Instalar"
    SetOutPath "$INSTDIR"
    
    ; 1. Copiamos el ejecutable real generado por Flet
    File "dist\\${EXECUTABLE}"

    ; 2. Creamos el desinstalador
    WriteUninstaller "$INSTDIR\\uninstall.exe"

    ; 3. Creamos el Acceso Directo en el Escritorio (¡Garantizado!)
    CreateShortcut "$DESKTOP\\${APPNAME}.lnk" "$INSTDIR\\${EXECUTABLE}" "" "$INSTDIR\\${EXECUTABLE}" 0
    
    ; 4. Creamos accesos en el Menú Inicio
    CreateDirectory "$SMPROGRAMS\\${APPNAME}"
    CreateShortcut "$SMPROGRAMS\\${APPNAME}\\${APPNAME}.lnk" "$INSTDIR\\${EXECUTABLE}" "" "$INSTDIR\\${EXECUTABLE}" 0
    CreateShortcut "$SMPROGRAMS\\${APPNAME}\\Desinstalar.lnk" "$INSTDIR\\uninstall.exe"
    
    ; 5. Registrar en Panel de Control para Desinstalación limpia
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "DisplayName" "${APPNAME}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "UninstallString" '"$INSTDIR\\uninstall.exe"'
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "DisplayIcon" '"$INSTDIR\\${EXECUTABLE}"'
SectionEnd

Section "Uninstall"
    Delete "$INSTDIR\\${EXECUTABLE}"
    Delete "$INSTDIR\\uninstall.exe"
    RMDir "$INSTDIR"
    
    Delete "$DESKTOP\\${APPNAME}.lnk"
    Delete "$SMPROGRAMS\\${APPNAME}\\${APPNAME}.lnk"
    Delete "$SMPROGRAMS\\${APPNAME}\\Desinstalar.lnk"
    RMDir "$SMPROGRAMS\\${APPNAME}"
    
    DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}"
SectionEnd
"""

# Guardamos el script
with open("setup.nsi", "w", encoding="utf-8") as f:
    f.write(nsi_script)

print("⚙️ Invocando al motor NSIS para crear el Setup...")

# Basado en tu log anterior, esta es la ruta exacta de NSIS en tu PC:
nsis_path = r"C:\Program Files (x86)\NSIS\makensis.exe"

try:
    subprocess.run([nsis_path, "setup.nsi"], check=True)
    print("✅ ¡Instalador 'Instalador_HidroSistem_10.9.exe' creado con éxito!")
except Exception as e:
    print(f"❌ Error al compilar: {e}")