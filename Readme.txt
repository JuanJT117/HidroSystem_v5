Instrucciones para ejecución de HidroSistem en Anaconda

1- Instalar anaconda, cualquier versión.
2- Seleccionar la opción "Environments" en el menú izquierdo
3- Seleccionar "Import", se ubica en el menú inferior junto a Create, Clone, Blackup y Remove
4- Buscar el archivo "environment.yml" dentro de la carpeta de desarrollo de la aplicación.
5. Una vez generado el Environment hay que seleccionarlo y dar clic sobre el.
6- Ya seleccionado regresamos al menú "Home" y buscamos la aplicación VS Code, si no esta instalada es necesario instalarla desde ese mismo menú.
7- Una vez dentro de VS Code en el menú superior seleccionamos "File" (Archivo) y damos click en "Add Folder to Workspace", buscamos la carpeta del proyecto para integrarla al explorador de archivos.
8- En el menú se desplegara la carpeta, sub carpetas y archivos que se encuentran dentro de las mismas, buscamos el archivo llamado "main.py" lo abrimos de manera que podamos visualizar el código en VS Code
9- YA abierto se habilitara un icono de "Play" en la esquina superior derecha correspondiente a "Run Python File" dando click en el se ejecutara el programa.

Instrucciones para la creación del ejecutable .exe de HidroSistem en Anaconda

1- Abrir anaconda y activar el Environment del proyecto
2- Abrir "CMD.exe Prompt" o "Powershell Prompt"
3- Navegar mediante CD hasta la carpeta donde esta el archivo "main.py"
4- Ejecutar el siguiente código:

flet pack main.py --name "HidroSistem_v9.1" --add-data "assets;assets" --hidden-import "pandas" --hidden-import "numpy" --hidden-import "scipy" --hidden-import "statsmodels" --hidden-import "sklearn" --hidden-import "pmdarima" --hidden-import "matplotlib" --hidden-import "seaborn" --hidden-import "folium" --hidden-import "geopy" --hidden-import "shapefile" --hidden-import "shapely" --hidden-import "tabulate" --hidden-import "requests" --icon "assets/icon.ico"

NOTA: este código es funcional únicamente para la Versión "HidroSistem_v9.1" 

5- En la capeta creada después de la ejecución del código  llamada "dist" se encontrara el archivo .exe del programa

Instrucciones para el respaldo del Environment 

1- Abrir anaconda y el Environment del proyecto 
2- Abrir "CMD.exe Prompt" o "Powershell Prompt"
3- Navegar mediante CD hasta la carpeta donde se dese guardar el archivo, preferentemente en la misma carpeta de desarrollo, en la carpeta llamada "assets"
4- ejecutar el siguiente código:

conda env export --no-builds > environment.yml