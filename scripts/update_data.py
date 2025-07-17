import sys
import subprocess
import os

def run_script(script_path):
    """Ejecuta un script de Python usando el mismo intérprete que este script."""
    # sys.executable es la ruta exacta al python.exe del entorno virtual (.venv310)
    interpreter = sys.executable
    print(f"▶️ Ejecutando: {script_path} con {os.path.basename(interpreter)}")
    
    # Usamos subprocess.run para más control y para ver errores
    result = subprocess.run([interpreter, script_path], capture_output=True, text=True)
    
    # Si hubo un error en el script hijo, lo mostramos
    if result.returncode != 0:
        print(f"❌ ERROR en {script_path}:")
        print(result.stderr)
    else:
        print(f"✅ {script_path} completado.")
    print("-" * 50)


if __name__ == "__main__":
    print("🔄 Iniciando pipeline de actualización de datos...")
    print("-" * 50)
    
    # Definimos la lista de scripts a ejecutar en orden
    scripts_a_ejecutar = [
        "scripts/download_data.py",
        "scripts/calc_indicators.py",
        "scripts/prepare_data_classification.py"
    ]
    
    # Ejecutamos cada script
    for script in scripts_a_ejecutar:
        run_script(script)
        
    print("🎉 Pipeline de actualización finalizado.")