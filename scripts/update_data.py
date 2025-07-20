# scripts/update_data.py

import sys
import subprocess
import os

def run_script(script_path):
    """Ejecuta un script de Python usando el mismo intérprete que este script."""
    interpreter = sys.executable
    print(f"▶️ Ejecutando: {script_path} con {os.path.basename(interpreter)}")
    result = subprocess.run([interpreter, script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR en {script_path}:\n{result.stderr}")
    else:
        print(f"✅ {script_path} completado.")
    print("-" * 50)

if __name__ == "__main__":
    print("🔄 Iniciando pipeline de actualización de datos...")
    print("-" * 50)
    
    scripts_a_ejecutar = [
        "scripts/download_data.py",
        "scripts/calc_indicators.py",
        "scripts/prepare_data.py"
    ]
    
    for script in scripts_a_ejecutar:
        run_script(script)
        
    print("🎉 Pipeline de actualización finalizado.")