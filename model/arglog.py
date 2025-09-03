from datetime import datetime
import os

def write_log_with_timestamp(log_file_path, content, level="INFO"):
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{level.upper():<8}] {content}\n"
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(line)

