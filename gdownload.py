import os
import re
import subprocess
import sys

def download_files_from_log(log_file, output_dir):
    """
    Membaca file log, mengekstrak ID file Google Drive dan nama filenya,
    lalu mengunduhnya menggunakan gdown.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Direktori dibuat: {output_dir}")

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File log '{log_file}' tidak ditemukan.", file=sys.stderr)
        return

    print(f"Memulai proses unduh dari {log_file}...")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Regex untuk mengekstrak nama file dan ID dari URL
        match = re.search(r'\s*([TDS][\d_]+\.txt)\s*—\s*https://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)/view', line)
        
        if not match:
            print(f"Melewati baris tidak valid: {line}")
            continue

        filename = match.group(1)
        file_id = match.group(2)
        
        output_path = os.path.join(output_dir, filename)

        print(f"-> Menemukan file: {filename} (ID: {file_id})")

        if os.path.exists(output_path):
            print(f"  -> File '{filename}' sudah ada. Dilewati.")
            continue

        # Membuat perintah gdown
        command = ['gdown', '--id', file_id, '-O', output_path]

        try:
            print(f"  -> Mengunduh ke {output_path}...")
            # Menjalankan perintah
            subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
            print(f"  -> Berhasil mengunduh {filename}")
        except FileNotFoundError:
            print("\nError: Perintah 'gdown' tidak ditemukan.", file=sys.stderr)
            print("Silakan install dengan 'pip install gdown'", file=sys.stderr)
            return
        except subprocess.CalledProcessError as e:
            print(f"  -> Gagal mengunduh {filename}. Error: {e.stderr.strip()}", file=sys.stderr)
        except Exception as e:
            print(f"  -> Terjadi error tak terduga saat mengunduh {filename}: {e}", file=sys.stderr)

    print("\nProses unduh selesai.")

if __name__ == "__main__":
    LOG_FILE = 'gold2007.txt'
    OUTPUT_DIRECTORY = './Dataset/DUC2007/gold_summaries'
    
    download_files_from_log(LOG_FILE, OUTPUT_DIRECTORY)
