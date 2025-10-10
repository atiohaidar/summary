# summary

## Cara Menggunakan Aplikasi

1. **Persiapan Data**
	- Pastikan dataset sudah tersedia di folder `Dataset/` sesuai struktur yang ada (misal: `DUC2006/raw_data/` dan `DUC2006/gold_summaries/`).

2. **Instalasi Dependensi**
	- Pastikan Python sudah terpasang di komputer Anda.
	- Install semua library yang dibutuhkan (misal: transformers, torch, dll). Jika ada file `requirements.txt`, jalankan:
	  ```bash
	  pip install -r requirements.txt
	  ```

3. **Menjalankan Program**
	- Untuk menjalankan proses summarization, gunakan perintah berikut di terminal:
	  ```bash
	  python main_summarization.py
	  ```
	- Program akan memproses data dan menghasilkan ringkasan otomatis sesuai model yang digunakan (misal: BertSum atau Pegasus).

4. **Hasil Ringkasan**
	- Hasil ringkasan akan disimpan di folder `results/` atau sesuai pengaturan pada script.

5. **Catatan**
	- Jika ingin menggunakan model tertentu, pastikan script yang dijalankan sudah sesuai (misal: `bertsum_summarization.py` untuk BertSum, `pegasus_summarization.py` untuk Pegasus).
	- Untuk menambah atau mengubah dataset, letakkan file pada folder yang sesuai di dalam `Dataset/`.

---
Jika mengalami kendala, pastikan struktur folder dan dependensi sudah benar. Untuk pertanyaan lebih lanjut, silakan hubungi pengembang.