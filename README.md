
# summary

## Cara Menggunakan Aplikasi

### 1. Persiapan Data
- Pastikan dataset sudah tersedia di folder `Dataset/` sesuai struktur berikut:

```
Dataset/
	DUC2006/
		raw_data/
			D0601A/  # Folder berisi file artikel mentah
		gold_summaries/
			T1_1.txt, T1_2.txt, ...  # File ringkasan referensi
	DUC2007/
		...
```

### 2. Instalasi Dependensi
- Pastikan Python sudah terpasang di komputer Anda.
- Install semua library yang dibutuhkan dengan:
	```bash
	pip install -r requirements.txt
	```

### 3. Menjalankan Program
- Jalankan proses summarization dengan perintah:
	```bash
	python main_summarization.py
	```
- Setelah dijalankan, akan muncul **menu interaktif**:
	- Pilih apakah ingin memproses SEMUA data atau hanya BEBERAPA data saja (misal: 1 file).
	- Jika memilih beberapa data, masukkan jumlah data yang ingin diuji.

### 4. Hasil Ringkasan
- Hasil ringkasan dan skor evaluasi akan otomatis disimpan di dalam folder `results/` dengan subfolder nama waktu (timestamp), misal:
	```
	results/20251010_153012/
		DUC2006_bertsum_summaries.txt
		DUC2006_bertsum_scores.txt
		DUC2006_pegasus_summaries.txt
		DUC2006_pegasus_scores.txt
		overall_summary.txt
	```
- Setiap kali menjalankan program, hasil baru akan tersimpan di folder baru, sehingga hasil sebelumnya tidak tertimpa.

### 5. Log & Progress
- Selama proses berjalan, program akan menampilkan log progress, status per sample, serta penggunaan CPU dan RAM.
- Jika proses terasa lama, itu normal karena model summarization (terutama Pegasus) cukup berat, apalagi jika dijalankan di CPU.

### 6. Catatan Penting
- **Script ini hanya untuk inference (pengujian ringkasan), BUKAN untuk training/fine-tuning model.**
- Untuk menambah atau mengubah dataset, letakkan file pada folder yang sesuai di dalam `Dataset/`.
- Jika ingin menggunakan model tertentu, pastikan script yang dijalankan sudah sesuai (misal: `bertsum_summarization.py` untuk BertSum, `pegasus_summarization.py` untuk Pegasus).
- Proses akan jauh lebih cepat jika menggunakan GPU. Jika hanya menggunakan CPU, proses bisa sangat lama terutama untuk model besar.

---
Jika mengalami kendala, pastikan struktur folder dan dependensi sudah benar. Untuk pertanyaan lebih lanjut, silakan hubungi pengembang.