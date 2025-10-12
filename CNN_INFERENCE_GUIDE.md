# 🚀 CNN/DailyMail Inference Guide

## 📖 Deskripsi

Script `cnn_inference.py` digunakan untuk melakukan inference (prediksi) text summarization pada dataset CNN/DailyMail menggunakan model BertSum dan/atau Pegasus.

## 🎯 Fitur Utama

- ✅ **Pilih dataset**: test.csv, validation.csv, atau train.csv
- ✅ **Pilih jumlah sampel**: 1 sampel sampai seluruh dataset
- ✅ **Pilih model**: BertSum, Pegasus, atau keduanya
- ✅ **Evaluasi otomatis**: ROUGE scores untuk mengukur kualitas summary
- ✅ **Hasil lengkap**: Artikel asli, reference summary, generated summary, dan scores

## 🚀 Cara Penggunaan

### 1. Jalankan Script

```bash
python cnn_inference.py
```

### 2. Ikuti Menu Interaktif

#### Menu 1: Pilih Dataset
```
📋 SELECT DATASET FILE:
1. test.csv  
2. validation.csv
3. train.csv

Pilih dataset (1/2/3) [default: 1]: 1
```

**Rekomendasi**: Pilih `1` (test.csv) untuk evaluasi standar

#### Menu 2: Pilih Jumlah Sampel
```
🎯 PILIH JUMLAH SAMPEL:
Berapa artikel yang ingin di-inference?
  - 1 sample: ~1-2 menit (testing cepat)
  - 5 samples: ~5-10 menit
  - 10 samples: ~10-20 menit
  - 50+ samples: lebih lama (tergantung resource)

Masukkan jumlah sampel [default: 1]: 1
```

**Tips**:
- Untuk **testing cepat**: 1-5 samples
- Untuk **evaluasi**: 50-100 samples
- Untuk **full evaluation**: 1000+ samples (butuh waktu lama!)

#### Menu 3: Pilih Model
```
🤖 PILIH MODEL SUMMARIZATION:
1. BertSum (Extractive) - Cepat, pilih kalimat dari artikel
2. Pegasus (Abstractive) - Lambat, generate kalimat baru
3. Keduanya (Perbandingan) - Paling lambat tapi lengkap

Pilih model (1/2/3) [default: 3]: 3
```

**Perbedaan**:
- **BertSum**: Extractive (pilih kalimat penting), lebih cepat (~30-60 detik/sample)
- **Pegasus**: Abstractive (buat kalimat baru), lebih lambat (~60-120 detik/sample)
- **Keduanya**: Untuk perbandingan kualitas

### 3. Tunggu Proses Inference

Script akan menampilkan progress:

```
📂 Loading CNN/DailyMail dataset from: /workspaces/summary/cnn_dailymail/test.csv
✅ Total samples in file: 11490
🎯 Selected samples: 1

================================================================================
📄 SAMPLE #1 INFO
================================================================================
ID: 92c514c913c0bdfe25341af9fd72b29db544099b

📰 ARTICLE (first 500 chars):
Ever noticed how plane seats appear to be getting smaller and smaller?...

📊 Article length: 3254 characters

✨ REFERENCE SUMMARY:
Experts question if packed out planes are putting passengers at risk...
================================================================================

[Bertsum] Memproses sample ke-1 dari 1...
[Pegasus] Memproses sample ke-1 dari 1...
```

### 4. Lihat Hasil

Hasil akan tersimpan di folder `results_cnn/YYYYMMDD_HHMMSS/`:

```
results_cnn/
└── 20251012_143520/
    ├── cnn_bertsum_summaries.txt   # Summary lengkap BertSum
    ├── cnn_bertsum_scores.txt      # ROUGE scores BertSum
    ├── cnn_pegasus_summaries.txt   # Summary lengkap Pegasus
    └── cnn_pegasus_scores.txt      # ROUGE scores Pegasus
```

## 📊 Format Output

### File Summaries (`cnn_bertsum_summaries.txt`)

```
CNN/DailyMail Inference Results - BERTSUM
================================================================================
Total Samples: 1
Date: 2025-10-12 14:35:20
================================================================================

================================================================================
SAMPLE #1
================================================================================
ID: 92c514c913c0bdfe25341af9fd72b29db544099b

ORIGINAL ARTICLE:
[Artikel lengkap...]

--------------------------------------------------------------------------------
REFERENCE SUMMARY (Ground Truth):
[Reference summary dari dataset...]

--------------------------------------------------------------------------------
GENERATED SUMMARY (BERTSUM):
[Summary hasil model...]

ROUGE SCORES:
  ROUGE-1: 0.4523
  ROUGE-2: 0.2341
  ROUGE-L: 0.3892
```

### File Scores (`cnn_bertsum_scores.txt`)

```
CNN/DailyMail ROUGE Scores - BERTSUM
================================================================================
Total Samples: 1
Date: 2025-10-12 14:35:20
================================================================================

AVERAGE SCORES:
  ROUGE-1: 0.4523
  ROUGE-2: 0.2341
  ROUGE-L: 0.3892

================================================================================

INDIVIDUAL SCORES:

Sample #1:
  ROUGE-1: 0.4523
  ROUGE-2: 0.2341
  ROUGE-L: 0.3892
```

## 📈 Interpretasi ROUGE Scores

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) mengukur overlap antara summary hasil model dengan reference summary:

| Score Range | Kualitas |
|-------------|----------|
| 0.0 - 0.2   | Poor     |
| 0.2 - 0.4   | Fair     |
| 0.4 - 0.6   | Good     |
| 0.6 - 0.8   | Very Good|
| 0.8 - 1.0   | Excellent|

**Metrics**:
- **ROUGE-1**: Overlap kata individual (precision kata)
- **ROUGE-2**: Overlap 2 kata berurutan (coherence)
- **ROUGE-L**: Longest common subsequence (struktur kalimat)

## ⚡ Tips & Best Practices

### 1. Testing Awal
```bash
# Test dengan 1 sample dulu untuk validasi
# Pastikan tidak ada error
python cnn_inference.py
# Input: 1, 1, 3
```

### 2. Evaluasi Cepat (5-10 Samples)
```bash
python cnn_inference.py
# Input: 1, 5, 3
```

### 3. Evaluasi Standar (50-100 Samples)
```bash
python cnn_inference.py
# Input: 1, 50, 3
# Waktu: ~1-2 jam
```

### 4. Full Evaluation (1000+ Samples)
```bash
# Jalankan di background dengan nohup
nohup python cnn_inference.py > inference.log 2>&1 &
# Input: 1, 1000, 3
# Waktu: ~10-20 jam
```

## 🔧 Troubleshooting

### Error: Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solusi**:
- Kurangi jumlah sampel
- Gunakan BertSum saja (lebih ringan)
- Restart kernel/environment

### Error: File Not Found
```
FileNotFoundError: cnn_dailymail/test.csv
```
**Solusi**:
- Pastikan file CSV ada di folder `cnn_dailymail/`
- Extract file ZIP jika belum

### Model Download Lambat
**Pertama kali** BertSum dan Pegasus akan download model (~1-2 GB)
- **Solusi**: Sabar menunggu, ini hanya sekali

## 📝 Contoh Use Case

### Use Case 1: Quick Test
**Tujuan**: Cek apakah script berjalan dengan baik
```
Dataset: 1 (test.csv)
Samples: 1
Model: 3 (keduanya)
Waktu: ~2 menit
```

### Use Case 2: Model Comparison
**Tujuan**: Bandingkan BertSum vs Pegasus
```
Dataset: 1 (test.csv)
Samples: 10
Model: 3 (keduanya)
Waktu: ~15-20 menit
```

### Use Case 3: Research Evaluation
**Tujuan**: Evaluasi performa model secara detail
```
Dataset: 1 (test.csv)
Samples: 100
Model: 3 (keduanya)
Waktu: ~2-3 jam
```

## 📞 Support

Jika ada pertanyaan atau issue:
1. Cek file log error
2. Pastikan semua dependencies ter-install
3. Cek resource (RAM/CPU) mencukupi

---

**Happy Summarizing! 🎉**
