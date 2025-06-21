# Laporan Proyek Machine Learning - Sakti Mujahid Imani

## Project Overview

Sistem rekomendasi adalah komponen penting dalam platform digital modern, digunakan untuk mempersonalisasi pengalaman pengguna dengan memberikan saran berdasarkan preferensi atau perilaku mereka sebelumnya. Dalam proyek ini, saya membangun sistem rekomendasi buku berdasarkan data Book-Crossing dari Kaggle.

Permasalahan utama yang ingin diselesaikan adalah bagaimana membantu pengguna menemukan buku yang relevan di tengah banyaknya pilihan yang tersedia. Proyek ini penting karena dapat digunakan dalam e-commerce buku, platform perpustakaan digital, atau sistem pembelajaran adaptif untuk menyarankan bacaan yang sesuai dengan minat pengguna.

Dataset ini digunakan karena memiliki kombinasi interaksi pengguna (user-item rating) dan metadata buku (judul dan penulis), yang sangat cocok untuk dua pendekatan utama sistem rekomendasi: **Content-Based Filtering** dan **Collaborative Filtering**.

Referensi:

- Ricci, F., Rokach, L., & Shapira, B. (2011). _Recommender Systems Handbook_. Springer. https://doi.org/10.1007/978-0-387-85820-3
- Kaggle. (n.d.). _Book-Crossing Dataset_. Diakses dari https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset

## Business Understanding

### Problem Statements

- Pengguna kesulitan menemukan buku yang sesuai dengan preferensi pribadi karena banyaknya pilihan yang tersedia.
- Sistem pencarian konvensional tidak dapat memahami selera pengguna secara mendalam tanpa pemrosesan lanjutan.
- Banyak pengguna hanya menemukan buku populer tanpa pernah menerima rekomendasi yang relevan secara personal.

### Goals

- Membangun sistem rekomendasi yang mampu menyarankan buku berdasarkan interaksi pengguna atau informasi konten buku.
- Mengimplementasikan dua pendekatan sistem rekomendasi dan membandingkan hasilnya.
- Menyediakan Top-N Recommendation secara personal untuk meningkatkan relevansi bacaan.

### Solution Statements

- **Content-Based Filtering:** Rekomendasi berdasarkan kemiripan fitur konten buku, seperti penulis dan judul.
- **Collaborative Filtering (User-Based):** Rekomendasi berdasarkan kesamaan preferensi antar pengguna.

## Data Understanding

Dataset yang digunakan adalah hasil gabungan dari `BX-Books.csv` dan `BX-Book-Ratings.csv`, yang kemudian disimpan sebagai `dataset.csv`.

- **Jumlah baris:** 1.031.173
- **Jumlah user unik:** 92.107
- **Jumlah buku unik:** 270.168
- **Sumber data:** [Kaggle - Book-Crossing Dataset](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset)

### Variabel:

- `User-ID`: ID unik pengguna
- `ISBN`: ID unik buku
- `Book-Rating`: Nilai rating pengguna (skala 0–10)
- `Book-Title`: Judul buku
- `Book-Author`: Nama penulis buku

### EDA (Exploratory Data Analysis):

- Sebagian besar rating bernilai 0 → tidak digunakan dalam model.
- Banyak pengguna hanya memberikan sedikit rating → perlu disaring.
- Distribusi rating menunjukkan dominasi pada nilai tinggi (8–10).
- Beberapa buku sangat sering dirating, menunjukkan popularitas.

## Data Preparation

### Teknik yang Digunakan:

1. **Menghapus rating bernilai 0** – karena tidak memberikan sinyal preferensi eksplisit.
2. **Menghapus duplikat user–book** – untuk menghindari bias dalam data.
3. **Normalisasi teks (`lowercase`)** – agar pencarian konten lebih akurat.
4. **Filter pengguna aktif (1000 user teratas)** – untuk mengurangi sparsity dan meningkatkan relevansi.

Selain itu, pada tahap modeling dilakukan ekstraksi fitur konten menggunakan **TF-IDF** pada gabungan judul dan penulis buku untuk Content-Based Filtering.

Langkah-langkah ini diterapkan secara berurutan di notebook dan `project.py`.

## Modeling

### 1. Content-Based Filtering

- Pada tahap ini, dilakukan ekstraksi fitur menggunakan **TF-IDF** pada gabungan kolom `Book-Title + Book-Author`.
- Mengukur kemiripan antar buku menggunakan `linear_kernel`.
- Rekomendasi diberikan berdasarkan kemiripan konten dengan buku yang disukai pengguna.

**Contoh Output:**
Jika pengguna menyukai buku _"harry potter and the chamber of secrets"_, maka sistem akan merekomendasikan buku-buku berikut (hasil aktual notebook):

| Book-Title                                | Book-Author   |
| ----------------------------------------- | ------------- |
| harry potter and the prisoner of azkaban  | j. k. rowling |
| harry potter and the goblet of fire       | j. k. rowling |
| harry potter and the order of the phoenix | j. k. rowling |
| harry potter and the sorcerer's stone     | j. k. rowling |
| harry potter and the prisoner of azkaban  | j. k. rowling |

### 2. Collaborative Filtering (User-Based)

- Membuat pivot table: user x book rating.
- Menggunakan cosine similarity antar pengguna.
- Mengambil 5 pengguna paling mirip dan menghitung rata-rata buku yang mereka sukai untuk diberikan sebagai rekomendasi.

**Contoh Output:**
Untuk user ID 277427, sistem menyarankan buku-buku berikut (hasil aktual notebook):

| Recommended Book          |
| ------------------------- |
| violets are blue          |
| me talk pretty one day    |
| the bonesetter's daughter |
| the lovely bones: a novel |
| the red tent              |

### Kelebihan & Kekurangan

| Metode                  | Kelebihan                                               | Kekurangan                                              |
| ----------------------- | ------------------------------------------------------- | ------------------------------------------------------- |
| Content-Based Filtering | Tidak butuh data pengguna lain (cocok untuk cold start) | Terbatas pada fitur konten yang tersedia                |
| Collaborative Filtering | Rekomendasi berdasarkan preferensi nyata pengguna lain  | Tidak bekerja optimal untuk user/buku baru (cold start) |

## Evaluation

### Metrik: Precision@K

Metrik ini digunakan untuk mengukur seberapa banyak item yang relevan dari K item yang direkomendasikan.

**Formula:**

$$
\text{Precision@K} = \frac{\text{Jumlah item relevan di Top-K}}{K}
$$

**Cara kerja:**

- Untuk setiap user, sistem menghasilkan K rekomendasi.
- Kemudian dibandingkan dengan data rating aktual (rating ≥ 7 dianggap relevan).
- Precision dihitung sebagai rasio dari buku relevan di antara Top-K hasil.

**Hasil Evaluasi:**

- Precision@5 diuji pada 10 pengguna sampel.
- Rata-rata Precision@5 untuk Content-Based Filtering: **0.04**
- Rata-rata Precision@5 untuk Collaborative Filtering: **0.42**

---
