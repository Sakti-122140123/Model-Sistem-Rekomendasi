import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dataset.csv")

print("Jumlah baris:", len(df))
print("Jumlah user unik:", df['User-ID'].nunique())
print("Jumlah buku unik:", df['ISBN'].nunique())

plt.figure(figsize=(8, 4))
sns.countplot(x='Book-Rating', data=df, palette='viridis')
plt.title("Distribusi Rating Buku")
plt.xlabel("Rating")
plt.ylabel("Jumlah")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

top_books = df['Book-Title'].value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_books.values, y=top_books.index, palette='magma')
plt.title("10 Buku yang Paling Sering Dirating")
plt.xlabel("Jumlah Rating")
plt.ylabel("Judul Buku")
plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer

df = df[df["Book-Rating"] > 0]

df = df.drop_duplicates(subset=["User-ID", "ISBN"])

df["Book-Title"] = df["Book-Title"].str.lower()
df["Book-Author"] = df["Book-Author"].str.lower()

top_users = df["User-ID"].value_counts().head(1000).index
df = df[df["User-ID"].isin(top_users)]

books = df.drop_duplicates(subset='ISBN').head(20000).reset_index(drop=True)

books['content'] = books['Book-Title'] + ' ' + books['Book-Author']

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['content'])

indices = pd.Series(books.index, index=books['Book-Title']).drop_duplicates()

print("Jumlah baris setelah preparation:", len(df))
df.head()

from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

def recommend_cbf(title, top_n=5):
    if title not in indices:
        return f"Judul '{title}' tidak ditemukan dalam subset data."
    idx = indices[title]
    sim_scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_indices = sim_scores.argsort()[-top_n-1:-1][::-1]
    return books.loc[sim_indices, ['Book-Title', 'Book-Author']]

from sklearn.metrics.pairwise import cosine_similarity

pivot = df.pivot_table(index='User-ID', columns='Book-Title', values='Book-Rating').fillna(0)

user_sim = cosine_similarity(pivot)
user_sim_df = pd.DataFrame(user_sim, index=pivot.index, columns=pivot.index)

def recommend_cf(user_id, top_n=5):
    if user_id not in user_sim_df.index:
        return f"User-ID {user_id} tidak ditemukan."
    
    similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:6]
    weighted_scores = pivot.loc[similar_users.index].mean().sort_values(ascending=False)
    recommended_books = weighted_scores.head(top_n).index
    return pd.DataFrame({'Recommended Book': recommended_books})

print("📚 Rekomendasi berdasarkan Content-Based Filtering:")
print(recommend_cbf("harry potter and the chamber of secrets"))

print("\n📚 Rekomendasi berdasarkan Collaborative Filtering:")
print(recommend_cf(277427))

def precision_at_k_cf(user_id, top_k=5, threshold=7):
    if user_id not in user_sim_df.index:
        return None
    recommended = recommend_cf(user_id, top_k)
    if isinstance(recommended, str):
        return None
    recommended_books = recommended['Recommended Book'].tolist()
    actual_rated = df[(df['User-ID'] == user_id) & (df['Book-Rating'] >= threshold)]
    relevant_books = actual_rated['Book-Title'].tolist()
    relevan_count = sum([1 for book in recommended_books if book in relevant_books])
    return relevan_count / top_k

def precision_at_k_cbf(user_id, top_k=5, threshold=7):
    rated_books = df[(df['User-ID'] == user_id) & (df['Book-Rating'] >= threshold)]
    if rated_books.empty:
        return None

    sample_title = rated_books.iloc[0]['Book-Title']
    recommended = recommend_cbf(sample_title, top_k)
    if isinstance(recommended, str):
        return None
    recommended_books = recommended['Book-Title'].tolist()

    user_books = df[df['User-ID'] == user_id]
    relevant_books = user_books[user_books['Book-Rating'] >= threshold]['Book-Title'].tolist()
    relevan_count = sum([1 for book in recommended_books if book in relevant_books])
    return relevan_count / top_k

sample_users = df['User-ID'].unique()[:10]

cf_scores = [precision_at_k_cf(user, 5) for user in sample_users if precision_at_k_cf(user, 5) is not None]
cf_avg = sum(cf_scores) / len(cf_scores)

cbf_scores = [precision_at_k_cbf(user, 5) for user in sample_users if precision_at_k_cbf(user, 5) is not None]
cbf_avg = sum(cbf_scores) / len(cbf_scores)

print(f"🎯 Rata-rata Precision@5 - Content-Based Filtering: {cbf_avg:.2f}")
print(f"🎯 Rata-rata Precision@5 - Collaborative Filtering: {cf_avg:.2f}")