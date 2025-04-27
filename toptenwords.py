def get_top_words(model, feature_names, n_top_words=10):
    top_words = []
    for topic_idx, topic in enumerate(model.components_):
        top_word_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words_for_topic = [feature_names[i] for i in top_word_indices]
        top_words.append(top_words_for_topic)
    return top_words

# Step 5: Retrieve top 10 words for each main cluster
top_words_per_cluster = get_top_words(nmf, feature_names, n_top_words=10)

# Step 6: Calculate top words for misc cluster
misc_docs_indices = np.where(misc_cluster)[0]
if len(misc_docs_indices) > 0:
    misc_tfidf = tfidf[misc_docs_indices].sum(axis=0).A1  # Sum across documents
    misc_top_indices = np.argsort(misc_tfidf)[:-11:-1]  # Top 10 indices
    misc_top_words = [feature_names[i] for i in misc_top_indices]
    misc_weights = misc_tfidf[misc_top_indices]
else:
    misc_top_words = []
    misc_weights = np.array([])

# Step 7: Plotting the top 10 words for each cluster including misc
all_clusters_words = top_words_per_cluster + [misc_top_words]
all_clusters_weights = [nmf.components_[i][[vectorizer.vocabulary_.get(word, 0) for word in words]] for i, words in enumerate(top_words_per_cluster)]
all_clusters_weights.append(misc_weights)

fig, axes = plt.subplots(n_topics + 1, 1, figsize=(10, 6 * (n_topics + 1)))
for i, ax in enumerate(axes):
    words = all_clusters_words[i]
    weights = all_clusters_weights[i]
    if len(words) > 0 and len(weights) == len(words):
        ax.barh(words[::-1], weights[::-1], color='skyblue')
    ax.set_title(f'Top 10 Words for Cluster {i}' if i < n_topics else 'Top 10 Words for Misc Cluster')
    ax.set_xlabel('Weight')
    ax.invert_yaxis()  # Highest weight on top

plt.tight_layout()
plt.show()
