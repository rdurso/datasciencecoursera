def get_top_words(model, feature_names, n_top_words=10):
    top_words = []
    for topic_idx, topic in enumerate(model.components_):
        top_word_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words_for_topic = [feature_names[i] for i in top_word_indices]
        top_words.append(top_words_for_topic)
    return top_words

# Step 5: Retrieve top 10 words for each main cluster
top_words_per_cluster = get_top_words(nmf, feature_names, n_top_words=10)

# Step 6: Plotting the top 10 words for each cluster
fig, axes = plt.subplots(n_topics, 1, figsize=(10, 6 * n_topics))
for i, ax in enumerate(axes):
    words = top_words_per_cluster[i]
    weights = nmf.components_[i][[vectorizer.vocabulary_[word] for word in words]]
    ax.barh(words[::-1], weights[::-1], color='skyblue')
    ax.set_title(f'Top 10 Words for Cluster {i}')
    ax.set_xlabel('Weight')
    ax.invert_yaxis()  # Highest weight on top

plt.tight_layout()
plt.show()
