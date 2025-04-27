# Step 4: Function to get top N words for each topic/cluster based on NMF weights
def get_top_words(model, feature_names, n_top_words=10):
    top_words = []
    for topic_idx, topic in enumerate(model.components_):
        top_word_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words_for_topic = [feature_names[i] for i in top_word_indices]
        top_words.append(top_words_for_topic)
    return top_words

# Step 5: Retrieve top 10 words for each main cluster
top_words_per_cluster = get_top_words(nmf, feature_names, n_top_words=10)

# Step 6: Calculate document assignments for main clusters
cluster_assignments = []
for i in range(len(documents)):
    if misc_cluster[i]:
        cluster_assignments.append(n_topics)  # Assign to misc cluster (index after main topics)
    else:
        cluster_assignments.append(main_cluster_assignments[i])

# Step 7: Calculate top words and proportions for misc cluster
misc_docs_indices = np.where(misc_cluster)[0]
if len(misc_docs_indices) > 0:
    misc_tfidf = tfidf[misc_docs_indices].sum(axis=0).A1  # Sum across documents for ranking
    misc_top_indices = np.argsort(misc_tfidf)[:-11:-1]  # Top 10 indices
    misc_top_words = [feature_names[i] for i in misc_top_indices]
else:
    misc_top_words = []

# Step 8: Calculate proportions of documents containing each word per cluster
all_clusters_words = top_words_per_cluster + [misc_top_words]
all_clusters_proportions = []
total_clusters = n_topics + 1  # Include misc cluster

for cluster_id in range(total_clusters):
    if cluster_id < n_topics:
        cluster_docs_indices = np.where(np.array(cluster_assignments) == cluster_id)[0]
    else:
        cluster_docs_indices = misc_docs_indices
    
    total_docs_in_cluster = len(cluster_docs_indices)
    if total_docs_in_cluster == 0:
        proportions = np.zeros(10)  # No documents, proportions are 0
    else:
        words = all_clusters_words[cluster_id]
        proportions = []
        for word in words:
            if word in vectorizer.vocabulary_:
                word_idx = vectorizer.vocabulary_[word]
                docs_with_word = np.sum(tfidf[cluster_docs_indices, word_idx].toarray() > 0)
                proportion = docs_with_word / total_docs_in_cluster if total_docs_in_cluster > 0 else 0
                proportions.append(proportion)
            else:
                proportions.append(0)
    all_clusters_proportions.append(np.array(proportions))

# Step 9: Plotting the top 10 words for each cluster with proportions
fig, axes = plt.subplots(total_clusters, 1, figsize=(10, 6 * total_clusters))
for i, ax in enumerate(axes):
    words = all_clusters_words[i]
    proportions = all_clusters_proportions[i]
    if len(words) > 0 and len(proportions) == len(words):
        ax.barh(words[::-1], proportions[::-1], color='skyblue')
    ax.set_title(f'Top 10 Words for Cluster {i}' if i < n_topics else 'Top 10 Words for Misc Cluster')
    ax.set_xlabel('Proportion of Documents with Word')
    ax.set_xlim(0, 1)  # Proportion range is 0 to 1
    ax.invert_yaxis()  # Highest proportion on top

plt.tight_layout()
plt.show()
