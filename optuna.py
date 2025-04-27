def objective(trial):
    # Suggest hyperparameters
    n_components = trial.suggest_int('n_components', 2, 20)
    alpha_W = trial.suggest_float('alpha_W', 0.0, 1.0)
    beta_loss = trial.suggest_categorical('beta_loss', ['frobenius', 'kullback-leibler'])
    max_iter = trial.suggest_int('max_iter', 100, 500)
    
    # Initialize and fit NMF
    nmf = NMF(
        n_components=n_components,
        alpha_W=alpha_W,
        beta_loss=beta_loss,
        max_iter=max_iter,
        random_state=42
    ).fit(tfidf)
    
    # Extract top words per topic for coherence
    top_words_per_topic = []
    for topic in nmf.components_:
        top_words = [feature_names[i] for i in topic.argsort()[-10:][::-1]]
        top_words_per_topic.append(top_words)
    
    # Calculate coherence (C_V)
    coherence_model = CoherenceModel(
        topics=top_words_per_topic,
        texts=texts,
        dictionary=dictionary,
        coherence='c_v'
    )
    coherence_score = coherence_model.get_coherence()
    
    return coherence_score

# Create a study and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
