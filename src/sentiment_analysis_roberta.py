
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt
from IPython.display import display, HTML

# Configuration du modèle
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Charger les données
df = pd.read_csv('path/to/your/csv/file')

# Fonction pour calculer les scores de polarité avec RoBERTa en utilisant PyTorch
def polarity_scores_roberta(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {'score_neg': 0.0, 'score_neu': 0.0, 'score_pos': 0.0}

    # Tokenisation et encodage du texte
    encoding = tokenizer.encode_plus(
        text,
        max_length=512,  # Adapter en fonction de la longueur maximale du modèle
        add_special_tokens=True,
        return_tensors='pt',
        padding='max_length',
        truncation=True
    )

    # Récupération des tenseurs encodés
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    token_type_ids = encoding['token_type_ids'] if 'token_type_ids' in encoding else None

    # Passage à travers le modèle
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        scores = torch.softmax(output.logits[0], dim=0).numpy()

    # Création du dictionnaire de scores
    scores_dict = {
        'score_neg': scores[0],
        'score_neu': scores[1],
        'score_pos': scores[2]
    }
    
    return scores_dict

# Calcul des scores de sentiment pour chaque avis et stockage dans un dictionnaire
sentiment_scores = {}
for index, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing Reviews"):
    text = row['review_text']
    roberta_scores = polarity_scores_roberta(text)
    sentiment_scores[text] = roberta_scores

# Création du DataFrame à partir du dictionnaire de scores
results_df = pd.DataFrame(sentiment_scores).T

# Ajout de la colonne 'sentiment_label' au DataFrame
results_df['sentiment_label'] = results_df.apply(lambda x: np.argmax(x), axis=1)
results_df['sentiment_label'] = results_df['sentiment_label'].replace({0: 'Négatif', 1: 'Neutre', 2: 'Positif'})

# Affichage du tableau des avis et des sentiments associés
display(HTML(results_df.to_html()))

# Résumé de la distribution des sentiments
sentiment_summary = results_df['sentiment_label'].value_counts()

# Affichage du résumé textuel de la distribution des sentiments
print("\nRésumé de la distribution des sentiments:")
print("-" * 30)
for sentiment, count in sentiment_summary.items():
    print(f"{sentiment.capitalize()}: {count} avis")

# Visualisation de la distribution des sentiments
plt.figure(figsize=(8, 6))
plt.bar(sentiment_summary.index, sentiment_summary.values, color=['green', 'red', 'grey'])
plt.xlabel('Sentiment')
plt.ylabel('Nombre')
plt.title('Distribution des Sentiments')
plt.show()

# Données de scores de sentiment
sentiment_scores = results_df[['score_neg', 'score_neu', 'score_pos']].values

# Liste des étiquettes pour les avis
avis_labels = [f"{i+1}" for i in range(len(results_df))]

# Couleurs pour chaque catégorie de sentiment
colors = ['red', 'grey', 'green']  # Rouge pour négatif, Gris pour neutre, Vert pour positif

# Création du graphique
plt.figure(figsize=(12, 8))

# Position des barres
positions = np.arange(len(results_df))

# Tracé des barres pour chaque catégorie de sentiment
bottom = None
for i, (sentiment, label) in enumerate(zip(sentiment_scores.T, avis_labels)):
    plt.bar(positions, sentiment, label=label, color=colors[i], bottom=bottom)
    if bottom is None:
        bottom = sentiment
    else:
        bottom += sentiment

# Configuration de l'axe x
plt.xlabel('Avis')
plt.ylabel('Score de Sentiment')
plt.title('Scores de Sentiment pour les Avis')
plt.xticks(positions, avis_labels)  # Étiquettes d'avis sur l'axe x

# Ajout d'une légende
plt.legend(title='Sentiment', labels=['Négatif', 'Neutre', 'Positif'])

# Affichage du graphique
plt.show()
