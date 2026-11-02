# Générateur d'excuses bidons

Un petit GPT que j'ai entraîné de zéro pour inventer des excuses absurdes.
On lui donne un début de phrase (ou rien du tout) et il écrit une excuse.

Quelques excuses écrites par le modèle :

> Excuse-moi pour ce matin, mais mon oreiller a remplacé ma patience par une pastèque.

> Je n'ai pas appelé ma mère car une poule ambitieuse a mangé ma carte de bus.

> Je n'ai pas pu rendre mon devoir parce que j'étais occupé à garder le moral de ma plante (c'est la troisième fois ce mois-ci).

> Je ne suis pas venu au sport car un hérisson a fait une crise existentielle, comme chaque mardi.

## L'idée

En cours, on a vu comment marche un Transformer et comment GPT génère du texte token par token.
Je voulais refaire un GPT moi-même, en partant d'une feuille blanche, mais sur un sujet un peu drôle.
Je me suis inspiré de la vidéo d'Andrej Karpathy
[« Let's build GPT: from scratch, in code, spelled out »](https://www.youtube.com/watch?v=kCc8FmEb1nY)
et j'ai suivi le même chemin : partir d'un modèle tout simple et ajouter les briques une par une.

La différence avec la vidéo : lui découpe le texte lettre par lettre, moi j'ai utilisé le tokenizer de GPT-2 (`tiktoken`), donc mes tokens sont des bouts de mots.

## Comment j'ai fait, de A à Z

### 1. Les données

Il n'existe pas de base d'excuses drôles en français (j'ai cherché sur Kaggle, Hugging Face et GitHub).
J'ai donc généré les excuses avec **Claude** : à peu près 10 000 excuses, une par ligne, dans `data/excuses.txt`.
Ça fait environ 1 million de caractères, soit 388 201 tokens.

### 2. Le texte devient des nombres

Un modèle ne sait calculer qu'avec des nombres. `tiktoken` (le découpeur de GPT-2) transforme le texte en numéros (`encode`) et les numéros en texte (`decode`).
Son dictionnaire a 50 257 morceaux. Par exemple « mon chat a mangé mes devoirs » donne 9 tokens : `mon` `␣chat` `␣a` `␣mang` `é` `␣mes` `␣dev` `oir` `s`.

Ensuite je coupe : 90 % du texte pour l'entraînement (train), 10 % pour vérifier qu'il n'apprend pas par cœur (val).

### 3. Les lots

À chaque pas, je prends 16 morceaux de 64 tokens au hasard dans le texte (`get_batch`).
`x` c'est le morceau, `y` c'est le même morceau décalé d'un cran : sous chaque token, le token qui vient vraiment après.
Le texte sert donc à la fois d'exercice et de corrigé.

J'ai choisi 64 parce que les excuses font 38 tokens en moyenne et 71 au maximum.

### 4. Le modèle, brique par brique (`transformer_model.py`)

J'ai ajouté les briques une à une, en entraînant 300 pas à chaque fois pour comparer :

| Étape | Ce que j'ai ajouté | Loss après 300 pas |
|---|---|---|
| Bigramme | une fiche (embedding) par token + une couche qui donne un score à chaque token suivant | 2,21 |
| Une tête d'attention | la position (le « dossard ») + une tête : chaque token regarde ceux d'avant | 3,55 |
| 4 têtes + feed forward | 4 têtes de 32 au lieu d'une, et une petite couche pour « réfléchir » | 2,73 |
| 4 blocs complets | connexions résiduelles (`x = x + ...`), LayerNorm, dropout, et 4 blocs empilés | **1,08** |

Ce qui m'a surpris : ajouter de l'attention a d'abord fait **pire** que le bigramme.
La sortie de l'attention remplaçait la fiche du token au lieu de l'enrichir.
C'est le résiduel (`x = x + self.sa(self.ln1(x))`) qui règle ça : la fiche passe tout droit et chaque couche ajoute juste des corrections.

Le modèle final :

| Réglage | Valeur |
|---|---|
| tokenizer | GPT-2 (50 257 tokens) |
| taille d'une fiche (`n_embd`) | 128 |
| têtes par bloc (`n_head`) | 4 |
| blocs (`n_layer`) | 4 |
| fenêtre (`block_size`) | 64 tokens |
| dropout | 0,1 |
| nombre de poids | 13 716 049 |

Presque tous les poids (12,9 millions) sont dans la table des fiches et la dernière couche, à cause des 50 257 tokens de GPT-2. Les 4 blocs n'en font que 790 000.

### 5. L'entraînement

AdamW, learning rate 0,001, 3 000 pas, sur mon PC sans carte graphique (environ 0,6 seconde par pas, donc une demi-heure).
Tous les 250 pas, je mesure la loss sur train et sur val (moyenne de 50 lots, dropout coupé) :

| Pas | Train | Val |
|---|---|---|
| 0 | 10,99 | 11,00 |
| 250 | 1,25 | 1,28 |
| 500 | 0,69 | 0,72 |
| 1 000 | 0,53 | 0,59 |
| 2 000 | 0,50 | 0,59 |
| 2 999 | **0,48** | **0,55** |

Au départ, 10,99 c'est le hasard (ln 50 257 ≈ 10,8). L'essentiel est appris en 500 pas, ensuite ça baisse doucement.
À la fin, l'écart entre train et val commence à s'ouvrir un peu : c'est le début du par-cœur, donc 3 000 pas c'est un bon moment pour s'arrêter.
La loss est très basse parce que les excuses se ressemblent beaucoup entre elles (beaucoup ont la même structure).

### 6. Sauvegarder

Je sauvegarde les poids et les réglages du modèle dans `gpt_excuses.pt`, pour ne pas devoir réentraîner à chaque fois.

### 7. Générer une excuse

La fonction `excuse()` part d'un retour à la ligne (le début d'une excuse), ajoute le début qu'on lui donne, puis :
- prend les scores du modèle pour le token suivant ;
- les divise par la **température** (plus bas = plus sage, plus haut = plus fou) ;
- garde seulement les 50 meilleurs (**top-k**) pour éviter les choix absurdes ;
- tire un token au sort et recommence ;
- s'arrête quand il écrit un retour à la ligne : l'excuse est finie.

## Les fichiers

| Fichier | Ce qu'il y a dedans |
|---|---|
| `excuses.ipynb` | tout le projet : données, tokenizer, lots, entraînement, sauvegarde, génération (les cellules « test » sont les étapes intermédiaires, en commentaire) |
| `transformer_model.py` | le modèle : `BigramLanguageModel`, `Head`, `MultiHeadAttention`, `FeedForward`, `Block`, `GPTLanguageModel` |
| `data/excuses.txt` | les ~10 000 excuses |
| `gpt_excuses.pt` | le modèle entraîné (~55 Mo) |
| `app.py` | une petite interface pour tester vite fait |

## Tester

```
pip install -r requirements.txt
python app.py
```

Puis ouvrir l'adresse affichée (`http://127.0.0.1:7860`).

`app.py`, c'est juste une petite interface simple (avec Gradio) pour pouvoir essayer le générateur sans ouvrir le notebook. Ce n'est pas le but du projet : le vrai travail est dans `excuses.ipynb` et `transformer_model.py`.
Le curseur « niveau de mauvaise foi », c'est la température.

## Les limites

- Il fait encore quelques fautes (« un pont s'est faisait sieste »).
- Il recombine surtout les tournures qu'il a vues : comme les données ont des structures qui reviennent souvent, ses excuses aussi.
- Le tokenizer de GPT-2 a été fait pour l'anglais : il coupe souvent les accents en deux tokens, donc le français lui coûte plus cher.
