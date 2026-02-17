# HR Policy RAG Assistant
## Assistant RH basé sur RAG (Retrieval-Augmented Generation)

Ce projet implémente un assistant RH intelligent basé sur une architecture RAG (Retrieval-Augmented Generation) permettant d’interroger des règlements intérieurs d’établissements publics (hôpitaux / universités) et d’obtenir des réponses contextualisées, sourcées et non hallucinées.

L’objectif est de démontrer comment combiner la recherche sémantique (SentenceTransformers), l’indexation vectorielle (FAISS), et la génération contrôlée via un modèle de langage, le tout exposé à travers une API Flask.

## Motivation du projet

Les règlements intérieurs des établissements publics sont souvent longs, juridiquement structurés et complexes à consulter manuellement. Ils couvrent des sujets tels que :

- les congés et autorisations d’absence  
- le temps de travail  
- les obligations professionnelles  
- la discipline  
- la sécurité et l’hygiène  
- les droits des agents  

Ce projet propose un assistant conversationnel capable de répondre à des questions RH à partir de ces documents, en citant explicitement ses sources et en refusant de répondre lorsque l’information n’est pas présente dans le corpus.

L’architecture est conçue pour limiter les hallucinations grâce à une génération strictement conditionnée par le contexte extrait des documents.

## Corpus documentaire

Le projet utilise des règlements intérieurs d’établissements publics français à des fins de démonstration technique.

Les documents incluent notamment :

- règlements intérieurs hospitaliers  
- livrets d’accueil professionnels  
- chartes d’usage du système d’information  
- dispositions relatives aux congés, au temps de travail, à la discipline et aux obligations des personnels  

Le corpus est utilisé uniquement dans un cadre académique et expérimental.

## Architecture technique

L’assistant repose sur une architecture RAG en trois étapes principales.

### 1. Nettoyage et découpage

- Extraction du texte à partir des fichiers PDF  
- Nettoyage du texte (normalisation, suppression des artefacts)  
- Découpage en segments sémantiques (chunks)  
- Enrichissement avec des métadonnées (document source, page, section)  

### 2. Embeddings et indexation

- Modèle d’embeddings : sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  
- Vectorisation de chaque chunk  
- Normalisation des vecteurs  
- Indexation via FAISS (IndexFlatIP) pour recherche par similarité cosinus  

Cette étape permet une recherche rapide et scalable des passages les plus pertinents.

### 3. Génération contrôlée

Pipeline d’inférence :

1. L’utilisateur pose une question  
2. Les chunks les plus similaires sont récupérés via FAISS  
3. Un filtrage par seuil de similarité est appliqué  
4. Un bloc de contexte est construit à partir des extraits retenus  
5. Le modèle de langage génère une réponse à partir de ce contexte  

Le prompt impose les contraintes suivantes :

- répondre uniquement à partir du contexte fourni  
- ne pas inventer d’information  
- citer explicitement les sources utilisées  

### Exemple de réponse

```json
{
  "answer": "Les congés annuels sont définis à l’article 7 du règlement intérieur...",
  "sources": [
    "reglement-interieur.pdf (page 69)",
    "livret-d-accueil.pdf (page 5)"
  ]
}
```
## Fonctionnalités principales

- Recherche multi-documents  
- Filtrage par seuil de similarité  
- Citation automatique des sources  
- Refus de réponse si information absente  
- Architecture modulaire  
- API Flask prête pour intégration ou déploiement  

## Stack technique

- Python  
- Flask  
- SentenceTransformers  
- FAISS  
- NumPy  
- OpenAI API
- Jupyter Notebooks  

## Sécurité et bonnes pratiques

- Les clés API ne sont jamais versionnées  
- Les réponses sont strictement conditionnées au contexte extrait  
- Les artefacts d’index sont séparés des documents sources  
- Push protection GitHub activée  

## Conclusion

Ce projet démontre comment concevoir un assistant RH basé sur des documents réglementaires réels à l’aide d’une architecture RAG robuste. Il met en évidence l’ingénierie de données textuelles longues, la recherche sémantique vectorielle et la génération contrôlée afin de produire des réponses fiables et traçables.

