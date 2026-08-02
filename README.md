# Video LLM Lab local avec Ollama

Application Streamlit locale pour comparer des stratégies d'analyse vidéo avec un modèle multimodal servi par Ollama.

## Lancement

```bash
pip install -r requirements.txt
streamlit run app.py
```

Par défaut l'application utilise `http://localhost:11434` et le modèle `qwen2.5vl:7b`.
La limite de téléversement Streamlit est configurée à 500 Mo pour accepter de grandes vidéos.
Les paramètres sont sauvegardés dans `video_llm_lab_settings.json`, ou dans le chemin défini par `VIDEO_LLM_LAB_SETTINGS`.

Au démarrage, l'application interroge `/api/tags` et propose les modèles Ollama installés dans le panneau **Paramètres**. Le bouton **Actualiser les modèles Ollama** permet de relancer le scan après un `ollama pull`.

Après le chargement d’une vidéo, le bouton général **Exécuter toutes les phases 1 à 7** enchaîne automatiquement l’application des paramètres, l’extraction des vignettes, la transcription, les analyses A/B/C, le traitement two-pass des keyframes et la comparaison finale. Les boutons de chaque onglet restent disponibles pour relancer une phase séparément.

Pour les analyses A/B/C, l'application utilise `/api/chat`, ce qui laisse Ollama appliquer le template propre au modèle sélectionné. Elle interroge également `/api/show` : les modèles qui annoncent la capacité `vision` peuvent recevoir les images des cas A et C, tandis que les modèles texte restent utilisables pour le cas B. Un transcript vide est refusé afin d'éviter qu'un modèle réponde simplement qu'il attend le dialogue.

Les consignes des analyses A, B, C ainsi que celles des deux appels D1/D3 du mode two-pass sont visibles et modifiables avant leur envoi au LLM. Sous chaque résultat A, B, C et dans les étapes D et de comparaison, l'expander **Log exact des appels LLM** montre l'URL, le modèle, le prompt effectivement composé (avec transcript et contexte automatiques), les options, l'heure et les métadonnées HTTP de chaque appel. Une zone de texte distincte affiche pour chaque appel la totalité de la réponse HTTP brute du LLM, sans tronquer ni normaliser son contenu. Pour ne pas saturer la page, les longues chaînes base64 des images envoyées sont remplacées dans l'affichage par leur taille et leur empreinte SHA-256.

Certains modèles raisonnants remplissent d'abord `message.thinking`. S'ils atteignent la limite `num_predict` avant de produire `message.content`, Ollama renvoie `done_reason: "length"` et la réponse finale est vide. L'application affiche alors une erreur explicite conseillant d'augmenter `num_predict` ou de réduire le transcript; le raisonnement incomplet reste disponible dans le log brut.

## Réimplémenter l'option « two-pass keyframes »

La méthodologie complète, indépendante de Streamlit et d'Ollama, est décrite dans
[`docs/METHODOLOGIE_TWO_PASS_KEYFRAMES.md`](docs/METHODOLOGIE_TWO_PASS_KEYFRAMES.md).
Le document détaille le contrat JSON entre les deux passes, l'extraction temporelle,
les appels au modèle, un pseudo-code portable, les contrôles de robustesse et les
critères permettant de valider une implémentation dans un autre logiciel.

L'onglet **Test protocole LLM** permet d'envoyer librement un prompt (et, si nécessaire, une image) à `/api/chat` ou `/api/generate`. Chaque essai produit dans `ollama_protocol_tests/` une trace JSON contenant la requête, les métadonnées HTTP, le JSON décodé et la réponse brute. Les images en base64 ne sont pas recopiées dans la trace : leur taille et leur empreinte SHA-256 suffisent à identifier exactement l'entrée. Ces fichiers facilitent la comparaison du comportement des modèles et l'étude des champs ou marqueurs comme `<think>`.
