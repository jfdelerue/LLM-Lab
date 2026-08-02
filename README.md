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

Pour les analyses A/B/C, l'application utilise `/api/chat`, ce qui laisse Ollama appliquer le template propre au modèle sélectionné. Elle interroge également `/api/show` : les modèles qui annoncent la capacité `vision` peuvent recevoir les images des cas A et C, tandis que les modèles texte restent utilisables pour le cas B. Un transcript vide est refusé afin d'éviter qu'un modèle réponde simplement qu'il attend le dialogue.

Sous chaque résultat A, B, C et dans les étapes D et de comparaison, l'expander **Log exact des appels LLM** montre l'URL, le modèle, le prompt, les options, l'heure et les métadonnées HTTP de chaque appel. Une zone de texte distincte affiche pour chaque appel la totalité de la réponse HTTP brute du LLM, sans tronquer ni normaliser son contenu. Pour ne pas saturer la page, les longues chaînes base64 des images envoyées sont remplacées dans l'affichage par leur taille et leur empreinte SHA-256.

Certains modèles raisonnants remplissent d'abord `message.thinking`. S'ils atteignent la limite `num_predict` avant de produire `message.content`, Ollama renvoie `done_reason: "length"` et la réponse finale est vide. L'application affiche alors une erreur explicite conseillant d'augmenter `num_predict` ou de réduire le transcript; le raisonnement incomplet reste disponible dans le log brut.

L'onglet **Test protocole LLM** permet d'envoyer librement un prompt (et, si nécessaire, une image) à `/api/chat` ou `/api/generate`. Chaque essai produit dans `ollama_protocol_tests/` une trace JSON contenant la requête, les métadonnées HTTP, le JSON décodé et la réponse brute. Les images en base64 ne sont pas recopiées dans la trace : leur taille et leur empreinte SHA-256 suffisent à identifier exactement l'entrée. Ces fichiers facilitent la comparaison du comportement des modèles et l'étude des champs ou marqueurs comme `<think>`.
