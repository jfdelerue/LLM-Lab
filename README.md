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

L'onglet **Test protocole LLM** permet d'envoyer librement un prompt (et, si nécessaire, une image) à `/api/chat` ou `/api/generate`. Chaque essai produit dans `ollama_protocol_tests/` une trace JSON contenant la requête, les métadonnées HTTP, le JSON décodé et la réponse brute. Les images en base64 ne sont pas recopiées dans la trace : leur taille et leur empreinte SHA-256 suffisent à identifier exactement l'entrée. Ces fichiers facilitent la comparaison du comportement des modèles et l'étude des champs ou marqueurs comme `<think>`.
