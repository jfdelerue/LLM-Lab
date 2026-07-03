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
