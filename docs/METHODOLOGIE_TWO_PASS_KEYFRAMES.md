# Méthodologie « two-pass keyframes »

Ce document décrit l'option **two-pass keyframes** (deux passes sur images clés) de
façon indépendante de l'interface Streamlit. Le même pipeline peut être intégré à
un logiciel desktop, un service web, un moteur de montage ou un traitement batch.

## 1. Principe

Envoyer toutes les images d'une vidéo en haute résolution à un modèle multimodal
coûte beaucoup de mémoire, de temps et de contexte. La méthode sépare donc le
traitement en deux appels visuels :

1. **Passe 1 — sélection (D1)** : parcourir la vidéo avec des vignettes légères et
   demander au modèle quels instants contiennent l'information visuelle utile ;
2. **Extraction (D2)** : relire la vidéo source aux instants retenus et produire
   seulement ces images en haute qualité ;
3. **Passe 2 — analyse (D3)** : envoyer les images haute qualité, les raisons de
   leur sélection et le transcript au modèle pour obtenir l'analyse finale.

D2 n'est pas un appel au LLM : c'est le pont déterministe entre les deux passes.
La sélection réduit le volume envoyé en haute résolution, sans perdre la vue
chronologique globale nécessaire au choix des moments importants.

## 2. Données d'entrée et paramètres

Entrées minimales :

- la vidéo source ;
- un modèle multimodal acceptant plusieurs images ordonnées ;
- idéalement un transcript horodaté, facultatif si l'analyse est purement visuelle.

Paramètres équivalents à ceux de cette application :

| Paramètre | Valeur par défaut | Rôle |
|---|---:|---|
| `thumbnail_interval_sec` | 2 s | Pas temporel de l'échantillonnage D1 |
| `thumbnail_max_frames` | 48 | Plafond du nombre de vignettes D1 |
| `thumbnail_largest_side_px` | 400 px | Plus grand côté des vignettes D1 |
| `thumbnail_jpeg_quality` | 85 | Qualité JPEG des vignettes D1 |
| `two_pass_max_keyframes` | 12 | Nombre maximal de sélections utilisées |
| `two_pass_high_quality_largest_side_px` | 1280 px | Plus grand côté des images D3 |
| `two_pass_high_quality_jpeg_quality` | 90 | Qualité JPEG des images D3 |
| `two_pass_context_before_sec` | 0 s | Image supplémentaire avant chaque instant |
| `two_pass_context_after_sec` | 0 s | Image supplémentaire après chaque instant |

La durée couverte par l'échantillonnage est approximativement :

```text
min(durée_vidéo, (thumbnail_max_frames - 1) × thumbnail_interval_sec)
```

Pour couvrir une longue vidéo, il faut augmenter le plafond, augmenter l'intervalle,
ou découper la vidéo en fenêtres qui exécutent chacune D1 avant une consolidation.

## 3. Préparation commune

### 3.1 Vignettes basse résolution

Lire une image aux instants `i × intervalle`, pour `i = 0..max_frames-1`, tant que
l'instant appartient à la vidéo. Conserver pour chaque vignette :

```json
{
  "frame_index": 12,
  "timestamp_sec": 22.0,
  "path": "thumb_000012_22.00s.jpg",
  "width": 400,
  "height": 225
}
```

`frame_index` est ici l'indice de la **vignette échantillonnée**, pas le numéro de
frame du flux vidéo. Le timestamp en secondes est l'identifiant temporel de
référence et doit accompagner explicitement les images dans le prompt si l'API ne
transmet pas leurs métadonnées.

Le redimensionnement conserve le ratio et ne doit pas agrandir une image déjà plus
petite :

```text
facteur = min(1, largest_side_px / max(largeur, hauteur))
```

### 3.2 Transcript

Un transcript horodaté aide D1 à repérer les instants où l'image peut confirmer ou
modifier le sens des paroles. Dans l'implémentation actuelle, le contexte D1 est
réduit à 3 000 caractères en gardant le début et la fin ; D3 utilise la limite
configurable `transcript_context_max_chars` avec la même stratégie.

Pour une meilleure transposition, sélectionner plutôt les segments proches des
timestamps candidats ou résumer le transcript par fenêtres. Cela évite qu'un
événement situé au milieu disparaisse lors de la réduction.

## 4. Passe 1 (D1) : sélectionner les instants

Envoyer au modèle, dans l'ordre chronologique :

- toutes les vignettes D1 ;
- la correspondance explicite `frame_index → timestamp_sec` ;
- le transcript réduit ;
- l'objectif métier (par exemple identifier les gestes, objets, textes ou
  changements de scène utiles) ;
- une obligation de retourner uniquement du JSON conforme au contrat ci-dessous.

### Contrat de sortie recommandé

```json
{
  "selected_keyframes": [
    {
      "frame_index": 12,
      "timestamp_sec": 22.0,
      "priority": "high",
      "reason": "Un objet nouveau apparaît dans les mains de la personne.",
      "suggested_focus": "Lire l'étiquette et identifier l'objet."
    }
  ]
}
```

Sémantique des champs :

- `frame_index` permet de rattacher le choix à la vignette présentée ;
- `timestamp_sec` pilote D2 et doit être un nombre fini compris dans la vidéo ;
- `priority` aide à tronquer intelligemment si le modèle dépasse la limite ;
- `reason` explique pourquoi l'instant est utile ;
- `suggested_focus` dirige l'attention de D3 vers le détail à examiner.

Le logiciel cible doit parser strictement le JSON, puis **valider** le schéma. Il
est préférable de recalculer le timestamp à partir de `frame_index` lorsque les deux
valeurs ne correspondent pas, plutôt que de faire confiance à un timestamp inventé
par le modèle. Rejeter les indices absents, timestamps non finis ou hors limites,
supprimer les doublons, trier par priorité puis limiter à `max_keyframes`. Si la
sortie est invalide, effectuer une unique demande de correction JSON ou appliquer
une sélection de secours déterministe (changements de scène, puis répartition
uniforme).

## 5. D2 : réextraire depuis la source

Pour chaque sélection conservée, former la liste :

```text
timestamp - context_before, timestamp, timestamp + context_after
```

Puis :

1. borner chaque instant à `[0, durée_vidéo]` ;
2. dédupliquer les timestamps (l'application les arrondit au centième de seconde) ;
3. rechercher l'instant dans la **vidéo source**, jamais dans la vignette D1 ;
4. décoder la frame, conserver le ratio, réduire son plus grand côté à la taille HQ ;
5. encoder en JPEG et mémoriser le timestamp réellement décodé si le backend peut
   le fournir.

Une recherche vidéo peut atterrir sur une frame voisine à cause des keyframes du
codec. Pour une précision forte, demander au décodeur une recherche au PTS puis
décoder en avant jusqu'au timestamp visé. Le contexte avant/après est utile pour un
geste bref, mais peut produire jusqu'à trois images par sélection : dimensionner la
limite d'images D3 en conséquence.

## 6. Passe 2 (D3) : analyser les images haute qualité

Envoyer dans un seul message multimodal, ou en lots suivis d'une synthèse :

- les images HQ en ordre chronologique ;
- leur table `index → timestamp` ;
- le JSON D1 validé, notamment `reason` et `suggested_focus` ;
- le transcript réduit ou les segments temporels voisins ;
- une consigne demandant, pour chaque image, ce qu'elle montre, ce qu'elle ajoute
  au transcript, les détails/textes visibles et son niveau d'utilité ;
- une demande de synthèse finale répondant à l'objectif métier.

Les images encodées en base64 ne contiennent pas leur nom de fichier : sans table
de timestamps dans le texte du message, le modèle peut difficilement associer son
observation à un instant fiable. Cette association doit donc être explicite.

## 7. Pseudo-code portable

```text
function analyse_two_pass(video, transcript, config):
    metadata = inspect_video(video)

    thumbnails = []
    for i from 0 to config.thumbnail_max_frames - 1:
        ts = i * config.thumbnail_interval_sec
        if ts > metadata.duration: break
        frame = decode_at(video, ts)
        if frame is missing: break
        thumbnails.append({
            frame_index: i + 1,
            timestamp_sec: ts,
            image: resize_and_encode(frame, 400, quality=85)
        })

    d1_prompt = selection_instruction
              + timestamp_table(thumbnails)
              + reduce_transcript(transcript, 3000)
    d1_raw = multimodal_call(d1_prompt, images(thumbnails))
    selection = validate_and_normalize_json(d1_raw, thumbnails, metadata)
    selection = prioritize_deduplicate_and_limit(selection, 12)

    hq_frames = []
    for candidate in selection:
        for ts in [candidate.ts - before, candidate.ts, candidate.ts + after]:
            ts = clamp(ts, 0, metadata.duration)
            if rounded(ts, 2) was already extracted: continue
            frame = decode_at(video, ts)
            if frame exists:
                hq_frames.append({
                    timestamp_sec: ts,
                    image: resize_and_encode(frame, 1280, quality=90)
                })

    if hq_frames is empty: return controlled_error_or_fallback()

    d3_prompt = analysis_instruction
              + timestamp_table(hq_frames)
              + validated_selection_json(selection)
              + relevant_transcript(transcript, hq_frames)
    return multimodal_call(d3_prompt, images(hq_frames))
```

## 8. Adaptation à une autre API

Avec Ollama, chaque passe utilise `/api/chat`, un message utilisateur contenant
`content` et un tableau `images` en base64, avec `stream: false`. Pour une autre API,
seule la couche `multimodal_call` change : URLs d'images, octets multipart ou blocs
image peuvent remplacer le base64. Le pipeline doit rester découplé en quatre
composants testables : décodeur vidéo, sélection D1, validateur/planificateur D2 et
analyse D3.

Prévoir également :

- un timeout long, une limite de taille et un nombre maximal d'images par appel ;
- un traitement par lots si le modèle refuse toutes les images, suivi d'une
  consolidation textuelle ;
- la journalisation du modèle, des paramètres, prompts, réponses brutes, timestamps
  et empreintes SHA-256 des images, sans recopier le base64 dans les logs ;
- l'idempotence : un cache indexé par empreinte vidéo + configuration + version des
  prompts évite de refaire D1 et D2 ;
- une politique claire si le modèle ne possède pas la capacité vision.

## 9. Tests d'acceptation

Une réimplémentation est fonctionnellement équivalente si elle vérifie au minimum :

1. une vidéo courte produit des vignettes ordonnées avec timestamps exacts ;
2. une sortie D1 mal formée, hors limites ou dupliquée ne provoque jamais une
   extraction arbitraire ;
3. `max_keyframes` est appliqué avant l'ajout du contexte avant/après ;
4. D2 repart bien de la vidéo originale et respecte taille, ratio et qualité ;
5. les timestamps négatifs sont ramenés à zéro et les doublons sont supprimés ;
6. D3 reçoit les images dans le même ordre que la table temporelle ;
7. transcript vide, absence de sélection et échec de décodage ont un comportement
   explicite et testable ;
8. le budget d'images, le timeout et les erreurs du fournisseur sont observables ;
9. sur un jeu vidéo de référence, D3 apporte davantage de détails utiles qu'une
   analyse des seules vignettes, pour un coût inférieur à l'envoi de toutes les
   frames en haute résolution.

## 10. Limites à connaître

- D1 ne peut pas sélectionner un événement absent de l'échantillonnage ; un
  intervalle de deux secondes peut manquer une action très brève.
- Un modèle génératif peut produire un JSON valide mais factuellement incohérent :
  la validation et la normalisation sont obligatoires en production.
- La première et la dernière moitié d'un transcript ne représentent pas toujours
  son contenu central ; une sélection temporelle des segments est préférable.
- « Keyframe » désigne ici une **image sémantiquement sélectionnée**, pas forcément
  une I-frame du codec vidéo.
- La qualité JPEG et la résolution doivent être ajustées au modèle : au-delà de sa
  résolution effective, une image plus grande augmente surtout le coût.
