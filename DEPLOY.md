# Hébergement gratuit (Jungle in English — ML Web)

L’application est prête pour un **conteneur Docker** : modèles **démo** générés au build (`bootstrap_demo`). Pour vos vrais `.joblib` / `.pkl`, ajoutez-les au dépôt **avant** le build ou montez un volume (selon la plateforme).

---

## Option 1 — [Render](https://render.com) (gratuit, veille après inactivité)

1. Créez un compte et connectez le dépôt GitHub contenant ce projet.
2. **New → Web Service** → choisissez le dépôt.
3. **Runtime : Docker**, fichier : `Dockerfile` à la racine.
4. Laissez **Health Check Path** : `/api/health` (déjà dans `render.yaml` si vous importez le Blueprint).
5. Déployez. URL du type `https://jungle-ml-web.onrender.com`.

**Blueprint** : dans le dashboard Render → *Blueprints* → lier le repo ; Render lit `render.yaml`.

**Limite** : 512 Mo RAM — si LightGBM plante au démarrage, passez à un plan payant ou allégez les modèles.

---

## Option 2 — [Fly.io](https://fly.io) (quota gratuit, puis facturation légère possible)

1. Installez la CLI : https://fly.io/docs/hands-on/install-flyctl/
2. Dans le dossier du projet : `fly auth login` puis `fly launch --dockerfile Dockerfile` (adaptez le nom `app` dans `fly.toml` si besoin).
3. `fly deploy`

L’app écoute sur le port **8000** (voir `fly.toml` → `internal_port`).

---

## Option 3 — [Railway](https://railway.app)

1. New Project → **Deploy from GitHub** → ce repo.
2. Détection **Dockerfile** automatique ; sinon définissez la commande :

   `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

3. Variable d’environnement `PORT` : Railway la fournit en général automatiquement.

---

## Vérifications après déploiement

- Page d’accueil : `/`
- Santé : `/api/health`
- Documentation API : `/docs`

---

## Fichiers ajoutés

| Fichier        | Rôle                                      |
|----------------|-------------------------------------------|
| `Dockerfile`   | Build Python + génération des artefacts |
| `.dockerignore`| Image plus légère                         |
| `render.yaml`  | Déploiement Render (plan gratuit)         |
| `fly.toml`     | Déploiement Fly.io                        |

En local Docker (daemon démarré) :

```bash
docker build -t jungle-ml-web .
docker run -p 8000:8000 -e PORT=8000 jungle-ml-web
```

Puis ouvrir http://localhost:8000 .
