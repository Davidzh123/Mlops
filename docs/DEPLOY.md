# Déploiement sur VPS (Oracle Cloud / DigitalOcean)

Déploiement de la stack du projet — **MLflow + API FastAPI + frontend Streamlit** — avec
`docker compose`. (Ce projet n'utilise ni MySQL ni Airflow : 2–4 Go de RAM suffisent.)

> ⚠️ Sécurité : l'API, MLflow et le frontend sont exposés **sans authentification**. Ce
> déploiement est prévu pour une **démo de cours**, pas pour de la production. N'exposez les
> ports que le temps de l'évaluation, de préférence restreints à votre adresse IP.

## 1. Créer le VPS

- **Oracle Cloud Always Free** : instance Compute, Ubuntu 22.04, shape `VM.Standard.A1.Flex`
  (ARM), 2 OCPU / 6–12 Go RAM. Télécharger la clé SSH.
- **DigitalOcean** (GitHub Student Pack, 200 $ de crédit) : Droplet Ubuntu 22.04, 4 Go RAM,
  authentification par clé SSH.

Notez l'**IP publique** de la machine.

## 2. Se connecter en SSH

```bash
ssh ubuntu@VOTRE_IP        # Oracle
ssh root@VOTRE_IP          # DigitalOcean
```

## 3. Installer Docker, git et make

```bash
sudo apt-get update
sudo apt-get install -y git make ca-certificates curl
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker
docker --version && docker compose version && make --version
```

## 4. Cloner le dépôt

```bash
git clone https://github.com/Davidzh123/Mlops.git
cd Mlops
```

## 5. Récupérer le jeu de données

Le CSV n'est pas versionné : il faut le télécharger depuis Kaggle (token requis).

```bash
mkdir -p ~/.kaggle
mv ~/kaggle.json ~/.kaggle/kaggle.json      # déposez votre token sur le VPS
chmod 600 ~/.kaggle/kaggle.json
make data                                   # -> data/raw/bank_fraud.csv
```

> Sans ce fichier, le service `train` ne trouvera pas les données. (Alternative démo sans
> Kaggle : `uv run python scripts/make_sample_data.py` pour un petit échantillon synthétique.)

## 6. Lancer la stack

```bash
make docker-up                 # build + démarre mlflow, api, frontend
make docker-run                # entraîne -> écrit model.joblib dans le volume partagé
docker compose restart api     # l'API recharge le modèle
```

## 7. Ouvrir les ports

| Port | Service |
|------|---------|
| 8000 | API FastAPI |
| 8501 | Frontend Streamlit |
| 5000 | MLflow UI |

(Ne pas exposer d'autres ports.)

**Pare-feu du fournisseur**
- Oracle : VCN → Security Lists → règles Ingress TCP pour 8000, 8501, 5000 (source votre IP de préférence).
- DigitalOcean : si Cloud Firewall actif, mêmes ports en entrée.

**Pare-feu système (Oracle Ubuntu uniquement)**
```bash
sudo iptables -I INPUT -p tcp --dport 8000 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 8501 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 5000 -j ACCEPT
sudo netfilter-persistent save
```
DigitalOcean (si `ufw` actif) :
```bash
sudo ufw allow 8000,8501,5000/tcp
```

## 8. Accéder aux services

- API (docs) : `http://VOTRE_IP:8000/docs`
- Frontend : `http://VOTRE_IP:8501`
- MLflow : `http://VOTRE_IP:5000`

## 9. Gérer et arrêter

```bash
docker compose ps              # état des services
docker compose logs -f api     # suivre les logs d'un service
make docker-down               # arrêter (conserve les volumes)
docker compose down -v         # arrêter ET effacer les volumes
```

## Dépannage

- **503 sur /predict** : aucun modèle dans le volume. Lancez `make docker-run` puis
  `docker compose restart api`.
- **Un port ne répond pas** : vérifiez les DEUX pare-feu (fournisseur + système). Testez en
  local sur le VPS : `curl http://localhost:8000/health`.
- **`docker: permission denied`** : reconnectez-vous en SSH après `usermod -aG docker`.
- **Build lent / manque de mémoire** : les images installent toutes les dépendances ML ;
  privilégiez 4 Go de RAM.
