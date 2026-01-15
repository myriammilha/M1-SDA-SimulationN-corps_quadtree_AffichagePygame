# Projet : N-Body Simulation with QuadTree (Barnes–Hut) - M1 SDA

Ce projet implémente une **simulation N-corps gravitationnelle** optimisée à l’aide d’un **quadtree** (algorithme de **Barnes–Hut**).

Deux versions sont fournies :
- **Python + Pygame** : simulation locale avec affichage temps réel
- **JavaScript (p5.js)** : version web pour visualisation dans un navigateur
---

## Structure du projet
```
.
├── Code_python/
│   ├── main.py        # Simulation Python finale
│   ├── tryagain.py    # Version expérimentale initiale
│   └── venv/          # Environnement virtuel Python (non versionné)
│
├── Code_js/
│   ├── sketch.js      # Simulation N-body (p5.js)
│   ├── quadtree.js    # Implémentation du quadtree en JS
│   └── index.html     # Point d’entrée web
│
└── README.md

```
---

### Principe algorithmique

La simulation repose sur l’algorithme **Barnes–Hut** :

- L’espace est subdivisé récursivement en **quadrants (quadtree)**
- Les corps lointains sont approximés par leur **centre de masse**
- La complexité passe de **O(N²)** à **O(N log N)**

---

## Technologies utilisées
* **Python 3** – logique de simulation
* **NumPy** – calculs vectoriels et numériques
* **Pygame** – affichage graphique temps réel
* **JavaScript (ES6)** – version web de la simulation
* **p5.js** – rendu graphique dans le navigateur
* **HTML5** – page de lancement web
* **Git / GitHub** – gestion de versions

---

## Version Python (Pygame)

### Prérequis
- Python ≥ 3.10
- `numpy`
- `pygame`

### Installation (recommandée avec venv)
```bash
sudo apt install python3-venv python3-full
cd Code_python
python3 -m venv venv
source venv/bin/activate
pip install numpy pygame
```

### Lancer la simulation

```bash
python3 tryagain.py
```
 OU

```bash
make run
```


### Contrôles

* Fermeture de la fenêtre : clic sur la croix
* Les particules se déplacent selon la gravité
* Le quadtree est affiché :

  * vert : feuille
  * rouge : nœud interne

---

## Version JavaScript (Web)

### Prérequis

* Un navigateur moderne
* Aucun package à installer

### Lancer la version web

Depuis le dossier `Code_js` :

```bash
cd Code_js
python3 -m http.server 8000
```

Puis ouvrir dans le navigateur :

```
http://localhost:8000
```

---

## Remarques importantes

* Les fichiers **`.js` sont indépendants** de la version Python
* `tryagain.py` est une version de travail / prototype
* La version Python reconstruit le quadtree à chaque frame pour garantir la cohérence
* L’intégration numérique utilise un schéma d’Euler explicite

---

## Cadre académique

Projet réalisé dans le cadre du **Master 1 – Structures de données**, 
illustrant l’utilisation de structures hiérarchiques pour l’optimisation algorithmique.
