// sketch.js

let qtree;

// Mise en place de la toile (setup)
function setup() {
  createCanvas(400, 400, WEBGL);
  background(255);

  let boundary = new Rectangle3D(0, 0, 0, 200, 200, 200);
  qtree = new QuadTree3D(boundary, 4);

  for (let i = 0; i < 300; i++) {
    let x = randomGaussian(width / 2, width / 8);
    let y = randomGaussian(height / 2, height / 8);
    let z = randomGaussian(50, 100);
    let p = new Point3D(x, y, z);
    qtree.insert(p);
  }
}

// Dessin de la scène (draw)
function draw() {
  background(0);
  orbitControl();
  rotateX(PI / 6);
  rotateY(PI / 6);

  qtree.show();

  stroke(0, 255, 0);
  rectMode(CENTER);
  let range = new Rectangle3D(mouseX - width / 2, mouseY - height / 2, 0, 25, 25, 25);

  if (mouseX < width && mouseY < height) {
    rect(range.x, range.y, range.z, range.w * 2, range.h * 2);
    let points = qtree.query(range);

    // Vérification que points est bien un tableau avant de l'itérer
    if (Array.isArray(points)) {
      for (let p of points) {
        strokeWeight(4);
        point(p.x, p.y, p.z);
      }
    }
  }
}






/*
// sketch.js

let qtree;

// Mise en place de la toile (setup)
function setup() {
  createCanvas(400, 400, WEBGL); // Crée une toile (canvas) de dimensions 400x400 en mode 3D (WEBGL).
  background(255); // Définit la couleur d'arrière-plan de la toile en blanc.
  let boundary = new Rectangle3D(0, 0, 0, 200, 200, 200); // Initialise une limite (boundary) en 3D avec une position initiale en (0, 0, 0) et une taille de 200x200x200.
  qtree = new QuadTree3D(boundary, 4); // Initialise un quadtree en 3D avec la limite et une capacité maximale de 4 points par noeud


  //  Insertion de points aléatoires (for loop dans setup) :
  //  Utilise une boucle pour générer 300 points aléatoires en 3D, en utilisant la fonction randomGaussian pour répartir les points autour du centre de la toile.
  //  Insère chaque point dans le quadtree créé.
  //  randomGaussian génère des valeurs centrées autour d'une moyenne

  for (let i = 0; i < 300; i++) {
    let x = randomGaussian(width / 2, width / 8);
    let y = randomGaussian(height / 2, height / 8);
    let z = randomGaussian(0, 100); // Adjust the z-coordinate range for 3D
    let p = new Point3D(x, y, z);
    qtree.insert(p);
  }
}

// Dessin de la scène (draw)
function draw() {
  background(0); // Réinitialise la toile à une couleur d'arrière-plan noire.
  orbitControl(); // Active le contrôle orbital de la souris pour permettre la rotation de la scène.
  rotateX(PI / 6); // Effectue une rotation de la scène pour un meilleur angle de vue.
  rotateY(PI / 6);

  // Affiche la structure du quadtree avec ses limites et points.
  qtree.show();

  stroke(0, 255, 0);
  rectMode(CENTER);
  let range = new Rectangle3D(mouseX - width / 2, mouseY - height / 2, 0, 25, 25, 25); // Définit une zone de recherche en 3D basée sur la position de la souris.

  if (mouseX < width && mouseY < height) {
    box(range.w * 2, range.h * 2, range.d * 2); // Dessine un cube représentant la zone de recherche.
    let points = qtree.query(range); // Effectue une requête dans le quadtree pour obtenir les points à l'intérieur de la zone de recherche.
    for (let p of points) { // Dessine les points trouvés à l'intérieur de la zone de recherche.
      strokeWeight(4);
      point(p.x, p.y, p.z);
    }
  }
}
*/
