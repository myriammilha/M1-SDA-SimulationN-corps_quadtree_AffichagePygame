// quadtree3d.js

class Point3D {
  constructor(x, y, z) {
    this.x = x;
    this.y = y;
    this.z = z;
  }
}

class Rectangle3D {
  constructor(x, y, z, w, h, d) {
    this.x = x;
    this.y = y;
    this.z = z;
    this.w = w;
    this.h = h;
    this.d = d;
  }

  contains(point) {
    console.log("Checking containment:", point);
    return (
      point.x >= this.x - this.w &&
      point.x < this.x + this.w &&
      point.y >= this.y - this.h &&
      point.y < this.y + this.h &&
      point.z >= this.z - this.d &&
      point.z < this.z + this.d
    );
  }

  intersects(range) {
    return !(
      range.x - range.w > this.x + this.w ||
      range.x + range.w < this.x - this.w ||
      range.y - range.h > this.y + this.h ||
      range.y + range.h < this.y - this.h ||
      range.z - range.d > this.z + this.d ||
      range.z + range.d < this.z - this.d
    );
  }
}

class QuadTree3D {
  constructor(boundary, n) {
    this.boundary = boundary;
    this.capacity = n;
    this.points = [];
    this.divided = false;
  }

  subdivide() {
    let x = this.boundary.x;
    let y = this.boundary.y;
    let z = this.boundary.z;
    let w = this.boundary.w;
    let h = this.boundary.h;
    let d = this.boundary.d;

    let ne = new Rectangle3D(x + w / 2, y - h / 2, z + d / 2, w / 2, h / 2, d / 2);
    this.northeast = new QuadTree3D(ne, this.capacity);

    let nw = new Rectangle3D(x - w / 2, y - h / 2, z + d / 2, w / 2, h / 2, d / 2);
    this.northwest = new QuadTree3D(nw, this.capacity);

    let se = new Rectangle3D(x + w / 2, y + h / 2, z + d / 2, w / 2, h / 2, d / 2);
    this.southeast = new QuadTree3D(se, this.capacity);

    let sw = new Rectangle3D(x - w / 2, y + h / 2, z + d / 2, w / 2, h / 2, d / 2);
    this.southwest = new QuadTree3D(sw, this.capacity);

    this.divided = true;
  }

  insert(point) {
    if (!this.boundary.contains(point)) {
      return false;
    }

    if (this.points.length < this.capacity) {
      this.points.push(point);
      return true;
    } else {
      if (!this.divided) {
        this.subdivide();
      }

      if (this.northeast.insert(point)) return true;
      if (this.northwest.insert(point)) return true;
      if (this.southeast.insert(point)) return true;
      if (this.southwest.insert(point)) return true;
    }
  }

  query(range, found) {
    if (!found) {
      found = [];
    }

    if (!this.boundary.intersects(range)) {
      return;
    }
    else {
      for (let p of this.points) {
        if (range.contains(p)) {
          found.push(p);
        }
      }
      if (this.divided) {
        this.northwest.query(range, found);
        this.northeast.query(range, found);
        this.southwest.query(range, found);
        this.southeast.query(range, found);
      }
    }

    return found;
  }

  show() {
    stroke(255);
    noFill();
    strokeWeight(1);
    rectMode(CENTER);
    rect(
      this.boundary.x,
      this.boundary.y,
      this.boundary.z,
      this.boundary.w * 2,
      this.boundary.h * 2
    );

    for (let p of this.points) {
      strokeWeight(2);
      point(p.x, p.y, p.z);
    }

    if (this.divided) {
      this.northeast.show();
      this.northwest.show();
      this.southeast.show();
      this.southwest.show();
    }
  }
}
