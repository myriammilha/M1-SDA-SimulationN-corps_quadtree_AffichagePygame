from numpy import *
from numpy import zeros, array, sqrt
from random import random,seed
from time import time
from math import pi, cos, sin
import random
import pygame
import numpy as np


pygame.init()
screen_size = (1000, 800)
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption("N-Body Simulation")
clock = pygame.time.Clock()



seed(323)

G = 1.0 # Gravitational Constant
THRESHOLD = 1 # Number of bodies after which to subdivide Quad
MAXDEPTH = 10
THETA = 0.5 # Barnes-Hut ratio of accuracy
ETA = 0.5 # Softening factor


NUM_CHECKS = 0 # Counter

class QuadTree:
    """ Class container for for N points stored as a 2D Quadtree """
    root = None
    def __init__(self, bbox, N, theta = THETA):
        self.bbox = bbox
        self.N = N
        self.theta = theta
        self.root = Quad(bbox)
     
    

    def find_quad_of_body(self, body_idx, node):
        """ Find the quad that contains the body with the given index. """
        if node is None:
            return None

        if body_idx in node.bods:
            return node

        if not node.leaf:
      
            for child in node.children:
                if child is not None and child.bbox.inside(POS[body_idx]):
                    return self.find_quad_of_body(body_idx, child)
        return None

    def has_moved_significantly(self, body_idx, old_quad):

        if old_quad is None:
            return True  # Consider it as significantly moved
        return not old_quad.bbox.inside(POS[body_idx])

    def find_new_quad(self, body_idx, start_node):

        if start_node.bbox.inside(POS[body_idx]):
            if start_node.leaf:
                return start_node
            else:
                for child in start_node.children:
                    if child and child.bbox.inside(POS[body_idx]):
                        return self.find_new_quad(body_idx, child)
        else:
            if start_node == self.root:
                return None 


    def move_body(self, body_idx, old_quad, new_quad):

        old_quad.remove_body(body_idx)
        

        old_quad.update_mass_and_com(body_idx)


        if old_quad.is_empty() and old_quad != self.root:
            old_quad.collapse()
        
        # Add the body to the new quad
        new_quad.addBody(body_idx, new_quad.depth)

        # Update center of mass and mass for the new quad after addition
        new_quad.update_mass_and_com(body_idx)

    def update_tree(self):
        for body_idx in range(N):
            old_quad = self.find_quad_of_body(body_idx, self.root)

            if old_quad is None:
                # The body is outside the root quad, we ignore it for now
                continue

            if self.has_moved_significantly(body_idx, old_quad):
                new_quad = self.find_new_quad(body_idx, self.root)
                print(new_quad)
                if new_quad:
                    self.move_body(body_idx, old_quad, new_quad)


        
    def reset(self):
        self.root = None


    def updateSys(self, dt):
        self.calculateBodyAccels()
        global VEL
        VEL += ACC * dt

    def calculateBodyAccels(self):
        # Update ACC table based on current POS
        for k in range(self.N):
            # Check if the body is within the root quad
            if self.find_quad_of_body(k, self.root) is not None:
                ACC[k] = self.calculateBodyAccel(k)
            else:
                # If the body is outside the root quad, set its acceleration to zero
                ACC[k] = zeros(2, dtype=float)
    def calculateBodyAccel(self, bodI):
        return self.calculateBodyAccelR(bodI, self.root)

    def calculateBodyAccelR(self, bodI, node):
        # Calculate acceleration on body I
        # key difference is that body is ignored in calculations
        if node is None:
            return zeros(2, dtype=float)
        acc = zeros(2,dtype=float)
        if (node.leaf):
            # print "Leaf"
            # Leaf node, no children
            for k in node.bods:
                if k != bodI: # Skip same body
                    acc += getForce( POS[bodI] ,1.0,POS[k],MASS[k])
        else:
            s = max( node.bbox.sideLength )
            d = node.center - POS[bodI]
            r = sqrt(d.dot(d))
            # print "s/r = %g/%g = %g" % (s,r,s/r)
            if (r > 0 and s/r < self.theta):
                # Far enough to do approximation
                acc += getForce( POS[bodI] ,1.0, node.com, node.mass)
            else:
                # Too close to approximate, recurse down tree
                for k in range(4):
                    if node.children[k] != None:
                        acc += self.calculateBodyAccelR(bodI, node.children[k])
        # print "ACC : %s" % acc
        return acc

    
def getForce(p1,m1,p2,m2):
    # need to d
    global NUM_CHECKS
    d = p2-p1
    r = sqrt(d.dot(d)) + ETA
    f = array( d * G*m1*m2 / r**3 )
    NUM_CHECKS += 1
    return f


class Quad:
    """ A rectangle of space, contains point bodies """
    def __init__(self,bbox,bod = None,depth=0):
        self.bbox = bbox
        self.center = bbox.center
        self.leaf = True # Whether is a parent or not
        self.depth = depth
        if bod != None: # want to capture 0 int also
            self.setToBody(bod)
            self.N = 1
        else:
            self.bods = []
            self.mass = 0.
            self.com = array([0,0], dtype=float)
            self.N = 0
            
        self.children = [None]*4 # top-left,top-right,bot-left,bot-right
    def remove_body(self, body_idx):
       """ Remove a body from this quad """
       if body_idx in self.bods:
          self.bods.remove(body_idx)

    def addBody(self, idx, depth):
        # Check if the body already exists in this quad
        if idx in self.bods:
            return  # Body is already in this quad, no need to add again

        # Add the body to this quad
        if len(self.bods) > 0 or not self.leaf:
            # This quad is not empty
            if depth >= MAXDEPTH:
                self.bods.append(idx)
            else:
                # Subdivide the quad if necessary
                if not self.leaf:
                    # The quad is already subdivided, just add the body to the appropriate child
                    quadIdx = self.getQuadIndex(idx)
                    if self.children[quadIdx] is None:
                        # Create a new child quad if it doesn't exist
                        subBBox = self.bbox.getSubQuad(quadIdx)
                        self.children[quadIdx] = Quad(subBBox, depth=depth+1)
                    # Recursively add the body to the child quad
                    self.children[quadIdx].addBody(idx, depth+1)
                else:
                    # Move existing bodies to children and add the new body
                    subBods = [idx] + self.bods
                    self.bods = []
                    self.leaf = False
                    for bod in subBods:
                        quadIdx = self.getQuadIndex(bod)
                        if self.children[quadIdx] is None:
                            subBBox = self.bbox.getSubQuad(quadIdx)
                            self.children[quadIdx] = Quad(subBBox, depth=depth+1)
                        self.children[quadIdx].addBody(bod, depth+1)

        else:
            # This quad is empty, add the body directly
            self.setToBody(idx)

        # Update the center of mass and mass of the quad
        self.update_mass_and_com(idx)

    def update_mass_and_com(self, idx):
        bodyMass = MASS[idx]
        if self.mass == 0:
            self.com = POS[idx].copy()
        else:
            self.com = (self.com * self.mass + POS[idx] * bodyMass) / (self.mass + bodyMass)
        self.mass += bodyMass
        self.N += 1  # Increment the number of bodies

    
        
    def setToBody(self,idx):
        self.bods = [idx]
        self.mass = float( MASS[idx].copy() )
        self.com  = POS[idx].copy()

    def getQuadIndex(self,idx):
        return self.bbox.getQuadIdx(POS[idx])
    def is_empty(self):
        return len(self.bods) == 0 and all(child is None or child.is_empty() for child in self.children)
    def collapse(self):
        # This should remove all children of this quad if it is an internal node
        # and there are no bodies within it and its children.
        self.children = [None]*4
        self.leaf = True
        

class BoundingBox:
    def __init__(self,box,dim=2):
        assert(dim*2 == len(box))
        self.box = array(box,dtype=float)
        self.center = array( [(self.box[2]+self.box[0])/2, (self.box[3]+self.box[1])/2] , dtype=float)
        self.dim = dim
        self.sideLength = self.max() - self.min()

    def max(self):
        return self.box[self.dim:]
    def min(self):
        return self.box[:self.dim]
    def inside(self,p):
        # p = [x,y]
        if any(p < self.min()) or any(p > self.max()):
            return False
        else:
            return True
    def getQuadIdx(self,p):
        # y goes up
        # 0 1
        # 2 3
        if p[0] > self.center[0]: # x > mid
            if p[1] > self.center[1]: # y > mid
                return 1
            else:
                return 3
        else:
            if p[1] > self.center[1]: # y > mid
                return 0
            else:
                return 2
    def getSubQuad(self,idx):
        # 0 1
        # 2 3
        # [x  y x2 y2]
        #  0  1  2  3
        b = array([None,None,None,None])
        if idx % 2 == 0:
            # Even #, left half
            b[::2] = [self.box[0], self.center[0]] # x - midx
        else:
            b[::2] = [self.center[0], self.box[2]] # midx - x2
        if idx < 2:
            # Upper half (0 1)
            b[1::2] = [self.center[1], self.box[3]] # midy - y2
        else:
            b[1::2] = [self.box[1], self.center[1]] # y - midy
        return BoundingBox(b,self.dim)


def draw_bodies_pygame():
    for i in range(N):
        if BOUNDS.inside(POS[i]):
            x, y = convert_to_screen_coords(POS[i])
            pygame.draw.circle(screen, (0, 0, 0), (x, y), int(MASS[i] * 2))

def draw_bbox_pygame(node):
    if node is not None:
        x0, y0 = convert_to_screen_coords(node.bbox.min())
        x1, y1 = convert_to_screen_coords(node.bbox.max())
        pygame.draw.rect(screen, (0, 0, 255), (x0, y0, x1 - x0, y1 - y0), 1)
        if node.leaf:
            color = (0, 255, 0)  # Green for leaf nodes
        else:
            color = (255, 0, 0)  # Red for parent nodes
        pygame.draw.rect(screen, color, (x0, y0, x1 - x0, y1 - y0), 1)
        for child in node.children:
            draw_bbox_pygame(child)

def convert_to_screen_coords(p):
    screen_pos = (p - BOUNDS.min()) / (BOUNDS.max() - BOUNDS.min()) * np.array(screen_size)
    return np.trunc(screen_pos).astype(int)
def draw_quadtree(node, surface):
    if node is None:
        return

    top_left = convert_to_screen_coords(node.bbox.min())
    bottom_right = convert_to_screen_coords(node.bbox.max())
    width = bottom_right[0] - top_left[0]
    height = bottom_right[1] - top_left[1]

    if node.leaf:
        color = (0, 255, 0)  # Green for leaf nodes
    else:
        color = (255, 0, 0)  # Red for parent nodes

    # Debugging: Print the coordinates and dimensions
    #print(f"Drawing quad: Top Left: {top_left}, Width: {width}, Height: {height}")

    pygame.draw.rect(surface, color, (top_left[0], top_left[1], width, height), 1)

    for child in node.children:
        draw_quadtree(child, surface)


N = 10
BOUNDS = BoundingBox([0,0,10,10])



# Global variables
# 2D Position
MASS = zeros(N,dtype=float)
POS = zeros((N,2),dtype=float)
VEL = zeros((N,2),dtype=float)
ACC = zeros((N,2),dtype=float)
for i in range(N):
    MASS[i] = 1 
    POS[i] = BOUNDS.min() + array([random.random(), random.random()]) * BOUNDS.sideLength


# Calculate the center of mass for the entire system
total_mass = MASS.sum()
center_of_mass = np.average(POS, axis=0, weights=MASS)
# Maximum speed calculation
DT =0.001
T = 0
max_speed = min(BOUNDS.sideLength) / (2 * DT)
# Define the central point of the simulation (where the horizontal and vertical lines intersect)
central_point = np.array([BOUNDS.sideLength[0] / 2, BOUNDS.sideLength[1] / 2])
sys = QuadTree(BOUNDS, N)
# After initializing the QuadTree
for i in range(N):
    sys.root.addBody(i, 0)  # Add each body to the QuadTree

# Calculate the initial angles for each body
angles = np.arctan2(POS[:, 1] - central_point[1], POS[:, 0] - central_point[0])

angular_speed = 0.01
# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
##    # Update the positions based on the angles
    for i in range(N):
        radius = np.linalg.norm(POS[i] - central_point)
        angles[i] += angular_speed  # Update the angle

        # Update POS[i] to be on the circle
        POS[i][0] = central_point[0] + radius * np.cos(angles[i])
        POS[i][1] = central_point[1] + radius * np.sin(angles[i])

    # Update the simulation
    sys.update_tree()
    sys.updateSys(DT)

    # Drawing
    screen.fill((255, 255, 255))
    draw_bodies_pygame()
    draw_quadtree(sys.root, screen)
    #draw_bbox_pygame(sys.root)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()

