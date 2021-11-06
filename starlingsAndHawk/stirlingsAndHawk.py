import os
import glob

import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import cv2

L  = 32
N  = 10000
R  = 1
RB = 4
V0 = 2
A  = .15
DT = 1

NO_FRAMES = 200
FPS = 6
DPI = 300

WITH_PREDAOTR = False

PATH_PREFIX = os.path.dirname(os.path.realpath(__file__))

class Birds():
    def __init__(self, withPredator = False):
        self.positions = (np.random.uniform(0, L, N)
                          + 1j*np.random.uniform(0, L, N))
        self.directions = np.random.uniform(-np.pi, np.pi, N)
        self.velocities = V0 * np.exp(1j*self.directions)

        positions2D = np.column_stack((self.positions.real,
                                       self.positions.imag))
        neighboursTree = cKDTree(positions2D, boxsize=[L,L])
        self.neighboursDistances = np.array(neighboursTree.sparse_distance_matrix(
            neighboursTree,
            max_distance=R,
            output_type='coo_matrix'
            ).todense()).ravel()
        self.neighboursDistances = np.reshape(self.neighboursDistances, (N, N))

        self.withPredator = withPredator
        if self.withPredator:
            self.predatorPos = (np.random.uniform(0, L)
                                + 1j*np.random.uniform(0, L))
            distancesToVictims = np.abs(self.positions - self.predatorPos)
            closestVictimPos = self.positions[
                # returns index of closest victim
                np.argmin(distancesToVictims)
                ]
            self.predatorDir = np.angle(closestVictimPos
                                        - self.predatorPos)
            self.predatorVel = V0 * np.exp(1j*self.predatorDir)

    def new_directions(self):
        self.neighboursDistances[self.neighboursDistances > 0] = 1
        sum_velocities = np.add(
            np.array(np.dot(self.neighboursDistances, self.velocities)).ravel(),
            self.velocities
            )
        new_directions = np.angle(sum_velocities) + A*np.random.uniform(-np.pi,
                                                                        np.pi,
                                                                        N)
        return new_directions

    def update(self):
        self.directions = self.new_directions()
        new_positions   = np.array(self.positions + DT*self.velocities)
        new_velocities  = np.array(V0*np.exp(1j*self.directions))

        # Periodic conditions
        new_positions[new_positions.real > L] -= L
        new_positions[new_positions.imag > L] -= 1j*L
        new_positions[new_positions.real < 0] += L
        new_positions[new_positions.imag < 0] += 1j*L

        if self.withPredator:
            # Periodic conditions
            predPosX = (self.predatorPos.real + DT*self.predatorVel.real) % L
            predPosY = (self.predatorPos.imag + DT*self.predatorVel.imag) % L
            predPosX += L if predPosX < 0 else 0
            predPosY += L if predPosY < 0 else 0
            self.predatorPos = predPosX + 1j*predPosY
            threatenedBirds = np.where(
                # Birds in radius RB around predator
                np.abs(self.positions - self.predatorPos) < RB
            )
            threatenedDir = np.angle(self.positions[threatenedBirds]
                                     - self.predatorPos)
            self.directions[threatenedBirds] = threatenedDir
            new_velocities[threatenedBirds] = np.array(V0*np.exp(1j*threatenedDir))
            
            distancesToVictims = np.abs(self.positions - self.predatorPos)
            closestVictimPos = self.positions[
                # returns index of closest victim
                np.argmin(distancesToVictims)
                ]
            self.predatorDir = np.angle(closestVictimPos
                                        - self.predatorPos)
            self.predatorVel = V0 * np.exp(1j*self.predatorDir)

        self.positions  = np.array(new_positions, dtype=complex)
        self.velocities = np.array(new_velocities, dtype=complex)

        positions2D = np.column_stack((self.positions.real, self.positions.imag))
        neighboursTree = cKDTree(positions2D, boxsize=[L,L])
        self.neighboursDistances = np.array(neighboursTree.sparse_distance_matrix(
            neighboursTree,
            max_distance=R,
            output_type='coo_matrix'
            ).todense()).ravel()
        self.neighboursDistances = np.reshape(self.neighboursDistances, (N, N))

    def save_frame(self):
        X = self.positions.real
        Y = self.positions.imag
        U = self.velocities.real
        V = self.velocities.imag
        plt.clf()
        plt.quiver(X, Y, U, V, self.directions)
        if self.withPredator:
            plt.quiver(self.predatorPos.real,
                    self.predatorPos.imag,
                    self.predatorVel.real,
                    self.predatorVel.imag)
        plt.xlim([0, L])
        plt.ylim([0, L])

        i = 0
        nameIndex = '000'
        while os.path.exists(os.path.join(
            PATH_PREFIX,
            r"starlingsSimulation\\frame%s.png" % nameIndex
            )):
            i += 1
            nameIndex = (3-len(str(i)))*'0' + str(i)
        # leading zeros in frame name
        nameIndex = (3-len(str(i)))*'0' + str(i)
        figName = os.path.join(
            PATH_PREFIX,
            r"starlingsSimulation\\frame%s.png" % nameIndex
        )
        plt.savefig(figName, dpi=DPI)

    def show_frame(self):
        X = self.positions.real
        Y = self.positions.imag
        U = self.velocities.real
        V = self.velocities.imag
        plt.quiver(X, Y, U, V, self.directions)
        if self.withPredator:
            plt.quiver(self.predatorPos.real,
                    self.predatorPos.imag,
                    self.predatorVel.real,
                    self.predatorVel.imag)
        plt.xlim([0, L])
        plt.ylim([0, L])
        plt.show()

def create_video():
    image_folder = os.path.join(PATH_PREFIX, 'starlingsSimulation')
    video_name = os.path.join(PATH_PREFIX, 'video.avi')

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, FPS, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

def main():
    # Empty the frames directory    
    files = glob.glob(os.path.join(PATH_PREFIX, 'starlingsSimulation/*'))
    for f in files:
        os.remove(f)
    
    # Simulate and save
    swarm = Birds(WITH_PREDAOTR)
    swarm.save_frame()
    for _ in range(NO_FRAMES):
        swarm.update()
        swarm.save_frame()
    create_video()

if __name__=='__main__':
    main()