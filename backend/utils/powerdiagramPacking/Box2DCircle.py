# simulate circles using box2d

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Box2D import *

import matplotlib
# matplotlib.use("Agg")


class Box2DSimulator(object):
    def __init__(self, positions, radii, attractions, attraction_magnitude=50, gravitiy_magnitude=200, iterations=0, # parameters you can change. # you need not to set iterations if you want to manually run the simulator using step()
                 density=1.0, bullet=False, time_step=0.005, # recommend not to change these
                 velocity_iterations=6, position_iterations=2): # recommend not to change these
        self.positions = positions
        self.radii = radii
        self.attractions = attractions
        self.attraction_magnitude = attraction_magnitude
        self.gravity_magnitude = gravitiy_magnitude
        self.init_gravity_magnitude = gravitiy_magnitude
        self.init_attraction_magnitude = attraction_magnitude
        self.iterations = iterations
        self.density = density
        self.bullet = bullet
        self.gravity_center = np.array([0, 0], dtype=np.float32)
        self.time_step = time_step
        self.velocity_iterations = velocity_iterations
        self.position_iterations = position_iterations
        self.world = b2World(gravity=(0, 0), doSleep=True)
        self.size_mag = 100
        self.bodies = []
        self.init_bodies()
        self.init_joints()
        ## maybe useful for debugging
        # self.applied_gravities = None
        # self.applied_attractions = None
        # self.applied_repulsions = None
        # self.attraction_potential = None
        # self.repulsion_potential = None
        # self.gravity_potential = None

    def init_bodies(self):
        self.size_mag = 1/np.min(self.radii)/1.1
        for i in range(len(self.positions)):
            # convert to float
            position = [float(self.positions[i][0]), float(self.positions[i][1])]
            position[0] *= self.size_mag
            position[1] *= self.size_mag
            body = self.world.CreateDynamicBody(position=position, angle=0, linearDamping=0, angularDamping=0,
                                                bullet=self.bullet)
            body.CreateCircleFixture(radius=self.radii[i]* self.size_mag, density=self.density, friction=0)
            # print(f'scaled_radius={self.radii[i]* self.size_mag}')
            self.bodies.append(body)
            # set gravity center
            self.gravity_center += position
        self.gravity_center /= len(self.positions)
    # def init_polygon(self):
    #     # create a polygon
    #     vertices = [(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)]
    #     shape = b2PolygonShape(vertices=vertices)
    #     body = self.world.CreateDynamicBody(position=(0, 0), angle=0, linearDamping=0, angularDamping=0, bullet=self.bullet)
    #     body.CreateFixture(shape=shape, density=self.density)
    def init_joints(self):
        # keep the distance between the connected bodies
        for (id1, id2) in self.attractions:
            body1 = self.bodies[id1]
            body2 = self.bodies[id2]
            distance = np.linalg.norm(body1.position - body2.position)
            joint = self.world.CreateDistanceJoint(bodyA=body1, bodyB=body2, anchorA=body1.position, anchorB=body2.position)
            joint.length = distance


    def apply_attractions(self):
        # attraction = [body_id1, body_id2, magnitude of force]
        for (id1, id2) in self.attractions:
            # get the direction of the force
            body1 = self.bodies[id1]
            body2 = self.bodies[id2]
            direction = body2.position - body1.position
            # normalize the direction
            norm = np.linalg.norm(direction)
            direction = direction / norm
            # apply the force
            body1.ApplyForceToCenter(direction * self.attraction_magnitude * self.size_mag, True)
            body2.ApplyForceToCenter(-direction * self.attraction_magnitude * self.size_mag, True)

    def apply_gravity(self):
        # gravities = []
        # positions = []
        # print(f'gravity_center={self.gravity_center}')
        for body in self.bodies:
            direction = self.gravity_center - body.position
            # positions.append(body.position)
            norm = np.linalg.norm(direction)
            if norm < 1e-5:
                continue
            direction = direction / norm
            body.ApplyForceToCenter(direction * self.gravity_magnitude * self.size_mag, True)
            # print(direction * self.gravity_magnitude * self.size_mag)
            # gravities.append(direction * self.gravity_magnitude * self.size_mag)
        # get the average gravity
        # print(f'positions={positions}')
        # print(f'gravities={gravities}')
        # gravities = np.array(gravities)
        # self.avg_gravity = np.mean(gravities, axis=0)
        # print(f'avg_gravity={self.avg_gravity}')

    def clear_velocities(self):
        for body in self.bodies:
            body.linearVelocity = (0, 0)
            body.angularVelocity = 0

    def step(self):
        self.world.Step(self.time_step, self.velocity_iterations, self.position_iterations)

    def run(self):
        # linear simulated annealing
        alpha_min = 0.1
        alpha = 1
        alpha_decay = (1 - (alpha_min ** (1 / self.iterations)))
        alpha_target = 0
        draw = 0
        for i in range(self.iterations):
            if draw:
                plt.cla()
            self.clear_velocities()
            self.apply_gravity()
            self.apply_attractions()
            self.step()
            alpha += alpha_decay * (alpha_target - alpha)
            self.attraction_magnitude = alpha * self.init_attraction_magnitude
            self.gravity_magnitude = alpha * self.init_gravity_magnitude
            if draw:
                positions = self.get_positions()
                if len(positions) < 10:
                    continue
                plt.xlim(-0.4,0.4)
                plt.ylim(-0.4,0.4)
                for j in range(len(positions)):
                    circle = plt.Circle(positions[j], self.radii[j])
                    plt.gca().add_patch(circle)
                plt.pause(0.005)

    def get_positions(self):
        positions = []
        for body in self.bodies:
            position = body.position.copy()
            position[0] /= self.size_mag
            position[1] /= self.size_mag
            positions.append(position)
        return positions

    def get_radii(self):
        radii = []
        for body in self.bodies:
            radii.append(body.fixtures[0].shape.radius)
        return radii

    def get_velocities(self):
        velocities = []
        for body in self.bodies:
            velocities.append(body.linearVelocity)
        velocities = np.array(velocities)
        avg_velocity = np.mean(velocities, axis=0)
        print(f'avg_velocity={avg_velocity}')
        return velocities

    def get_momentums(self):
        momentums = []
        for body in self.bodies:
            momentums.append(body.mass * body.linearVelocity)
        return momentums



if __name__ == "__main__":
    positions = np.random.rand(10, 2)*5
    radii = np.random.rand(10)
    # build delauany graph
    from scipy.spatial import Delaunay
    tri = Delaunay(positions)
    # get the edges
    edges = []
    for simplex in tri.simplices:
        edges.append((simplex[0], simplex[1]))
        edges.append((simplex[1], simplex[2]))
        edges.append((simplex[2], simplex[0]))
    # build attractions
    attractions = []
    # remove duplicates
    edges = list(set(edges))
    for edge in edges:
        attractions.append((edge[0], edge[1]))
    # build simulator
    simulator = Box2DSimulator(positions, radii, attractions,  attraction_magnitude=100, gravitiy_magnitude=100)
    # run simulator
    iterations = 1000
    for i in range(iterations):
        plt.cla()
        simulator.apply_gravity()
        simulator.apply_attractions()
        simulator.step()
        # get positions
        positions = simulator.get_positions()
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        # draw circles
        for j in range(len(positions)):
            circle = plt.Circle(positions[j], radii[j])
            plt.gca().add_patch(circle)
        plt.pause(0.01)

