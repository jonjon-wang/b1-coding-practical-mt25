from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from .terrain import generate_reference_and_limits
import pandas as pd

"""Simulated submarine dynamics and mission utilities.

Provides:

Submarine: simple discrete-time plant (x/y, vertical dynamics).
Mission: reference depth and cave limits for trajectory planning.
ClosedLoop: runs a controller against the Submarine and records a Trajectory.
"""

class Submarine:
    """Simple submarine plant for discrete-time simulation.

    Attributes:
    mass (float): mass used for vertical acceleration calculation.
    drag (float): damping on vertical velocity.
    actuator_gain (float): gain from control input to force.
    dt (float): timestep (s).
    pos_x, pos_y (float): position states.
    vel_x, vel_y (float): velocity states.
    """
    def __init__(self):

        self.mass = 1
        self.drag = 0.1
        self.actuator_gain = 1

        self.dt = 1 # Time step for discrete time simulation

        self.pos_x = 0
        self.pos_y = 0
        self.vel_x = 1 # Constant velocity in x direction
        self.vel_y = 0


    def transition(self, action: float, disturbance: float):
        self.pos_x += self.vel_x * self.dt
        self.pos_y += self.vel_y * self.dt

        force_y = -self.drag * self.vel_y + self.actuator_gain * (action + disturbance)
        acc_y = force_y / self.mass
        self.vel_y += acc_y * self.dt

    def get_depth(self) -> float:
        return self.pos_y
    
    def get_position(self) -> tuple:
        return self.pos_x, self.pos_y
    
    def reset_state(self):
        self.pos_x = 0
        self.pos_y = 0
        self.vel_x = 1
        self.vel_y = 0
    
class Trajectory:
    """Stores an (T,2) array of (x,y) positions and plotting helpers.

    position: np.ndarray shaped (T, 2) where columns are x and y.
    """
    def __init__(self, position: np.ndarray):
        self.position = position  
        
    def plot(self):
        plt.plot(self.position[:, 0], self.position[:, 1])
        plt.show()

    def plot_completed_mission(self, mission: Mission):
        x_values = np.arange(len(mission.reference))
        min_depth = np.min(mission.cave_depth)
        max_height = np.max(mission.cave_height)

        plt.fill_between(x_values, mission.cave_height, mission.cave_depth, color='blue', alpha=0.3)
        plt.fill_between(x_values, mission.cave_depth, min_depth*np.ones(len(x_values)), 
                         color='saddlebrown', alpha=0.3)
        plt.fill_between(x_values, max_height*np.ones(len(x_values)), mission.cave_height, 
                         color='saddlebrown', alpha=0.3)
        plt.plot(self.position[:, 0], self.position[:, 1], label='Trajectory')
        plt.plot(mission.reference, 'r', linestyle='--', label='Reference')
        plt.legend(loc='upper right')
        plt.show()

@dataclass
class Mission:
    """Mission data: reference depth and cave geometry.

    Fields:
    reference: 1D array of desired depths (float).
    cave_height: 1D array of cave top heights.
    cave_depth: 1D array of cave bottom depths.
    """
    reference: np.ndarray
    cave_height: np.ndarray
    cave_depth: np.ndarray

    @classmethod
    def random_mission(cls, duration: int, scale: float):
        (reference, cave_height, cave_depth) = generate_reference_and_limits(duration, scale)
        return cls(reference, cave_height, cave_depth)

    @classmethod
    def from_csv(cls, file_name: str):
        """Load mission from a CSV file with columns: reference, cave_height, cave_depth"""
        df = pd.read_csv(file_name)
        reference = df['reference'].to_numpy()
        cave_height = df['cave_height'].to_numpy()
        cave_depth = df['cave_depth'].to_numpy()
        return cls(reference, cave_height, cave_depth)


class ClosedLoop:
    """Simulate closed-loop operation.

    Args:
    mission: Mission instance (defines T).
    disturbances: 1D numpy array, length >= T, additive disturbances per step.
    Returns:
    Trajectory with recorded positions.
    Raises:
    ValueError if disturbances length < mission duration.
    """
    def __init__(self, plant: Submarine, controller):
        self.plant = plant
        self.controller = controller
        self.errorlog = None

    def simulate(self,  mission: Mission, disturbances: np.ndarray) -> Trajectory:

        T = len(mission.reference)
        if len(disturbances) < T:
            raise ValueError("Disturbances must be at least as long as mission duration")
        
        positions = np.zeros((T, 2))
        actions = np.zeros(T)
        errorlog = np.zeros(T)

        self.controller.reset()
        self.plant.reset_state()

        for t in range(T):
            positions[t] = self.plant.get_position()
            observation_t = self.plant.get_depth()
            reference_t = float(mission.reference[t])

            error = reference_t - observation_t
            errorlog[t] = abs(error)
            actions[t] = self.controller.update(error)

            self.plant.transition(actions[t], disturbances[t])

        self.errorlog = errorlog

        return Trajectory(positions)
        
    def simulate_with_random_disturbances(self, mission: Mission, variance: float = 0.5, seed: Optional[int] = None) -> Trajectory:
        """Simulate with Gaussian random disturbances.

        Args:
            mission: Mission instance defining the duration T.
            variance: standard deviation of the normal disturbance (float).
            seed: optional integer seed for reproducible random draws. If None,
                a non-deterministic generator is used.

        Returns:
            Trajectory recorded from the simulation.
        """
        # Use a Generator for reproducible draws when seed is provided
        rng = np.random.default_rng(seed)
        disturbances = rng.normal(0, variance, len(mission.reference))
        return self.simulate(mission, disturbances)

    def get_avg_error(self) -> float:
        if self.errorlog is not None:
            erroravg = np.mean(self.errorlog)
            return erroravg
        else:
            raise ValueError("No error value available, please run simulate before accessing.")
