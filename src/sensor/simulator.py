"""
Synthetic Vibration Sensor Simulation.
Models a damped harmonic oscillator driven by Gaussian noise.
Implements a 4th-order Runge-Kutta (RK4) numerical integrator yielding data as a generator.
"""

import math
import random
import time
from typing import Generator, Tuple, Optional

from src.config import SAMPLE_RATE_HZ, BASE_OMEGA_N, BASE_ZETA
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VibrationSensor:
    """
    Simulates a physical vibration sensor mounted on industrial equipment.
    
    The physics model follows a stochastic second-order differential equation:
        x''(t) + 2*zeta*omega_n*x'(t) + omega_n^2*x(t) = F(t)
        
    Where:
        - x(t) is displacement, x'(t) is velocity, x''(t) is acceleration.
        - F(t) is a Gaussian noise force representing normal operation vibrations.
        - omega_n is the natural frequency.
        - zeta is the damping ratio.
    """

    def __init__(self, sample_rate_hz: int = SAMPLE_RATE_HZ):
        self.sample_rate = sample_rate_hz
        self.dt = 1.0 / self.sample_rate
        
        # Baseline physical parameters
        self.base_omega_n = BASE_OMEGA_N
        self.base_zeta = BASE_ZETA
        
        # Fault injection state
        self.fault_start_time: Optional[float] = None
        self.fault_end_time: Optional[float] = None
        self.fault_omega_n: Optional[float] = None
        self.fault_zeta: Optional[float] = None

    def inject_fault(self, start_time: float, duration: float, 
                     new_omega_n: Optional[float] = None, 
                     new_zeta: Optional[float] = None) -> None:
        """
        Schedules a physical anomaly simulating severe structural degradation/bearing failure.
        """
        self.fault_start_time = start_time
        self.fault_end_time = start_time + duration
        
        self.fault_omega_n = new_omega_n if new_omega_n is not None else self.base_omega_n * 2.5
        self.fault_zeta = new_zeta if new_zeta is not None else self.base_zeta * 0.1
        
        logger.info(
            f"Fault scheduled from t={start_time}s to t={self.fault_end_time}s. "
            f"Omega: {self.fault_omega_n:.2f}, Zeta: {self.fault_zeta:.4f}"
        )

    def _is_anomaly(self, current_sim_time: float) -> bool:
        """Checks if the current simulation time falls within the fault window."""
        if self.fault_start_time and self.fault_end_time:
            return self.fault_start_time <= current_sim_time <= self.fault_end_time
        return False

    def stream_samples(self, real_time: bool = True) -> Generator[Tuple[float, float, bool], None, None]:
        """
        Continuous generator that yields simulated sensor readings.
        
        :param real_time: If True, sleeps for self.dt on each step to match real sampling frequency (e.g., 1000 Hz).
        Yields: (current_unix_timestamp, acceleration, ground_truth_anomaly)
        """
        logger.info(f"Starting sensor stream at {self.sample_rate} Hz (Real-time pacing: {real_time})...")
        
        # Initial physics state
        x = 0.0  # Initial displacement (position)
        v = 0.0  # Initial velocity
        t_sim = 0.0  # Simulation relative time in seconds (used for physics calculation and anomaly checking)
        
        # Real-world base timestamp (prevents the 1970 date partitioning bug)
        start_wall_timestamp = time.time()
        
        def get_derivatives(pos: float, vel: float, current_force: float, 
                            current_zeta: float, current_omega: float) -> Tuple[float, float]:
            accel = current_force - 2.0 * current_zeta * current_omega * vel - (current_omega ** 2) * pos
            return vel, accel

        while True:
            # Calculate real-world Unix timestamp for S3 partitioning
            current_timestamp = start_wall_timestamp + t_sim
            
            # 1. Determine current physical parameters based on relative time
            is_anomaly = self._is_anomaly(t_sim)
            omega = self.fault_omega_n if is_anomaly else self.base_omega_n
            zeta = self.fault_zeta if is_anomaly else self.base_zeta
            
            # 2. Generate random excitation force (Gaussian noise)
            force = random.gauss(0, 1.0)
            
            # 3. Runge-Kutta 4th Order (RK4) integration steps
            k1_x, k1_v = get_derivatives(x, v, force, zeta, omega)
            k2_x, k2_v = get_derivatives(x + 0.5 * self.dt * k1_x, v + 0.5 * self.dt * k1_v, force, zeta, omega)
            k3_x, k3_v = get_derivatives(x + 0.5 * self.dt * k2_x, v + 0.5 * self.dt * k2_v, force, zeta, omega)
            k4_x, k4_v = get_derivatives(x + self.dt * k3_x, v + self.dt * k3_v, force, zeta, omega)
            
            # 4. Update physics state
            x += (self.dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
            v += (self.dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
            
            # 5. Calculate acceleration at this step
            acceleration = force - 2.0 * zeta * omega * v - (omega ** 2) * x
            
            # Yield real Unix timestamp, acceleration reading, and ground truth flag
            yield (current_timestamp, acceleration, is_anomaly)
            
            # Advance relative simulation clock
            t_sim += self.dt

            # Control thread execution rate (1000 Hz = sleep 0.001s per loop)
            if real_time:
                time.sleep(self.dt)