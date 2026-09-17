"""Simulation dynamics for a 4-way signalized traffic intersection."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np


class Approach(IntEnum):
    """Cardinal approach directions to the intersection."""

    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class Phase(IntEnum):
    """Traffic signal phase controlling green signals."""

    NORTH_SOUTH = 0  # Green for North & South; Red for East & West
    EAST_WEST = 1  # Green for East & West; Red for North & South


APPROACH_NAMES: Dict[Approach, str] = {
    Approach.NORTH: "North",
    Approach.SOUTH: "South",
    Approach.EAST: "East",
    Approach.WEST: "West",
}

PHASE_NAMES: Dict[Phase, str] = {
    Phase.NORTH_SOUTH: "NORTH_SOUTH",
    Phase.EAST_WEST: "EAST_WEST",
}


@dataclass
class ApproachState:
    """Internal state of a single approach lane."""

    direction: Approach
    arrival_rate: float
    max_capacity: int
    queue: deque[int]  # Stores individual waiting time in steps for each queued vehicle

    @property
    def queue_length(self) -> int:
        """Current number of vehicles queued at this approach."""
        return len(self.queue)

    @property
    def max_wait_time(self) -> int:
        """Maximum wait time among all queued vehicles in this approach."""
        return self.queue[0] if self.queue else 0

    @property
    def total_wait_time(self) -> int:
        """Sum of wait times across all queued vehicles in this approach."""
        return sum(self.queue)

    @property
    def mean_wait_time(self) -> float:
        """Average wait time for queued vehicles in this approach."""
        return float(self.total_wait_time / self.queue_length) if self.queue else 0.0

    def step_wait(self) -> None:
        """Increment waiting time for all vehicles currently queued."""
        for i in range(len(self.queue)):
            self.queue[i] += 1

    def add_arrivals(self, count: int) -> int:
        """Add new vehicle arrivals to the queue up to maximum capacity.

        Returns the number of vehicles successfully added (not dropped due to overflow).
        """
        added = 0
        for _ in range(count):
            if len(self.queue) < self.max_capacity:
                self.queue.append(0)  # Initial wait time is 0 steps
                added += 1
            else:
                break
        return added

    def discharge(self, max_vehicles: int) -> Tuple[int, List[int]]:
        """Discharge up to `max_vehicles` from the front of the queue (FIFO).

        Returns:
            Tuple[int, List[int]]: (count discharged, list of wait times of departed vehicles)
        """
        discharged_waits: List[int] = []
        num_to_discharge = min(max_vehicles, len(self.queue))
        for _ in range(num_to_discharge):
            discharged_waits.append(self.queue.popleft())
        return num_to_discharge, discharged_waits


@dataclass
class IntersectionStepTelemetry:
    """Telemetry recorded during a single simulation step."""

    action: int
    current_phase: Phase
    phase_switched: bool
    phase_duration: int
    arrivals: Dict[Approach, int]
    departures: Dict[Approach, int]
    queue_lengths: Dict[Approach, int]
    max_wait_times: Dict[Approach, int]
    mean_wait_times: Dict[Approach, float]
    total_departures: int
    total_arrivals: int
    total_queue: int
    overflow: bool


class TrafficIntersection:
    """Core physics and queuing model for a 4-way signalized intersection.

    Simulates vehicle arrivals (stochastic Poisson process), signal phase state,
    saturation discharge flow, and individual vehicle delay accumulation.
    """

    def __init__(
        self,
        arrival_rates: Tuple[float, float, float, float] = (0.3, 0.3, 0.2, 0.2),
        departure_rate: int = 2,
        max_queue: int = 30,
        max_wait_limit: int = 100,
        min_green_steps: int = 2,
    ) -> None:
        """Initialize the traffic intersection model.

        Args:
            arrival_rates: Expected vehicle arrival rate per step for (North, South, East, West).
            departure_rate: Maximum vehicle throughput per green approach per step.
            max_queue: Maximum queue storage capacity per approach lane.
            max_wait_limit: Maximum expected wait time used for normalization.
            min_green_steps: Minimum green duration before a switch occurs without short-cycle penalty.
        """
        if len(arrival_rates) != 4:
            raise ValueError(f"arrival_rates must have 4 elements, got {len(arrival_rates)}")
        if any(r < 0.0 for r in arrival_rates):
            raise ValueError(f"Arrival rates must be non-negative, got {arrival_rates}")
        if departure_rate <= 0:
            raise ValueError(f"departure_rate must be positive, got {departure_rate}")
        if max_queue <= 0:
            raise ValueError(f"max_queue must be positive, got {max_queue}")

        self.arrival_rates = arrival_rates
        self.departure_rate = departure_rate
        self.max_queue = max_queue
        self.max_wait_limit = max_wait_limit
        self.min_green_steps = min_green_steps

        self.approaches: Dict[Approach, ApproachState] = {
            Approach.NORTH: ApproachState(Approach.NORTH, arrival_rates[0], max_queue, deque()),
            Approach.SOUTH: ApproachState(Approach.SOUTH, arrival_rates[1], max_queue, deque()),
            Approach.EAST: ApproachState(Approach.EAST, arrival_rates[2], max_queue, deque()),
            Approach.WEST: ApproachState(Approach.WEST, arrival_rates[3], max_queue, deque()),
        }

        self.current_phase: Phase = Phase.NORTH_SOUTH
        self.phase_duration: int = 0
        self.total_switches: int = 0
        self.cumulative_departures: int = 0
        self.cumulative_arrivals: int = 0
        self.cumulative_delay: int = 0

    def reset(
        self,
        arrival_rates: Optional[Tuple[float, float, float, float]] = None,
        initial_queues: Optional[Dict[Approach, int]] = None,
    ) -> None:
        """Reset intersection state, clearing queues and resetting counters."""
        if arrival_rates is not None:
            if len(arrival_rates) != 4 or any(r < 0.0 for r in arrival_rates):
                raise ValueError(f"Invalid arrival_rates: {arrival_rates}")
            self.arrival_rates = arrival_rates

        rates = self.arrival_rates
        for idx, direction in enumerate(
            [Approach.NORTH, Approach.SOUTH, Approach.EAST, Approach.WEST]
        ):
            initial_count = 0
            if initial_queues and direction in initial_queues:
                initial_count = min(initial_queues[direction], self.max_queue)
            initial_deque = deque([0] * initial_count)
            self.approaches[direction] = ApproachState(
                direction=direction,
                arrival_rate=rates[idx],
                max_capacity=self.max_queue,
                queue=initial_deque,
            )

        self.current_phase = Phase.NORTH_SOUTH
        self.phase_duration = 0
        self.total_switches = 0
        self.cumulative_departures = 0
        self.cumulative_arrivals = sum(len(a.queue) for a in self.approaches.values())
        self.cumulative_delay = 0

    def step(
        self,
        action: int,
        rng: np.random.Generator,
    ) -> IntersectionStepTelemetry:
        """Advance the intersection simulation by one discrete time step.

        Process:
        1. Evaluate signal phase transition (switch vs maintain).
        2. Discharge queued vehicles on green approaches up to `departure_rate`.
        3. Increment waiting time for all remaining queued vehicles.
        4. Sample stochastic Poisson arrivals and append to approach queues.
        5. Record step telemetry.

        Args:
            action: 0 for NORTH_SOUTH green, 1 for EAST_WEST green.
            rng: NumPy random generator instance.

        Returns:
            IntersectionStepTelemetry: Detailed metrics from this step.
        """
        target_phase = Phase(action)
        phase_switched = target_phase != self.current_phase

        if phase_switched:
            self.current_phase = target_phase
            self.phase_duration = 1
            self.total_switches += 1
        else:
            self.phase_duration += 1

        # 2. Discharge vehicles from green approaches
        step_departures: Dict[Approach, int] = {}
        for direction, approach in self.approaches.items():
            is_green = (
                direction in (Approach.NORTH, Approach.SOUTH)
                and self.current_phase == Phase.NORTH_SOUTH
            ) or (
                direction in (Approach.EAST, Approach.WEST)
                and self.current_phase == Phase.EAST_WEST
            )
            if is_green:
                discharged_count, discharged_waits = approach.discharge(self.departure_rate)
                step_departures[direction] = discharged_count
                self.cumulative_delay += sum(discharged_waits)
            else:
                step_departures[direction] = 0

        # 3. Accumulate waiting time for all remaining vehicles
        for approach in self.approaches.values():
            approach.step_wait()

        # 4. Generate stochastic arrivals
        step_arrivals: Dict[Approach, int] = {}
        overflow_occurred = False
        for direction, approach in self.approaches.items():
            num_arrivals = int(rng.poisson(lam=approach.arrival_rate))
            step_arrivals[direction] = num_arrivals
            if approach.queue_length + num_arrivals > self.max_queue:
                overflow_occurred = True
            approach.add_arrivals(num_arrivals)

        # 5. Compute metrics
        queue_lengths = {d: a.queue_length for d, a in self.approaches.items()}
        max_waits = {d: a.max_wait_time for d, a in self.approaches.items()}
        mean_waits = {d: a.mean_wait_time for d, a in self.approaches.items()}

        step_total_departures = sum(step_departures.values())
        step_total_arrivals = sum(step_arrivals.values())
        self.cumulative_departures += step_total_departures
        self.cumulative_arrivals += step_total_arrivals

        return IntersectionStepTelemetry(
            action=action,
            current_phase=self.current_phase,
            phase_switched=phase_switched,
            phase_duration=self.phase_duration,
            arrivals=step_arrivals,
            departures=step_departures,
            queue_lengths=queue_lengths,
            max_wait_times=max_waits,
            mean_wait_times=mean_waits,
            total_departures=step_total_departures,
            total_arrivals=step_total_arrivals,
            total_queue=sum(queue_lengths.values()),
            overflow=overflow_occurred,
        )
