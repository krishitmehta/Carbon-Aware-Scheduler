"""
Picks the cleanest time to run a job.

A job has a length, a power draw and a window: the earliest it may start and the time it must be
finished by. It runs in one go, without pauses. If the intensity for each half-hour is known (or
forecast), the CO2 from starting at a given time is

    kg CO2 = power in kW x 0.5 hours x (sum of the intensities over the job's half-hours) / 1000

best_start() works this out for every possible start time using running totals and returns the
cheapest one. schedule_jobs() does the same for several jobs, and when there is a limit on how
much power the site can draw at once it solves a small optimisation problem (one yes/no variable
per job and start time) with scipy's milp solver.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

SLOT = pd.Timedelta(minutes=30)
SLOT_HOURS = 0.5


@dataclass(frozen=True)
class Job:
    name: str
    duration_slots: int            # length in half-hours
    power_kw: float
    earliest: pd.Timestamp         # first allowed start
    deadline: pd.Timestamp         # job must have finished by this time

    @classmethod
    def from_hours(cls, name, duration_hours, power_kw, earliest, deadline) -> "Job":
        slots = duration_hours * 2
        if slots != int(slots) or slots < 1:
            raise ValueError("duration must be a positive multiple of 0.5 hours")
        return cls(name, int(slots), float(power_kw), pd.Timestamp(earliest), pd.Timestamp(deadline))


def _window(intensity: pd.Series, job: Job) -> pd.Series:
    win = intensity.loc[job.earliest: job.deadline - SLOT]
    if len(win) < job.duration_slots:
        raise ValueError(f"window too short for job '{job.name}'")
    if not (win.index.to_series().diff().dropna() == SLOT).all():
        raise ValueError("intensity series has gaps inside the job window")
    if win.isna().any():
        raise ValueError("intensity series has missing values inside the job window")
    return win


def window_costs(intensity: pd.Series, job: Job) -> pd.Series:
    """kgCO2 for every feasible start time of the job, indexed by start time."""
    win = _window(intensity, job)
    d = job.duration_slots
    csum = np.concatenate([[0.0], np.cumsum(win.to_numpy(dtype=float))])
    sums = csum[d:] - csum[:-d]
    kg = job.power_kw * SLOT_HOURS * sums / 1000.0
    return pd.Series(kg, index=win.index[: len(sums)], name=job.name)


def best_start(intensity: pd.Series, job: Job) -> tuple[pd.Timestamp, float]:
    costs = window_costs(intensity, job)
    start = costs.idxmin()
    return start, float(costs.min())


def emissions_kg(intensity: pd.Series, start: pd.Timestamp, job: Job) -> float:
    """Emissions of running the job from `start` under the given intensity series."""
    end = start + (job.duration_slots - 1) * SLOT
    return float(job.power_kw * SLOT_HOURS * intensity.loc[start:end].sum() / 1000.0)


def schedule_jobs(jobs: list[Job], intensity: pd.Series, capacity_kw: float | None = None) -> dict:
    """Return {job name: start time}. With capacity_kw, total power at any time is limited."""
    if len({j.name for j in jobs}) != len(jobs):
        raise ValueError("job names must be unique")
    costs = [window_costs(intensity, j) for j in jobs]
    if capacity_kw is None:
        return {j.name: c.idxmin() for j, c in zip(jobs, costs)}

    grid = intensity.loc[min(j.earliest for j in jobs): max(j.deadline for j in jobs) - SLOT]
    var_job, var_slot, var_cost = [], [], []
    for ji, (job, c) in enumerate(zip(jobs, costs)):
        for pos, value in zip(grid.index.get_indexer(c.index), c.to_numpy()):
            var_job.append(ji); var_slot.append(pos); var_cost.append(value)
    n = len(var_cost)

    assign = coo_matrix((np.ones(n), (var_job, np.arange(n))), shape=(len(jobs), n))
    rows, cols, vals = [], [], []
    for k, (ji, pos) in enumerate(zip(var_job, var_slot)):
        for t in range(pos, pos + jobs[ji].duration_slots):
            rows.append(t); cols.append(k); vals.append(jobs[ji].power_kw)
    capacity = coo_matrix((vals, (rows, cols)), shape=(len(grid), n))

    res = milp(c=np.array(var_cost), integrality=np.ones(n), bounds=Bounds(0, 1),
               constraints=[LinearConstraint(assign, 1, 1), LinearConstraint(capacity, -np.inf, capacity_kw)])
    if not res.success:
        raise RuntimeError(f"No feasible schedule: {res.message}")
    return {jobs[var_job[k]].name: grid.index[var_slot[k]] for k in np.flatnonzero(np.round(res.x))}
