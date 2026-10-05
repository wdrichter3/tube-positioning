"""Available device drivers."""
from .agent import AgentDriver
from .files import FolderDriver
from .oak import OakDriver
from .orbbec import OrbbecDriver
from .simulator import SimulatorDriver
from .uvc import UvcDriver

DRIVERS = {d.driver_id: d for d in (OakDriver, AgentDriver, OrbbecDriver, UvcDriver, FolderDriver, SimulatorDriver)}


def describe():
    out = []
    for d in DRIVERS.values():
        ok, msg = d.available()
        out.append({"id": d.driver_id, "label": d.label, "description": d.description,
                    "has_depth": d.has_depth, "available": ok, "availability": msg,
                    "schema": d.full_schema()})
    return out
