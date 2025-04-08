import random
from typing import Optional

from among_them import consts
from among_them.models.location import Location


class Task:
    def __init__(self, name: str, location: Optional[Location], completed: bool = False):
        self.name = name
        self.completed = completed
        self.location = location

    def complete(self, location: Location) -> str:
        if self.completed:
            return f"Task {self.name} already completed!"
        if self.location != location:
            return f"Task {self.name} cannot be completed in {location.value}!"
        self.completed = True
        return f"Task {self.name} completed!"

    def __str__(self):
        return f"{'[DONE]' if self.completed else '[TODO]'} | {self.name}"


def get_crewmate_tasks() -> list[Task]:
    return get_short_tasks()


def get_impostor_tasks() -> list[Task]:
    return [Task(name="Vote out or kill all crewmates", location=None)]


def get_impostor_pretend_tasks_at_location(location: Location) -> list[Task]:
    return [[task for task in SHORT_TASKS if task.location == location][0]]


def get_short_tasks() -> list[Task]:
    return random.sample(SHORT_TASKS, k=consts.NUM_TASKS)


SHORT_TASKS = [
    Task(name="Empty the cafeteria trash", location=Location.CAFETERIA),
    Task(name="Start the coffee maker in the cafeteria", location=Location.CAFETERIA),
    Task(name="Fix wiring in cafeteria", location=Location.CAFETERIA),
    Task(name="Empty the storage trash chute", location=Location.STORAGE),
    Task(name="Fix wiring in storage", location=Location.STORAGE),
    Task(name="Clean the floor in storage", location=Location.STORAGE),
    Task(name="Fix wiring in electrical", location=Location.ELECTRICAL),
    Task(name="Reset breakers in electrical", location=Location.ELECTRICAL),
    Task(name="Route power to defence in electrical", location=Location.ELECTRICAL),
    Task(name="Route power to attack in electrical", location=Location.ELECTRICAL),
    Task(name="Fix wiring in admin", location=Location.ADMIN),
    Task(name="Clean the floor in admin", location=Location.ADMIN),
    Task(name="Fix wiring in navigation", location=Location.NAVIGATION),
    Task(name="Adjust course in navigation", location=Location.NAVIGATION),
    Task(name="Check headings in navigation", location=Location.NAVIGATION),
    Task(name="Chart course in navigation", location=Location.NAVIGATION),
    Task(name="Fix wiring in weapons", location=Location.WEAPONS),
    Task(name="Clear asteroids in weapons", location=Location.WEAPONS),
    Task(name="Calibrate targeting system in weapons", location=Location.WEAPONS),
    Task(name="Fix wiring in shields", location=Location.SHIELDS),
    Task(name="Prime shields", location=Location.SHIELDS),
    Task(name="Fix wiring in o2", location=Location.O2),
    Task(name="Clean oxygenator filter in o2", location=Location.O2),
    Task(name="Water plants in o2", location=Location.O2),
    Task(name="Fix wiring in medbay", location=Location.MEDBAY),
    Task(name="Run diagnostics in medbay", location=Location.MEDBAY),
    Task(name="Submit scan in medbay", location=Location.MEDBAY),
    Task(name="Check catalyzer in upper engine", location=Location.UPPER_ENGINE),
    Task(name="Replace compression coil in upper engine", location=Location.UPPER_ENGINE),
    Task(name="Align engine output in upper engine", location=Location.UPPER_ENGINE),
    Task(name="Check catalyzer in lower engine", location=Location.LOWER_ENGINE),
    Task(name="Replace compression coil in lower engine", location=Location.LOWER_ENGINE),
    Task(name="Process data in communications", location=Location.COMMUNICATIONS),
    Task(name="Fix wiring in reactor", location=Location.REACTOR)
]
