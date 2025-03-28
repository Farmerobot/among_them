from among_them.models.location import Location
import random
from among_them import consts


class Task:
    def __init__(self, name: str, location: Location, completed: bool = False):
        self.name = name
        self.completed = completed
        self.location = location

    def complete(self, location: Location) -> str:
        pass


class ShortTask(Task):
    def complete(self, location: Location) -> str:
        if self.completed:
            return f"Task {self.name} already completed!"
        if self.location != location:
            return f"Task {self.name} cannot be completed in {location.value}!"
        self.completed = True
        return f"Task {self.name} completed!"

    def __str__(self):
        return f"{'[DONE]' if self.completed else '[TODO]'} | {self.name}"


class LongTask(Task):
    def __init__(
        self,
        name: str,
        location: Location,
        completed: bool = False,
        turns_left: int = 2,
    ):
        super().__init__(name, location, completed)
        self.turns_left = turns_left

    def complete(self, location: Location) -> str:
        if self.completed:
            return f"Task {self.name} already completed!"
        if self.location != location:
            return f"Task {self.name} cannot be completed in {location.value}!"
        self.turns_left -= 1
        if self.turns_left == 0:
            self.completed = True
            return f"Task {self.name} completed!"
        return f"Task {self.name} requires {self.turns_left} more turns to complete."

    def __str__(self):
        return f"{'[DONE]' if self.completed else '[TODO]'} | {self.name} | {self.turns_left} turns left to finish"


def get_crewmate_tasks() -> list[Task]:
    return get_short_tasks() + get_long_tasks()


def get_impostor_tasks() -> list[ShortTask]:
    return [ShortTask(name="Eliminate all crewmates", location=None)]


def get_impostor_pretend_tasks_at_location(location: Location) -> list[ShortTask]:
    return [[task for task in SHORT_TASKS if task.location == location][0]]


def get_short_tasks() -> list[ShortTask]:
    return random.sample(SHORT_TASKS, k=consts.NUM_SHORT_TASKS)


def get_long_tasks() -> list[LongTask]:
    return random.sample(LONG_TASKS, k=consts.NUM_LONG_TASKS)


LONG_TASKS = [
    LongTask(
        name="Align engine output in upper engine",
        location=Location.UPPER_ENGINE,
    ),
    LongTask(name="Chart course in navigation", location=Location.NAVIGATION),
    LongTask(name="Clear asteroids in weapons", location=Location.WEAPONS),
    LongTask(
        name="Route power to defence in electrical",
        location=Location.ELECTRICAL,
    ),
    LongTask(
        name="Route power to attack in electrical",
        location=Location.ELECTRICAL,
    ),
    LongTask(name="Prime shields", location=Location.SHIELDS),
    LongTask(
        name="Process data in communications",
        location=Location.COMMUNICATIONS,
    ),
    LongTask(name="Run diagnostics in medbay", location=Location.MEDBAY),
    LongTask(name="Submit scan in medbay", location=Location.MEDBAY),
]

SHORT_TASKS = [
    ShortTask(name="Empty the cafeteria trash", location=Location.CAFETERIA),
    ShortTask(
        name="Start the coffee maker in the cafeteria",
        location=Location.CAFETERIA,
    ),
    ShortTask(name="Fix wiring in cafeteria", location=Location.CAFETERIA),
    ShortTask(name="Empty the storage trash chute", location=Location.STORAGE),
    ShortTask(name="Fix wiring in storage", location=Location.STORAGE),
    ShortTask(name="Clean the floor in storage", location=Location.STORAGE),
    ShortTask(name="Fix wiring in electrical", location=Location.ELECTRICAL),
    ShortTask(name="Reset breakers in electrical", location=Location.ELECTRICAL),
    ShortTask(name="Fix wiring in admin", location=Location.ADMIN),
    ShortTask(name="Clean the floor in admin", location=Location.ADMIN),
    ShortTask(name="Fix wiring in navigation", location=Location.NAVIGATION),
    ShortTask(name="Adjust course in navigation", location=Location.NAVIGATION),
    ShortTask(name="Check headings in navigation", location=Location.NAVIGATION),
    ShortTask(name="Fix wiring in weapons", location=Location.WEAPONS),
    ShortTask(
        name="Calibrate targeting system in weapons",
        location=Location.WEAPONS,
    ),
    ShortTask(name="Fix wiring in shields", location=Location.SHIELDS),
    ShortTask(name="Fix wiring in o2", location=Location.O2),
    ShortTask(name="Clean oxygenator filter in o2", location=Location.O2),
    ShortTask(name="Water plants in o2", location=Location.O2),
    ShortTask(name="Fix wiring in medbay", location=Location.MEDBAY),
    ShortTask(
        name="Check catalyzer in upper engine",
        location=Location.UPPER_ENGINE,
    ),
    ShortTask(
        name="Check catalyzer in lower engine",
        location=Location.LOWER_ENGINE,
    ),
    ShortTask(
        name="Replace compression coil in upper engine",
        location=Location.UPPER_ENGINE,
    ),
    ShortTask(
        name="Replace compression coil in lower engine",
        location=Location.LOWER_ENGINE,
    ),
]
