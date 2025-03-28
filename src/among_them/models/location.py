from enum import Enum


class Location(str, Enum):
    CAFETERIA = "Cafeteria"
    REACTOR = "Reactor"
    UPPER_ENGINE = "Upper Engine"
    LOWER_ENGINE = "Lower Engine"
    SECURITY = "Security"
    MEDBAY = "Medbay"
    ELECTRICAL = "Electrical"
    STORAGE = "Storage"
    ADMIN = "Admin"
    COMMUNICATIONS = "Communications"
    O2 = "O2"
    WEAPONS = "Weapons"
    SHIELDS = "Shields"
    NAVIGATION = "Navigation"

    def __repr__(self):
        return self.value


DOORS: dict[Location, list] = {
    Location.CAFETERIA: [
        Location.MEDBAY,
        Location.ADMIN,
        Location.WEAPONS,
    ],
    Location.REACTOR: [
        Location.UPPER_ENGINE,
        Location.SECURITY,
        Location.LOWER_ENGINE,
    ],
    Location.UPPER_ENGINE: [
        Location.REACTOR,
        Location.SECURITY,
        Location.MEDBAY,
    ],
    Location.LOWER_ENGINE: [
        Location.REACTOR,
        Location.SECURITY,
        Location.ELECTRICAL,
    ],
    Location.SECURITY: [
        Location.UPPER_ENGINE,
        Location.REACTOR,
        Location.LOWER_ENGINE,
    ],
    Location.MEDBAY: [
        Location.UPPER_ENGINE,
        Location.CAFETERIA,
    ],
    Location.ELECTRICAL: [
        Location.LOWER_ENGINE,
        Location.STORAGE,
    ],
    Location.STORAGE: [
        Location.ELECTRICAL,
        Location.ADMIN,
        Location.COMMUNICATIONS,
        Location.SHIELDS,
    ],
    Location.ADMIN: [Location.CAFETERIA, Location.STORAGE],
    Location.COMMUNICATIONS: [
        Location.STORAGE,
        Location.SHIELDS,
    ],
    Location.O2: [
        Location.SHIELDS,
        Location.WEAPONS,
        Location.NAVIGATION,
    ],
    Location.WEAPONS: [
        Location.CAFETERIA,
        Location.O2,
        Location.NAVIGATION,
    ],
    Location.SHIELDS: [
        Location.STORAGE,
        Location.COMMUNICATIONS,
        Location.O2,
        Location.NAVIGATION,
    ],
    Location.NAVIGATION: [
        Location.WEAPONS,
        Location.O2,
        Location.SHIELDS,
    ],
}

ROOM_COORDINATES = {
    Location.CAFETERIA: (2.2, 1.8),
    Location.REACTOR: (0.4, 1.2),
    Location.UPPER_ENGINE: (0.75, 1.75),
    Location.LOWER_ENGINE: (0.75, 0.6),
    Location.SECURITY: (1.1, 1.2),
    Location.MEDBAY: (1.5, 1.4),
    Location.ELECTRICAL: (1.5, 0.8),
    Location.STORAGE: (2.1, 0.5),
    Location.ADMIN: (2.7, 0.9),
    Location.COMMUNICATIONS: (2.7, 0.3),
    Location.O2: (2.8, 1.3),
    Location.WEAPONS: (3.1, 1.8),
    Location.SHIELDS: (3.1, 0.6),
    Location.NAVIGATION: (3.8, 1.3),
}
