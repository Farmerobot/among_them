from enum import Enum


class Location(Enum):
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


# Small map - 4 rooms
SMALL_MAP_LOCATIONS = [
    Location.CAFETERIA,
    Location.STORAGE,
    Location.ELECTRICAL,
    Location.MEDBAY,
]

# Medium map - 7 rooms
MEDIUM_MAP_LOCATIONS = [
    Location.CAFETERIA,
    Location.STORAGE,
    Location.ELECTRICAL,
    Location.MEDBAY,
    Location.SECURITY,
    Location.ADMIN,
    Location.WEAPONS,
]

# Large map - all 14 rooms (original)
LARGE_MAP_LOCATIONS = [
    Location.CAFETERIA,
    Location.REACTOR,
    Location.UPPER_ENGINE,
    Location.LOWER_ENGINE,
    Location.SECURITY,
    Location.MEDBAY,
    Location.ELECTRICAL,
    Location.STORAGE,
    Location.ADMIN,
    Location.COMMUNICATIONS,
    Location.O2,
    Location.WEAPONS,
    Location.SHIELDS,
    Location.NAVIGATION,
]

# Small map - 4 rooms
DOORS_SMALL: dict[Location, list] = {
    Location.CAFETERIA: [
        Location.STORAGE,
        Location.MEDBAY,
    ],
    Location.STORAGE: [
        Location.CAFETERIA,
        Location.ELECTRICAL,
    ],
    Location.ELECTRICAL: [
        Location.STORAGE,
        Location.MEDBAY,
    ],
    Location.MEDBAY: [
        Location.CAFETERIA,
        Location.ELECTRICAL,
    ],
}

# Medium map - 7 rooms
DOORS_MEDIUM: dict[Location, list] = {
    Location.CAFETERIA: [
        Location.MEDBAY,
        Location.ADMIN,
        Location.WEAPONS,
    ],
    Location.STORAGE: [
        Location.ELECTRICAL,
        Location.ADMIN,
    ],
    Location.ELECTRICAL: [
        Location.STORAGE,
        Location.SECURITY,
    ],
    Location.MEDBAY: [
        Location.CAFETERIA,
        Location.SECURITY,
        Location.ELECTRICAL,
        Location.STORAGE,
    ],
    Location.SECURITY: [
        Location.ELECTRICAL,
        Location.MEDBAY,
    ],
    Location.ADMIN: [
        Location.CAFETERIA,
        Location.STORAGE,
    ],
    Location.WEAPONS: [
        Location.CAFETERIA,
    ],
}

# Large map - full 14 rooms (original implementation)
DOORS_LARGE: dict[Location, list] = {
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

# Coordinates for small map - 4 rooms
ROOM_COORDINATES_SMALL = {
    Location.CAFETERIA: (1.0, 1.0),
    Location.STORAGE: (2.0, 1.0),
    Location.ELECTRICAL: (2.0, 2.0),
    Location.MEDBAY: (1.0, 2.0),
}

# Coordinates for medium map - 7 rooms
ROOM_COORDINATES_MEDIUM = {
    Location.CAFETERIA: (1.0, 2.0),
    Location.STORAGE: (2.0, 1.0),
    Location.ELECTRICAL: (3.0, 1.0),
    Location.MEDBAY: (2.0, 2.0),
    Location.SECURITY: (3.0, 2.0),
    Location.ADMIN: (1.5, 1.0),
    Location.WEAPONS: (0.5, 1.5),
}

# Original coordinates for large map - 14 rooms
ROOM_COORDINATES_LARGE = {
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

# Select map based on MAP_SIZE from consts.py
def get_map(map_size: int) -> tuple[dict[Location, list[Location]], dict[Location, tuple[float, float]], list[Location]]:
    """Returns the doors, room coordinates, and active locations for a given map size."""   
    if map_size == 0:
        return DOORS_SMALL, ROOM_COORDINATES_SMALL, SMALL_MAP_LOCATIONS
    elif map_size == 1:
        return DOORS_MEDIUM, ROOM_COORDINATES_MEDIUM, MEDIUM_MAP_LOCATIONS
    else:  # MAP_SIZE == 2 or any other value defaults to large
        return DOORS_LARGE, ROOM_COORDINATES_LARGE, LARGE_MAP_LOCATIONS
