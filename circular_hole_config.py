import os


OUTPUT_FOLDER_NAME             = 'circular_hole'

# Simulation parameters 
NUMBER_OF_PHONONS              = 500
OUTPUT_TRAJECTORIES_OF_FIRST   = NUMBER_OF_PHONONS
NUMBER_OF_TIMESTEPS            = 60000
TIMESTEP                       = 2e-12
T                              = 4

# Domain Size
THICKNESS                      = 150e-9
WIDTH                          = 1000e-9
LENGTH                         = 2000e-9

# Walls:
INCLUDE_RIGHT_SIDEWALL           = True
INCLUDE_LEFT_SIDEWALL            = True
INCLUDE_TOP_SIDEWALL             = False
INCLUDE_BOTTOM_SIDEWALL          = False

# Cold sides:
COLD_SIDE_POSITION_TOP           = True
COLD_SIDE_POSITION_BOTTOM        = False
COLD_SIDE_POSITION_RIGHT         = False
COLD_SIDE_POSITION_LEFT          = False

# Hot sides:
HOT_SIDE_POSITION_TOP            = False
HOT_SIDE_POSITION_BOTTOM         = True
HOT_SIDE_POSITION_RIGHT          = False
HOT_SIDE_POSITION_LEFT           = False

# Sources
PHONON_SOURCES = [Source(x=0, y=0, z=0, size_x=WIDTH, size_y=0, size_z=THICKNESS, angle_distribution="random_up")]

# Holes
HOLES = [CircularHole(x=0, y=LENGTH/2, diameter=WIDTH/2)]
