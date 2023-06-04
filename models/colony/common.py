import numpy

# random state
random_state = numpy.random.RandomState()

# reference moles
CHAMBER_MOLES = 120

# lysis moles
LYSIS_MOLES = 3 * CHAMBER_MOLES

# lysis moles
REPLICATION_MOLES = CHAMBER_MOLES // 2

# high pressure
HIGH_PRESSURE_MOLES = (2 * LYSIS_MOLES // 3)

# max matrix moles
MATRIX_MOLES = 4 * CHAMBER_MOLES

# matrix motility factor
MATRIX_MOTILITY = 10

# max number of genes
PLASMID_GENES = 15

# conjugation distance
CONJUGATION_DISTANCE = 10
