# These are the parameters used in the z-parametrization for Bs -> K l nu
INV_GEV_FM = 0.1973
SCALE_R1 = 0.31174
GEV_TO_R1 = SCALE_R1 / INV_GEV_FM
MB = 5.27962
MPION = 0.1349766
MBS = 5.36682
MBSTAR = 5.32465
MBSTAR0 = 5.63
MK = 0.497611 # neutral kaon
#MK = 0.493677 # charged kaon

# pair production threshold
TCUT = (MB + MPION)**2
#print "TCUT", TCUT

# maximum momentum-transfer allowed in the semileptonic Bs -> K l nu decay
TMINUS = (MBS - MK)**2
#print "MBS", MBS, "MK", MK
#print "TMINUS", TMINUS

# parameter to central the semileptonic region in z-plane.
T0 = TCUT * (1 - (1 - TMINUS / TCUT)**0.5)
#T0 = TCUT - (TCUT * (TCUT - TMINUS))**0.5
#print "T0", T0

# Fermi constant
GF = 1.1663787 * 0.00001 # GeV^-2 PDG16 p119
# muon mass
MMU = 105.6583745 * 0.001 # GeV PDG16 p 32
# tau mass
MTAU = 1776.86 * 0.001 #GeV PDG 16 p32

#1GeV-1 = 6.58*10**-25 s = 6.58*10**-13 ps
GEV_TO_PS = 6.58*10**(-13)

VUB_EX = (3.47, 0.22)
VUB_INC = (4.41, 0.22)
