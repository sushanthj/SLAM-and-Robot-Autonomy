import numpy as np

ang = 16
ang2 = 16

while ang > np.pi:
    ang -= 2*np.pi

while ang < -np.pi:
    ang += 2*np.pi

print(ang)

print("***********")

if ang2 > np.pi:
    mod = ang2%np.pi
    print(mod)