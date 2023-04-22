import imp
import numpy as np
import time
from autolab_core import RigidTransform
from frankapy import FrankaArm
from argparse import ArgumentParser
import sys

def main(fa):
    fa.run_guide_mode(20, block=False)
    start = time.time()
    while (time.time() - start < 20):
        T_ee_world = fa.get_pose()
        joints = fa.get_joints()
        print("T_ee_world", T_ee_world)
        print("joints", joints)
        time.sleep(0.01)
    
    fa.stop_skill()

if __name__ == "__main__" :
    fa = FrankaArm()
    fa.reset_joints()
    main(fa)