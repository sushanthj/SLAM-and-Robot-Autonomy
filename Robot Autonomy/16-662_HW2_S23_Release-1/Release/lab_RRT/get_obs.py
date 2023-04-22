import imp
import numpy as np
import time
from autolab_core import RigidTransform
from frankapy import FrankaArm
from argparse import ArgumentParser
import sys

def main(fa):
    fa.run_guide_mode(2000, block=False)
    print("Starting Guide Mode...\n")

    while True:

        # take input from the user
        user_input = int(input("Please Enter a Command:\n1 - Print EE Pose\n2 - Print Joint Angle\n3 - End Guide Mode\n"))

        if user_input == 1:
            # printing pose...
            ee_world_pose = fa.get_pose()
            print(ee_world_pose)

        elif user_input == 2:
            # printing joints...
            joint_world = fa.get_joints()
            print(joint_world)

        elif user_input == 3:
            # kills the user guide mode...
            fa.stop_skill()

            reset = int(input("Reset the arm? [0 = No, 1 = Yes]: "))
            if reset:
                fa.reset_joints()

            break

if __name__ == "__main__" :
    fa = FrankaArm()
    main(fa)