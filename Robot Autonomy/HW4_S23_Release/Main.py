import FrankaRollout as Simulator


if __name__ == "__main__":

    # String path to the MJCF model
    xml_filepath = "./franka_emika_panda/panda_with_hand_torque.xml"

    # Create a simulation instance
    joints = [1, 3] # Joint numbering starts from 0, we will be actuating the second and fourth joints
    sim = Simulator.FrankaSim(xml_filepath, joints)


############################## MODIFY THE CALLS BELOW ###########################
    # TODO: Uncomment the correct method call below
    # sim.CEM()
    sim.REPS()
############################## MODIFY THE CALLS ABOVE ###########################

