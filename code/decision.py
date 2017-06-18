import numpy as np

from supporting_functions import distance_between


# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # We are home, lets dance.
    if Rover.mode == 'home_dance':
        # We are home, lets dance
        if Rover.vel > 0:
            Rover.brake = Rover.brake_set
            Rover.steer = 0
            Rover.throttle = 0
        elif Rover.increment and Rover.dance_frames <= 20:
            Rover.steer = -15
            Rover.dance_frames += 1
            Rover.brake = 0
            Rover.throttle = 0
            if Rover.dance_frames > 20:
                Rover.increment = False
        else:
            Rover.steer = 15
            Rover.dance_frames -= 1
            Rover.brake = 0
            Rover.throttle = 0
            if Rover.dance_frames < 0:
                Rover.increment = True
        return Rover
    
    # Check if we have vision of a rock and make decisions
    if Rover.rock_angles is not None and len(Rover.rock_angles) > 1:
        # Begin stopping
        if Rover.mode != 'pickup':
            Rover.throttle = 0
            Rover.brake = 3
            Rover.steer = 0
            Rover.mode = 'pickup'

    # We are getting unstuck going home.
    if Rover.try_home_frames > 0:
        Rover.try_home_frames -= 1
        # We have tried to get unstuck for a while, try to get to home again.
        if Rover.try_home_frames <= 0:
            Rover.mode = 'home'

    # Some times the rover might get stuck in a corner with no way to go out, so reduce the number of forward pixels
    # to find a way out.
    if Rover.mode == 'unstuck' or Rover.mode == 'stop':
        Rover.unstuck_frames += 1
        if Rover.unstuck_frames > 500:
            Rover.go_forward = 100
    else:
        Rover.unstuck_frames = 0

    # Reset the go_forward threshold.
    if Rover.go_forward == 100:
        Rover.low_forward_frames += 1
        if Rover.low_forward_frames >= 500:
            Rover.go_forward = 500
    else:
        Rover.low_forward_frames = 0

    # The Rover might go in circles when in a wide open area, this is to safe guard against that.
    if Rover.mode == 'forward' \
            and not Rover.picking_up \
            and (Rover.steer > 13.5 or Rover.steer < -13.5) \
            and Rover.vel > 0.2:
        Rover.max_steer_frames += 1
        if Rover.max_steer_frames > 500:
            Rover.mode = 'unstuck'
            Rover.brake = 0
            Rover.steer = 0
            Rover.throttle = 0
            Rover.stuck_yaw = Rover.yaw
    else:
        Rover.max_steer_frames = 0

    # Set the starting position.
    if Rover.start_pos is None:
        Rover.start_pos = Rover.pos
        Rover.recover_pos = Rover.pos
        Rover.recover_yaw = Rover.yaw
    else:
        #  Check if we are near the start.
        distance_to_start = distance_between(Rover.pos, Rover.start_pos)
        Rover.distance_to_start = distance_to_start
        map_filled = np.count_nonzero(Rover.worldmap)
        if not Rover.mode == 'unstuck' and map_filled > 7000 and distance_to_start < 5:
            Rover.ready_for_home = True
            if Rover.try_home_frames <= 0:
                Rover.mode = 'home'

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:

        if is_stuck(Rover):
            Rover.stuck_frames += 1
            if Rover.stuck_frames > 50:
                Rover.throttle = 0
                Rover.brake = 0
                Rover.steer = 0
                Rover.stuck_yaw = Rover.yaw
                Rover.mode = 'unstuck'
                if Rover.ready_for_home:
                    Rover.try_home_frames = 500
        else:
            Rover.stuck_frames = 0

        # Check for Rover.mode status
        if Rover.mode == 'forward':
            # Check the extent of navigable terrain
            if len(Rover.nav_angles) >= Rover.stop_forward:  
                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle 
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
                # Set steering to (average + 14) angle clipped to the range +/- 15
                Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi) + 14, -15, 15)
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = -15 # Could be more clever here about which way to turn
                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                    Rover.mode = 'forward'
        elif Rover.mode == 'pickup':
            # If not picking up, send pickup.
            if Rover.vel == 0 and not Rover.picking_up and Rover.near_sample:
                Rover.send_pickup = True
            if Rover.picking_up:  # Confirmed pickup, set mode to forward.
                Rover.mode = 'forward'
                Rover.current_sample_pos = None
            else:
                # If we see the rock, go towards it.
                if Rover.rock_angles is not None and len(Rover.rock_angles) > 1:
                    rock_distance = np.mean(Rover.rock_dists)
                    if Rover.near_sample:
                        Rover.throttle = 0
                        Rover.brake = Rover.brake_set
                        Rover.steer = 0
                    elif rock_distance < 15:
                        if Rover.vel < Rover.rock_approach_vel:
                            Rover.throttle = Rover.throttle_crawl
                            Rover.brake = 0
                        else:
                            Rover.throttle = 0
                            Rover.brake = 8
                        Rover.steer = np.clip(np.mean(Rover.rock_angles * 180/np.pi) - 10, -15, 15)
                    else:
                        if Rover.vel < Rover.rock_approach_vel:
                            Rover.throttle = Rover.throttle_crawl
                        else:
                            Rover.throttle = 0
                            Rover.brake = 6
                        Rover.steer = np.clip(np.mean(Rover.rock_angles * 180/np.pi) - 10, -15, 15)
                        Rover.brake = 0
                elif Rover.current_sample_pos is not None:
                    # We dont see the rock, but know its position, turn towards it to see it.
                    rock_distance = distance_between(Rover.current_sample_pos, Rover.pos)
                    target_yaw = np.arctan2(int(Rover.current_sample_pos[1]) - (int(Rover.pos[1])),
                                    int(Rover.current_sample_pos[0]) - (int(Rover.pos[0])))
                    if target_yaw < 0:
                        target_yaw += np.pi * 2

                    target_yaw = target_yaw * 180/np.pi
                    if Rover.near_sample:
                        Rover.throttle = 0
                        Rover.brake = Rover.brake_set
                        Rover.steer = 0
                    elif abs(target_yaw - Rover.yaw) <= 5 or abs(target_yaw - Rover.yaw) >= 255:
                        if rock_distance < 10:
                            if Rover.vel < Rover.rock_approach_vel:
                                Rover.throttle = Rover.throttle_crawl
                                Rover.brake = 0
                            else:
                                Rover.throttle = 0
                                Rover.brake = 8
                            Rover.steer = 0
                        else:
                            if Rover.vel < Rover.rock_approach_vel:
                                Rover.throttle = Rover.throttle_crawl
                                Rover.brake = 0
                            else:
                                Rover.throttle = 0
                                Rover.brake = 6
                            Rover.steer = 0
                    elif rock_distance > 1:
                        if Rover.vel > 0:
                            Rover.steer = 0
                            Rover.throttle = 0
                            Rover.brake = Rover.brake_set
                        else:
                            if abs(Rover.yaw - target_yaw) > 180:
                                if target_yaw - Rover.yaw < 0:
                                    Rover.steer = 2
                                else:
                                    Rover.steer = -2
                            else:
                                if target_yaw - Rover.yaw < 0:
                                    Rover.steer = -2
                                else:
                                    Rover.steer = 2
                            Rover.throttle = 0
                            Rover.brake = Rover.brake = 0
                    else:
                        # We might have overshot, just keep moving although we should think about a way to go
                        # back and get it.
                        Rover.steer = 0
                        Rover.throttle = 0
                        Rover.brake = 0
                        Rover.mode = 'forward'
                else:
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.throttle = 0
                    Rover.current_sample_pos = None
                    Rover.mode = 'forward'
        elif Rover.mode == 'unstuck':
            # Uh oh, we are stuck, rotate to find a way out.
            Rover.throttle = 0
            Rover.brake = 0
            Rover.steer = - 15
            if abs(Rover.yaw - Rover.stuck_yaw) > 10:
                Rover.mode = 'stop'
        elif Rover.mode == 'home':
            # We are all done here, lets get home.
            target_yaw = np.arctan2(int(Rover.start_pos[1]) - (int(Rover.pos[1])),
                                    int(Rover.start_pos[0]) - (int(Rover.pos[0])))
            if target_yaw < 0:
                target_yaw += np.pi * 2

            target_yaw = target_yaw * 180/np.pi

            yaw_diff = target_yaw - Rover.yaw

            # We are almost looking at the destination, go towards it.
            if abs(yaw_diff) <= 5 or abs(yaw_diff) >= 355:
                # Move towards target
                if Rover.distance_to_start < 1:
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.throttle = 0
                    Rover.mode = 'home_dance'
                else:
                    Rover.steer = 0
                    Rover.brake = 0
                    Rover.throttle = Rover.throttle_quarter
            else:
                # Stop and turn towards the target.
                if Rover.vel > 0:
                    Rover.steer = 0
                    Rover.throttle = 0
                    Rover.brake = Rover.brake_set
                else:
                    if abs(yaw_diff) > 180:
                        if yaw_diff < 0:
                            Rover.steer = 2
                        else:
                            Rover.steer = -2
                    else:
                        if yaw_diff < 0:
                            Rover.steer = -2
                        else:
                            Rover.steer = 2
                    Rover.throttle = 0
                    Rover.brake = Rover.brake = 0

    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0
        
    # # If in a state where want to pickup a rock send pickup command
    # if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
    #     Rover.send_pickup = True
    
    return Rover


# Checks if the rover is stuck.
def is_stuck(Rover):
    return not Rover.picking_up and Rover.vel < 0.1 and Rover.throttle != 0
