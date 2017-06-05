import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Identify pixels in range.
# Threshold of 120 < R < 220, 100 < G < 180 and -1 < B < 90 does a nice job of identifying rock samples
def color_range(img, rgb_thresh_bottom=(120, 100, -1), rgb_thresh_top=(220, 180, 90)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Requires that each pixel be in between the range for all three threshold values of RGB.
    # in_range will now contain a boolean array with 'True' where the values are in range given.
    in_range = (img[:,:,0] > rgb_thresh_bottom[0]) & (img[:,:,0] < rgb_thresh_top[0]) \
            & (img[:,:,1] > rgb_thresh_bottom[1]) & (img[:,:,1] < rgb_thresh_top[1]) \
            & (img[:,:,2] > rgb_thresh_bottom[2]) & (img[:,:,2] < rgb_thresh_top[2])
    
    # Index the arry of zeros with the boolean array and set to 1
    color_select[in_range] = 1
    # Return the binary image
    return color_select

# Define a function to convert to rover-centric coordinates
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = np.absolute(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[0]).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to apply a rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    # Apply a rotation
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = xpix * np.cos(yaw_rad) - ypix * np.sin(yaw_rad)
    ypix_rotated = xpix * np.sin(yaw_rad) + ypix * np.cos(yaw_rad)
    # Return the result  
    return xpix_rotated, ypix_rotated

# Define a function to perform a translation
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):
    # Apply a scaling and a translation
    xpix_translated = np.int_(xpos + (xpix_rot / scale))
    ypix_translated = np.int_(ypos + (ypix_rot / scale))
    # Return the result  
    return xpix_translated, ypix_translated

# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # NOTE: camera image is coming to you in Rover.img
    img = Rover.img
    # 1) Define source and destination points for perspective transform
    bottom_offset = 6
    # Define calibration box in source (actual) and destination (desired) coordinates
    # These source and destination points are defined to warp the image
    # to a grid where each 10x10 pixel square represents 1 square meter
    # The destination box will be 2*dst_size on each side
    dst_size = 5
    
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[img.shape[1]/2 - dst_size, img.shape[0] - bottom_offset],
                  [img.shape[1]/2 + dst_size, img.shape[0] - bottom_offset],
                  [img.shape[1]/2 + dst_size, img.shape[0] - 2*dst_size - bottom_offset], 
                  [img.shape[1]/2 - dst_size, img.shape[0] - 2*dst_size - bottom_offset],
                  ])
    # 2) Apply perspective transform
    warped = perspect_transform(img, source, destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    navigable_threshed = color_thresh(warped)
    rock_threshed = color_range(warped)
    obsticle_threshed = ~navigable_threshed
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    Rover.vision_image[:,:,0] = obsticle_threshed * 255
    Rover.vision_image[:,:,1] = rock_threshed * 255
    Rover.vision_image[:,:,2] = navigable_threshed * 255
    # 5) Convert map image pixel values to rover-centric coords
    navigable_xpix, navigable_ypix = rover_coords(navigable_threshed)
    rock_xpix, rock_ypix = rover_coords(rock_threshed)
    obsticle_xpix, obsticle_ypix = rover_coords(obsticle_threshed)
    
    max_distance = 50
    good_navigable = np.sqrt(navigable_xpix**2 + navigable_ypix**2) < max_distance
    good_navigable_x = navigable_xpix[good_navigable]
    good_navigable_y = navigable_ypix[good_navigable]
    
    good_rock = np.sqrt(rock_xpix**2 + rock_ypix**2) < max_distance
    good_rock_x = rock_xpix[good_rock]
    good_rock_y = rock_ypix[good_rock]
    
    good_obsticle = np.sqrt(obsticle_xpix**2 + obsticle_ypix**2) < max_distance
    good_obsticle_x = obsticle_xpix[good_obsticle]
    good_obsticle_y = obsticle_ypix[good_obsticle]
    # 6) Convert rover-centric pixel values to world coordinates
    scale = 2 * dst_size
    navigable_xpix_world, navigable_ypix_world = pix_to_world(good_navigable_x, \
                                                              good_navigable_y, \
                                                              Rover.pos[0], \
                                                              Rover.pos[1], \
                                                              Rover.yaw, \
                                                              Rover.worldmap.shape[0], \
                                                              scale)
    rock_xpix_world, rock_ypix_world = pix_to_world(good_rock_x, \
                                                    good_rock_y, \
                                                    Rover.pos[0], \
                                                    Rover.pos[1], \
                                                    Rover.yaw, \
                                                    Rover.worldmap.shape[0], \
                                                    scale)
    obsticle_xpix_world, obsticle_ypix_world = pix_to_world(good_obsticle_x, \
                                                            good_obsticle_y, \
                                                            Rover.pos[0], \
                                                            Rover.pos[1], \
                                                            Rover.yaw, \
                                                            Rover.worldmap.shape[0], \
                                                            scale)
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    Rover.worldmap[navigable_ypix_world, navigable_xpix_world, 2] += 1
    Rover.worldmap[rock_ypix_world, rock_xpix_world, 1] += 1
    Rover.worldmap[navigable_ypix_world, navigable_xpix_world, 0] -= 1
    Rover.worldmap[obsticle_ypix_world, obsticle_xpix_world, 0] += 1
    # 8) Convert rover-centric pixel positions to polar coordinates
    distances, angles  = to_polar_coords(navigable_xpix, navigable_ypix)
    avg_angle = np.mean(angles)
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    Rover.nav_dists = distances
    Rover.nav_angles = angles
    
    return Rover