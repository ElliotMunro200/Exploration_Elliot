# Python program to illustrate
# corner detection with
# Harris Corner Detection Method
import copy

# organizing imports
import cv2
import numpy as np

def NMS(R_thresh, width):
    # make a numpy array of zeros of size of R_thresh
    # for each point in the image:
    # with a window of size width=corner cluster width (odd number),
    # it over the image and suppress (--> 0) the anchor point if not the maximal value in the window
    pad = int((width-1)/2)
    R_thresh_padded = np.zeros((R_thresh.shape[0] + 2 * pad, R_thresh.shape[1] + 2 * pad))
    R_thresh_padded[pad:R_thresh.shape[0] + pad, pad:R_thresh.shape[1] + pad] = R_thresh
    #print(R_thresh.shape)
    #print(pad)
    #print(R_thresh_padded.shape)
    #print(R_thresh_padded)
    h = R_thresh.shape[0]
    w = R_thresh.shape[1]
    for r in range(0+pad,h+pad):
        for c in range(0+pad,w+pad):
            anchor = R_thresh_padded[r,c]
            #print(r,c,anchor)
            window_max = R_thresh_padded[r-pad:r+pad+1,c-pad:c+pad+1].max()
            #print(window_max)
            if anchor < window_max:
                R_thresh_padded[r,c] = 0
            else:
                continue
    R_thresh_NMS = R_thresh_padded[pad:R_thresh.shape[0]+pad, pad:R_thresh.shape[1]+pad]
    #print(R_thresh_NMS)
    return R_thresh_NMS
def img_cH_NMS(image, blockSize=2, ksize=5, k=0.07, T=0.01, NMS_width=13):

    operatedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    operatedImage = np.float32(operatedImage)
    # apply the cv2.cornerHarris method to detect the corners with appropriate values as input parameters
    R = cv2.cornerHarris(operatedImage, blockSize, ksize, k)
    # Results are marked through the dilated corners, makes the corner response smoother,
    # which can later be corrected by NMS.
    R = cv2.dilate(R, None) # (249,203), numpy array
    R_thresh = copy.deepcopy(R)
    R_thresh[R_thresh <= T * R_thresh.max()] = 0 # thresholded R, R.max() = 12416213000.0

    R_thresh_NMS = NMS(R_thresh, NMS_width)
    #R_thresh_NMS = cv2.erode(R_thresh_NMS, None)
    corner_bool = R_thresh_NMS > 0
    corners = np.nonzero(R_thresh_NMS)
    # Making new corner detected image
    image[corner_bool] = [0, 0, 255]

    return image, corners

if __name__ == "__main__":
    # all
    dir = f"/home/elliot/Pictures/Elliots_Architecture/Local_Area_Determination"
    image_types = ["png", "npy", "randnumpy"]
    selection = 1
    image_type = image_types[selection]
    # png
    image_number = 2
    # npy
    rank = 1
    episode = 1
    timestep = 1000
    # randnumpy
    width = 3

    if image_type == "png":
        print(f"Using image type: {image_type}")
        # test: 0; floor plans: 1-4; maze: 5; shapes/objects/people: 6-10; stars: 11-15
        images = ["CornerHarris_GfG_test", "4_floor_plans", "floor_plan", "thick_floor_plan", "small_floor_plan",
                  "maze", "shapes-01-mask", "cog", "leaf", "man", "man_2", "starfish_sharp", "starfish_sharp_2",
                  "starfish_soft", "6_pointed_star", "pointed_stars"]
        img_path = f"{dir}/{images[image_number]}.{image_type}"
        # reading in the image
        image = cv2.imread(img_path)  # e.g. (249, 203, 3), numpy array
        print(image.shape)
        image, corners = img_cH_NMS(image)

        print(f"Showing Harris NMS corners on image {img_path}")
        #print(f"Corners: {corners}")
        # the window showing output image with corners
        cv2.imshow('Image with Borders', image)
        # De-allocate any associated memory usage
        if cv2.waitKey(0) & 0xff == 27: # press any key
            cv2.destroyAllWindows()

    elif image_type == "npy":
        print(f"Using image type: {image_type}")
        occ_map_file = f"{rank}-{episode}-Occ-{timestep}"
        img_path = f"{dir}/Occupancy_Maps/{occ_map_file}.{image_type}"
        occ_map = np.load(img_path)
        image = (occ_map)*255
        image = np.float32(np.flip(image,axis=0))
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image, corners = img_cH_NMS(image)

        print(f"Showing Harris NMS corners on image {img_path}")
        #print(f"Corners: {corners}")
        cv2.imshow('Image with Borders', image)
        print(occ_map)
        if cv2.waitKey(0) & 0xff == 27: # press any key
            cv2.destroyAllWindows()

    elif image_type == "randnumpy":
        print(f"Using image type: {image_type}")
        test_numpy = np.random.randint(0,5,(5,5))
        NMS(test_numpy, width=width)

    else:
        print(f"Image type {image_type} not recognized")
