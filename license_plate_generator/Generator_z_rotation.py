import os
import random
import cv2
import argparse
import numpy as np

def rotate_image(image, angle):
    """
    Rotates an image around its center.

    Args:
        image (numpy.ndarray): The input image.
        angle (float): The rotation angle in degrees (positive for counter-clockwise).

    Returns:
        numpy.ndarray: The rotated image.
    """
    # Get image dimensions
    (h, w) = image.shape[:2]
    # Calculate the center of the image
    center = (w // 2, h // 2)

    # Get the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0) # 1.0 is scale factor

    # Perform the rotation
    # Note: This will crop parts of the image if it rotates beyond the original bounds.
    # To avoid cropping, you would need to calculate new dimensions for the rotated image
    # and adjust the translation part of the rotation matrix (M).
    rotated_image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return rotated_image


def image_augmentation(img, type2=False, apply_blur=True): # เพิ่ม apply_blur=True
    """
    Performs image augmentation including perspective transformation, brightness adjustment,
    and blurring.
    """
    # Perspective transformation
    # Get image dimensions (height, width, channels)
    h, w, _ = img.shape
    # Define source points (corners of the original image)
    pts1 = np.float32([[0, 0], [0, h], [w, 0], [w, h]])

    begin, end = 30, 90
    # Define destination points for perspective transformation with random offsets
    pts2 = np.float32([[random.randint(begin, end), random.randint(begin, end)],
                       [random.randint(begin, end), h - random.randint(begin, end)],
                       [w - random.randint(begin, end), random.randint(begin, end)],
                       [w - random.randint(begin, end), h - random.randint(begin, end)]])
    # Get the perspective transformation matrix
    M = cv2.getPerspectiveTransform(pts1, pts2)

    # Apply perspective transformation
    img = cv2.warpPerspective(img, M, (w, h))

    # Brightness adjustment
    # Convert image from RGB to HSV color space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = np.array(img, dtype=np.float64) # Convert to float64 for calculations
    random_bright = .4 + np.random.uniform() # Generate a random brightness factor
    img[:, :, 2] = img[:, :, 2] * random_bright # Adjust V (Value/Brightness) channel
    img[:, :, 2][img[:, :, 2] > 255] = 255 # Clip values to max 255
    img = np.array(img, dtype=np.uint8) # Convert back to uint8
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR) # Convert back to BGR

    # Blur - เพิ่มเงื่อนไข if apply_blur
    if apply_blur:
        blur_value = random.randint(0,4) * 2 + 1 # Generate odd blur kernel size (1, 3, 5, 7, 9)
        img = cv2.blur(img,(blur_value, blur_value)) # Apply blur

    # Crop the image based on type
    if type2:
        return img[130:280, 220:560, :]
    return img[130:280, 120:660, :]


def overlay_image_alpha(background_img, foreground_img_rgba, x_offset, y_offset):
    """
    Overlays a foreground image (with alpha channel) onto a background image.
    Handles transparency and ensures the foreground fits within the background bounds.
    
    Args:
        background_img (numpy.ndarray): The background image (3 channels, BGR).
        foreground_img_rgba (numpy.ndarray): The foreground image (4 channels, BGRA).
        x_offset (int): X-coordinate offset for placing the foreground.
        y_offset (int): Y-coordinate offset for placing the foreground.
        
    Returns:
        numpy.ndarray: The background image with the foreground overlaid.
    """
    
    # Check if foreground_img_rgba has 4 channels
    if foreground_img_rgba is None or foreground_img_rgba.shape[2] != 4:
        # print(f"Warning: Foreground image at offset ({x_offset},{y_offset}) is not a 4-channel BGRA image or is None. Skipping overlay.")
        return background_img

    # Extract color channels (BGR) and alpha channel from foreground
    foreground_bgr = foreground_img_rgba[:, :, :3]
    alpha_channel = foreground_img_rgba[:, :, 3] / 255.0  # Normalize alpha to 0-1

    # Get dimensions
    h_fg, w_fg = foreground_bgr.shape[:2]
    h_bg, w_bg = background_img.shape[:2]

    # Calculate region of interest (ROI) on the background where foreground will be placed
    y1, y2 = max(0, y_offset), min(h_bg, y_offset + h_fg)
    x1, x2 = max(0, x_offset), min(w_bg, x_offset + w_fg)

    # Calculate corresponding ROI on the foreground (in case foreground is clipped by background edges)
    y1_fg = y1 - y_offset
    y2_fg = y2 - y_offset
    x1_fg = x1 - x_offset
    x2_fg = x2 - x_offset

    # Get the ROI from the background
    background_roi = background_img[y1:y2, x1:x2]

    # Get the ROI from the foreground BGR and alpha channels
    foreground_roi = foreground_bgr[y1_fg:y2_fg, x1_fg:x2_fg]
    alpha_roi = alpha_channel[y1_fg:y2_fg, x1_fg:x2_fg]

    # Reshape alpha for broadcasting (to multiply with 3 color channels of BGR)
    alpha_roi_reshaped = alpha_roi[:, :, np.newaxis]

    # Blend the images using alpha compositing formula:
    # Output = Foreground_Color * Alpha + Background_Color * (1 - Alpha)
    blended_roi = (foreground_roi * alpha_roi_reshaped + 
                   background_roi * (1 - alpha_roi_reshaped)).astype(np.uint8)

    # Place the blended ROI back into the background image
    background_img[y1:y2, x1:x2] = blended_roi

    return background_img


class ImageGenerator:
    """
    Generates synthetic Thai license plate images.
    """
    def __init__(self, save_path):
        self.save_path = save_path
        
        # Get the directory of the current script to form absolute paths
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Load Plate background image (general default)
        plate_path = os.path.join(script_dir, "plate.jpg")
        self.plate = cv2.imread(plate_path, cv2.IMREAD_COLOR) # Ensure loading as BGR (3 channels)
        if self.plate is None:
            print(f"ERROR: Failed to load plate image: {plate_path}")
            exit() # Exit if a critical asset like plate.jpg is missing

        # Initialize the dictionary to store special plate backgrounds
        self.special_plates = {} 

        # Load special plate background images and add to special_plates dictionary
        # --- Add all your special plate loading here ---
        # Example: AQ Plate
        aq_plate_path = os.path.join(script_dir, "AQ_plate.jpg")
        aq_plate_img = cv2.imread(aq_plate_path, cv2.IMREAD_COLOR)
        if aq_plate_img is None:
            print(f"WARNING: Failed to load AQ plate image: {aq_plate_path}. 'AQ' plates might use default 'plate.jpg'.")
        self.special_plates['AQ'] = aq_plate_img 

        # BJ Plate
        bj_plate_path = os.path.join(script_dir, "BJ_plate.jpg")
        bj_plate_img = cv2.imread(bj_plate_path, cv2.IMREAD_COLOR)
        if bj_plate_img is None:
            print(f"WARNING: Failed to load BJ plate image: {bj_plate_path}. 'BJ' plates might use default 'plate.jpg'.")
        self.special_plates['BJ'] = bj_plate_img
        
        # BU Plate
        bu_plate_path = os.path.join(script_dir, "BU_plate.jpg")
        bu_plate_img = cv2.imread(bu_plate_path, cv2.IMREAD_COLOR)
        if bu_plate_img is None:
            print(f"WARNING: Failed to load BU plate image: {bu_plate_path}. 'BU' plates might use default 'plate.jpg'.")
        self.special_plates['BU'] = bu_plate_img

        # BY Plate
        by_plate_path = os.path.join(script_dir, "BY_plate.jpg")
        by_plate_img = cv2.imread(by_plate_path, cv2.IMREAD_COLOR)
        if by_plate_img is None:
            print(f"WARNING: Failed to load BY plate image: {by_plate_path}. 'BY' plates might use default 'plate.jpg'.")
        self.special_plates['BY'] = by_plate_img 

        # CD Plate
        cd_plate_path = os.path.join(script_dir, "CD_plate.jpg")
        cd_plate_img = cv2.imread(cd_plate_path, cv2.IMREAD_COLOR)
        if cd_plate_img is None:
            print(f"WARNING: Failed to load CD plate image: {cd_plate_path}. 'CD' plates might use default 'plate.jpg'.")
        self.special_plates['CD'] = cd_plate_img

        # CQ Plate
        cq_plate_path = os.path.join(script_dir, "CQ_plate.jpg")
        cq_plate_img = cv2.imread(cq_plate_path, cv2.IMREAD_COLOR)
        if cq_plate_img is None:
            print(f"WARNING: Failed to load CQ plate image: {cq_plate_path}. 'CQ' plates might use default 'plate.jpg'.")
        self.special_plates['CQ'] = cq_plate_img

        ae_plate_path = os.path.join(script_dir, "AE_plate.jpg")
        ae_plate_img = cv2.imread(ae_plate_path, cv2.IMREAD_COLOR)
        if ae_plate_img is None:
            print(f"WARNING: Failed to load AE plate image: {ae_plate_path}. 'AE' plates might use default 'plate.jpg'.")
        self.special_plates['AE'] = ae_plate_img

        aj_plate_path = os.path.join(script_dir, "AJ_plate.jpg")
        aj_plate_img = cv2.imread(aj_plate_path, cv2.IMREAD_COLOR)
        if aj_plate_img is None:
            print(f"WARNING: Failed to load AJ plate image: {aj_plate_path}. 'AJ' plates might use default 'plate.jpg'.")
        self.special_plates['AJ'] = aj_plate_img

        af_plate_path = os.path.join(script_dir, "AF_plate.jpg")
        af_plate_img = cv2.imread(af_plate_path, cv2.IMREAD_COLOR)
        if af_plate_img is None:
            print(f"WARNING: Failed to load AF plate image: {af_plate_path}. 'AF' plates might use default 'plate.jpg'.")
        self.special_plates['AF'] = af_plate_img

        cj_plate_path = os.path.join(script_dir, "CJ_plate.jpg")
        cj_plate_img = cv2.imread(cj_plate_path, cv2.IMREAD_COLOR)
        if cj_plate_img is None:
            print(f"WARNING: Failed to load CJ plate image: {cj_plate_path}. 'CJ' plates might use default 'plate.jpg'.")
        self.special_plates['CJ'] = cj_plate_img

        as_plate_path = os.path.join(script_dir, "AS_plate.jpg")
        as_plate_img = cv2.imread(as_plate_path, cv2.IMREAD_COLOR)
        if as_plate_img is None:
            print(f"WARNING: Failed to load AS plate image: {as_plate_path}. 'AS' plates might use default 'plate.jpg'.")
        self.special_plates['AS'] = as_plate_img

        cc_plate_path = os.path.join(script_dir, "CC_plate.jpg")
        cc_plate_img = cv2.imread(cc_plate_path, cv2.IMREAD_COLOR)
        if cc_plate_img is None:
            print(f"WARNING: Failed to load CC plate image: {cc_plate_path}. 'CC' plates might use default 'plate.jpg'.")
        self.special_plates['CC'] = cc_plate_img

        bb_plate_path = os.path.join(script_dir, "BB_plate.jpg")
        bb_plate_img = cv2.imread(bb_plate_path, cv2.IMREAD_COLOR)
        if bb_plate_img is None:
            print(f"WARNING: Failed to load BB plate image: {bb_plate_path}. 'BB' plates might use default 'plate.jpg'.")
        self.special_plates['BB'] = bb_plate_img

        cn_plate_path = os.path.join(script_dir, "CN_plate.jpg")
        cn_plate_img = cv2.imread(cn_plate_path, cv2.IMREAD_COLOR)
        if cn_plate_img is None:
            print(f"WARNING: Failed to load CN plate image: {cn_plate_path}. 'CN' plates might use default 'plate.jpg'.")
        self.special_plates['CN'] = cn_plate_img

        at_plate_path = os.path.join(script_dir, "AT_plate.jpg")
        at_plate_img = cv2.imread(at_plate_path, cv2.IMREAD_COLOR)
        if at_plate_img is None:
            print(f"WARNING: Failed to load AT plate image: {at_plate_path}. 'AT' plates might use default 'plate.jpg'.")
        self.special_plates['AT'] = at_plate_img

        cb_plate_path = os.path.join(script_dir, "CB_plate.jpg")
        cb_plate_img = cv2.imread(cb_plate_path, cv2.IMREAD_COLOR)
        if cb_plate_img is None:
            print(f"WARNING: Failed to load CB plate image: {cb_plate_path}. 'CB' plates might use default 'plate.jpg'.")
        self.special_plates['CB'] = cb_plate_img

        be_plate_path = os.path.join(script_dir, "BE_plate.jpg")
        be_plate_img = cv2.imread(be_plate_path, cv2.IMREAD_COLOR)
        if be_plate_img is None:
            print(f"WARNING: Failed to load BE plate image: {be_plate_path}. 'BE' plates might use default 'plate.jpg'.")
        self.special_plates['BE'] = be_plate_img

        bt_plate_path = os.path.join(script_dir, "BT_plate.jpg")
        bt_plate_img = cv2.imread(bt_plate_path, cv2.IMREAD_COLOR)
        if bt_plate_img is None:
            print(f"WARNING: Failed to load BT plate image: {bt_plate_path}. 'BT' plates might use default 'plate.jpg'.")
        self.special_plates['BT'] = bt_plate_img

        cp_plate_path = os.path.join(script_dir, "CP_plate.jpg")
        cp_plate_img = cv2.imread(cp_plate_path, cv2.IMREAD_COLOR)
        if cp_plate_img is None:
            print(f"WARNING: Failed to load CP plate image: {cp_plate_path}. 'CP' plates might use default 'plate.jpg'.")
        self.special_plates['CP'] = cp_plate_img

        al_plate_path = os.path.join(script_dir, "AL_plate.jpg")
        al_plate_img = cv2.imread(al_plate_path, cv2.IMREAD_COLOR)
        if al_plate_img is None:
            print(f"WARNING: Failed to load AL plate image: {al_plate_path}. 'AL' plates might use default 'plate.jpg'.")
        self.special_plates['AL'] = al_plate_img

        bl_plate_path = os.path.join(script_dir, "BL_plate.jpg")
        bl_plate_img = cv2.imread(bl_plate_path, cv2.IMREAD_COLOR)
        if bl_plate_img is None:
            print(f"WARNING: Failed to load BL plate image: {bl_plate_path}. 'BL' plates might use default 'plate.jpg'.")
        self.special_plates['BL'] = bl_plate_img

        ac_plate_path = os.path.join(script_dir, "AC_plate.jpg")
        ac_plate_img = cv2.imread(ac_plate_path, cv2.IMREAD_COLOR)
        if ac_plate_img is None:
            print(f"WARNING: Failed to load AC plate image: {ac_plate_path}. 'AC' plates might use default 'plate.jpg'.")
        self.special_plates['AC'] = ac_plate_img

        ak_plate_path = os.path.join(script_dir, "AK_plate.jpg")
        ak_plate_img = cv2.imread(ak_plate_path, cv2.IMREAD_COLOR)
        if ak_plate_img is None:
            print(f"WARNING: Failed to load AK plate image: {ak_plate_path}. 'AK' plates might use default 'plate.jpg'.")
        self.special_plates['AK'] = ak_plate_img

        bd_plate_path = os.path.join(script_dir, "BD_plate.jpg")
        bd_plate_img = cv2.imread(bd_plate_path, cv2.IMREAD_COLOR)
        if bd_plate_img is None:
            print(f"WARNING: Failed to load BD plate image: {bd_plate_path}. 'BD' plates might use default 'plate.jpg'.")
        self.special_plates['BD'] = bd_plate_img

        bq_plate_path = os.path.join(script_dir, "BQ_plate.jpg")
        bq_plate_img = cv2.imread(bq_plate_path, cv2.IMREAD_COLOR)
        if bq_plate_img is None:
            print(f"WARNING: Failed to load BQ plate image: {bq_plate_path}. 'BQ' plates might use default 'plate.jpg'.")
        self.special_plates['BQ'] = bq_plate_img

        cv_plate_path = os.path.join(script_dir, "CV_plate.jpg")
        cv_plate_img = cv2.imread(cv_plate_path, cv2.IMREAD_COLOR)
        if cv_plate_img is None:
            print(f"WARNING: Failed to load CV plate image: {cv_plate_path}. 'CV' plates might use default 'plate.jpg'.")
        self.special_plates['CV'] = cv_plate_img

        # --- IMPORTANT: Commented out if you don't have DV_plate.jpg ---
        # dv_plate_path = os.path.join(script_dir, "DV_plate.jpg")
        # dv_plate_img = cv2.imread(dv_plate_path, cv2.IMREAD_COLOR)
        # if dv_plate_img is None:
        #     print(f"WARNING: Failed to load DV plate image: {dv_plate_path}. 'DV' plates might use default 'plate.jpg'.")
        # self.special_plates['DV'] = dv_plate_img
        # --- End of DV_plate.jpg comment block ---

        av_plate_path = os.path.join(script_dir, "AV_plate.jpg")
        av_plate_img = cv2.imread(av_plate_path, cv2.IMREAD_COLOR)
        if av_plate_img is None:
            print(f"WARNING: Failed to load AV plate image: {av_plate_path}. 'AV' plates might use default 'plate.jpg'.")
        self.special_plates['AV'] = av_plate_img

        bf_plate_path = os.path.join(script_dir, "BF_plate.jpg")
        bf_plate_img = cv2.imread(bf_plate_path, cv2.IMREAD_COLOR)
        if bf_plate_img is None:
            print(f"WARNING: Failed to load BF plate image: {bf_plate_path}. 'BF' plates might use default 'plate.jpg'.")
        self.special_plates['BF'] = bf_plate_img

        bp_plate_path = os.path.join(script_dir, "BP_plate.jpg")
        bp_plate_img = cv2.imread(bp_plate_path, cv2.IMREAD_COLOR)
        if bp_plate_img is None:
            print(f"WARNING: Failed to load BP plate image: {bp_plate_path}. 'BP' plates might use default 'plate.jpg'.")
        self.special_plates['BP'] = bp_plate_img

        br_plate_path = os.path.join(script_dir, "BR_plate.jpg")
        br_plate_img = cv2.imread(br_plate_path, cv2.IMREAD_COLOR)
        if br_plate_img is None:
            print(f"WARNING: Failed to load BR plate image: {br_plate_path}. 'BR' plates might use default 'plate.jpg'.")
        self.special_plates['BR'] = br_plate_img

        bw_plate_path = os.path.join(script_dir, "BW_plate.jpg")
        bw_plate_img = cv2.imread(bw_plate_path, cv2.IMREAD_COLOR)
        if bw_plate_img is None:
            print(f"WARNING: Failed to load BW plate image: {bw_plate_path}. 'BW' plates might use default 'plate.jpg'.")
        self.special_plates['BW'] = bw_plate_img

        bx_plate_path = os.path.join(script_dir, "BX_plate.jpg")
        bx_plate_img = cv2.imread(bx_plate_path, cv2.IMREAD_COLOR)
        if bx_plate_img is None:
            print(f"WARNING: Failed to load BX plate image: {bx_plate_path}. 'BX' plates might use default 'plate.jpg'.")
        self.special_plates['BX'] = bx_plate_img

        cf_plate_path = os.path.join(script_dir, "CF_plate.jpg")
        cf_plate_img = cv2.imread(cf_plate_path, cv2.IMREAD_COLOR)
        if cf_plate_img is None:
            print(f"WARNING: Failed to load CF plate image: {cf_plate_path}. 'CF' plates might use default 'plate.jpg'.")
        self.special_plates['CF'] = cf_plate_img

        cr_plate_path = os.path.join(script_dir, "CR_plate.jpg")
        cr_plate_img = cv2.imread(cr_plate_path, cv2.IMREAD_COLOR)
        if cr_plate_img is None:
            print(f"WARNING: Failed to load CR plate image: {cr_plate_path}. 'CR' plates might use default 'plate.jpg'.")
        self.special_plates['CR'] = cr_plate_img

        bk_plate_path = os.path.join(script_dir, "BK_plate.jpg")
        bk_plate_img = cv2.imread(bk_plate_path, cv2.IMREAD_COLOR)
        if bk_plate_img is None:
            print(f"WARNING: Failed to load BK plate image: {bk_plate_path}. 'BK' plates might use default 'plate.jpg'.")
        self.special_plates['BK'] = bk_plate_img

        bv_plate_path = os.path.join(script_dir, "BV_plate.jpg")
        bv_plate_img = cv2.imread(bv_plate_path, cv2.IMREAD_COLOR)
        if bv_plate_img is None:
            print(f"WARNING: Failed to load BV plate image: {bv_plate_path}. 'BV' plates might use default 'plate.jpg'.")
        self.special_plates['BV'] = bv_plate_img

        co_plate_path = os.path.join(script_dir, "CO_plate.jpg")
        co_plate_img = cv2.imread(co_plate_path, cv2.IMREAD_COLOR)
        if co_plate_img is None:
            print(f"WARNING: Failed to load CO plate image: {co_plate_path}. 'CO' plates might use default 'plate.jpg'.")
        self.special_plates['CO'] = co_plate_img

        ct_plate_path = os.path.join(script_dir, "CT_plate.jpg")
        ct_plate_img = cv2.imread(ct_plate_path, cv2.IMREAD_COLOR)
        if ct_plate_img is None:
            print(f"WARNING: Failed to load CT plate image: {ct_plate_path}. 'CT' plates might use default 'plate.jpg'.")
        self.special_plates['CT'] = ct_plate_img

        cw_plate_path = os.path.join(script_dir, "CW_plate.jpg")
        cw_plate_img = cv2.imread(cw_plate_path, cv2.IMREAD_COLOR)
        if cw_plate_img is None:
            print(f"WARNING: Failed to load CW plate image: {cw_plate_path}. 'CW' plates might use default 'plate.jpg'.")
        self.special_plates['CW'] = cw_plate_img

        # --- Removed self.special_plate_provinces as per your request ---

        # Helper function to load images from a directory
        def load_images_from_dir(directory_name):
            full_path = os.path.join(script_dir, directory_name)
            images = []
            names = []
            if os.path.exists(full_path):
                file_list = os.listdir(full_path)
                # Sort files to ensure consistent order (e.g., 0.png, 1.png, ..., 9.png)
                file_list.sort() 
                for file_name in file_list:
                    # Filter out non-image files if any
                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(full_path, file_name)
                        # Load PNG with IMREAD_UNCHANGED to get Alpha Channel
                        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) 
                        if img is None:
                            print(f"WARNING: Failed to load image: {img_path}")
                        images.append(img)
                        names.append(os.path.splitext(file_name)[0]) # Get name without extension
            else:
                print(f"ERROR: Directory not found: {full_path}")
                exit() # Exit if a required directory is missing
            return images, names

        # Loading Number 0-9
        self.Number, self.number_list = load_images_from_dir("num")
            
        # Loading Number 1-9 (numbers that start a sequence)
        self.NumberS, self.numberS_list = load_images_from_dir("numS")

        # Loading Front Number
        self.NumberF, self.numberF_list = load_images_from_dir("num_front")
            
        # Loading Char
        self.Char1, self.char_list = load_images_from_dir("char1")

        # Loading Province
        self.Province, self.province_list = load_images_from_dir("province")
            
    def Type_4(self, num, save=False, plate_type=None, z_rotation=0, no_blur=False): # เพิ่ม no_blur=False
        """
        Generates license plates of Type 4 format.
        Format: [NumberF][Char1][Char1][NumberS][Number][Number][Number][Province]
        e.g., 1กข1234กรุงเทพมหานคร

        Args:
            num (int): Number of images to generate.
            save (bool): True to save images, False to display.
            plate_type (str, optional): Specific plate background type (e.g., 'AQ', 'BJ').
                                        If None, uses random selection as before.
            z_rotation (int, optional): The rotation angle in degrees for Z-axis.
                                        If 0, no rotation is applied.
            no_blur (bool, optional): If True, no blurring augmentation will be applied.
        """
        top_w, top_h = 35, 80
        top_char_w, top_char_h = 40, 80
        bot_w, bot_h = 200, 40

        # Resize all loaded images once to improve performance
        # Filter out None values and resize. Expecting 4-channel for overlay.
        numberF_resized = [cv2.resize(number, (top_w, top_h)) for number in self.NumberF if number is not None]
        char_resized = [cv2.resize(char1, (top_char_w, top_char_h)) for char1 in self.Char1 if char1 is not None]
        numberS_resized = [cv2.resize(number, (top_w, top_h)) for number in self.NumberS if number is not None]
        number2_resized = [cv2.resize(number, (top_w, top_h)) for number in self.Number if number is not None]
        province1_resized = [cv2.resize(province, (bot_w, bot_h)) for province in self.Province if province is not None]
        
        # Ensure lists are not empty before getting length
        if not char_resized:
            print("ERROR: 'char_resized' list is empty. Check char1 folder content.")
            return
        if not province1_resized:
            print("ERROR: 'province1_resized' list is empty. Check province folder content.")
            return
        if not numberF_resized:
            print("ERROR: 'numberF_resized' list is empty. Check num_front folder content.")
            return
        if not numberS_resized:
            print("ERROR: 'numberS_resized' list is empty. Check numS folder content.")
            return
        if not number2_resized:
            print("ERROR: 'number2_resized' list is empty. Check num folder content.")
            return

        char_len = len(char_resized) - 1
        province1_len = len(province1_resized) - 1
        numberF_len = len(numberF_resized) - 1
        numberS_len = len(numberS_resized) - 1
        number2_len = len(number2_resized) - 1

        for i, Iter in enumerate(range(num)):
            label = str()

            Plate_to_use = self.plate # Default plate
            selected_province_name = None 
            rand_province_idx = -1 

            # Determine which plate background to use based on plate_type argument
            if plate_type and plate_type in self.special_plates and self.special_plates[plate_type] is not None:
                Plate_to_use = self.special_plates[plate_type]
                
                # --- MODIFIED LOGIC: Use plate_type directly as province name ---
                selected_province_name = plate_type # Use the plate_type as the desired province name
                try:
                    rand_province_idx = self.province_list.index(selected_province_name)
                except ValueError:
                    print(f"WARNING: Province image '{selected_province_name}' for plate type '{plate_type}' not found in province list. Falling back to random province for text.")
                    # Fallback to random if the specified province image isn't found
                    rand_province_idx = random.randint(0, province1_len)
                    selected_province_name = self.province_list[rand_province_idx]
                # --- END MODIFIED LOGIC ---
            else:
                # Original random province selection and 50% chance for special plates based on province name
                rand_province_idx = random.randint(0, province1_len)
                selected_province_name = self.province_list[rand_province_idx]
                if selected_province_name in self.special_plates and \
                   self.special_plates[selected_province_name] is not None and \
                   random.random() < 0.5:
                    Plate_to_use = self.special_plates[selected_province_name]
            
            # Error check for rand_province_idx in case of unexpected scenarios (e.g. empty province list)
            if rand_province_idx == -1: # This should be handled by the logic above, but as a safeguard
                rand_province_idx = random.randint(0, province1_len)
                selected_province_name = self.province_list[rand_province_idx]

            Plate = cv2.resize(Plate_to_use.copy(), (330, 150))
            # --- End Plate/Province Logic ---

            b_width, b_height = 400, 800         
            random_R, random_G, random_B = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            background = np.zeros((b_width, b_height, 3), np.uint8)
            cv2.rectangle(background, (0, 0), (b_height, b_width), (random_R, random_G, random_B), -1)
            
            # row -> y , col -> x (coordinates for placing characters on the plate)
            row, col = 10, 25

            # number Front
            rand_int = random.randint(0, numberF_len)
            label += self.numberF_list[rand_int]
            Plate = overlay_image_alpha(Plate, numberF_resized[rand_int], col, row)
            col += top_w

            # character 1
            rand_int = random.randint(0, char_len)
            label += self.char_list[rand_int]
            Plate = overlay_image_alpha(Plate, char_resized[rand_int], col, row)
            col += top_char_w
            
            # character 2
            rand_int = random.randint(0, char_len)
            label += self.char_list[rand_int]
            Plate = overlay_image_alpha(Plate, char_resized[rand_int], col, row)
            col += top_w + 20 # Add extra space between character and starting number

            # number Start (1-9)
            rand_int = random.randint(0, numberS_len)
            label += self.numberS_list[rand_int]
            Plate = overlay_image_alpha(Plate, numberS_resized[rand_int], col, row)
            col += top_w

            # number 2 (0-9)
            rand_int = random.randint(0, number2_len)
            label += self.number_list[rand_int]
            Plate = overlay_image_alpha(Plate, number2_resized[rand_int], col, row)
            col += top_w

            # number 3 (0-9)
            rand_int = random.randint(0, number2_len)
            label += self.number_list[rand_int]
            Plate = overlay_image_alpha(Plate, number2_resized[rand_int], col, row)
            col += top_w

            # number 4 (0-9)
            rand_int = random.randint(0, number2_len)
            label += self.number_list[rand_int]
            Plate = overlay_image_alpha(Plate, number2_resized[rand_int], col, row)

            row, col = 97, 65 # new line for province (adjust coordinates as needed)
            
            # province (use the selected province for text)
            label += selected_province_name
            Plate = overlay_image_alpha(Plate, province1_resized[rand_province_idx], col, row)

            # --- Apply Z-rotation to the Plate image before placing it on background ---
            if z_rotation != 0:
                Plate = rotate_image(Plate, z_rotation)
            # --- End Z-rotation ---

            # Place the generated plate onto the background
            s_width, s_height = int((b_width - Plate.shape[0]) / 2), int((b_height - Plate.shape[1]) / 2)
            # Ensure Plate is 3-channel BGR before placing on 3-channel background
            if Plate.shape[2] == 4:
                Plate_bgr = cv2.cvtColor(Plate, cv2.COLOR_BGRA2BGR)
            else:
                Plate_bgr = Plate
                
            background[s_width:Plate.shape[0] + s_width, s_height:Plate.shape[1] + s_height, :] = Plate_bgr
            
            # Apply image augmentation to the final background image
            # ส่งค่า apply_blur โดยใช้ not no_blur (ถ้า no_blur เป็น True, apply_blur จะเป็น False)
            background = image_augmentation(background, type2=True, apply_blur=not no_blur)

            # Save or display the image
            if save:
                output_filename = os.path.join(self.save_path, label + ".jpg")
                cv2.imwrite(output_filename, background)
            else:
                cv2.imshow(label, background)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


# Argument parsing setup
parser = argparse.ArgumentParser(description="Generate synthetic Thai license plate images.")
parser.add_argument("-i", "--img_dir", help="Directory to save generated images",
                    type=str, default="../CRNN/DB/")
parser.add_argument("-n", "--num", help="Number of images to generate",
                    type=int, required=True)
parser.add_argument("-s", "--save", help="Set to True to save images, False to display them",
                    action='store_true', default=False)
parser.add_argument("-z", "--z_rotation", help="Rotation angle in degrees (Z-axis). Positive for counter-clockwise.",
                    type=int, default=0)
parser.add_argument("-p", "--plate_type", help="Specific plate background type (e.g., 'AQ', 'BJ'). If not provided, random.",
                    type=str, default=None) 
parser.add_argument("--no_blur", help="Set to True to disable blurring augmentation.", # เพิ่ม argument ใหม่
                    action='store_true', default=False)
args = parser.parse_args()

# Main execution block
img_dir = args.img_dir
# Ensure the save directory exists if saving
if args.save and not os.path.exists(img_dir):
    os.makedirs(img_dir)
    print(f"Created output directory: {img_dir}")

A = ImageGenerator(img_dir)

num_img = args.num
Save = args.save

# Pass all relevant arguments to Type_4, including z_rotation, plate_type, and no_blur
A.Type_4(num_img, save=Save, plate_type=args.plate_type, z_rotation=args.z_rotation, no_blur=args.no_blur)
print("Type 4 finish")