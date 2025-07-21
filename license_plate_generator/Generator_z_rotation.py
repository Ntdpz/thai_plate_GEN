import os
import random
import cv2
import argparse
import numpy as np
import math

# --- ฟังก์ชันช่วยเหลือใหม่: การหมุน 3 มิติ ---
def apply_3d_rotation(image, rx, ry, rz):
    """
    ทำการหมุน 3 มิติ (roll, pitch, yaw) บนภาพโดยใช้ Perspective Transform

    อาร์กิวเมนต์:
        image (numpy.ndarray): รูปภาพอินพุต (BGR หรือ BGRA)
        rx (float): มุมการหมุนรอบแกน X (Pitch) เป็นองศา
        ry (float): มุมการหมุนรอบแกน Y (Yaw) เป็นองศา
        rz (float): มุมการหมุนรอบแกน Z (Roll) เป็นองศา

    คืนค่า:
        numpy.ndarray: รูปภาพที่หมุนแล้วพร้อม Perspective Transform
    """
    h, w = image.shape[:2]
    
    # แปลงมุมจากองศาเป็นเรเดียน
    rx = math.radians(rx)
    ry = math.radians(ry)
    rz = math.radians(rz)

    # เมทริกซ์การหมุน (จากจุดศูนย์กลางภาพ)
    # หมุนรอบแกน X (Pitch)
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(rx), -math.sin(rx)],
        [0, math.sin(rx), math.cos(rx)]
    ])

    # หมุนรอบแกน Y (Yaw)
    Ry = np.array([
        [math.cos(ry), 0, math.sin(ry)],
        [0, 1, 0],
        [-math.sin(ry), 0, math.cos(ry)]
    ])

    # หมุนรอบแกน Z (Roll)
    Rz = np.array([
        [math.cos(rz), -math.sin(rz), 0],
        [math.sin(rz), math.cos(rz), 0],
        [0, 0, 1]
    ])

    # รวมเมทริกซ์การหมุน: R = Rz @ Ry @ Rx
    R = Rz @ Ry @ Rx

    # สร้างจุด 3 มิติของมุมภาพ (อ้างอิงจากจุดศูนย์กลาง)
    # จุด (x, y, z) โดยที่ z เป็น 0 สำหรับภาพ 2 มิติเริ่มต้น
    points_3d = np.array([
        [-w/2, -h/2, 0],  # Top-left
        [-w/2, h/2, 0],   # Bottom-left
        [w/2, -h/2, 0],   # Top-right
        [w/2, h/2, 0]     # Bottom-right
    ])

    # จุด 3 มิติที่หมุนแล้ว
    rotated_points = points_3d @ R.T # R.T คือ transpose ของ R

    # การฉายภาพแบบ Perspective (สมมติกล้องอยู่ห่างออกไป)
    # จุดกำเนิดภาพอยู่ที่ (0,0) (Top-left)
    # F = ระยะโฟกัส (ค่าที่ใหญ่ขึ้น = การบิดเบือนน้อยลง)
    F = w * 1.5 # ปรับค่า F เพื่อควบคุมผลกระทบของ Perspective
    
    # สร้างเมทริกซ์กล้องแบบง่ายๆ (การฉายภาพแบบ Perspective)
    K = np.array([
        [F, 0, w/2],
        [0, F, h/2],
        [0, 0, 1]
    ])

    # แปลงจุด 3 มิติที่หมุนแล้วให้เป็น 2 มิติ (จุดบนระนาบภาพ)
    projected_points = []
    for p in rotated_points:
        z_prime = p[2] + F 
        if z_prime == 0: 
            z_prime = 1e-6 
        
        x_prime = (p[0] * F) / z_prime + w/2
        y_prime = (p[1] * F) / z_prime + h/2
        projected_points.append([x_prime, y_prime])
        
    projected_points = np.float32(projected_points)

    # จุดต้นฉบับบนภาพ 2 มิติ (ก่อนหมุน)
    pts1 = np.float32([[0, 0], [0, h], [w, 0], [w, h]]) 
    
    # รับเมทริกซ์ Perspective Transform
    M = cv2.getPerspectiveTransform(pts1, projected_points)

    # ใช้ Perspective Transform กับรูปภาพ
    rotated_image = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated_image
# --- สิ้นสุดฟังก์ชันช่วยเหลือใหม่ ---

def image_augmentation(img, type2=False):
    h, w, _ = img.shape
    
    if random.random() < 0.5:
        pts1 = np.float32([[0, 0], [0, h], [w, 0], [w, h]])
        begin, end = 30, 90
        pts2 = np.float32([[random.randint(begin, end), random.randint(begin, end)],
                           [random.randint(begin, end), h - random.randint(begin, end)],
                           [w - random.randint(begin, end), random.randint(begin, end)],
                           [w - random.randint(begin, end), h - random.randint(begin, end)]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M, (w, h))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = np.array(img, dtype=np.float64)
    random_bright = .4 + np.random.uniform()
    img[:, :, 2] = img[:, :, 2] * random_bright
    img[:, :, 2][img[:, :, 2] > 255] = 255
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    if random.random() < 0.5:
        blur_value = random.randint(0,4) * 2 + 1
        img = cv2.blur(img,(blur_value, blur_value))

    if type2:
        return img[130:280, 220:560, :]
    return img[130:280, 120:660, :]


def overlay_image_alpha(background_img, foreground_img_rgba, x_offset, y_offset):
    if foreground_img_rgba is None or foreground_img_rgba.shape[2] != 4:
        return background_img

    foreground_bgr = foreground_img_rgba[:, :, :3]
    alpha_channel = foreground_img_rgba[:, :, 3] / 255.0

    h_fg, w_fg = foreground_bgr.shape[:2]
    h_bg, w_bg = background_img.shape[:2]

    y1, y2 = max(0, y_offset), min(h_bg, y_offset + h_fg)
    x1, x2 = max(0, x_offset), min(w_bg, x_offset + w_fg)

    y1_fg = y1 - y_offset
    y2_fg = y2 - y_offset
    x1_fg = x1 - x_offset
    x2_fg = x2 - x_offset

    background_roi = background_img[y1:y2, x1:x2]
    foreground_roi = foreground_bgr[y1_fg:y2_fg, x1_fg:x2_fg]
    alpha_roi = alpha_channel[y1_fg:y2_fg, x1_fg:x2_fg]

    alpha_roi_reshaped = alpha_roi[:, :, np.newaxis]

    blended_roi = (foreground_roi * alpha_roi_reshaped + 
                   background_roi * (1 - alpha_roi_reshaped)).astype(np.uint8)

    background_img[y1:y2, x1:x2] = blended_roi

    return background_img


class ImageGenerator:
    def __init__(self, save_path):
        self.save_path = save_path
        
        script_dir = os.path.dirname(os.path.abspath(__file__))

        plate_path = os.path.join(script_dir, "plate.jpg")
        self.plate = cv2.imread(plate_path, cv2.IMREAD_COLOR)
        if self.plate is None:
            print(f"ERROR: Failed to load plate image: {plate_path}")
            exit()

        self.special_plates = {} 
        # โหลดป้ายพิเศษทั้งหมดของคุณที่นี่ (เหมือนโค้ดเดิม)
        # ตัวอย่าง:
        aq_plate_path = os.path.join(script_dir, "AQ_plate.jpg")
        aq_plate_img = cv2.imread(aq_plate_path, cv2.IMREAD_COLOR)
        if aq_plate_img is None:
            print(f"WARNING: Failed to load AQ plate image: {aq_plate_path}. 'AQ' plates might use default 'plate.jpg'.")
        self.special_plates['AQ'] = aq_plate_img 

        bj_plate_path = os.path.join(script_dir, "BJ_plate.jpg")
        bj_plate_img = cv2.imread(bj_plate_path, cv2.IMREAD_COLOR)
        if bj_plate_img is None:
            print(f"WARNING: Failed to load BJ plate image: {bj_plate_path}. 'BJ' plates might use default 'plate.jpg'.")
        self.special_plates['BJ'] = bj_plate_img
        
        bu_plate_path = os.path.join(script_dir, "BU_plate.jpg")
        bu_plate_img = cv2.imread(bu_plate_path, cv2.IMREAD_COLOR)
        if bu_plate_img is None:
            print(f"WARNING: Failed to load BU plate image: {bu_plate_path}. 'BU' plates might use default 'plate.jpg'.")
        self.special_plates['BU'] = bu_plate_img

        by_plate_path = os.path.join(script_dir, "BY_plate.jpg")
        by_plate_img = cv2.imread(by_plate_path, cv2.IMREAD_COLOR)
        if by_plate_img is None:
            print(f"WARNING: Failed to load BY plate image: {by_plate_path}. 'BY' plates might use default 'plate.jpg'.")
        self.special_plates['BY'] = by_plate_img 

        cd_plate_path = os.path.join(script_dir, "CD_plate.jpg")
        cd_plate_img = cv2.imread(cd_plate_path, cv2.IMREAD_COLOR)
        if cd_plate_img is None:
            print(f"WARNING: Failed to load CD plate image: {cd_plate_path}. 'CD' plates might use default 'plate.jpg'.")
        self.special_plates['CD'] = cd_plate_img

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

        av_plate_path = os.path.join(script_dir, "AV_plate.jpg")
        av_plate_img = cv2.imread(av_plate_path, cv2.IMREAD_COLOR)
        if av_plate_img is None:
            print(f"WARNING: Failed to load AV plate image: {av_plate_path}. 'AV' plates might use default 'plate.jpg'.")
        self.special_plates['AV'] = av_plate_img

        da_plate_path = os.path.join(script_dir, "DA_plate.jpg")
        da_plate_img = cv2.imread(da_plate_path, cv2.IMREAD_COLOR)
        if da_plate_img is None:
            print(f"WARNING: Failed to load DA plate image: {da_plate_path}. 'DA' plates might use default 'plate.jpg'.")
        self.special_plates['DA'] = da_plate_img


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


        self.color_plates = {}
        white_plate_path = os.path.join(script_dir, "plate.jpg")
        self.color_plates['ขาว'] = cv2.imread(white_plate_path, cv2.IMREAD_COLOR)
        if self.color_plates['ขาว'] is None:
             print(f"ERROR: Failed to load default white plate image: {white_plate_path}. Defaulting to solid white.")
             self.color_plates['ขาว'] = np.full((150, 330, 3), 255, dtype=np.uint8) 

        self.color_plates['เหลือง'] = np.full((150, 330, 3), (0, 255, 255), dtype=np.uint8) 
        self.color_plates['เขียว'] = np.full((150, 330, 3), (0, 128, 0), dtype=np.uint8) 
        self.color_plates['แดง'] = np.full((150, 330, 3), (0, 0, 255), dtype=np.uint8) 
        
        for color, img in self.color_plates.items():
            if img is None:
                print(f"WARNING: Plate background for color '{color}' is missing. It will default to a solid color if specified.")


        def load_images_from_dir(directory_name):
            full_path = os.path.join(script_dir, directory_name)
            images = []
            names = []
            if os.path.exists(full_path):
                file_list = os.listdir(full_path)
                file_list.sort() 
                for file_name in file_list:
                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(full_path, file_name)
                        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) 
                        if img is None:
                            print(f"WARNING: Failed to load image: {img_path}")
                        images.append(img)
                        names.append(os.path.splitext(file_name)[0])
            else:
                print(f"ERROR: Directory not found: {full_path}")
                exit()
            return images, names

        self.Number, self.number_list = load_images_from_dir("num")
        self.NumberS, self.numberS_list = load_images_from_dir("numS")
        self.NumberF, self.numberF_list = load_images_from_dir("num_front")
        self.Char1, self.char_list = load_images_from_dir("char1")
        self.Province, self.province_list = load_images_from_dir("province")
        self.background_images, _ = load_images_from_dir("background_images")
        if not self.background_images:
            print("WARNING: ไม่พบรูปภาพในไดเรกทอรี 'background_images' จะใช้พื้นหลังสีทึบเป็นทางเลือกแทน")
            
    def Type_4(self, num, save=False, plate_type=None, rx_rotation=0, ry_rotation=0, rz_rotation=0, plate_color=None, plate_width=330, plate_height=150):
        """
        สร้างป้ายทะเบียนรูปแบบ Type 4
        รูปแบบ: [ตัวเลขด้านหน้า][ตัวอักษร1][ตัวอักษร1][ตัวเลขเริ่มต้น][ตัวเลข][ตัวเลข][ตัวเลข][จังหวัด]

        อาร์กิวเมนต์:
            num (int): จำนวนรูปภาพที่ต้องการสร้าง
            save (bool): True เพื่อบันทึกรูปภาพ, False เพื่อแสดงผล
            plate_type (str, optional): ประเภทพื้นหลังป้ายที่เฉพาะเจาะจง (เช่น 'AQ', 'BJ')
                                        หากเป็น None จะสุ่มเลือกเหมือนเดิม
            rx_rotation (int, optional): มุมการหมุนรอบแกน X (Pitch) เป็นองศา
            ry_rotation (int, optional): มุมการหมุนรอบแกน Y (Yaw) เป็นองศา
            rz_rotation (int, optional): มุมการหมุนรอบแกน Z (Roll) เป็นองศา
            plate_color (str, optional): สีพื้นหลังป้ายที่ต้องการ ('ขาว', 'เหลือง', 'เขียว', 'แดง')
                                        หากเป็น None จะใช้ตรรกะการเลือกป้ายตามปกติ
            plate_width (int, optional): ความกว้างของป้ายทะเบียนที่ต้องการ (ค่าเริ่มต้น 330)
            plate_height (int, optional): ความสูงของป้ายทะเบียนที่ต้องการ (ค่าเริ่มต้น 150)
        """
        original_plate_width = 330
        original_plate_height = 150

        scale_w = plate_width / original_plate_width
        scale_h = plate_height / original_plate_height

        top_w = int(35 * scale_w)
        top_h = int(80 * scale_h)
        top_char_w = int(40 * scale_w)
        top_char_h = int(80 * scale_h)
        bot_w = int(200 * scale_w)
        bot_h = int(40 * scale_h)

        row_char = int(10 * scale_h)
        col_char_start = int(25 * scale_w)
        col_province_start = int(65 * scale_w)
        row_province = int(97 * scale_h)

        numberF_resized = [cv2.resize(number, (top_w, top_h)) for number in self.NumberF if number is not None]
        char_resized = [cv2.resize(char1, (top_char_w, top_char_h)) for char1 in self.Char1 if char1 is not None]
        numberS_resized = [cv2.resize(number, (top_w, top_h)) for number in self.NumberS if number is not None]
        number2_resized = [cv2.resize(number, (top_w, top_h)) for number in self.Number if number is not None]
        province1_resized = [cv2.resize(province, (bot_w, bot_h)) for province in self.Province if province is not None]
        
        if not char_resized: print("ERROR: ลิสต์ 'char_resized' ว่างเปล่า"); return
        if not province1_resized: print("ERROR: ลิสต์ 'province1_resized' ว่างเปล่า"); return
        if not numberF_resized: print("ERROR: ลิสต์ 'numberF_resized' ว่างเปล่า"); return
        if not numberS_resized: print("ERROR: ลิสต์ 'numberS_resized' ว่างเปล่า"); return
        if not number2_resized: print("ERROR: ลิสต์ 'number2_resized' ว่างเปล่า"); return

        char_len = len(char_resized) - 1
        province1_len = len(province1_resized) - 1
        numberF_len = len(numberF_resized) - 1
        numberS_len = len(numberS_resized) - 1
        number2_len = len(number2_resized) - 1

        for i, Iter in enumerate(range(num)):
            label = str()

            # --- ย้าย: ตรวจสอบให้แน่ใจว่าได้เลือกจังหวัดก่อนเสมอ ---
            rand_province_idx = random.randint(0, province1_len)
            selected_province_name = self.province_list[rand_province_idx]
            # --- สิ้นสุดส่วนที่ย้าย ---

            # --- ตรรกะการเลือกพื้นหลังป้าย (อัปเดต) ---
            Plate_base_img = None
            if plate_color and plate_color in self.color_plates:
                # สร้างรูปภาพสีทึบตามขนาดที่กำหนด
                color_bgr = self.color_plates[plate_color][0, 0] # ดึงสี BGR จากภาพสีทึบที่โหลดไว้
                Plate_base_img = np.full((plate_height, plate_width, 3), color_bgr, dtype=np.uint8)
            elif plate_type and plate_type in self.special_plates and self.special_plates[plate_type] is not None:
                Plate_base_img = self.special_plates[plate_type].copy()
            else:
                # บล็อกนี้จะใช้ชื่อจังหวัดที่สุ่มไว้แล้ว ถ้ามี
                if selected_province_name in self.special_plates and \
                   self.special_plates[selected_province_name] is not None and \
                   random.random() < 0.5:
                    Plate_base_img = self.special_plates[selected_province_name].copy()
                else:
                    Plate_base_img = self.plate.copy() # ใช้ plate.jpg เป็นค่าเริ่มต้น

            if Plate_base_img is None: 
                Plate_base_img = self.plate.copy()
            
            # --- ปรับขนาด Plate ตาม plate_width, plate_height ที่กำหนด ---
            Plate = cv2.resize(Plate_base_img, (plate_width, plate_height))

            # ขนาดพื้นหลังใหญ่ (ยังคงเป็น 800x400)
            b_width, b_height = 400, 800         
            
            if self.background_images:
                chosen_bg_img = random.choice(self.background_images).copy()
                if chosen_bg_img.shape[2] == 4:
                    chosen_bg_img = cv2.cvtColor(chosen_bg_img, cv2.COLOR_BGRA2BGR)
                background = cv2.resize(chosen_bg_img, (b_height, b_width))
            else:
                random_R, random_G, random_B = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                background = np.zeros((b_width, b_height, 3), np.uint8)
                cv2.rectangle(background, (0, 0), (b_height, b_width), (random_R, random_G, random_B), -1)
            
            # ใช้พิกัดที่ปรับขนาดแล้ว
            col = col_char_start

            rand_int = random.randint(0, numberF_len)
            label += self.numberF_list[rand_int]
            Plate = overlay_image_alpha(Plate, numberF_resized[rand_int], col, row_char)
            col += top_w

            rand_int = random.randint(0, char_len)
            label += self.char_list[rand_int]
            Plate = overlay_image_alpha(Plate, char_resized[rand_int], col, row_char)
            col += top_char_w
            
            rand_int = random.randint(0, char_len)
            label += self.char_list[rand_int]
            Plate = overlay_image_alpha(Plate, char_resized[rand_int], col, row_char)
            col += top_w + int(20 * scale_w) # ปรับระยะห่าง

            rand_int = random.randint(0, numberS_len)
            label += self.numberS_list[rand_int]
            Plate = overlay_image_alpha(Plate, numberS_resized[rand_int], col, row_char)
            col += top_w

            rand_int = random.randint(0, number2_len)
            label += self.number_list[rand_int]
            Plate = overlay_image_alpha(Plate, number2_resized[rand_int], col, row_char)
            col += top_w

            rand_int = random.randint(0, number2_len)
            label += self.number_list[rand_int]
            Plate = overlay_image_alpha(Plate, number2_resized[rand_int], col, row_char)
            col += top_w

            rand_int = random.randint(0, number2_len)
            label += self.number_list[rand_int]
            Plate = overlay_image_alpha(Plate, number2_resized[rand_int], col, row_char)

            # ใช้พิกัดที่ปรับขนาดแล้ว
            label += selected_province_name
            Plate = overlay_image_alpha(Plate, province1_resized[rand_province_idx], col_province_start, row_province)

            Plate = apply_3d_rotation(Plate, rx_rotation, ry_rotation, rz_rotation)
            
            s_width, s_height = int((b_width - Plate.shape[0]) / 2), int((b_height - Plate.shape[1]) / 2)
            
            if Plate.shape[2] == 4:
                Plate_bgr = cv2.cvtColor(Plate, cv2.COLOR_BGRA2BGR)
            else:
                Plate_bgr = Plate
                
            y1_bg, y2_bg = s_width, s_width + Plate_bgr.shape[0]
            x1_bg, x2_bg = s_height, s_height + Plate_bgr.shape[1]

            if y2_bg > b_width:
                Plate_bgr = Plate_bgr[:b_width - y1_bg, :]
                y2_bg = b_width
            if x2_bg > b_height:
                Plate_bgr = Plate_bgr[:, :b_height - x1_bg]
                x2_bg = b_height
            
            if Plate_bgr.shape[0] > 0 and Plate_bgr.shape[1] > 0:
                background[y1_bg:y2_bg, x1_bg:x2_bg, :] = Plate_bgr

            background = image_augmentation(background, type2=True)

            if save:
                output_filename = os.path.join(self.save_path, label + ".jpg")
                cv2.imwrite(output_filename, background)
            else:
                cv2.imshow(label, background)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


# ตั้งค่าการแยกวิเคราะห์อาร์กิวเมนต์
parser = argparse.ArgumentParser(description="สร้างรูปภาพป้ายทะเบียนไทยสังเคราะห์")
parser.add_argument("-i", "--img_dir", help="ไดเรกทอรีสำหรับบันทึกรูปภาพที่สร้าง",
                    type=str, default="../CRNN/DB/")
parser.add_argument("-n", "--num", help="จำนวนรูปภาพที่ต้องการสร้าง",
                    type=int, required=True)
parser.add_argument("-s", "--save", help="ตั้งค่าเป็น True เพื่อบันทึกรูปภาพ, False เพื่อแสดงผล",
                    action='store_true', default=False)
parser.add_argument("--rx", "--rx_rotation", help="มุมการหมุนรอบแกน X (Pitch) เป็นองศา",
                    type=int, default=0)
parser.add_argument("--ry", "--ry_rotation", help="มุมการหมุนรอบแกน Y (Yaw) เป็นองศา",
                    type=int, default=0)
parser.add_argument("--rz", "--rz_rotation", help="มุมการหมุนรอบแกน Z (Roll) เป็นองศา",
                    type=int, default=0)
parser.add_argument("-p", "--plate_type", help="ประเภทพื้นหลังป้ายที่เฉพาะเจาะจง (เช่น 'AQ', 'BJ') หากไม่ระบุจะสุ่มเลือก",
                    type=str, default=None) 
parser.add_argument("--pc", "--plate_color", help="สีพื้นหลังป้ายที่ต้องการ ('ขาว', 'เหลือง', 'เขียว', 'แดง')",
                    type=str, default=None)
parser.add_argument("--pw", "--plate_width", help="ความกว้างของป้ายทะเบียน (ค่าเริ่มต้น 330)",
                    type=int, default=330)
parser.add_argument("--ph", "--plate_height", help="ความสูงของป้ายทะเบียน (ค่าเริ่มต้น 150)",
                    type=int, default=150)
args = parser.parse_args()

# บล็อกการรันหลัก
img_dir = args.img_dir
if args.save and not os.path.exists(img_dir):
    os.makedirs(img_dir)
    print(f"สร้างไดเรกทอรีเอาต์พุต: {img_dir}")

A = ImageGenerator(img_dir)

num_img = args.num
Save = args.save

A.Type_4(num_img, save=Save, 
         plate_type=args.plate_type, 
         rx_rotation=args.rx, 
         ry_rotation=args.ry, 
         rz_rotation=args.rz,
         plate_color=args.pc,
         plate_width=args.pw,
         plate_height=args.ph)
print("Type 4 เสร็จสิ้น")