import os
import random
import cv2
import argparse
import numpy as np

def rotate_image(image, angle):
    """
    หมุนภาพรอบจุดศูนย์กลาง

    อาร์กิวเมนต์:
        image (numpy.ndarray): รูปภาพอินพุต
        angle (float): มุมการหมุนเป็นองศา (ค่าบวกสำหรับการหมุนทวนเข็มนาฬิกา)

    คืนค่า:
        numpy.ndarray: รูปภาพที่หมุนแล้ว
    """
    # รับขนาดของรูปภาพ
    (h, w) = image.shape[:2]
    # คำนวณจุดศูนย์กลางของรูปภาพ
    center = (w // 2, h // 2)

    # รับเมทริกซ์การหมุน
    M = cv2.getRotationMatrix2D(center, angle, 1.0) # 1.0 คือปัจจัยการปรับขนาด

    # ทำการหมุน
    # หมายเหตุ: การดำเนินการนี้จะตัดส่วนของรูปภาพออก หากหมุนเกินขอบเขตเดิม
    # เพื่อหลีกเลี่ยงการตัดภาพ คุณจะต้องคำนวณขนาดใหม่สำหรับรูปภาพที่หมุนแล้ว
    # และปรับส่วนการแปลของเมทริกซ์การหมุน (M)
    rotated_image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return rotated_image


def image_augmentation(img, type2=False):
    """
    ทำการเพิ่มประสิทธิภาพภาพ (image augmentation) รวมถึงการแปลงมุมมอง, การปรับความสว่าง,
    และการเบลอ
    """
    # รับขนาดของรูปภาพ (ความสูง, ความกว้าง, จำนวนช่องสี)
    h, w, _ = img.shape
    
    # การแปลงมุมมอง (perspective transformation) (สุ่มใช้ 50% ของเวลา)
    if random.random() < 0.5:
        # กำหนดจุดต้นฉบับ (มุมของรูปภาพเดิม)
        pts1 = np.float32([[0, 0], [0, h], [w, 0], [w, h]])

        begin, end = 30, 90
        # กำหนดจุดปลายทางสำหรับการแปลงมุมมองพร้อมค่าออฟเซ็ตแบบสุ่ม
        pts2 = np.float32([[random.randint(begin, end), random.randint(begin, end)],
                           [random.randint(begin, end), h - random.randint(begin, end)],
                           [w - random.randint(begin, end), random.randint(begin, end)],
                           [w - random.randint(begin, end), h - random.randint(begin, end)]])
        # รับเมทริกซ์การแปลงมุมมอง
        M = cv2.getPerspectiveTransform(pts1, pts2)

        # ใช้การแปลงมุมมอง
        img = cv2.warpPerspective(img, M, (w, h))

    # การปรับความสว่าง (ใช้เสมอ)
    # แปลงภาพจากปริภูมิสี RGB เป็น HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = np.array(img, dtype=np.float64) # แปลงเป็น float64 สำหรับการคำนวณ
    random_bright = .4 + np.random.uniform() # สร้างปัจจัยความสว่างแบบสุ่ม
    img[:, :, 2] = img[:, :, 2] * random_bright # ปรับช่อง V (Value/Brightness)
    img[:, :, 2][img[:, :, 2] > 255] = 255 # ตัดค่าไม่ให้เกิน 255
    img = np.array(img, dtype=np.uint8) # แปลงกลับเป็น uint8
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR) # แปลงกลับเป็น BGR

    # การเบลอ (สุ่มใช้ 50% ของเวลา)
    if random.random() < 0.5:
        blur_value = random.randint(0,4) * 2 + 1 # สร้างขนาดเคอร์เนลเบลอแบบคี่ (1, 3, 5, 7, 9)
        img = cv2.blur(img,(blur_value, blur_value)) # ใช้การเบลอ

    # ตัดรูปภาพตามประเภท
    if type2:
        return img[130:280, 220:560, :]
    return img[130:280, 120:660, :]


def overlay_image_alpha(background_img, foreground_img_rgba, x_offset, y_offset):
    """
    วางรูปภาพเบื้องหน้า (พร้อมช่องอัลฟ่า) ทับบนรูปภาพพื้นหลัง
    จัดการความโปร่งใสและตรวจสอบให้แน่ใจว่ารูปภาพเบื้องหน้าพอดีภายในขอบเขตของพื้นหลัง
    
    อาร์กิวเมนต์:
        background_img (numpy.ndarray): รูปภาพพื้นหลัง (3 ช่องสี, BGR)
        foreground_img_rgba (numpy.ndarray): รูปภาพเบื้องหน้า (4 ช่องสี, BGRA)
        x_offset (int): ออฟเซ็ตพิกัด X สำหรับวางรูปภาพเบื้องหน้า
        y_offset (int): ออฟเซ็ตพิกัด Y สำหรับวางรูปภาพเบื้องหน้า
        
    คืนค่า:
        numpy.ndarray: รูปภาพพื้นหลังที่มีรูปภาพเบื้องหน้าทับอยู่
    """
    
    # ตรวจสอบว่า foreground_img_rgba มี 4 ช่องสีหรือไม่
    if foreground_img_rgba is None or foreground_img_rgba.shape[2] != 4:
        # print(f"Warning: Foreground image at offset ({x_offset},{y_offset}) is not a 4-channel BGRA image or is None. Skipping overlay.")
        return background_img

    # แยกช่องสี (BGR) และช่องอัลฟ่าออกจากรูปภาพเบื้องหน้า
    foreground_bgr = foreground_img_rgba[:, :, :3]
    alpha_channel = foreground_img_rgba[:, :, 3] / 255.0  # ทำให้ค่าอัลฟ่าเป็น 0-1

    # รับขนาด
    h_fg, w_fg = foreground_bgr.shape[:2]
    h_bg, w_bg = background_img.shape[:2]

    # คำนวณขอบเขตที่น่าสนใจ (ROI) บนพื้นหลังที่จะวางรูปภาพเบื้องหน้า
    y1, y2 = max(0, y_offset), min(h_bg, y_offset + h_fg)
    x1, x2 = max(0, x_offset), min(w_bg, x_offset + w_fg)

    # คำนวณ ROI ที่สอดคล้องกันบนรูปภาพเบื้องหน้า (ในกรณีที่รูปภาพเบื้องหน้าถูกตัดโดยขอบพื้นหลัง)
    y1_fg = y1 - y_offset
    y2_fg = y2 - y_offset
    x1_fg = x1 - x_offset
    x2_fg = x2 - x_offset

    # รับ ROI จากพื้นหลัง
    background_roi = background_img[y1:y2, x1:x2]

    # รับ ROI จากช่อง BGR และอัลฟ่าของรูปภาพเบื้องหน้า
    foreground_roi = foreground_bgr[y1_fg:y2_fg, x1_fg:x2_fg]
    alpha_roi = alpha_channel[y1_fg:y2_fg, x1_fg:x2_fg]

    # ปรับรูปร่างอัลฟ่าสำหรับการบรอดแคสต์ (เพื่อคูณกับ 3 ช่องสีของ BGR)
    alpha_roi_reshaped = alpha_roi[:, :, np.newaxis]

    # ผสมรูปภาพโดยใช้สูตรการผสมอัลฟ่า:
    # Output = Foreground_Color * Alpha + Background_Color * (1 - Alpha)
    blended_roi = (foreground_roi * alpha_roi_reshaped + 
                   background_roi * (1 - alpha_roi_reshaped)).astype(np.uint8)

    # วาง ROI ที่ผสมแล้วกลับลงในรูปภาพพื้นหลัง
    background_img[y1:y2, x1:x2] = blended_roi

    return background_img


class ImageGenerator:
    """
    สร้างรูปภาพป้ายทะเบียนไทยสังเคราะห์
    """
    def __init__(self, save_path):
        self.save_path = save_path
        
        # รับไดเรกทอรีของสคริปต์ปัจจุบันเพื่อสร้างพาธแบบสัมบูรณ์
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # โหลดรูปภาพพื้นหลังป้าย (ค่าเริ่มต้นทั่วไป)
        plate_path = os.path.join(script_dir, "plate.jpg")
        self.plate = cv2.imread(plate_path, cv2.IMREAD_COLOR) # ตรวจสอบให้แน่ใจว่าโหลดเป็น BGR (3 ช่องสี)
        if self.plate is None:
            print(f"ERROR: Failed to load plate image: {plate_path}")
            exit() # ออกจากโปรแกรมหากไม่มีสินทรัพย์สำคัญเช่น plate.jpg

        # เริ่มต้นพจนานุกรมเพื่อเก็บรูปภาพพื้นหลังป้ายพิเศษ
        self.special_plates = {} 

        # โหลดรูปภาพพื้นหลังป้ายพิเศษและเพิ่มลงในพจนานุกรม special_plates
        # --- เพิ่มการโหลดป้ายพิเศษทั้งหมดของคุณที่นี่ ---
        # ตัวอย่าง: ป้าย AQ
        aq_plate_path = os.path.join(script_dir, "AQ_plate.jpg")
        aq_plate_img = cv2.imread(aq_plate_path, cv2.IMREAD_COLOR)
        if aq_plate_img is None:
            print(f"WARNING: Failed to load AQ plate image: {aq_plate_path}. 'AQ' plates might use default 'plate.jpg'.")
        self.special_plates['AQ'] = aq_plate_img 

        # ป้าย BJ
        bj_plate_path = os.path.join(script_dir, "BJ_plate.jpg")
        bj_plate_img = cv2.imread(bj_plate_path, cv2.IMREAD_COLOR)
        if bj_plate_img is None:
            print(f"WARNING: Failed to load BJ plate image: {bj_plate_path}. 'BJ' plates might use default 'plate.jpg'.")
        self.special_plates['BJ'] = bj_plate_img
        
        # ป้าย BU
        bu_plate_path = os.path.join(script_dir, "BU_plate.jpg")
        bu_plate_img = cv2.imread(bu_plate_path, cv2.IMREAD_COLOR)
        if bu_plate_img is None:
            print(f"WARNING: Failed to load BU plate image: {bu_plate_path}. 'BU' plates might use default 'plate.jpg'.")
        self.special_plates['BU'] = bu_plate_img

        # ป้าย BY
        by_plate_path = os.path.join(script_dir, "BY_plate.jpg")
        by_plate_img = cv2.imread(by_plate_path, cv2.IMREAD_COLOR)
        if by_plate_img is None:
            print(f"WARNING: Failed to load BY plate image: {by_plate_path}. 'BY' plates might use default 'plate.jpg'.")
        self.special_plates['BY'] = by_plate_img 

        # ป้าย CD
        cd_plate_path = os.path.join(script_dir, "CD_plate.jpg")
        cd_plate_img = cv2.imread(cd_plate_path, cv2.IMREAD_COLOR)
        if cd_plate_img is None:
            print(f"WARNING: Failed to load CD plate image: {cd_plate_path}. 'CD' plates might use default 'plate.jpg'.")
        self.special_plates['CD'] = cd_plate_img

        # ป้าย CQ
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

        # --- สำคัญ: คอมเมนต์ออก หากคุณไม่มี DV_plate.jpg ---
        # dv_plate_path = os.path.join(script_dir, "DV_plate.jpg")
        # dv_plate_img = cv2.imread(dv_plate_path, cv2.IMREAD_COLOR)
        # if dv_plate_img is None:
        #     print(f"WARNING: Failed to load DV plate image: {dv_plate_path}. 'DV' plates might use default 'plate.jpg'.")
        # self.special_plates['DV'] = dv_plate_img
        # --- สิ้นสุดบล็อกคอมเมนต์ DV_plate.jpg ---

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

        # --- ลบ self.special_plate_provinces ตามที่คุณร้องขอ ---

        # ฟังก์ชันช่วยเหลือสำหรับโหลดรูปภาพจากไดเรกทอรี
        def load_images_from_dir(directory_name):
            full_path = os.path.join(script_dir, directory_name)
            images = []
            names = []
            if os.path.exists(full_path):
                file_list = os.listdir(full_path)
                # จัดเรียงไฟล์เพื่อให้มั่นใจในลำดับที่สอดคล้องกัน (เช่น 0.png, 1.png, ..., 9.png)
                file_list.sort() 
                for file_name in file_list:
                    # กรองไฟล์ที่ไม่ใช่รูปภาพออก หากมี
                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(full_path, file_name)
                        # โหลด PNG ด้วย IMREAD_UNCHANGED เพื่อให้ได้ช่อง Alpha
                        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) 
                        if img is None:
                            print(f"WARNING: Failed to load image: {img_path}")
                        images.append(img)
                        names.append(os.path.splitext(file_name)[0]) # รับชื่อโดยไม่มีนามสกุล
            else:
                print(f"ERROR: Directory not found: {full_path}")
                exit() # ออกจากโปรแกรมหากไม่มีไดเรกทอรีที่จำเป็น
            return images, names

        # โหลดตัวเลข 0-9
        self.Number, self.number_list = load_images_from_dir("num")
            
        # โหลดตัวเลข 1-9 (ตัวเลขที่ขึ้นต้นลำดับ)
        self.NumberS, self.numberS_list = load_images_from_dir("numS")

        # โหลดตัวเลขด้านหน้า
        self.NumberF, self.numberF_list = load_images_from_dir("num_front")
            
        # โหลดตัวอักษร
        self.Char1, self.char_list = load_images_from_dir("char1")

        # โหลดจังหวัด
        self.Province, self.province_list = load_images_from_dir("province")

        # --- ใหม่: โหลดรูปภาพพื้นหลังทั่วไป ---
        self.background_images, _ = load_images_from_dir("background_images")
        if not self.background_images:
            print("WARNING: ไม่พบรูปภาพในไดเรกทอรี 'background_images' จะใช้พื้นหลังสีทึบเป็นทางเลือกแทน")
        # --- สิ้นสุดใหม่ ---
            
    def Type_4(self, num, save=False, plate_type=None, z_rotation=0):
        """
        สร้างป้ายทะเบียนรูปแบบ Type 4
        รูปแบบ: [ตัวเลขด้านหน้า][ตัวอักษร1][ตัวอักษร1][ตัวเลขเริ่มต้น][ตัวเลข][ตัวเลข][ตัวเลข][จังหวัด]
        เช่น 1กข1234กรุงเทพมหานคร

        อาร์กิวเมนต์:
            num (int): จำนวนรูปภาพที่ต้องการสร้าง
            save (bool): True เพื่อบันทึกรูปภาพ, False เพื่อแสดงผล
            plate_type (str, optional): ประเภทพื้นหลังป้ายที่เฉพาะเจาะจง (เช่น 'AQ', 'BJ')
                                        หากเป็น None จะสุ่มเลือกเหมือนเดิม
            z_rotation (int, optional): มุมการหมุนเป็นองศา (แกน Z)
                                        หากเป็น 0 จะไม่มีการหมุน
        """
        top_w, top_h = 35, 80
        top_char_w, top_char_h = 40, 80
        bot_w, bot_h = 200, 40

        # ปรับขนาดรูปภาพที่โหลดทั้งหมดครั้งเดียวเพื่อปรับปรุงประสิทธิภาพ
        # กรองค่า None ออกและปรับขนาด คาดหวัง 4 ช่องสีสำหรับการวางทับ
        numberF_resized = [cv2.resize(number, (top_w, top_h)) for number in self.NumberF if number is not None]
        char_resized = [cv2.resize(char1, (top_char_w, top_char_h)) for char1 in self.Char1 if char1 is not None]
        numberS_resized = [cv2.resize(number, (top_w, top_h)) for number in self.NumberS if number is not None]
        number2_resized = [cv2.resize(number, (top_w, top_h)) for number in self.Number if number is not None]
        province1_resized = [cv2.resize(province, (bot_w, bot_h)) for province in self.Province if province is not None]
        
        # ตรวจสอบว่าลิสต์ไม่ว่างก่อนที่จะหาความยาว
        if not char_resized:
            print("ERROR: ลิสต์ 'char_resized' ว่างเปล่า โปรดตรวจสอบเนื้อหาในโฟลเดอร์ char1")
            return
        if not province1_resized:
            print("ERROR: ลิสต์ 'province1_resized' ว่างเปล่า โปรดตรวจสอบเนื้อหาในโฟลเดอร์ province")
            return
        if not numberF_resized:
            print("ERROR: ลิสต์ 'numberF_resized' ว่างเปล่า โปรดตรวจสอบเนื้อหาในโฟลเดอร์ num_front")
            return
        if not numberS_resized:
            print("ERROR: ลิสต์ 'numberS_resized' ว่างเปล่า โปรดตรวจสอบเนื้อหาในโฟลเดอร์ numS")
            return
        if not number2_resized:
            print("ERROR: ลิสต์ 'number2_resized' ว่างเปล่า โปรดตรวจสอบเนื้อหาในโฟลเดอร์ num")
            return

        char_len = len(char_resized) - 1
        province1_len = len(province1_resized) - 1
        numberF_len = len(numberF_resized) - 1
        numberS_len = len(numberS_resized) - 1
        number2_len = len(number2_resized) - 1

        for i, Iter in enumerate(range(num)):
            label = str()

            Plate_to_use = self.plate # ป้ายเริ่มต้น
            selected_province_name = None 
            rand_province_idx = -1 

            # กำหนดว่าจะใช้พื้นหลังป้ายใดตามอาร์กิวเมนต์ plate_type
            if plate_type and plate_type in self.special_plates and self.special_plates[plate_type] is not None:
                Plate_to_use = self.special_plates[plate_type]
                
                # --- ตรรกะที่แก้ไข: ใช้ plate_type โดยตรงเป็นชื่อจังหวัด ---
                selected_province_name = plate_type # ใช้ plate_type เป็นชื่อจังหวัดที่ต้องการ
                try:
                    rand_province_idx = self.province_list.index(selected_province_name)
                except ValueError:
                    print(f"WARNING: ไม่พบรูปภาพจังหวัด '{selected_province_name}' สำหรับประเภทป้าย '{plate_type}' ในลิสต์จังหวัด จะใช้จังหวัดแบบสุ่มสำหรับข้อความแทน")
                    # กลับไปใช้แบบสุ่ม หากไม่พบรูปภาพจังหวัดที่ระบุ
                    rand_province_idx = random.randint(0, province1_len)
                    selected_province_name = self.province_list[rand_province_idx]
                # --- สิ้นสุดตรรกะที่แก้ไข ---
            else:
                # การเลือกจังหวัดแบบสุ่มเดิม และโอกาส 50% สำหรับป้ายพิเศษตามชื่อจังหวัด
                rand_province_idx = random.randint(0, province1_len)
                selected_province_name = self.province_list[rand_province_idx]
                if selected_province_name in self.special_plates and \
                   self.special_plates[selected_province_name] is not None and \
                   random.random() < 0.5:
                    Plate_to_use = self.special_plates[selected_province_name]
            
            # ตรวจสอบข้อผิดพลาดสำหรับ rand_province_idx ในกรณีที่ไม่คาดคิด (เช่น ลิสต์จังหวัดว่างเปล่า)
            if rand_province_idx == -1: # ควรได้รับการจัดการโดยตรรกะข้างต้น แต่เป็นมาตรการป้องกัน
                rand_province_idx = random.randint(0, province1_len)
                selected_province_name = self.province_list[rand_province_idx]

            Plate = cv2.resize(Plate_to_use.copy(), (330, 150))
            # --- สิ้นสุดตรรกะป้าย/จังหวัด ---

            b_width, b_height = 400, 800         
            
            # --- ใหม่: ตรรกะการเลือกพื้นหลัง ---
            if self.background_images:
                # สุ่มเลือกรูปภาพพื้นหลัง
                chosen_bg_img = random.choice(self.background_images).copy()
                # ปรับขนาดรูปภาพพื้นหลังที่เลือกให้เป็นขนาดเป้าหมาย
                # ตรวจสอบให้แน่ใจว่าเป็น 3 ช่องสีเพื่อความสอดคล้องกัน หากถูกโหลดพร้อมช่องอัลฟ่า
                if chosen_bg_img.shape[2] == 4:
                    chosen_bg_img = cv2.cvtColor(chosen_bg_img, cv2.COLOR_BGRA2BGR)
                background = cv2.resize(chosen_bg_img, (b_height, b_width))
            else:
                # สำรองข้อมูลเป็นสีสุ่มทึบ หากไม่มีรูปภาพพื้นหลังถูกโหลด
                random_R, random_G, random_B = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                background = np.zeros((b_width, b_height, 3), np.uint8)
                cv2.rectangle(background, (0, 0), (b_height, b_width), (random_R, random_G, random_B), -1)
            # --- สิ้นสุดใหม่ ---
            
            # row -> y , col -> x (พิกัดสำหรับการวางตัวอักษรบนป้าย)
            row, col = 10, 25

            # ตัวเลขด้านหน้า
            rand_int = random.randint(0, numberF_len)
            label += self.numberF_list[rand_int]
            Plate = overlay_image_alpha(Plate, numberF_resized[rand_int], col, row)
            col += top_w

            # ตัวอักษร 1
            rand_int = random.randint(0, char_len)
            label += self.char_list[rand_int]
            Plate = overlay_image_alpha(Plate, char_resized[rand_int], col, row)
            col += top_char_w
            
            # ตัวอักษร 2
            rand_int = random.randint(0, char_len)
            label += self.char_list[rand_int]
            Plate = overlay_image_alpha(Plate, char_resized[rand_int], col, row)
            col += top_w + 20 # เพิ่มช่องว่างพิเศษระหว่างตัวอักษรและตัวเลขเริ่มต้น

            # ตัวเลขเริ่มต้น (1-9)
            rand_int = random.randint(0, numberS_len)
            label += self.numberS_list[rand_int]
            Plate = overlay_image_alpha(Plate, numberS_resized[rand_int], col, row)
            col += top_w

            # ตัวเลข 2 (0-9)
            rand_int = random.randint(0, number2_len)
            label += self.number_list[rand_int]
            Plate = overlay_image_alpha(Plate, number2_resized[rand_int], col, row)
            col += top_w

            # ตัวเลข 3 (0-9)
            rand_int = random.randint(0, number2_len)
            label += self.number_list[rand_int]
            Plate = overlay_image_alpha(Plate, number2_resized[rand_int], col, row)
            col += top_w

            # ตัวเลข 4 (0-9)
            rand_int = random.randint(0, number2_len)
            label += self.number_list[rand_int]
            Plate = overlay_image_alpha(Plate, number2_resized[rand_int], col, row)

            row, col = 97, 65 # บรรทัดใหม่สำหรับจังหวัด (ปรับพิกัดตามความจำเป็น)
            
            # จังหวัด (ใช้จังหวัดที่เลือกสำหรับข้อความ)
            label += selected_province_name
            Plate = overlay_image_alpha(Plate, province1_resized[rand_province_idx], col, row)

            # --- ใช้การหมุนแกน Z กับรูปภาพป้ายก่อนวางบนพื้นหลัง ---
            if z_rotation != 0:
                Plate = rotate_image(Plate, z_rotation)
            # --- สิ้นสุดการหมุนแกน Z ---

            # วางป้ายที่สร้างขึ้นลงบนพื้นหลัง
            s_width, s_height = int((b_width - Plate.shape[0]) / 2), int((b_height - Plate.shape[1]) / 2)
            # ตรวจสอบให้แน่ใจว่า Plate เป็น BGR 3 ช่องสีก่อนวางบนพื้นหลัง BGR 3 ช่องสี
            if Plate.shape[2] == 4:
                Plate_bgr = cv2.cvtColor(Plate, cv2.COLOR_BGRA2BGR)
            else:
                Plate_bgr = Plate
                
            background[s_width:Plate.shape[0] + s_width, s_height:Plate.shape[1] + s_height, :] = Plate_bgr
            
            # ใช้การเพิ่มประสิทธิภาพภาพกับรูปภาพพื้นหลังสุดท้าย
            background = image_augmentation(background, type2=True)

            # บันทึกหรือแสดงรูปภาพ
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
                    type=str, default="../DB/train/")
parser.add_argument("-n", "--num", help="จำนวนรูปภาพที่ต้องการสร้าง",
                    type=int, required=True)
parser.add_argument("-s", "--save", help="ตั้งค่าเป็น True เพื่อบันทึกรูปภาพ, False เพื่อแสดงผล",
                    action='store_true', default=False)
parser.add_argument("-z", "--z_rotation", help="มุมการหมุนเป็นองศา (แกน Z) ค่าบวกสำหรับการหมุนทวนเข็มนาฬิกา",
                    type=int, default=0)
parser.add_argument("-p", "--plate_type", help="ประเภทพื้นหลังป้ายที่เฉพาะเจาะจง (เช่น 'AQ', 'BJ') หากไม่ระบุจะสุ่มเลือก",
                    type=str, default=None) 
args = parser.parse_args()

# บล็อกการรันหลัก
img_dir = args.img_dir
# ตรวจสอบให้แน่ใจว่าไดเรกทอรีสำหรับบันทึกมีอยู่ หากมีการบันทึก
if args.save and not os.path.exists(img_dir):
    os.makedirs(img_dir)
    print(f"สร้างไดเรกทอรีเอาต์พุต: {img_dir}")

A = ImageGenerator(img_dir)

num_img = args.num
Save = args.save

# ส่งผ่านอาร์กิวเมนต์ที่เกี่ยวข้องทั้งหมดไปยัง Type_4 รวมถึง z_rotation และ plate_type
A.Type_4(num_img, save=Save, plate_type=args.plate_type, z_rotation=args.z_rotation)
print("Type 4 เสร็จสิ้น")