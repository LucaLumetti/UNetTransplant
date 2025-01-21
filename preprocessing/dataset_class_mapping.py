DATASET_IDX = {
    "AMOS": 0,
    "SegThor": 1,
    "Skull": 2,
    "TotalSegmentator": 3,
    "ZhimingCui": 4,
    "BTCV_Abdomen": 5,
    "BTCV_Cervix": 6,
    "AbdomenCT-1K": 7,
}

DATASET_ORIGINAL_LABELS = {
    "AMOS": {
        "0": "background",
        "1": "spleen",  # 1
        "2": "right kidney",  # 2
        "3": "left kidney",  # 2
        "4": "gall bladder",  # 3
        "5": "esophagus",  # 4
        "6": "liver",  # 5
        "7": "stomach",  # 6
        "8": "arota",  # 7
        "9": "postcava",  # 8
        "10": "pancreas",  # 9
        "11": "right adrenal gland",  # 10
        "12": "left adrenal gland",  # 10
        "13": "duodenum",  # 11
        "14": "bladder",  # 12
        "15": "prostate/uterus",  # 13
    },
    "SegThor": {
        "0": "background",
        "1": "esophagus",  # 14
        "2": "hearth",  # 15
        "3": "trachea",  # 16
        "4": "aorta",  # 17
    },
    "Skull": {"0": "background", "1": "skull"},  # 18
    "TotalSegmentator": {
        "1": "spleen",  # 19
        "2": "kidney_right",  # 20
        "3": "kidney_left",  # 20
        "4": "gallbladder",  # 21
        "5": "liver",  # 22
        "6": "stomach",  # 23
        "7": "pancreas",  # 24
        "8": "adrenal_gland_right",  # 25
        "9": "adrenal_gland_left",  # 25
        "10": "lung_upper_lobe_left",  # 26
        "11": "lung_lower_lobe_left",  # 26
        "12": "lung_upper_lobe_right",  # 26
        "13": "lung_middle_lobe_right",  # 26
        "14": "lung_lower_lobe_right",  # 26
        "15": "esophagus",  # 27
        "16": "trachea",  # 28
        "17": "thyroid_gland",  # 29
        "18": "small_bowel",  # 30
        "19": "duodenum",  # 31
        "20": "colon",  # 32
        "21": "urinary_bladder",  # 33
        "22": "prostate",  # 34
        "23": "kidney_cyst_left",  # 35
        "24": "kidney_cyst_right",  # 35
        "25": "sacrum",  # 36
        "26": "vertebrae_S1",  # 37
        "27": "vertebrae_L5",  # 37
        "28": "vertebrae_L4",  # 37
        "29": "vertebrae_L3",  # 37
        "30": "vertebrae_L2",  # 37
        "31": "vertebrae_L1",  # 37
        "32": "vertebrae_T12",  # 37
        "33": "vertebrae_T11",  # 37
        "34": "vertebrae_T10",  # 37
        "35": "vertebrae_T9",  # 37
        "36": "vertebrae_T8",  # 37
        "37": "vertebrae_T7",  # 37
        "38": "vertebrae_T6",  # 37
        "39": "vertebrae_T5",  # 37
        "40": "vertebrae_T4",  # 37
        "41": "vertebrae_T3",  # 37
        "42": "vertebrae_T2",  # 37
        "43": "vertebrae_T1",  # 37
        "44": "vertebrae_C7",  # 37
        "45": "vertebrae_C6",  # 37
        "46": "vertebrae_C5",  # 37
        "47": "vertebrae_C4",  # 37
        "48": "vertebrae_C3",  # 37
        "49": "vertebrae_C2",  # 37
        "50": "vertebrae_C1",  # 37
        "51": "heart",  # 38
        "52": "aorta",  # 39
        "53": "pulmonary_vein",  # 40
        "54": "brachiocephalic_trunk",  # 41
        "55": "subclavian_artery_right",  # 42
        "56": "subclavian_artery_left",  # 42
        "57": "common_carotid_artery_right",  # 43
        "58": "common_carotid_artery_left",  # 43
        "59": "brachiocephalic_vein_left",  # 44
        "60": "brachiocephalic_vein_right",  # 44
        "61": "atrial_appendage_left",  # 45
        "62": "superior_vena_cava",  # 46
        "63": "inferior_vena_cava",  # 46
        "64": "portal_vein_and_splenic_vein",  # 47
        "65": "iliac_artery_left",  # 48
        "66": "iliac_artery_right",  # 48
        "67": "iliac_vena_left",  # 48
        "68": "iliac_vena_right",  # 48
        "69": "humerus_left",  # 49
        "70": "humerus_right",  # 49
        "71": "scapula_left",  # 50
        "72": "scapula_right",  # 50
        "73": "clavicula_left",  # 51
        "74": "clavicula_right",  # 51
        "75": "femur_left",  # 52
        "76": "femur_right",  # 52
        "77": "hip_left",  # 53
        "78": "hip_right",  # 53
        "79": "spinal_cord",  # 54
        "80": "gluteus_maximus_left",  # 55
        "81": "gluteus_maximus_right",  # 55
        "82": "gluteus_medius_left",  # 55
        "83": "gluteus_medius_right",  # 55
        "84": "gluteus_minimus_left",  # 55
        "85": "gluteus_minimus_right",  # 55
        "86": "autochthon_left",  # 56
        "87": "autochthon_right",  # 56
        "88": "iliopsoas_left",  # 57
        "89": "iliopsoas_right",  # 57
        "90": "brain",  # 58
        "91": "skull",  # 59
        "92": "rib_left_1",  # 60
        "93": "rib_left_2",  # 60
        "94": "rib_left_3",  # 60
        "95": "rib_left_4",  # 60
        "96": "rib_left_5",  # 60
        "97": "rib_left_6",  # 60
        "98": "rib_left_7",  # 60
        "99": "rib_left_8",  # 60
        "100": "rib_left_9",  # 60
        "101": "rib_left_10",  # 60
        "102": "rib_left_11",  # 60
        "103": "rib_left_12",  # 60
        "104": "rib_right_1",  # 60
        "105": "rib_right_2",  # 60
        "106": "rib_right_3",  # 60
        "107": "rib_right_4",  # 60
        "108": "rib_right_5",  # 60
        "109": "rib_right_6",  # 60
        "110": "rib_right_7",  # 60
        "111": "rib_right_8",  # 60
        "112": "rib_right_9",  # 60
        "113": "rib_right_10",  # 60
        "114": "rib_right_11",  # 60
        "115": "rib_right_12",  # 60
        "116": "sternum",  # 61
        "117": "costal_cartilages",  # 62
    },
    "ZhimingCui": {
        "0": "background",
        "1": "teeth",  # 63
    },
    "BTCV_Abdomen": {
        "1": "spleen",
        "2": "right kidney",
        "3": "left kidney",
        "4": "gallbladder",
        "5": "esophagus",
        "6": "liver",
        "7": "stomach",
        "8": "aorta",
        "9": "inferior vena cava",
        "10": "portal vein and splenic vein",
        "11": "pancreas",
        "12": "right adrenal gland",
        "13": "left adrenal gland",
    },
    "BTCV_Cervix": {
        "1": "bladder",
        "2": "uterus",
        "3": "rectum",
        "4": "small bowel",
    },
    "AbdomenCT-1K": {
        "1": "liver",
        "2": "kidney",
        "3": "spleen",
        "4": "pancreas",
    },
    "ToothFairy2": {
        "1": "Lower Jawbone",
        "2": "Upper Jawbone",
        "3": "Left Inferior Alveolar Canal",
        "4": "Right Inferior Alveolar Canal",
        "5": "Left Maxillary Sinus",
        "6": "Right Maxillary Sinus",
        "7": "Pharynx",
        "8": "Bridge",
        "9": "Crown",
        "10": "Implant",
        "11": "Upper Right Central Incisor",
        "12": "Upper Right Lateral Incisor",
        "13": "Upper Right Canine",
        "14": "Upper Right First Premolar",
        "15": "Upper Right Second Premolar",
        "16": "Upper Right First Molar",
        "17": "Upper Right Second Molar",
        "18": "Upper Right Third Molar (Wisdom Tooth)",
        "19": "Upper Left Central Incisor",
        "20": "Upper Left Lateral Incisor",
        "21": "Upper Left Canine",
        "22": "Upper Left First Premolar",
        "23": "Upper Left Second Premolar",
        "24": "Upper Left First Molar",
        "25": "Upper Left Second Molar",
        "26": "Upper Left Third Molar (Wisdom Tooth)",
        "27": "Lower Left Central Incisor",
        "28": "Lower Left Lateral Incisor",
        "29": "Lower Left Canine",
        "30": "Lower Left First Premolar",
        "31": "Lower Left Second Premolar",
        "32": "Lower Left First Molar",
        "33": "Lower Left Second Molar",
        "34": "Lower Left Third Molar (Wisdom Tooth)",
        "35": "Lower Right Central Incisor",
        "36": "Lower Right Lateral Incisor",
        "37": "Lower Right Canine",
        "38": "Lower Right First Premolar",
        "39": "Lower Right Second Premolar",
        "40": "Lower Right First Molar",
        "41": "Lower Right Second Molar",
        "42": "Lower Right Third Molar (Wisdom Tooth)",
    },
}

DATASET_MAPPING_LABELS = {
    "AMOS": {
        1: 1,
        2: 2,
        3: 2,
        4: 3,
        5: 4,
        6: 5,
        7: 6,
        8: 7,
        9: 8,
        10: 9,
        11: 10,
        12: 10,
        13: 11,
        14: 12,
        15: 13,
    },
    "SegThor": {1: 14, 2: 15, 3: 16, 4: 17},
    "Skull": {
        1: 18,
    },
    "TotalSegmentator": {
        1: 19,
        2: 20,
        3: 20,
        4: 21,
        5: 22,
        6: 23,
        7: 24,
        8: 25,
        9: 25,
        10: 26,
        11: 26,
        12: 26,
        13: 26,
        14: 26,
        15: 27,
        16: 28,
        17: 29,
        18: 30,
        19: 31,
        20: 32,
        21: 33,
        22: 34,
        23: 35,
        24: 35,
        25: 36,
        26: 37,
        27: 37,
        28: 37,
        29: 37,
        30: 37,
        31: 37,
        32: 37,
        33: 37,
        34: 37,
        35: 37,
        36: 37,
        37: 37,
        38: 37,
        39: 37,
        40: 37,
        41: 37,
        42: 37,
        43: 37,
        44: 37,
        45: 37,
        46: 37,
        47: 37,
        48: 37,
        49: 37,
        50: 37,
        51: 38,
        52: 39,
        53: 40,
        54: 41,
        55: 42,
        56: 42,
        57: 43,
        58: 43,
        59: 44,
        60: 44,
        61: 45,
        62: 46,
        63: 46,
        64: 47,
        65: 48,
        66: 48,
        67: 48,
        68: 48,
        69: 49,
        70: 49,
        71: 50,
        72: 50,
        73: 51,
        74: 51,
        75: 52,
        76: 52,
        77: 53,
        78: 53,
        79: 54,
        80: 55,
        81: 55,
        82: 55,
        83: 55,
        84: 55,
        85: 55,
        86: 56,
        87: 56,
        88: 57,
        89: 57,
        90: 58,
        91: 59,
        92: 60,
        93: 60,
        94: 60,
        95: 60,
        96: 60,
        97: 60,
        98: 60,
        99: 60,
        100: 60,
        101: 60,
        102: 60,
        103: 60,
        104: 60,
        105: 60,
        106: 60,
        107: 60,
        108: 60,
        109: 60,
        110: 60,
        111: 60,
        112: 60,
        113: 60,
        114: 60,
        115: 60,
        116: 61,
        117: 62,
    },
    "ZhimingCui": {1: 63},
    "BTCV_Abdomen": {
        1: 64,
        2: 65,
        3: 65,
        4: 66,
        5: 67,
        6: 68,
        7: 69,
        8: 70,
        9: 71,
        10: 72,
        11: 73,
        12: 74,
        13: 74,
    },
    "BTCV_Cervix": {
        1: 75,
        2: 76,
        3: 77,
        4: 78,
    },
    "AbdomenCT-1K": {
        1: 79,
        2: 80,
        3: 81,
        4: 82,
    },
}
