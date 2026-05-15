LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

MOUTH_OUTER = [61, 291, 0, 17]
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308
LIP_TOP_OUTER = 0
LIP_BOTTOM_OUTER = 17

SMILE_LEFT_CORNER = 61
SMILE_RIGHT_CORNER = 291
SMILE_UPPER_LIP = 13
SMILE_LOWER_LIP = 14

LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [336, 296, 334, 293, 300]
LEFT_EYEBROW_TOP = 105
RIGHT_EYEBROW_TOP = 334

CHIN = 152
FOREHEAD_TOP = 10
LEFT_TEMPLE = 234
RIGHT_TEMPLE = 454
LEFT_JAW = 172
RIGHT_JAW = 397
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
LEFT_CHEEKBONE = 50
RIGHT_CHEEKBONE = 280
NOSE_TIP = 1
NOSE_BRIDGE = 168

POSE_LANDMARKS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_corner": 33,
    "right_eye_corner": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

POSE_3D_MODEL = [
    (0.0, 0.0, 0.0),
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1),
    (28.9, -28.9, -24.1),
]

COLOR_MESH = (60, 60, 60)
COLOR_IRIS = (120, 200, 240)
COLOR_BOX = (60, 152, 217)
COLOR_POSE_AXIS = (220, 220, 220)
COLOR_TEXT = (235, 235, 235)
COLOR_BG_OVERLAY = (4, 4, 4)
COLOR_TACTICAL = (23, 160, 212)
COLOR_LOCK = (96, 200, 96)
COLOR_WARN = (28, 134, 220)
COLOR_DANGER = (28, 28, 220)

UI_BG          = "#000000"
UI_SURFACE     = "#080808"
UI_SURFACE_2   = "#101010"
UI_SURFACE_3   = "#181818"
UI_SURFACE_4   = "#222222"
UI_BORDER      = "#1f1f1f"
UI_BORDER_HI   = "#333333"
UI_BORDER_HOT  = "#5c4a14"

UI_TEXT        = "#e5e2da"
UI_TEXT_MID    = "#a8a8a8"
UI_TEXT_DIM    = "#6a6a6a"
UI_TEXT_FAINT  = "#3a3a3a"

UI_ACCENT      = "#d4a017"
UI_ACCENT_HI   = "#f5bf32"
UI_ACCENT_LO   = "#1a1308"
UI_ACCENT_TXT  = "#0a0700"

UI_GOOD        = "#84cc16"
UI_WARN        = "#f59e0b"
UI_BAD         = "#dc2626"
UI_INFO        = "#94a3b8"

UI_PANEL       = UI_SURFACE
UI_PANEL_2     = UI_SURFACE_2
UI_ACCENT_2    = UI_ACCENT_HI

FONT_SANS = "Segoe UI"
FONT_MONO = "Consolas"

FACE_SHAPES = ["Oval", "Round", "Square", "Heart", "Diamond", "Oblong", "Triangle"]
EMOTIONS = ["Neutral", "Happy", "Surprised", "Sad", "Angry", "Focused"]
FILTERS = ["None", "Cartoon", "Sketch", "Edge", "Sepia", "Cool", "Warm", "Thermal", "Noir", "Infrared"]
GRID_MODES = ["None", "Rule of thirds", "Crosshair", "Grid 4x4", "Center dot"]
ANONYMIZE_MODES = ["Off", "Blur face", "Pixelate face", "Black bar (eyes)"]
