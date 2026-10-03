"""Build out/SWINQ_pitch.pdf: the pitch script with slide cues and judge Q&A."""
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from racket import config

pdfmetrics.registerFont(TTFont("Arial", r"C:\Windows\Fonts\arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", r"C:\Windows\Fonts\arialbd.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Italic", r"C:\Windows\Fonts\ariali.ttf"))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic")

NAVY, BLUE, GREY = colors.HexColor("#13202D"), colors.HexColor("#1F5FBF"), colors.HexColor("#5A6573")
title = ParagraphStyle("t", fontName="Arial-Bold", fontSize=22, leading=27, textColor=NAVY, spaceAfter=4)
sub = ParagraphStyle("s", fontName="Arial", fontSize=11, leading=15, textColor=GREY, spaceAfter=14)
h = ParagraphStyle("h", fontName="Arial-Bold", fontSize=13, leading=17, textColor=NAVY, spaceBefore=12, spaceAfter=2)
cue = ParagraphStyle("c", fontName="Arial-Italic", fontSize=9.5, leading=13, textColor=BLUE, spaceAfter=4)
body = ParagraphStyle("b", fontName="Arial", fontSize=11, leading=15.5, textColor=NAVY, alignment=TA_LEFT, spaceAfter=6)
small = ParagraphStyle("sm", fontName="Arial", fontSize=9.5, leading=13, textColor=NAVY)

SECTIONS = [
    ("1. Hook (15 s)", "Slide 1. Play stroke_3d.mp4 or the Blender render on a loop.",
     ["This is one forehand, rebuilt in 3D from nothing but the SWINQ dampener on the strings. "
      "No cameras, no motion-capture suit. One small IMU recording for under a second."]),
    ("2. The challenge (20 s)", "Slide 2: the data is hard.",
     ["The data is hard. We get 0.96 seconds at 416 Hz. The recording starts mid-swing, so there's no "
      "still moment to calibrate on. The racket peaks at <b>1720 degrees per second</b>. And the "
      "accelerometer hits its 16 g limit for the 18 samples just before impact, exactly when the swing is fastest."]),
    ("3. What we built (40 s)", "Slides 3 and 4: how it works, then the numbers.",
     ["Orientation comes from the gyroscope, integrated with exact rotation steps. On a synthetic swing at "
      "500 degrees per second it's accurate to 0.03 degrees, and our unit tests check that.",
      "We repaired the saturation with physics. Along the handle, the accelerometer measures centripetal "
      "force, so we can predict it from the gyro: acceleration equals ω² times the pivot radius. That "
      "relation holds with a correlation of 0.92, and it fills the clipped gap. The real peak was about 30 g, not 16.",
      "From the same sensor we get the numbers a player cares about: <b>head speed at impact of about 89 km/h</b>, "
      "<b>85° of racket rotation in the last 100 milliseconds</b>, plus impact timing, roll rate, and string "
      "vibration at about 580 Hz."]),
    ("4. Proof against the video (40 s)", "Slide 5: contact sheet, left half.",
     ["To check ourselves, we turned the two videos into a 3D reference. We calibrated both cameras using the "
      "racket itself as the calibration object, and solved the racket's pose in every frame.",
      "The sensor-only racket, in green, lands on the real racket in both camera views, "
      "<b>within 3 to 11 degrees</b>, through the start of the swing."]),
    ("5. Honest limitation (25 s)", "Slide 6: point at the right half of the contact sheet, then the ruled-out list.",
     ["Then it drifts, and we want to be upfront about that. Through the fast part of the swing, the gyro "
      "measures two to three times more rotation than the video shows. We ruled out the axis convention "
      "(all 48 options), the sample rate, sensor bias and mounting angle. The accelerometer independently "
      "confirms the gyro's scale. Our best explanation is the sensor-to-video time alignment and heavy motion "
      "blur at 30 fps. Every test is in the repo."]),
    ("6. Why it matters and what's next (20 s)", "Slide 7: open, tested, ready to extend.",
     ["Everything is open: a tested Python package, an orientation CSV per sample, and a Blender script that "
      "animates it. Orientation is the hard half of 6-DoF tracking. With the saturation repair and the "
      "pivot-radius model, the next step to full position tracking is already set up. "
      "<b>That's motion capture from a tennis dampener.</b>"]),
]

QA = [
    ("Did you use the video in the model?",
     "Only to set the racket's starting orientation, which a gyro can't know when the recording starts "
     "mid-swing. Everything after that is sensor-only."),
    ("Why not use gravity for the start?",
     "There's no still period, so the accelerometer never isolates gravity. Swing forces reach up to 30 g."),
    ("How do you know the saturation repair is right?",
     "The model fits the samples just before clipping with a correlation of 1.00 and joins the measured "
     "curve smoothly. It's physics, not curve-fitting."),
    ("Why 580 Hz for the strings?",
     "The data shows 165 Hz, but at 416 Hz sampling that's an alias. The real frequency is 251 or 581 Hz, "
     "and 581 is typical for a string bed."),
    ("What would you do with more time?",
     "Get the true time sync between sensor and video, then use the video only as a soft check and extend "
     "to full position tracking."),
]

TIPS = ["Open with the animation already playing.",
        "Say the head speed slowly, because it's the number people remember.",
        "Don't apologise for the limitation; present it as rigour.",
        "Total running time is about 2.5 minutes."]


def build(path):
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm,
                            title="SWINQ: 3D stroke from sensor data — pitch")
    story = [Paragraph("SWINQ: one sensor, one forehand, in 3D", title),
             Paragraph("Pitch script, about 2.5 minutes · Hack For Humanity, Sports Telemetry challenge", sub)]
    for head, c, paras in SECTIONS:
        story += [Paragraph(head, h), Paragraph("Show: " + c, cue)]
        story += [Paragraph("“" + p + "”" if i == 0 and len(paras) == 1 else p, body)
                  for i, p in enumerate(paras)]
    story += [Spacer(1, 6), Paragraph("Likely judge questions", h)]
    rows = [[Paragraph("<b>Question</b>", small), Paragraph("<b>Answer</b>", small)]]
    rows += [[Paragraph(q, small), Paragraph(a, small)] for q, a in QA]
    t = Table(rows, colWidths=[55 * mm, 115 * mm])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#D5D9DE")),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF2F7")),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story += [t, Paragraph("Delivery tips", h)]
    story += [Paragraph("• " + tip, body) for tip in TIPS]
    doc.build(story)


if __name__ == "__main__":
    out = config.OUT_DIR / "SWINQ_pitch.pdf"
    build(out)
    print(f"wrote {out}")
