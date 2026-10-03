"""Build out/SWINQ_pitch.pdf from the deck's slides and speaker notes.

Reads pitch_deck/project/deck.json (slide order) and each slide's <aside> notes,
so the PDF always matches the slides: per slide its title, time window, what to
say, and the plain-words explanation; then judge Q&A and delivery tips.
"""
import html
import json
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from racket import config

DECK = config.ROOT / "pitch_deck" / "project"

pdfmetrics.registerFont(TTFont("Arial", r"C:\Windows\Fonts\arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", r"C:\Windows\Fonts\arialbd.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Italic", r"C:\Windows\Fonts\ariali.ttf"))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic")

NAVY, BLUE, GREY = colors.HexColor("#13202D"), colors.HexColor("#1F5FBF"), colors.HexColor("#5A6573")
title = ParagraphStyle("t", fontName="Arial-Bold", fontSize=22, leading=27, textColor=NAVY, spaceAfter=4)
sub = ParagraphStyle("s", fontName="Arial", fontSize=11, leading=15, textColor=GREY, spaceAfter=10)
h = ParagraphStyle("h", fontName="Arial-Bold", fontSize=13, leading=17, textColor=NAVY, spaceBefore=12, spaceAfter=2)
cue = ParagraphStyle("c", fontName="Arial-Italic", fontSize=9.5, leading=13, textColor=BLUE, spaceAfter=4)
say = ParagraphStyle("say", fontName="Arial", fontSize=11.5, leading=16, textColor=NAVY, spaceAfter=5,
                     leftIndent=8, borderPadding=(4, 6, 4, 6), backColor=colors.HexColor("#F1F4F8"))
label = ParagraphStyle("l", fontName="Arial-Bold", fontSize=9, leading=12, textColor=GREY, spaceBefore=3, spaceAfter=6)
bullet = ParagraphStyle("bu", fontName="Arial", fontSize=9.8, leading=13.5, textColor=NAVY, leftIndent=10,
                        bulletIndent=0, spaceAfter=2)
small = ParagraphStyle("sm", fontName="Arial", fontSize=9.5, leading=13, textColor=NAVY)

QA = [
    ("Did you use the video in the model?",
     "The sensor gives all the motion. The video gives the starting angle (a gyro only knows changes) and, "
     "in the corrected version, the drift correction. The speeds and numbers are sensor-only, and we keep "
     "the pure sensor result."),
    ("Why not use gravity to find the start?",
     "The recording starts mid-swing; the racket is never still, so the sensor never feels gravity on its own."),
    ("How do you know the correction isn't just copying the video?",
     "We hid one video frame at a time, built the correction from the others, and checked the hidden frame: "
     "12.4° average error on frames it never saw."),
    ("How do you know the filled-in force is right?",
     "It's physics: spin speed predicts the outward pull. On the readings just before the sensor maxed out, "
     "the prediction matches almost perfectly."),
    ("What would you do with more time?",
     "Line up the sensor and camera clocks properly, then track the racket's position too, not just its angle."),
]

TIPS = ["Start the 3D animation before you begin talking.",
        "Say “89 kilometres per hour” slowly; it's the number people remember.",
        "On slide 6, play gp_correction.mp4: orange = before, green = after.",
        "Present the drift as rigour, not as an apology.",
        "Spoken script is about 2.5 minutes; the time windows add up to 3:00."]


def text_of(fragment):
    """Visible text of an HTML fragment (tags dropped, <br> as space)."""
    return html.unescape(re.sub(r"<[^>]+>", "", re.sub(r"<br\s*/?>", " ", fragment))).strip()


def slide_parts(sid):
    """(heading, time window, say text, list of plain-words bullets) for one slide."""
    s = (DECK / "slides" / f"{sid}.html").read_text(encoding="utf-8")
    heading = text_of(re.search(r"<h[12][^>]*>(.*?)</h[12]>", s, re.S).group(1))
    note = html.unescape(re.search(r"<aside>(.*?)</aside>", s, re.S).group(1))
    window = re.search(r"\[(.*?)\]", note).group(1)
    m = re.search(r'SAY: (".*?")(.*?)\n', note + "\n", re.S)
    say_text = m.group(1) + m.group(2)
    plain = note.split("IN PLAIN WORDS:", 1)[1] if "IN PLAIN WORDS:" in note else ""
    bullets = [b.strip()[2:] for b in plain.strip().splitlines() if b.strip().startswith("- ")]
    return heading, window, say_text, bullets


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build(path):
    deck = json.loads((DECK / "deck.json").read_text(encoding="utf-8"))
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="SWINQ: 3D stroke from sensor data — pitch")
    story = [Paragraph("SWINQ: one sensor, one forehand, in 3D", title),
             Paragraph("3-minute pitch script with plain-language notes · matches the slide deck", sub)]
    for i, sid in enumerate(deck["order"], 1):
        heading, window, say_text, bullets = slide_parts(sid)
        block = [Paragraph(f"Slide {i}: {esc(heading.title() if heading.isupper() else heading)}", h),
                 Paragraph(esc(window), cue), Paragraph("SAY", label), Paragraph(esc(say_text), say)]
        if bullets:
            block.append(Paragraph("IN PLAIN WORDS", label))
            block += [Paragraph(esc(b), bullet, bulletText="•") for b in bullets]
        story.append(KeepTogether(block))
    rows = [[Paragraph("<b>Question</b>", small), Paragraph("<b>Answer</b>", small)]]
    rows += [[Paragraph(esc(q), small), Paragraph(esc(a), small)] for q, a in QA]
    t = Table(rows, colWidths=[55 * mm, 119 * mm])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#D5D9DE")),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF2F7")),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story += [Spacer(1, 4), KeepTogether([Paragraph("Likely judge questions", h), t])]
    story += [Paragraph("Delivery tips", h)] + [Paragraph(esc(x), bullet, bulletText="•") for x in TIPS]
    doc.build(story)


if __name__ == "__main__":
    out = config.OUT_DIR / "SWINQ_pitch.pdf"
    build(out)
    print(f"wrote {out}")
