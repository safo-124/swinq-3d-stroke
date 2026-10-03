"""Rewrite the deck's speaker notes in plain language and simplify slide wording."""
import re
from pathlib import Path

D = Path(__file__).parent / "project" / "slides"

NOTES = {
"cover": """[0:00-0:20 | 20 s]

SAY: "This is one tennis forehand, rebuilt in 3D using only the small sensor clipped onto the racket strings. No cameras, no special suit. Just one tiny sensor, recording for less than a second."

IN PLAIN WORDS:
- The SWINQ sensor is the little rubber dampener players put on their strings. Inside it is a motion sensor, like the one in your phone that knows when you turn the screen.
- The picture shows our 3D racket at two moments: just before hitting the ball (left) and after (right).
- Brown = handle. Blue = racket head. Red arrow = the way the strings face. Grey line = the path the racket head travelled.
- Tip: start the 3D animation (stroke_3d_gp.mp4) or the Blender video before you start talking.""",

"challenge": """[0:20-0:45 | 25 s]

SAY: "The data is tough. We get less than one second of recording. It starts in the middle of the swing, so we never see the racket standing still. The racket spins at over 1,700 degrees per second. And just before it hits the ball, one of the sensors maxes out."

IN PLAIN WORDS:
- 0.96 s: the sensor records 416 readings per second, so we only get 400 readings in total.
- Why "starting mid-swing" is a problem: to know which way is up, you normally hold the sensor still and let it feel gravity. Here it is never still.
- 1720 degrees per second: almost five full turns every second.
- 16 g: the sensor that measures force can only count up to 16 times the force of gravity, like a bathroom scale that only goes up to 100 kg. During the fastest part of the swing the real force was higher, so the sensor just reads "maximum" for a moment.""",

"approach": """[0:45-1:20 | 35 s]

SAY: "We did three things. First, the sensor tells us how fast the racket is turning, and by adding up those small turns we know its angle at every moment. Second, where the force sensor maxed out, we filled the gap using physics: how fast it spins tells us how hard it must have pulled. Third, we used the two videos as a referee, to check our answer."

IN PLAIN WORDS:
- Card 1, turning into angle: imagine walking with your eyes closed and counting your turns: "a quarter turn left, a half turn right". If you count well, you always know which way you face. The sensor does this 416 times a second. We tested our maths on a made-up swing where we knew the right answer: it was off by only 0.03 degrees.
- Card 2, the maxed-out sensor: like a car in a corner, the faster you spin, the harder you are pushed outwards. We know how fast the racket spins, so we can work out the push the sensor should have felt and fill in the missing part.
- Card 3, video as referee: the sensor only knows how the racket turned, not where it started (like a step counter that knows how many steps you took, but not where you began). So we take the starting angle from the video, and then check everything after against it.""",

"numbers": """[1:20-1:45 | 25 s]

SAY: "From the sensor alone, a player learns: the racket head was moving at about 89 kilometres per hour when it hit the ball; it turned 85 degrees in the last tenth of a second; the real force peaked at about 30 g; and the strings rang at about 580 vibrations per second." (Say "89 kilometres per hour" slowly.)

IN PLAIN WORDS:
- 89 km/h: how fast the middle of the racket head was moving at impact. We get it from how fast the racket was turning and how far the head is from the turning point (the wrist and arm).
- 85 degrees: in the last tenth of a second before the hit, the racket turned almost a quarter turn. This is the "snap" of the swing.
- About 30 g: the true peak force, about twice what the sensor could show (it stopped at 16 g). This is the gap we filled in on the previous slide.
- 581 Hz: the strings vibrate after the ball hits, like a guitar string. The sensor reads too slowly to see this directly; it shows up as a slower "fake" vibration, like car wheels in a film that seem to spin backwards. Working backwards gives about 580 per second, which is normal for tennis strings. Say "likely".""",

"proof": """[1:45-2:10 | 25 s]

SAY: "To check our work, we used two videos of the same swing. Green is our sensor's racket, drawn on top of the real video. At the start of the swing it matches well. But as the swing speeds up, it drifts away from the real racket."

IN PLAIN WORDS:
- Top row = camera in front of the player. Bottom row = camera behind. Left to right = time, from half a second before the hit to half a second after.
- The number under each picture = how many degrees our racket is off from the real one.
- Why it drifts: remember the eyes-closed walk? Small counting mistakes add up over time. Here the sensor and the video disagree in the fastest part of the swing. Our best explanation: the video only takes 30 pictures a second and they are blurry, and the sensor and camera clocks may not be perfectly lined up.
- If asked: we checked and ruled out the obvious causes (how the sensor axes are set up, recording speed, sensor mounting angle). Details are in the notes of the next slide.""",

"limit": """[2:10-2:45 | 35 s]

SAY: "So we correct the drift. At every video frame we measure how far off we are. Then we draw a smooth curve through those errors and subtract it from the sensor's racket. We tested it fairly, on frames the correction never saw, and the error drops from 88 degrees to about 12." (Play gp_correction.mp4 here: orange = before, green = after.)

IN PLAIN WORDS:
- 88 degrees: on average, the uncorrected racket was pointing almost a right angle away from the real one.
- 12 degrees: after the correction, it is close to the real racket.
- How the correction works: think of checkpoints in a race. At each video frame we know exactly how far off we are. Between checkpoints, a smooth curve (a "Gaussian process", a standard statistics tool) estimates the error, and we remove it.
- Why "frames it never saw" matters: we hid one frame at a time, built the correction from the others, and checked whether it predicted the hidden frame. That is a fair test, not just copying the video.
- Biggest remaining errors: right at the moment of impact (the ball shakes the sensor) and in the fast turns after it.
- If asked "is it still sensor-only?": the corrected racket combines the sensor and the video. The speeds and numbers on slide 4 are from the sensor only, and we keep the pure sensor result too.
- If asked what we ruled out: every way the sensor axes could be set up (48), different recording speeds, and sensor tilt; the force sensor also agrees with the spin sensor.""",

"next": """[2:45-3:00 | 15 s]

SAY: "Everything we built is open and tested, and it runs from one small sensor. Knowing the racket's angle at every moment is the hardest half of full motion tracking. This is motion capture from a tennis dampener. Thank you."

IN PLAIN WORDS:
- Python package: our code, with 18 automatic tests that check the maths.
- orientation.csv: a spreadsheet with the racket's angle for every one of the 400 readings.
- Blender script: Blender is free 3D animation software; our script builds a racket and animates it from the spreadsheet.
- "Hardest half": full tracking means knowing both where the racket points (done) and where it is in space (next step). Knowing where it points is the harder part.""",
}

SLIDE_TEXT = {
    "approach": [
        ("Exact quaternion steps on the mean rate per interval. 0.03° error on a synthetic 500°/s swing, unit-tested.",
         "The sensor says how fast the racket turns. Adding up those turns, 416 times a second, gives its angle. Tested to 0.03°."),
        ("Physics repairs saturation", "Physics fills the gap"),
        ("Along the handle the accelerometer feels centripetal force: a = r·ω². Fit r from the gyro (corr 0.92) and fill the clipped gap.",
         "The force sensor tops out at 16 g. How fast the racket spins tells us how hard it pulled, so we fill in the missing part."),
        ("Both cameras calibrated with the racket itself. Video sets only the start pose; everything after is the sensor.",
         "Two cameras check our answer. The video gives only the starting angle; the motion comes from the sensor."),
    ],
    "numbers": [
        ("String vibration (seen aliased at 165 Hz at 416 Hz sampling)", "Strings vibrating after the hit (likely, per second)"),
        ("True peak acceleration, recovered above the 16 g clip", "True peak force, recovered above the sensor's 16 g limit"),
    ],
    "limit": [
        ("mean error with GP correction (leave-one-out; median 8.3°)", "mean error after correction, tested on frames it never saw"),
        ("mean error, sensor only", "mean error before correction"),
    ],
    "cover": [
        ("A full racket stroke rebuilt from the dampener's IMU alone: no cameras, no mocap suit.",
         "A full racket swing rebuilt from the small sensor on the strings: no cameras, no motion-capture suit."),
    ],
}


def main():
    words = 0
    for sid, note in NOTES.items():
        p = D / f"{sid}.html"
        s = p.read_text(encoding="utf-8")
        for old, new in SLIDE_TEXT.get(sid, []):
            assert old in s, (sid, old[:40])
            s = s.replace(old, new)
        assert len(note) < 4000, (sid, len(note))
        body = note.replace("&", "&amp;").replace("<", "&lt;")
        s, n = re.subn(r"<aside>.*?</aside>", lambda _: "<aside>" + body + "</aside>", s, flags=re.S)
        assert n == 1, sid
        p.write_text(s, encoding="utf-8")
        words += len(re.search(r'SAY: "(.*?)"', note, re.S).group(1).split())
    print(f"spoken words: {words} = about {words / 140:.1f} min at 140 words per minute")


if __name__ == "__main__":
    main()
