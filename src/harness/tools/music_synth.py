"""Tiny synthesizer for short public-domain music excerpts (stdlib only: wave + math).

Used when no local audio file matches a request, so the harness can play something real
without any music service, account or download. Excerpts are clearly labelled as
synthesized renditions of public-domain works.
"""

from __future__ import annotations

import math
import re
import struct
import wave
from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE = 22_050

# note name -> frequency (Hz), octave 3-5
_BASE = {"C": 261.63, "C#": 277.18, "Db": 277.18, "D": 293.66, "D#": 311.13, "Eb": 311.13,
         "E": 329.63, "F": 349.23, "F#": 369.99, "Gb": 369.99, "G": 392.00, "G#": 415.30,
         "Ab": 415.30, "A": 440.00, "A#": 466.16, "Bb": 466.16, "B": 493.88}


def _freq(note: str) -> float:
    if note == "R":
        return 0.0
    name, octave = note[:-1], int(note[-1])
    return _BASE[name] * (2 ** (octave - 4))


@dataclass(frozen=True)
class Piece:
    key: str
    title: str
    composer: str
    aliases: tuple[str, ...]
    notes: tuple[tuple[str, float], ...]  # (note, beats)
    tempo: int = 120


CATALOG: tuple[Piece, ...] = (
    Piece(
        "beethoven-5", "Symphony No. 5 in C minor, Op. 67 (opening motif)", "Ludwig van Beethoven",
        ("beethoven 5", "beethoven's 5th", "beethoven 5th", "fifth symphony", "symphony no. 5",
         "symphony no 5", "beethoven symphony 5", "beethoven"),
        (("G4", .5), ("G4", .5), ("G4", .5), ("Eb4", 2.5), ("R", .5),
         ("F4", .5), ("F4", .5), ("F4", .5), ("D4", 3.0), ("R", .5)) * 2,
        tempo=108,
    ),
    Piece(
        "ode-to-joy", "Symphony No. 9, 'Ode to Joy' theme", "Ludwig van Beethoven",
        ("ode to joy", "symphony no. 9", "beethoven 9", "ninth symphony"),
        (("E4", 1), ("E4", 1), ("F4", 1), ("G4", 1), ("G4", 1), ("F4", 1), ("E4", 1), ("D4", 1),
         ("C4", 1), ("C4", 1), ("D4", 1), ("E4", 1), ("E4", 1.5), ("D4", .5), ("D4", 2)),
    ),
    Piece(
        "fur-elise", "Für Elise (opening)", "Ludwig van Beethoven",
        ("fur elise", "für elise", "for elise"),
        (("E5", .5), ("D#5", .5), ("E5", .5), ("D#5", .5), ("E5", .5), ("B4", .5), ("D5", .5),
         ("C5", .5), ("A4", 1.5), ("R", .5), ("C4", .5), ("E4", .5), ("A4", .5), ("B4", 1.5)),
        tempo=132,
    ),
)


def _normalize(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"\bno\.\s*", "no. ", text)
    return re.sub(r"\s+", " ", text).strip()


def find_piece(query: str) -> Piece | None:
    """Best catalog match for a free-text query (longest alias contained in the query)."""
    q = _normalize(query)
    best: tuple[int, Piece] | None = None
    for piece in CATALOG:
        for alias in piece.aliases:
            if alias in q and (best is None or len(alias) > best[0]):
                best = (len(alias), piece)
    return best[1] if best else None


def render(piece: Piece, path: Path, *, volume: int = 70) -> Path:
    """Write ``piece`` as a mono 16-bit WAV file with a soft piano-like envelope."""
    amplitude = 0.6 * max(0, min(volume, 100)) / 100
    beat = 60.0 / piece.tempo
    frames = bytearray()
    for note, beats in piece.notes:
        freq = _freq(note)
        count = int(SAMPLE_RATE * beat * beats)
        for i in range(count):
            if freq == 0.0:
                sample = 0.0
            else:
                t = i / SAMPLE_RATE
                attack = min(1.0, i / (SAMPLE_RATE * 0.01))
                decay = math.exp(-2.5 * t / max(beat * beats, 0.05))
                tone = (math.sin(2 * math.pi * freq * t) + 0.3 * math.sin(4 * math.pi * freq * t)
                        + 0.1 * math.sin(6 * math.pi * freq * t)) / 1.4
                sample = amplitude * attack * decay * tone
            frames += struct.pack("<h", int(max(-1.0, min(1.0, sample)) * 32_000))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(bytes(frames))
    return path
