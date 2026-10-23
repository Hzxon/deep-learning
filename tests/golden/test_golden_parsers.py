"""Golden: filename parsing and the six-class label contract. Protected by testgate."""

from __future__ import annotations

from pathlib import Path

import pytest

from ser.data import LABEL_TO_ID, LABELS, parse_cremad, parse_ravdess, parse_savee, parse_tess


def test_label_set_and_ids_are_fixed() -> None:
    assert LABELS == ["angry", "disgust", "fear", "happy", "neutral", "sad"]
    assert LABEL_TO_ID == {"angry": 0, "disgust": 1, "fear": 2, "happy": 3, "neutral": 4, "sad": 5}


@pytest.mark.parametrize(
    ("emotion_code", "expected"),
    [("01", "neutral"), ("03", "happy"), ("04", "sad"), ("05", "angry"), ("06", "fear"), ("07", "disgust")],
)
def test_ravdess_kept_emotions(emotion_code: str, expected: str) -> None:
    path = Path(f"ravdess/Actor_07/03-01-{emotion_code}-01-01-01-07.wav")
    assert parse_ravdess(path) == ("RAVDESS_07", expected)


@pytest.mark.parametrize("emotion_code", ["02", "08"])
def test_ravdess_calm_and_surprise_discarded(emotion_code: str) -> None:
    assert parse_ravdess(Path(f"Actor_01/03-01-{emotion_code}-01-01-01-01.wav")) is None


def test_ravdess_speaker_falls_back_to_filename() -> None:
    assert parse_ravdess(Path("flat/03-01-05-02-02-01-3.wav")) == ("RAVDESS_03", "angry")


def test_ravdess_rejects_malformed_names() -> None:
    assert parse_ravdess(Path("Actor_01/03-01-05-01.wav")) is None
    assert parse_ravdess(Path("Actor_01/03-01-xx-01-01-01-01.wav")) is None


@pytest.mark.parametrize(
    ("code", "expected"),
    [("ANG", "angry"), ("DIS", "disgust"), ("FEA", "fear"), ("HAP", "happy"), ("NEU", "neutral"), ("SAD", "sad")],
)
def test_cremad_emotions(code: str, expected: str) -> None:
    assert parse_cremad(Path(f"cremad/1001_DFA_{code}_XX.wav")) == ("CREMA-D_1001", expected)


def test_cremad_rejects_unknown_and_short() -> None:
    assert parse_cremad(Path("1001_DFA_XYZ_XX.wav")) is None
    assert parse_cremad(Path("1001_DFA.wav")) is None


def test_tess_folder_layout() -> None:
    assert parse_tess(Path("tess/OAF_angry/OAF_back_angry.wav")) == ("TESS_OAF", "angry")
    assert parse_tess(Path("tess/YAF_Fear/YAF_back_fear.wav")) == ("TESS_YAF", "fear")


def test_tess_filename_fallback() -> None:
    assert parse_tess(Path("flat/YAF_dog_sad.wav")) == ("TESS_YAF", "sad")


@pytest.mark.parametrize("folder", ["OAF_Pleasant_surprise", "YAF_pleasant_surprised", "OAF_ps"])
def test_tess_pleasant_surprise_discarded(folder: str) -> None:
    assert parse_tess(Path(f"tess/{folder}/x.wav")) is None


def test_savee_official_layout() -> None:
    assert parse_savee(Path("AudioData/DC/a01.wav")) == ("SAVEE_DC", "angry")
    assert parse_savee(Path("AudioData/KL/sa03.wav")) == ("SAVEE_KL", "sad")
    assert parse_savee(Path("AudioData/JE/su02.wav")) is None


def test_savee_flat_layout() -> None:
    assert parse_savee(Path("ALL/JK_n10.wav")) == ("SAVEE_JK", "neutral")
    assert parse_savee(Path("ALL/je_h05.wav")) == ("SAVEE_JE", "happy")
    assert parse_savee(Path("ALL/XX_a01.wav")) is None
