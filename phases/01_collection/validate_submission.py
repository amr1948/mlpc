import sys
import zipfile
import tempfile
from pathlib import Path

import pandas as pd
from pydub import AudioSegment


# ------------------
# Configuration
# ------------------

MIN_DURATION = 15.0
MAX_DURATION = 35.0
MIN_TOLERANCE = 3.0
MAX_TOLERANCE = 5.0

MANDATORY_SCENES = 8
MAX_TOTAL_SCENES = 10

MAX_CLASSES_PER_SCENE = 5

ALLOWED_ENVIRONMENTS = {
    "kitchen",
    "bathroom",
    "toilet",
    "hallway",
    "bedroom",
    "living_room",
    "office",
}

ALLOWED_DEVICE_PLACEMENT = {"static", "mobile"}

ALLOWED_AUDIO_EXTENSIONS = {
    ".wav",
    ".m4a",
    ".mp3",
    ".mp4",
    ".3gp",
    ".ogg",
    ".flac",
    ".aac",
    ".caf",
}

REQUIRED_COLUMNS = [
    "filename",
    "target_classes",
    "non_target_sounds",
    "recording_device",
    "device_placement",
    "recording_environment",
    "scene_description",
]

TARGET_CLASSES = {
    "bell_ringing",
    "coffee_machine",
    "cutlery_dishes",
    "door_open_close",
    "footsteps",
    "keyboard_typing",
    "keychain",
    "light_switch",
    "microwave",
    "phone_ringing",
    "running_water",
    "toilet_flushing",
    "vacuum_cleaner",
    "wardrobe_drawer_open_close",
    "window_open_close",
}

REQUIRED_SPECIAL_CLASSES = {
    "bell_ringing",
    "keychain",
    "phone_ringing",
}


# ------------------
# Helpers
# ------------------

def error(msg):
    print(f"[ERROR] {msg}")
    sys.exit(1)


def warn(msg):
    print(f"[WARNING] {msg}")


def ok(msg):
    print(f"[OK] {msg}")


def normalize_label(text, lowercase=True):
    """
    Normalize canonical labels:
    - strip leading/trailing whitespace
    - remove surrounding quotes
    - replace internal whitespace with underscores
    - optionally lowercase
    """
    if pd.isna(text):
        return ""

    text = str(text).strip().strip('\'"')
    text = "_".join(text.split())  # replaces ANY whitespace block with single "_"

    if lowercase:
        text = text.lower()

    return text


def split_semicolon(field, lowercase=True):
    if pd.isna(field) or str(field).strip() == "":
        return []

    entries = [
        normalize_label(f, lowercase=lowercase)
        for f in str(field).split(";")
        if f.strip()
    ]

    return entries



def load_audio(path):
    try:
        return AudioSegment.from_file(path)
    except Exception:
        error(f"Could not read audio file: {path.name}")


# ------------------
# Main validation
# ------------------

def main(zip_path):

    if not zip_path.exists():
        error("Submission zip file does not exist.")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Extract zip
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(tmpdir)
        except zipfile.BadZipFile:
            error("Provided file is not a valid ZIP archive.")

        ok("ZIP archive extracted.")

        # Find audio files
        audio_files = [
            p for p in tmpdir.glob("**/*")
            if p.suffix.lower() in ALLOWED_AUDIO_EXTENSIONS
            and "__MACOSX" not in p.parts                       # Check for hidden macOS metadata folder
        ]

        metadata_files = list(tmpdir.glob("**/metadata.csv"))

        if len(metadata_files) != 1:
            error("Exactly one metadata.csv must be present.")

        if len(audio_files) < MANDATORY_SCENES:
            error("At least 8 scenes are required.")

        if len(audio_files) > MAX_TOTAL_SCENES:
            error("No more than 10 scenes allowed (8 mandatory + up to 2 bonus).")

        ok(f"Found {len(audio_files)} audio files.")

        # Duration check
        lower_bound = MIN_DURATION - MIN_TOLERANCE
        upper_bound = MAX_DURATION + MAX_TOLERANCE

        for audio_path in audio_files:
            audio = load_audio(audio_path)
            duration = len(audio) / 1000.0

            if not (lower_bound <= duration <= upper_bound):
                error(
                    f"{audio_path.name} duration {duration:.2f}s "
                    f"(allowed {lower_bound:.2f}–{upper_bound:.2f}s)."
                )

        ok("All audio durations valid.")

        # Load metadata
        try:
            # Find the header row by checking which line contains column names
            header_row = 0
            with open(metadata_files[0], "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line_lower = line.strip().lower()
                    has_column = any(col in line_lower for col in REQUIRED_COLUMNS)
                    if has_column:
                        header_row = idx
                        break
            
            if header_row > 0:
                warn(
                    f"Header row found at line {header_row + 1}. "
                    f"Skipping first {header_row} row(s)."
                )
            
            df = pd.read_csv(metadata_files[0], skiprows=header_row, header=0)
        except Exception:
            error("Could not read metadata.csv.")

        # Normalize column names (students may use spaces instead of underscores)
        df.columns = [
            normalize_label(col, lowercase=True)
            for col in df.columns
        ]

        # Column check
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                error(f"Missing required metadata column: '{col}'")

        ok("All required metadata columns present.")

        # Row count check
        if len(df) != len(audio_files):
            error("Number of metadata rows must match number of audio files.")

        if df["filename"].duplicated().any():
            error("Duplicate filenames found in metadata.csv.")

        # Filename consistency
        audio_names = {p.name for p in audio_files}
        metadata_names = set(df["filename"].astype(str).str.strip())

        if audio_names != metadata_names:

            missing_in_metadata = audio_names - metadata_names
            missing_audio_files = metadata_names - audio_names

            msg = ["Mismatch between metadata filenames and audio files."]

            if missing_in_metadata:
                msg.append(
                    "The following audio files are missing in metadata.csv: "
                    + ", ".join(sorted(missing_in_metadata))
                )

            if missing_audio_files:
                msg.append(
                    "The following filenames listed in metadata.csv were not found in the submission: "
                    + ", ".join(sorted(missing_audio_files))
                )

            error("\n".join(msg))

        ok("Filenames consistent.")

        # Global counters
        all_classes_used = set()
        multi_class_scenes = 0
        static_count = 0
        mobile_count = 0

        # Row validation
        for _, row in df.iterrows():

            filename = str(row["filename"]).strip()
            targets = split_semicolon(row["target_classes"])
            non_targets = split_semicolon(row["non_target_sounds"])
            environments = split_semicolon(row["recording_environment"])
            placement = str(row["device_placement"]).strip()

            if not targets:
                error(f"{filename}: no target classes specified.")

            # Remove duplicate target classes (preserve order)
            if len(set(targets)) != len(targets):
                warn(
                    f"{filename}: duplicate target classes found — duplicates removed. "
                    "If multiple occurrences of the same class are present, "
                    "describe their temporal order and repetition in the scene_description field."
                )
                targets = list(dict.fromkeys(targets))

            if len(targets) > MAX_CLASSES_PER_SCENE:
                error(
                    f"{filename}: more than {MAX_CLASSES_PER_SCENE} "
                    "distinct target classes in one scene."
                )

            for t in targets:
                if t not in TARGET_CLASSES:
                    error(f"{filename}: invalid target class '{t}'.")

            # Non-target must not overlap with target
            for nt in non_targets:
                if nt in TARGET_CLASSES:
                    error(
                        f"{filename}: '{nt}' appears in non_target_sounds "
                        "but is a defined target class."
                    )

            all_classes_used.update(targets)

            if len(targets) > 1:
                multi_class_scenes += 1

            # Device placement
            if placement not in ALLOWED_DEVICE_PLACEMENT:
                error(f"{filename}: device_placement must be 'static' or 'mobile'.")

            if placement == "static":
                static_count += 1
            else:
                mobile_count += 1

            # Environment
            for env in environments:
                if env not in ALLOWED_ENVIRONMENTS:
                    warn(f"{filename}: non-standard environment '{env}'.")

            # Scene description sanity
            description = str(row["scene_description"]).strip().strip('\'"')

            if len(description) < 40:
                warn(
                    f"{filename}: scene_description is very short. "
                    "Please provide a sufficiently detailed description (2–5 sentences)."
                )

        # -------- GLOBAL CONSTRAINTS --------

        if multi_class_scenes < 6:
            error("At least 6 scenes must contain more than one target class.")

        if static_count < 3:
            error("At least 3 scenes must use static device placement.")

        if mobile_count < 3:
            error("At least 3 scenes must use mobile device placement.")

        if len(all_classes_used) < 10:
            error("At least 10 different target classes must be covered.")

        special_count = len(all_classes_used & REQUIRED_SPECIAL_CLASSES)
        if special_count < 2:
            error(
                "At least two of the following classes must be included: "
                "bell_ringing, keychain, phone_ringing."
            )

        ok("All global constraints satisfied.")
        print("\n[SUCCESS] Submission passed all validation checks.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_submission.py <submission.zip>")
        sys.exit(1)

    main(Path(sys.argv[1]))
