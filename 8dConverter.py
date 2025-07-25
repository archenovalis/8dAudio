import os
import sys
from pydub import AudioSegment
import numpy as np

# === Configurable Parameters ===
pan_range = (0.18, 0.85)         # How far left/right it pans
rotation_speed = 0.25            # Rotations per second
gain_range = (0.05, -0.15)         # Front and rear gain

output_format = 'mp3'
output_bitrate = '256k'
output_codec = 'mp3'

def directional_weight(angle_deg, active_start, active_end):
    angle = angle_deg % 360
    if active_start < active_end:
        if not (active_start <= angle <= active_end):
            return 0
        t = (angle - active_start) / (active_end - active_start)
    else:
        if not (angle >= active_start or angle <= active_end):
            return 0
        t = (angle - active_start) % 360 / ((active_end - active_start) % 360)
    return np.sin(np.pi * t) ** 2  # smooth bell curve

def apply_8d_effect(filename, out_audio_path):
    audio = AudioSegment.from_file(filename)
    audio = audio.set_sample_width(4)

    if audio.channels == 1:
        audio = audio.set_channels(2)

    left, right = audio.split_to_mono()
    left_np = np.array(left.get_array_of_samples())
    right_np = np.array(right.get_array_of_samples())

    chunk_samples = int(audio.frame_rate * 0.02)
    total_samples = len(left_np)
    num_chunks = total_samples // chunk_samples

    left_chunks = []
    right_chunks = []

    for i in range(num_chunks):
        start = i * chunk_samples
        end = start + chunk_samples

        chunk_l = left_np[start:end].astype(np.float32)
        chunk_r = right_np[start:end].astype(np.float32)

        # === ROTATION ANGLE ===
        t = i * 0.02
        angle_rad = 2 * np.pi * rotation_speed * t
        angle_deg = (angle_rad * 180 / np.pi - 90) % 360  # ⬅️ ALIGN FRONT TO 0°

        # === PANNING ===
        pan = pan_range[0] + (pan_range[1] - pan_range[0]) * 0.5 * (1 + np.cos(angle_rad))

        # === GAIN BASED ON FRONT-BACK POSITION ONLY ===
        front_emphasis = directional_weight(angle_deg, 270, 90)  # Peaks at 0°, 0 at 90°/270°
        angle_gain = gain_range[1] + (gain_range[0] - gain_range[1]) * front_emphasis
        gain = 1.0 + angle_gain

        # === APPLY GAIN + PAN ===
        chunk_l *= (1 - pan) * gain
        chunk_r *= pan * gain

        left_chunks.append(chunk_l)
        right_chunks.append(chunk_r)

    # === Final Assembly ===
    left_final = np.clip(np.concatenate(left_chunks), -2147483648, 2147483647).astype(np.int32)
    right_final = np.clip(np.concatenate(right_chunks), -2147483648, 2147483647).astype(np.int32)

    left_seg = AudioSegment(
        left_final.tobytes(), frame_rate=audio.frame_rate, sample_width=audio.sample_width, channels=1
    )
    right_seg = AudioSegment(
        right_final.tobytes(), frame_rate=audio.frame_rate, sample_width=audio.sample_width, channels=1
    )

    final = AudioSegment.from_mono_audiosegments(left_seg, right_seg)
    final.export(out_audio_path, bitrate=output_bitrate, format=output_codec)

# === Batch Logic ===
def process_file(input_path, outpath):
    os.makedirs(outpath, exist_ok=True)
    filename = os.path.basename(input_path)
    out_audio_path = os.path.join(outpath, os.path.splitext(filename)[0] + '.' + output_format)
    apply_8d_effect(input_path, out_audio_path)

def process_files_in_folder(folder_path, outpath):  
    outpath = os.path.join(os.path.basename(folder_path), outpath)
    os.makedirs(outpath, exist_ok=True)
    for filename in os.listdir(folder_path):
        if filename.endswith((".mp3", ".webm", ".wav", ".flac")):
            process_file(os.path.join(folder_path, filename), outpath)

if __name__ == "__main__":
    outpath = 'out'
    if len(sys.argv) > 1:
        for input_path in sys.argv[1:]:
            if os.path.isdir(input_path):
                print(f"Processing folder: {input_path}")
                process_files_in_folder(input_path, outpath)
            elif os.path.isfile(input_path):
                print(f"Processing file: {input_path}")
                process_file(input_path, outpath)
            else:
                print(f"Invalid: {input_path}")
