import os
import sys
from pydub import AudioSegment
import numpy as np
from scipy.signal import butter, lfilter

pan_range = (0.2, 0.8)
output_format = 'mp3'
output_bitrate = '256k'
output_codec = 'mp3'
rotation_speed = 0.05  # Rotations per second


def butter_lowpass_filter(data, cutoff, fs, order=6):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False) # type: ignore
    y = lfilter(b, a, data)
    return y


def apply_8d_effect(filename, out_audio_path, temp_dir):
    base_dir = ''
    if os.path.basename(os.path.dirname(temp_dir)):
        base_dir = os.path.basename(os.path.dirname(temp_dir))+'\\'
    ext = os.path.splitext(filename)[1][1:]
    
    if ext != 'mp3':
        # Convert to mp3
        audio = AudioSegment.from_file(base_dir+filename, format=ext)
        
        ext = 'wav'
        # Get the new file path by changing the extension to .mp3
        filename = temp_dir+'\\'+os.path.splitext(filename)[0] + f'.{ext}'
        out_audio_path = os.path.splitext(out_audio_path)[0] + f'.{ext}'
        
        # Export the final audio
        audio.export(filename, format=ext, bitrate="320k", parameters=["-sample_fmt", "s16"])
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

    max_delay_samples = int(audio.frame_rate * 0.0007)  # ~0.7ms ITD

    for i in range(num_chunks):
        start = i * chunk_samples
        end = start + chunk_samples

        chunk_l = left_np[start:end].astype(np.float32)
        chunk_r = right_np[start:end].astype(np.float32)

        t = i * 0.02
        angle = 2 * np.pi * rotation_speed * t  # radians

        pan = 0.5 * (1 + np.cos(angle))
        gain = 1.0 + 0.15 * np.cos(angle)
        is_behind = np.cos(angle + np.pi) > 0.5

        itd = int(max_delay_samples * np.sin(angle))

        if is_behind:
            chunk_l = butter_lowpass_filter(chunk_l, 3000, audio.frame_rate)
            chunk_r = butter_lowpass_filter(chunk_r, 3000, audio.frame_rate)

        chunk_l *= (1 - pan) * gain
        chunk_r *= pan * gain

        chunk_l = np.roll(chunk_l, -itd)
        chunk_r = np.roll(chunk_r, itd)

        left_chunks.append(chunk_l)
        right_chunks.append(chunk_r)

    left_final = np.clip(np.concatenate(left_chunks), -2147483648, 2147483647).astype(np.int32)
    right_final = np.clip(np.concatenate(right_chunks), -2147483648, 2147483647).astype(np.int32)

    left_seg = AudioSegment(
        left_final.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=1
    )
    right_seg = AudioSegment(
        right_final.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=1
    )

    final = AudioSegment.from_mono_audiosegments(left_seg, right_seg)
    final.export(out_audio_path, bitrate=output_bitrate, format=output_codec)


def calculate_bitrate_from_samples(channels, bit_depth, sample_rate):
    return (channels * bit_depth * sample_rate) / 1000


def process_file(input_path, outpath, temp_dir):
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    filename = os.path.basename(input_path)
    out_audio_path = os.path.join(outpath, os.path.splitext(filename)[0] + '.' + output_format)
    apply_8d_effect(input_path, out_audio_path, temp_dir)

def process_files_in_folder(folder_path, outpath, temp_dir):
    """Process all files in a folder."""
    temp_dir = os.path.join(os.path.basename(folder_path), temp_dir)
    outpath = os.path.join(os.path.basename(folder_path), outpath)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(outpath, exist_ok=True)

    for filename in os.listdir(folder_path):
        if '.' in filename and (filename.endswith(".mp3") or filename.endswith(".webm") or filename.endswith(".wav") or filename.endswith(".flac")):
            process_file(filename, outpath, temp_dir)

if __name__ == "__main__":
    outpath = 'out'
    temp_dir = 'temp_audio'
    if len(sys.argv) > 1:
        for input_path in sys.argv[1:]:
            if os.path.isdir(input_path):
                print(f"Processing folder: {input_path}")
                process_files_in_folder(input_path, outpath, temp_dir)
            elif os.path.isfile(input_path) and (input_path.endswith(".mp3") or input_path.endswith(".webm") or input_path.endswith(".wav") or input_path.endswith(".flac")):
                print(f"Processing file: {input_path}")
                process_file(input_path, outpath, temp_dir)
            else:
                print(f"Invalid: {input_path}. Please provide a valid file or folder.")
