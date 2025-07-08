import os
import sys
import subprocess
from pydub import AudioSegment
import numpy as np

pan_range = (0.20, 0.80)
output_format = 'mp3'
output_bitrate = '256k'
output_codec = 'mp3'  # 'ipod' for AAC/m4a, 'mp3' for MP3

def apply_8d_effect(filename, out_audio_path, temp_dir):
    """
    Apply a smooth 8D audio effect with consistent volume during panning.
    The panning will complete within the specified duration (in seconds).
    """
    # Get the file extension of the input file (e.g., 'webm', 'wav', 'flac')
    
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
    
    # Load audio file
    audio = AudioSegment.from_file(filename)

    audio = audio.set_sample_width(4)
    
    # Convert to stereo if it's not already
    if audio.channels == 1:
        audio = audio.set_channels(2)

    # Split into left and right channels
    left_channel, right_channel = audio.split_to_mono()

    # Convert to numpy arrays for processing
    samples_left = np.array(left_channel.get_array_of_samples())
    samples_right = np.array(right_channel.get_array_of_samples())
    
    # Convert to stereo if it's not already
    if audio.channels == 1:
        audio = audio.set_channels(2)

    # Set chunk duration (20 ms in this case)
    chunk_samples = int(audio.frame_rate * 20)
    num_chunks = len(samples_left) // chunk_samples + 1

    # Initialize lists to store the processed chunks
    samples_left_panned_chunks = []
    samples_right_panned_chunks = []

    # Process audio in chunks
    for i in range(num_chunks):
        start_idx = i * chunk_samples
        end_idx = min((i + 1) * chunk_samples, len(samples_left))

        chunk_length = end_idx - start_idx

        # Generate the time vector for the current chunk
        t_chunk = np.linspace(0, 1, chunk_length)

        # Calculate the panning frequency for the chunk
        if i == num_chunks - 1:
            # For the last chunk, scale the frequency to fit the chunk's length
            panning_frequency = 3 * (chunk_length / chunk_samples)  # Scale frequency for last chunk
        else:
            panning_frequency = 3  # Normal 12 Hz for other chunks

        # Apply the panning effect for the current chunk
        pan_min, pan_max = pan_range
        pan_chunk = (pan_max - pan_min) * 0.5 * (np.sin(2 * np.pi * panning_frequency * t_chunk) + 1) + pan_min

        # Apply the panning to the left and right channels for the current chunk
        left_chunk = samples_left[start_idx:end_idx] * (1 - pan_chunk)
        right_chunk = samples_right[start_idx:end_idx] * pan_chunk

        # Append the panned chunks to the lists
        samples_left_panned_chunks.append(left_chunk)
        samples_right_panned_chunks.append(right_chunk)

    # Concatenate all the panned chunks
    samples_left_panned = np.concatenate(samples_left_panned_chunks)
    samples_right_panned = np.concatenate(samples_right_panned_chunks)

    # Clip the panned audio to avoid distortion
    combined_left = np.clip(samples_left_panned, -2147483648, 2147483647).astype(np.int32)
    combined_right = np.clip(samples_right_panned, -2147483648, 2147483647).astype(np.int32)

    # Create new AudioSegment instances for the panned audio
    left_channel = AudioSegment(
        combined_left.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=1
    )
    right_channel = AudioSegment(
        combined_right.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=1
    )

    # Combine the left and right channels back into stereo
    stereo_audio = AudioSegment.from_mono_audiosegments(left_channel, right_channel)
    
    # Get file details
    channels = audio.channels
    print(f'channels: {channels}')
    sample_rate = audio.frame_rate
    print(f'sample_rate: {sample_rate}')
    bit_depth = audio.sample_width * 8
    print(f'bit_depth: {bit_depth}')
    duration_seconds = len(audio) / 1000
    print(f'duration_seconds: {duration_seconds}')
    
    bitrate = calculate_bitrate_from_samples(
        channels, sample_rate, bit_depth
    )
    print(f'bitrate: {bitrate}')
    
    ext = 'flac'
    filename = os.path.splitext(filename)[0] + f'.{ext}'
    out_audio_path = os.path.splitext(filename)[0] + f'.{output_format}'
    
    # Export the final audio
    stereo_audio.export(out_audio_path, bitrate=output_bitrate, format=output_codec)
#    stereo_audio.export(out_audio_path, bitrate="320k", format=ext, parameters=["-sample_fmt", "s32"])
#    correct_metadata(filename, out_audio_path.replace(f'.{ext}', ".mp3"))


def correct_metadata(filename, out_audio_path):
    subprocess.run([
        'ffmpeg', '-i', filename,
        '-map_metadata', '-1',  # Remove metadata
        '-id3v2_version', '3',  # Set ID3 version
        '-write_id3v1', '1',  # Write ID3v1 metadata
        out_audio_path
    ])


def calculate_bitrate_from_samples(channels, bit_depth, sample_rate):
    """Calculate the bitrate based on sample length."""
    bitrate = channels * bit_depth * sample_rate
    bitrate_kbps = bitrate / 1000  # Convert to kbps
    return bitrate_kbps
    

def process(filename, outpath, title, temp_dir):
    """Apply 8D effect."""
    
    out_audio_path = os.path.join(outpath, os.path.basename(filename))
    
    # Apply the 8D effect to the input file (either downloaded or existing)
    apply_8d_effect(filename, out_audio_path, temp_dir)


def process_file(input_path, outpath, temp_dir):
    """Process a single file (apply 8D effect and add thumbnail)."""
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(outpath, exist_ok=True)
    
    filename = os.path.basename(input_path)
    title = os.path.splitext(filename)[0]
    
    process(filename, outpath, title, temp_dir)


def process_files_in_folder(folder_path, outpath, temp_dir):
    """Process all files in a folder."""
    temp_dir = os.path.join(os.path.basename(folder_path), temp_dir)
    outpath = os.path.join(os.path.basename(folder_path), outpath)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(outpath, exist_ok=True)

    for filename in os.listdir(folder_path):
        if '.' in filename:
            title = os.path.splitext(filename)[0]
            print(f"Processing file: {filename}")
            process(filename, outpath, title, temp_dir)

if __name__ == "__main__":
    outpath = 'out'
    temp_dir = 'temp_audio'
    # Check if the script was run with arguments (drag-and-drop or command line)
    if len(sys.argv) > 1:
        input_path = sys.argv[1]  # Use the first argument as the file or folder path
        
        for input_path in sys.argv[1:]:
            if os.path.isdir(input_path):
                print(f"Processing files in folder: {input_path}")
                process_files_in_folder(input_path, outpath, temp_dir)
            elif os.path.isfile(input_path) and (input_path.endswith(".mp3") or input_path.endswith(".webm") or input_path.endswith(".wav") or input_path.endswith(".flac")):
                print(f"Processing file: {input_path}")
                process_file(input_path, outpath, temp_dir)
            else:
                print(f"Invalid path: {input_path}. Please provide a valid file or folder.")
