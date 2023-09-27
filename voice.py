import torch
import torchaudio
import argparse
import os
from pydub import AudioSegment

# membuat dictionary untuk menyimpan path file reference WAV
REFERENCE_WAVS = {
    "taher": [
        "D:/machineLearning/DeepFake/voice_fix/reference/taher/aldi1-mono.wav",
        "D:/machineLearning/DeepFake/voice_fix/reference/taher/aldi2-mono.wav",
        "D:/machineLearning/DeepFake/voice_fix/reference/taher/aldi3-mono.wav",
    ],
    "dekan": [],
    "ryan": [
        'D:/machineLearning/DeepFake/voice_fix/reference/ryan/ryan1-mono.wav',
        'D:/machineLearning/DeepFake/voice_fix/reference/ryan/ryan2-mono.wav',
        'D:/machineLearning/DeepFake/voice_fix/reference/ryan/ryan3-mono.wav',
        'D:/machineLearning/DeepFake/voice_fix/reference/ryan/ryan4-mono.wav',
        'D:/machineLearning/DeepFake/voice_fix/reference/ryan/ryan5-mono.wav',
        'D:/machineLearning/DeepFake/voice_fix/reference/ryan/ryan6-mono.wav',
        'D:/machineLearning/DeepFake/voice_fix/reference/ryan/ryan7-mono.wav',
    ],
}

# membuat parser untuk CLI argumen
parser = argparse.ArgumentParser(description='Deepfake Voice')
parser.add_argument('--ref', required=True, help='Reference file.')
parser.add_argument('--src', required=True, help='Source file.')
args = parser.parse_args()

ref = args.ref
src = args.src
print(f'SOURCE IS {src}')

# mengecek apakah input reference ada di dictionary
if ref not in REFERENCE_WAVS:
    print("Error: Invalid reference name.")
    exit(1)

ref_wav_paths = REFERENCE_WAVS[ref]

# Preproses data audio dari stereo ke mono
sound = AudioSegment.from_wav(src)
sound = sound.set_channels(1)
src_mono = src.replace(".wav", "-mono.wav")
sound.export(src_mono, format="wav")
src_wav_path = src_mono
print(f'SOURCE\'s MONO IS SAVED TO {src_mono}')

# Load model KNN-VC
knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)

# Menghitung features untuk source audio
query_seq = knn_vc.get_features(src_wav_path)

# Menghitung matching set untuk reference audio
matching_set = knn_vc.get_matching_set(ref_wav_paths)

# Menerapkan model KNN-VC untuk voice matching
out_wav = knn_vc.match(query_seq, matching_set, topk=4)

# Membuat path folder dan nama file output 
output_directory = "D:/machineLearning/DeepFake/voice_fix/output"
base_filename = "output_audio"
file_number = 1

# Menghindari duplikasi nama file output
while os.path.exists(os.path.join(output_directory, f"{base_filename}_{file_number}.wav")):
    file_number += 1

new_file_path = os.path.join(output_directory, f"{base_filename}_{file_number}.wav")

# Menyimpan hasil output audio
torchaudio.save(new_file_path, out_wav[None], 16000)

print(f'OUTPUT IS SAVED TO {new_file_path}')