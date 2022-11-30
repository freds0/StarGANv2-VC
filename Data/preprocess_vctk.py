import argparse
import os
from scipy.io import wavfile
from pydub import AudioSegment
from pydub.silence import split_on_silence
from os import makedirs, listdir, walk
from os.path import join, exists
from tqdm import tqdm
import pandas as pd
import random

def split(sound):
    dBFS = sound.dBFS
    chunks = split_on_silence(sound,
        min_silence_len = 100,
        silence_thresh = dBFS-16,
        keep_silence = 100
    )
    return chunks


def combine(_src):
    audio = AudioSegment.empty()
    for i, filename in enumerate(listdir(_src)):
        if filename.endswith('.wav'):
            filename = join(_src, filename)
            audio += AudioSegment.from_wav(filename)
    return audio


def save_chunks(chunks, directory, sampling_rate=24000):
    if not exists(directory):
        makedirs(directory)
    counter = 0

    target_length = 5 * 1000
    output_chunks = [chunks[0]]
    for chunk in chunks[1:]:
        if len(output_chunks[-1]) < target_length:
            output_chunks[-1] += chunk
        else:
            # if the last output chunk is longer than the target length,
            # we can start a new one
            output_chunks.append(chunk)

    for chunk in output_chunks:
        chunk = chunk.set_frame_rate(sampling_rate)
        chunk = chunk.set_channels(1)
        counter = counter + 1
        chunk.export(join(directory, str(counter) + '.wav'), format="wav")


def downsampling(speakers, input_dir, output_dir, sampling_rate):
    for p in tqdm(speakers):
        directory = join(output_dir,'p' + str(p))
        if not exists(directory):
            audio = combine(input_dir + '/wav48/p' + str(p))
            chunks = split(audio)
            save_chunks(chunks, directory, sampling_rate)


def create_train_test_files(output_dir, speakers, split_size=0.1, output_train_filename='train_list.txt', output_test_filename='val_list.txt' ):
    data_list = []
    for path, subdirs, files in tqdm(walk(output_dir)):
        for name in files:
            if name.endswith(".wav"):
                speaker = int(path.split('/')[-1].replace('p', ''))
                if speaker in speakers:
                    data_list.append({"Path": join(path, name), "Speaker": int(speakers.index(speaker)) + 1})
    # Convert list to dataframe
    data_list = pd.DataFrame(data_list)
    # Shuffle dataframe
    data_list = data_list.sample(frac=1)
    # Define split index
    split_idx = round(len(data_list) * split_size)
    # Split dataframe
    test_data = data_list[:split_idx]
    train_data = data_list[split_idx:]

    # Save to file
    file_str = ""
    for index, k in train_data.iterrows():
        file_str += k['Path'] + "|" +str(k['Speaker'] - 1)+ '\n'
    text_file = open(join(output_dir, output_train_filename), "w")
    text_file.write(file_str)
    text_file.close()

    file_str = ""
    for index, k in test_data.iterrows():
        file_str += k['Path'] + "|" + str(k['Speaker'] - 1) + '\n'
    text_file = open(join(output_dir, output_test_filename), "w")
    text_file.write(file_str)
    text_file.close()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_dir', default='./')
    parser.add_argument('-i', '--input', default='VCTK-Corpus')
    parser.add_argument('-s', '--sampling_rate', default=24000)
    parser.add_argument('-o', '--output', default='Data')

    args = parser.parse_args()

    #speakers = [225,228,229,230,231,233,236,239,240,244,226,227,232,243,254,256,258,259,270,273]
    speakers = [225, 234, 245, 254, 263, 272, 281, 293, 303, 313, 330, 345, 376, 226, 236, 246, 255, 264, 273, 282, 294, 304, 314, 333, 347, 227, 237, 247, 256, 265, 274, 283, 295, 305, 315, 334, 351, 228, 238, 248, 257, 266, 275, 284, 297, 306, 316, 335, 360, 229, 239, 249, 258, 267, 276, 285, 298, 307, 317, 336, 361, 230, 240, 250, 259, 268, 277, 286, 299, 308, 318, 339, 362, 231, 241, 251, 260, 269, 278, 287, 300, 310, 323, 340, 363, 232, 243, 252, 261, 270, 279, 288, 301, 311, 326, 341, 364, 233, 244, 253, 262, 271, 280, 292, 302, 312, 329, 343, 374]

    input_dir = join(args.base_dir, args.input)
    output_dir = join(args.base_dir, args.output)

    print("Downsampling files...")
    downsampling(speakers, input_dir, output_dir, int(args.sampling_rate))

    print("Creating train/test files...")
    create_train_test_files(output_dir, speakers, split_size=0.1)

if __name__ == "__main__":
    main()
