import librosa as lr
import os
import soundfile as sf

# input_dir = ""

def stereo2mono(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        if os.path.splitext(filepath)[1] != '.wav':
            continue
        output_path = os.path.join(output_dir, filename)
        y_t, sr = lr.load(filepath, sr=None)
        y_t = lr.to_mono(y_t)
        sf.write(output_path, y_t, sr)
        print(output_path + " written.")

if __name__ == '__main__':
    input_dir = "data_syn/stereo"
    output_dir = "data_syn"
    stereo2mono(input_dir=input_dir, output_dir=output_dir)
