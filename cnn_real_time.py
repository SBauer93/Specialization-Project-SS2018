#!/usr/bin/env python3
"""Show a text-mode spectrogram using live microphone data."""
import argparse
import math
import numpy as np
import shutil
import time as t
from cnn_fft_model import model

usage_line = ' press <enter> to quit, +<enter> or -<enter> to change scaling '


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

try:
    columns, _ = shutil.get_terminal_size()
except AttributeError:
    columns = 80

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-l', '--list-devices', action='store_true',
                    help='list audio devices and exit')
parser.add_argument('-b', '--block-duration', type=float,
                    metavar='DURATION', default=50,
                    help='block size (default %(default)s milliseconds)')
parser.add_argument('-c', '--columns', type=int, default=columns,
                    help='width of spectrogram')
parser.add_argument('-d', '--device', type=int_or_str,
                    help='input device (numeric ID or substring)')
parser.add_argument('-g', '--gain', type=float, default=10,
                    help='initial gain factor (default %(default)s)')
parser.add_argument('-r', '--range', type=float, nargs=2,
                    metavar=('LOW', 'HIGH'), default=[100, 2000],
                    help='frequency range (default %(default)s Hz)')
args = parser.parse_args()

low, high = args.range

if high <= low:
    parser.error('HIGH must be greater than LOW')

try:
    import sounddevice as sd

    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)

    samplerate = sd.query_devices(args.device, 'input')['default_samplerate']

    model.load('cnn_fft.tflearn')

    delta_f = (high - low) / (args.columns - 1)
    fftsize = math.ceil(samplerate / delta_f)
    low_bin = math.floor(low / delta_f)

    start_time = 0
    sample_results = np.empty((0, 5))

    def callback(indata, frames, time, status):
        global start_time, sample_results
        if status:
            text = ' ' + str(status) + ' '
            print('\x1b[34;40m', text.center(args.columns, '#'),
                  '\x1b[0m', sep='')
        if any(indata) and indata[0, 0] > 0.01:
            if start_time == 0:
                start_time = t.time()
            X = np.empty((0, 2048))
            data = np.transpose(indata)
            X = np.append(X, data, axis=0)
            X_f = np.abs(np.fft.rfft(X, axis=1))
            X_f = np.float32(X_f)
            X_f = np.reshape(X_f, (-1, 1025, 1))
            predictions = model.predict_label(X_f)
            sample_results = np.append(sample_results, predictions, axis=0)
        else:
            if t.time() - start_time > 10 and start_time != 0:
                print('Time elapsed: ' + str(t.time() - start_time))
                print(sample_results.shape)
                sample_mean = np.mean(sample_results, axis=0)
                print('Predictions mean: ' + str(sample_mean))
                sample_results = np.empty((0, 5))
                start_time = 0

    with sd.InputStream(device=args.device, channels=1, callback=callback,
                        blocksize=2048,
                        samplerate=samplerate):
        while True:
            response = input()
            if response in ('', 'q', 'Q'):
                break
except KeyboardInterrupt:
    parser.exit('Interrupted by user')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))