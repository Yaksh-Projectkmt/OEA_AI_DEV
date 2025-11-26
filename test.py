from flask import Flask, request, jsonify
import requests,json
import time
import pandas as pd
import numpy as np
import tensorflow as tf

from scipy import signal
from scipy import sparse
from scipy.integrate import trapz
from scipy.sparse.linalg import spsolve
from scipy.signal import (find_peaks, firwin,medfilt,butter, filtfilt, welch, argrelextrema)
from statistics import mode, pvariance
import scipy.stats as stats
from sewar.full_ref import (mse, rmse, ergas, rase, sam)
from scipy.fft import rfft
import pywt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import colormaps
from scipy.interpolate import interp1d
import glob
import tools as st
import utils
import neurokit2 as nk
import cv2
import warnings
import threading
from biosppy.signals import ecg as hami
import concurrent.futures
import scipy
import os
from PIL import Image
import pytesseract
from collections import Counter
import easyocr
warnings.filterwarnings('ignore')
##gpus = tf.config.experimental.list_physical_devices('GPU')
##if gpus:
##    try:
##        for gpu in gpus:
##            tf.config.experimental.set_memory_growth(gpu, True)
##            tf.config.experimental.set_virtual_device_configuration(
##                gpu,
##                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])  # Limit memory to 4GB
##    except RuntimeError as e:
##        print(e)
##os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
results_lock = threading.RLock()


interpreter = tf.lite.Interpreter(model_path="Model\\BIDIRECTION_NEWCNN_PVC_PHYSICS__LBBB_RBBB_43_B0V2_CNN3_new_Freq_exctract_demo_new.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe' # add to your location path
reader = easyocr.Reader(['en']) # , 'la'

app = Flask(__name__)

def lowpass(ecg_signal, cutoff=0.3):
    b, a = signal.butter(3, cutoff, btype='lowpass', analog=False)
    low_passed = signal.filtfilt(b, a, ecg_signal)
    return low_passed

def baseline_construction_200(ecg_signal, kernel_Size=101):
    s_corrected = signal.detrend(ecg_signal)
    baseline_corrected = s_corrected - signal.medfilt(s_corrected, kernel_Size)
    return baseline_corrected

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

with tf.device('/CPU:0'):
    jr_model = load_tflite_model("Model\\JR_model_20_09_0_two_different_inputs.tflite")
    pac_model = load_tflite_model("Model\\low_PAC_GRU_EFF_TRANSFORMER_23.tflite")
    afib_model = load_tflite_model("Model\\afib_flutter_28_6.tflite")
    vfib_model = load_tflite_model("Model\\VFIB_Model_13MAY2024_0932.tflite")
    block_model = load_tflite_model("Model\\block_model_02_07_24.tflite")
    noise_model = load_tflite_model('Model\\NOISE_16_GPT.tflite')
    st_t_abn_model = load_tflite_model("Model\\st_w_t_abn_3 4.tflite")
    let_inf_moedel = load_tflite_model("Model\\31_5_mi_cwt.tflite")
##    ecgnoecgmodel = load_tflite_model('Model\\restingecgModel.tflite')
  
def predict_tflite_model(model:tuple, input_data:tuple):
    with results_lock:
        interpreter, input_details, output_details = model
        for i in range(len(input_data)):
            interpreter.set_tensor(input_details[i]['index'], input_data[i])
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])


# model_paths = 'tflite/restingecgModel.tflite'
   

# interpreter_ecg = tf.lite.Interpreter(model_path=model_paths)
# interpreter_ecg.allocate_tensors()


# input_details_ecg = interpreter_ecg.get_input_details()
# output_details_ecg = interpreter_ecg.get_output_details()

# def prediction_model_ECGnoECG(image_path):
#     with results_lock:
#         classes = ['ECG', 'No ECG']
#         image = Image.open(image_path)
#         input_arr = np.array(image, dtype=np.float32)
#         input_arr = tf.image.resize(input_arr, size=(224, 224), method=tf.image.ResizeMethod.BILINEAR)
#         input_arr = tf.expand_dims(input_arr, axis=0)
#         interpreter_ecg.set_tensor(input_details_ecg[0]['index'], input_arr)
#         interpreter_ecg.invoke()
#         output_data = interpreter_ecg.get_tensor(output_details_ecg[0]['index'])
#         idx = np.argmax(output_data[0])
#         return classes[idx]
    


# Noise detection
class NoiseDetection:
    def __init__(self, raw_data , frequency=200):
        self.frequency = frequency
        self.raw_data = raw_data

    def prediction_model(self,input_arr):
        classes = ['Noise', 'Normal']
        input_arr = tf.cast(input_arr, dtype=tf.float32)
        input_arr = tf.image.resize(input_arr, size=(224, 224), method=tf.image.ResizeMethod.BILINEAR)
        input_arr = (tf.expand_dims(input_arr, axis=0),)
        model_pred = predict_tflite_model(noise_model, input_arr)[0]
        idx = np.argmax(model_pred)
        return classes[idx]

    def plot_to_imagearray(self, ecg_signal):
        fig, ax = plt.subplots(num=1, clear=True)
        ax.plot(ecg_signal, color='black')
        ax.axis(False)
        random_letters = ''.join(random.choices(string.ascii_letters, k=12))
        plt.savefig(var + random_letters + ".jpg")
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data1 = data[:, :, ::-1]
        plt.close(fig)
        return data1

    def noise_model_check(self):
        steps_data = int(self.frequency * 2.5)
        total_data = self.raw_data.shape[0]
        start_data = 0
        normal_index,noise_index = [], []
        percentage = {'Normal': 0, 'Noise': 0}
        while start_data < total_data:
            end_data = start_data + steps_data
            fig, ax = plt.subplots(num=1, clear=True)
            if end_data - start_data == steps_data and end_data < total_data:
                # img_data = self.plot_to_imagearray(self.raw_data[start_data:end_data])
                img_data = pd.DataFrame(self.raw_data[start_data:end_data])
            else: 
                # img_data = self.plot_to_imagearray(self.raw_data[-steps_data:total_data])
                img_data = pd.DataFrame(self.raw_data[-steps_data:total_data])
                end_data = total_data -1

            ax.plot(img_data,color='black')
            ax.axis(False)
            # plt.savefig("temp_noise.jpg")
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            data1 = data[:, :, ::-1]
            plt.close(fig)

            model_result = self.prediction_model(data1) # img_data
            
            if model_result == 'Normal':
                normal_index.append((start_data, end_data))
                percentage['Normal'] += (end_data - start_data) / total_data
            else:
                noise_index.append((start_data, end_data))
                percentage['Noise'] += (end_data - start_data) / total_data

            start_data += steps_data

        Noise_label = 'Normal'
        if int(percentage['Noise']*100) >= 60:
            Noise_label = 'ARTIFACTS'
        # per_normal = int(percentage['Normal']*100)
        # per_noise = int(percentage['Noise']*100)
        # normal_index, noise_index, percentage,
        return Noise_label
  
# Peak detection
class pqrst_detection:
    
    def __init__(self, ecg_signal, fs=200, thres=0.5, lp_thres=0.2, rr_thres=0.12, width=(5, 50), JR=False):
        self.ecg_signal = ecg_signal
        self.fs = fs
        self.thres = thres
        self.lp_thres = lp_thres
        self.rr_thres = rr_thres
        self.width = width
        self.JR = JR

    def hamilton_segmenter(self):

        # check inputs
        if self.ecg_signal is None:
            raise TypeError("Please specify an input signal.")

        sampling_rate = float(self.fs)
        length = len(self.ecg_signal)
        dur = length / sampling_rate

        # algorithm parameters
        v1s = int(1.0 * sampling_rate)
        v100ms = int(0.1 * sampling_rate)
        TH_elapsed = np.ceil(0.36 * sampling_rate)
        sm_size = int(0.08 * sampling_rate)
        init_ecg = 10 # seconds for initialization
        if dur < init_ecg:
            init_ecg = int(dur)

        # filtering
        filtered, _, _ = st.filter_signal(
            signal=self.ecg_signal,
            ftype="butter",
            band="lowpass",
            order=4,
            frequency=20.0,
            sampling_rate=sampling_rate,
        )
        filtered, _, _ = st.filter_signal(
            signal=filtered,
            ftype="butter",
            band="highpass",
            order=4,
            frequency=3.0,
            sampling_rate=sampling_rate,
        )

        # diff
        dx = np.abs(np.diff(filtered, 1) * sampling_rate)

        # smoothing
        dx, _ = st.smoother(signal=dx, kernel="hamming", size=sm_size, mirror=True)

        # buffers
        qrspeakbuffer = np.zeros(init_ecg)
        noisepeakbuffer = np.zeros(init_ecg)
        peak_idx_test = np.zeros(init_ecg)
        noise_idx = np.zeros(init_ecg)
        rrinterval = sampling_rate * np.ones(init_ecg)

        a, b = 0, v1s
        all_peaks, _ = st.find_extrema(signal=dx, mode="max")
        for i in range(init_ecg):
            peaks, values = st.find_extrema(signal=dx[a:b], mode="max")
            try:
                ind = np.argmax(values)
            except ValueError:
                pass
            else:
                # peak amplitude
                qrspeakbuffer[i] = values[ind]
                # peak location
                peak_idx_test[i] = peaks[ind] + a

            a += v1s
            b += v1s

        # thresholds
        ANP = np.median(noisepeakbuffer)
        AQRSP = np.median(qrspeakbuffer)
        TH = 0.475
        DT = ANP + TH * (AQRSP - ANP)
        DT_vec = []
        indexqrs = 0
        indexnoise = 0
        indexrr = 0
        npeaks = 0
        offset = 0

        beats = []

        # detection rules
        # 1 - ignore all peaks that precede or follow larger peaks by less than 200ms
        lim = int(np.ceil(0.15 * sampling_rate))
        diff_nr = int(np.ceil(0.045 * sampling_rate))
        bpsi, bpe = offset, 0

        for f in all_peaks:
            DT_vec += [DT]
            # 1 - Checking if f-peak is larger than any peak following or preceding it by less than 200 ms
            peak_cond = np.array(
                (all_peaks > f - lim) * (all_peaks < f + lim) * (all_peaks != f)
            )
            peaks_within = all_peaks[peak_cond]
            if peaks_within.any() and (max(dx[peaks_within]) > dx[f]):
                continue

            # 4 - If the peak is larger than the detection threshold call it a QRS complex, otherwise call it noise
            if dx[f] > DT:
                # 2 - look for both positive and negative slopes in raw signal
                if f < diff_nr:
                    diff_now = np.diff(self.ecg_signal[0 : f + diff_nr])
                elif f + diff_nr >= len(self.ecg_signal):
                    diff_now = np.diff(self.ecg_signal[f - diff_nr : len(dx)])
                else:
                    diff_now = np.diff(self.ecg_signal[f - diff_nr : f + diff_nr])
                diff_signer = diff_now[diff_now > 0]
                if len(diff_signer) == 0 or len(diff_signer) == len(diff_now):
                    continue
                # RR INTERVALS
                if npeaks > 0:
                    # 3 - in here we check point 3 of the Hamilton paper
                    # that is, we check whether our current peak is a valid R-peak.
                    prev_rpeak = beats[npeaks - 1]

                    elapsed = f - prev_rpeak
                    # if the previous peak was within 360 ms interval
                    if elapsed < TH_elapsed:
                        # check current and previous slopes
                        if prev_rpeak < diff_nr:
                            diff_prev = np.diff(self.ecg_signal[0 : prev_rpeak + diff_nr])
                        elif prev_rpeak + diff_nr >= len(self.ecg_signal):
                            diff_prev = np.diff(self.ecg_signal[prev_rpeak - diff_nr : len(dx)])
                        else:
                            diff_prev = np.diff(
                                self.ecg_signal[prev_rpeak - diff_nr : prev_rpeak + diff_nr]
                            )

                        slope_now = max(diff_now)
                        slope_prev = max(diff_prev)

                        if slope_now < 0.5 * slope_prev:
                            # if current slope is smaller than half the previous one, then it is a T-wave
                            continue
                    if dx[f] < 3.0 * np.median(qrspeakbuffer):  # avoid retarded noise peaks
                        beats += [int(f) + bpsi]
                    else:
                        continue

                    if bpe == 0:
                        rrinterval[indexrr] = beats[npeaks] - beats[npeaks - 1]
                        indexrr += 1
                        if indexrr == init_ecg:
                            indexrr = 0
                    else:
                        if beats[npeaks] > beats[bpe - 1] + v100ms:
                            rrinterval[indexrr] = beats[npeaks] - beats[npeaks - 1]
                            indexrr += 1
                            if indexrr == init_ecg:
                                indexrr = 0

                elif dx[f] < 3.0 * np.median(qrspeakbuffer):
                    beats += [int(f) + bpsi]
                else:
                    continue

                npeaks += 1
                qrspeakbuffer[indexqrs] = dx[f]
                peak_idx_test[indexqrs] = f
                indexqrs += 1
                if indexqrs == init_ecg:
                    indexqrs = 0
            if dx[f] <= DT:
                tf = f + bpsi
                # RR interval median
                RRM = np.median(rrinterval)  # initial values are good?

                if len(beats) >= 2:
                    elapsed = tf - beats[npeaks - 1]

                    if elapsed >= 1.5 * RRM and elapsed > TH_elapsed:
                        if dx[f] > 0.5 * DT:
                            beats += [int(f) + offset]
                            # RR INTERVALS
                            if npeaks > 0:
                                rrinterval[indexrr] = beats[npeaks] - beats[npeaks - 1]
                                indexrr += 1
                                if indexrr == init_ecg:
                                    indexrr = 0
                            npeaks += 1
                            qrspeakbuffer[indexqrs] = dx[f]
                            peak_idx_test[indexqrs] = f
                            indexqrs += 1
                            if indexqrs == init_ecg:
                                indexqrs = 0
                    else:
                        noisepeakbuffer[indexnoise] = dx[f]
                        noise_idx[indexnoise] = f
                        indexnoise += 1
                        if indexnoise == init_ecg:
                            indexnoise = 0
                else:
                    noisepeakbuffer[indexnoise] = dx[f]
                    noise_idx[indexnoise] = f
                    indexnoise += 1
                    if indexnoise == init_ecg:
                        indexnoise = 0

            # Update Detection Threshold
            ANP = np.median(noisepeakbuffer)
            AQRSP = np.median(qrspeakbuffer)
            DT = ANP + 0.475 * (AQRSP - ANP)

        beats = np.array(beats)

        r_beats = []
        thres_ch = 1
        adjacency = 0.01 * sampling_rate
        for i in beats:
            error = [False, False]
            if i - lim < 0:
                window = self.ecg_signal[0 : i + lim]
                add = 0
            elif i + lim >= length:
                window = self.ecg_signal[i - lim : length]
                add = i - lim
            else:
                window = self.ecg_signal[i - lim : i + lim]
                add = i - lim
            # meanval = np.mean(window)
            w_peaks, _ = st.find_extrema(signal=window, mode="max")
            w_negpeaks, _ = st.find_extrema(signal=window, mode="min")
            zerdiffs = np.where(np.diff(window) == 0)[0]
            w_peaks = np.concatenate((w_peaks, zerdiffs))
            w_negpeaks = np.concatenate((w_negpeaks, zerdiffs))

            pospeaks = sorted(zip(window[w_peaks], w_peaks), reverse=True)
            negpeaks = sorted(zip(window[w_negpeaks], w_negpeaks))
        
            try:
                twopeaks = [pospeaks[0]]
            except IndexError:
                twopeaks = []
            try:
                twonegpeaks = [negpeaks[0]]
            except IndexError:
                twonegpeaks = []

            # getting positive peaks
            for i in range(len(pospeaks) - 1):
                if abs(pospeaks[0][1] - pospeaks[i + 1][1]) > adjacency:
                    twopeaks.append(pospeaks[i + 1])
                    break
            try:
                posdiv = abs(twopeaks[0][0] - twopeaks[1][0])
            except IndexError:
                error[0] = True

            # getting negative peaks
            for i in range(len(negpeaks) - 1):
                if abs(negpeaks[0][1] - negpeaks[i + 1][1]) > adjacency:
                    twonegpeaks.append(negpeaks[i + 1])
                    break
            try:
                negdiv = abs(twonegpeaks[0][0] - twonegpeaks[1][0])
            except IndexError:
                error[1] = True

            # choosing type of R-peak
            n_errors = sum(error)
            try:
                if not n_errors:
                    if posdiv > thres_ch * negdiv:
                        # pos noerr
                        r_beats.append(twopeaks[0][1] + add)
                    else:
                        # neg noerr
                        r_beats.append(twonegpeaks[0][1] + add)
                elif n_errors == 2:
                    if abs(twopeaks[0][1]) > abs(twonegpeaks[0][1]):
                        # pos allerr
                        r_beats.append(twopeaks[0][1] + add)
                    else:
                        # neg allerr
                        r_beats.append(twonegpeaks[0][1] + add)
                elif error[0]:
                    # pos poserr
                    r_beats.append(twopeaks[0][1] + add)
                else:
                    # neg negerr
                    r_beats.append(twonegpeaks[0][1] + add)
            except IndexError:
                continue

        rpeaks = sorted(list(set(r_beats)))
        rpeaks = np.array(rpeaks, dtype="int")

        return utils.ReturnTuple((rpeaks,), ("rpeaks",))

    def hr_count(self):
        cal_sec = 5
        if cal_sec != 0:
            hr = round(self.r_index.shape[0]*60/cal_sec)
            return hr
        return 0

    def fir_lowpass_filter(self, data, cutoff, numtaps=21):
        """A finite impulse response (FIR) lowpass filter to a given data using a
        specified cutoff frequency and number of filter taps.

        Args:
            data (array): The input data to be filtered
            cutoff (float): The cutoff frequency of the lowpass filter, specified in the same units as the
        sampling frequency of the input data. It determines the frequency below which the filter allows
        signals to pass through and above which it attenuates them
            numtaps (int, optional): the number of coefficients (taps) in the FIR filter. Defaults to 21.

        Returns:
            array: The filtered signal 'y' after applying a lowpass filter with a specified cutoff frequency
        and number of filter taps to the input signal 'data'.
        """
        b = firwin(numtaps, cutoff)
        y = signal.convolve(data, b, mode="same")
        return y

    def find_j_index(self):
        """The index of the maximum value in a given range of a file and returns a list of
        those indices.

        Args:
            signal (array): ECG signal values
            s_index (list/array): _description_
            fs (int, optional): sampling rate of the ECG signal, defaults to 200 (optional)

        Returns:
            list: Indices (j) where the maximum value is found in a specific range of the input
        ecg_signal (signal) defined by the start indices (s_index).
        """
        j = []
        increment = int(self.fs*0.05)
        for z in range (0,len(self.s_index)):
            data = []
            j_index = self.ecg_signal[self.s_index[z]:self.s_index[z]+increment]
            for k in range (0,len(j_index)):
                data.append(j_index[k])
            max_d = max(data)
            max_id = data.index(max_d)
            j.append(self.s_index[z]+max_id)
        return j

    def find_s_index(self, d):
            d = int(d)+1
            s = []
            for i in self.r_index:
                if i == len(self.ecg_signal):
                    s.append(i)
                    continue
                elif i+d<=len(self.ecg_signal):
                    s_array = self.ecg_signal[i:i+d]
                else:
                    s_array = self.ecg_signal[i:]
                if self.ecg_signal[i] > 0:
                    s_index = i+np.where(s_array == min(s_array))[0][0]
                else:
                    s_index = i+np.where(s_array == max(s_array))[0][0]
                    if abs(s_index - i) < d/2:
                        s_index_ = i+np.where(s_array == min(s_array))[0][0]
                        if abs(s_index_ - i) > d/2:
                            s_index = s_index_
                s.append(s_index)
            return np.sort(s)

    def find_q_index(self, d):
        """The Q wave index in an ECG signal given the R wave index and a specified
        distance.

        Args:
            ecg (array): ECG signal values
            R_index (array/list): R peak indices in the ECG signal
            d (int): The maximum distance (in samples) between the R peak and the Q wave onset that we want to find.

        Returns:
            list: Q-wave indices for each R-wave index in the ECG signal.
        """
        d = int(d) + 1
        q = []
        for i in self.r_index:
            if i == 0:
                q.append(i)
                continue
            elif 0 <= i - d:
                q_array = self.ecg_signal[i - d:i]
            else:
                q_array = self.ecg_signal[:i]
            if self.ecg_signal[i] > 0:
                q_index = i - (len(q_array) - np.where(q_array == min(q_array))[0][0])
            else:
                q_index = i - (len(q_array) - np.where(q_array == max(q_array))[0][0])
            q.append(q_index)
        return np.sort(q)

    def find_new_q_index(self, d):
        q = []
        for i in self.r_index:
            q_ = []
            if i == 0:
                q.append(i)
                continue
            if self.ecg_signal[i] > 0:
                c = i
                while c > 0 and self.ecg_signal[c-1] < self.ecg_signal[c]:
                    c -= 1                  
                if self.ecg_signal[i] * 0.01 > self.ecg_signal[c] or self.ecg_signal[c] < 0 or c == 0:
                    if abs(i-c) <= d:
                        q.append(c)
                        continue
                    else:
                        q_.append(c)
                while c > 0:
                    while c > 0 and self.ecg_signal[c-1] > self.ecg_signal[c]:
                        c -= 1
                    # q_.append(c)
                    while c > 0 and self.ecg_signal[c-1] < self.ecg_signal[c]:
                        c -= 1
                    if q_ and q_[-1] == c:
                        break
                    q_.append(c)
                    if self.ecg_signal[i] * 0.01 > self.ecg_signal[c] or self.ecg_signal[c] < 0 or c == 0:
                        break
            else:
                c = i
                while c > 0 and self.ecg_signal[c-1] > self.ecg_signal[c]:
                    c -= 1
                if self.ecg_signal[i] * 0.01 < self.ecg_signal[c] or self.ecg_signal[c] > 0 or c == 0:
                    if abs(i-c) <= d:
                        q.append(c)
                        continue
                    else:
                        q_.append(c)
                while c > 0:
                    while c > 0 and self.ecg_signal[c-1] < self.ecg_signal[c]:
                        c -= 1
                    # q_.append(c)
                    while c > 0 and self.ecg_signal[c-1] > self.ecg_signal[c]:
                        c -= 1
                    if q_ and q_[-1] == c:
                        break
                    q_.append(c)
                    if self.ecg_signal[i] * 0.01 < self.ecg_signal[c] or self.ecg_signal[c] > 0 or c == 0:
                        break
            if q_:
                a = 0
                for _q in q_[::-1]:
                    if abs(i-_q) <= d:
                        a = 1
                        q.append(_q)
                        break
                if a == 0:
                    q.append(q_[0])
        return np.sort(q)

    def find_new_s_index(self, d):
        s = []
        end_index = len(self.ecg_signal)
        for i in self.r_index:
            s_ = []
            if i == len(self.ecg_signal):
                s.append(i)
                continue
            if self.ecg_signal[i] > 0:
                c = i
                while c+1 < end_index and self.ecg_signal[c+1] < self.ecg_signal[c]:
                    c += 1
                if self.ecg_signal[i] * 0.01 > self.ecg_signal[c] or self.ecg_signal[c] < 0 or c == end_index-1:
                    if abs(i-c) <= d:
                        s.append(c)
                        continue
                    else:
                        s_.append(c)
                while c+1 < end_index:
                    while c+1 < end_index and self.ecg_signal[c+1] > self.ecg_signal[c]:
                        c += 1
                    while c+1 < end_index and self.ecg_signal[c+1] < self.ecg_signal[c]:
                        c += 1
                    if s_ and s_[-1] == c:
                        break
                    s_.append(c)
                    if self.ecg_signal[i] * 0.01 > self.ecg_signal[c] or self.ecg_signal[c] < 0 or c == end_index-1:
                        break
            else:
                c = i
                while c+1 < end_index and self.ecg_signal[c+1] > self.ecg_signal[c]:
                    c += 1
                if self.ecg_signal[i] * 0.01 < self.ecg_signal[c] or self.ecg_signal[c] > 0 or c == end_index-1:
                    if abs(i-c) <= d:
                        s.append(c)
                        continue
                    else:
                        s_.append(c)
                while c < end_index:
                    while c+1 < end_index and self.ecg_signal[c+1] > self.ecg_signal[c]:
                        c += 1
                    while c+1 < end_index and self.ecg_signal[c+1] < self.ecg_signal[c]:
                        c += 1
                    if s_ and s_[-1] == c:
                        break
                    s_.append(c)
                    if self.ecg_signal[i] * 0.01 < self.ecg_signal[c] or self.ecg_signal[c] > 0 or c == end_index-1:
                        break
            if s_:
                a = 0
                for _s in s_[::-1]:
                    if abs(i-_s) <= d:
                        a = 1
                        s.append(_s)
                        break
                if a == 0:
                    s.append(s_[0])
        return np.sort(s)

    def find_r_peaks(self):
        """Finds R-peaks in an ECG signal using the Hamilton segmenter algorithm.
    
        Args:
            ecg_signal (array): The ECG signal of numpy array
            fs (int, optional): sampling rate of the ECG signal, defaults to 200 (optional)
    
        Returns:
            list: the R-peak indices of the ECG signal using the Hamilton QRS complex detector algorithm.
        """
        r_ = []
        out = self.hamilton_segmenter()
        self.r_index = out["rpeaks"]
        heart_rate = self.hr_count()
        if self.JR: #---------------------
            diff_indexs = abs(round((self.fs * 0.4492537) + (heart_rate * -1.05518351) + 40.40601032654332))
        else: #---------------------
            diff_indexs = abs(round((self.fs * 0.4492537) + (heart_rate * -1.009) + 58.40601032654332))
    
        for r in self.r_index:
            if r - diff_indexs >= 0 and len(self.ecg_signal) >= r+diff_indexs:
                data = self.ecg_signal[r-diff_indexs:r+diff_indexs]
                abs_data = np.abs(data)
                r_.append(np.where(abs_data == max(abs_data))[0][0] + r-diff_indexs)
            else:
                r_.append(r)
            
        new_r = np.unique(r_) if r_ else self.r_index
        fs_diff = int((25*self.fs)/200)
        final_r = []
        if new_r.any(): final_r = [new_r[0]] + [new_r[j+1] for j, i in enumerate(np.diff(new_r)) if i >= fs_diff]
        return np.array(final_r)

    def pt_detection_1(self):
        """Detects peaks in a given signal within a specified range and returns the peak indices.

        Args:
            ecg_signal (array): ECG signal
            r_index (list/array): Indices representing the R-peaks in an ECG signal
            q_index (_type_): Indices representing the Q waves in an ECG signal
            s_index (_type_): Indices representing the S waves in an ECG signal
            width (_type_): In the find_peaks function to specify the minimum width of
        peaks to be detected. It is a positive integer value

        Returns:
            tuple: two lists: pt and p_t.
        """
        max_signal = max(self.ecg_signal)/100
        pt = []
        p_t = []
        for i in range (0, len(self.r_index)-1):
            aoi = self.ecg_signal[self.s_index[i]:self.q_index[i+1]]
            max_signal = max(self.ecg_signal)/100
            low = self.fir_lowpass_filter(aoi,self.lp_thres,30)
            if self.ecg_signal[self.r_index[i]]<0:
                max_signal=0.05
            else:
                max_signal=max_signal
            if aoi.any():
                peaks,_ = find_peaks(low,height=max_signal,width=self.width)
                peaks1=peaks+(self.s_index[i])
            else:
                peaks1 = [0]
            p_t.append(list(peaks1))
            pt.extend(list(peaks1))
            for i in range (len(p_t)):
                if not p_t[i]:
                    p_t[i] = [0]
        return pt, p_t

    def pt_detection_2(self):
        """Detects peaks in a given signal within a specified range and returns the peak indices.

        Args:
            ecg_signal (array): ECG signal
            r_index (list/array): Indices representing the R-peaks in an ECG signal
            q_index (_type_): Indices representing the Q waves in an ECG signal
            s_index (_type_): Indices representing the S waves in an ECG signal
            width (_type_): In the find_peaks function to specify the minimum width of
        peaks to be detected. It is a positive integer value

        Returns:
            tuple: two lists: pt and p_t.
        """
        pt = []
        p_t = []
        for i in range (0, len(self.r_index)-1):
            aoi = self.ecg_signal[self.s_index[i]:self.q_index[i+1]]
            if aoi.any():
                low = self.fir_lowpass_filter(aoi,self.lp_thres,30)
                if self.ecg_signal[self.r_index[i]]<0:
                    max_signal=0.05
                else:
                    max_signal= max(low)*0.2
                if aoi.any():
                    peaks,_ = find_peaks(low,height=max_signal,width=self.width)
                    peaks1=peaks+(self.s_index[i])
                else:
                    peaks1 = [0]
                p_t.append(list(peaks1))
                pt.extend(list(peaks1))
                for i in range (len(p_t)):
                    if not p_t[i]:
                        p_t[i] = [0]
            else:
                p_t.append([0])
        return pt, p_t

    def pt_detection_3(self):
        """Detects peaks in a given signal within a specified range and returns the peak indices.

        Args:
            ecg_signal (array): ECG signal
            r_index (list/array): Indices representing the R-peaks in an ECG signal
            q_index (_type_): Indices representing the Q waves in an ECG signal
            s_index (_type_): Indices representing the S waves in an ECG signal
            width (_type_): In the find_peaks function to specify the minimum width of
        peaks to be detected. It is a positive integer value

        Returns:
            tuple: two lists: pt and p_t.
        """
        pt = []
        p_t = []
        for i in range (0, len(self.r_index)-1):
            aoi = self.ecg_signal[self.s_index[i]:self.q_index[i+1]]
            low = self.fir_lowpass_filter(aoi,self.lp_thres,30)
            if aoi.any():
                peaks,_ = find_peaks(low,prominence=0.05,width=self.width)
                peaks1=peaks+(self.s_index[i])
            else:
                peaks1 = [0]
            p_t.append(list(peaks1))
            pt.extend(list(peaks1))
            for i in range (len(p_t)):
                if not p_t[i]:
                    p_t[i] = [0]

        return pt, p_t

    def pt_detection_4(self):
        """Detects peaks in a given signal within a specified range and returns the peak indices.

        Args:
            b_signal (array): ECG signal
            r_index (list/array): Indices representing the R-peaks in an ECG signal
            q_index (_type_): Indices representing the Q waves in an ECG signal
            s_index (_type_): Indices representing the S waves in an ECG signal
            width (_type_): In the find_peaks function to specify the minimum width of
        peaks to be detected. It is a positive integer value

        Returns:
            tuple: two lists: pt and p_t.
        """
        def all_peaks_7(arr):
            """The indices of all peaks in the array, where a peak is
            defined as a point that is higher than its neighboring points.

            Args:
                arr (array): An input array of numbers

            Returns:
                array: The function `all_peaks_7` returns a sorted numpy array of indices where peaks occur in
            the input array `arr`.
            """
            sign_arr = np.sign(np.diff(arr))
            pos = np.where(np.diff(sign_arr) == -2)[0] + 1
            neg = np.where(np.diff(sign_arr) == 2)[0] + 1
            all_peaks = np.sort(np.concatenate((pos, neg)))
            al = all_peaks.tolist()
            diff = {}
            P, Pa, Pb = [], [], []
            if len(al) > 2:
                for p in pos:
                    index = al.index(p)
                    if index == 0:
                        m, n, o = arr[0], arr[al[index]], arr[al[index+1]]
                    elif index == len(al)-1:
                        m, n, o = arr[al[index-1]], arr[al[index]], arr[-1]
                    else:
                        m, n, o = arr[al[index-1]], arr[al[index]], arr[al[index+1]]
                    diff[p] = [abs(n-m), abs(n-o)]
                th = np.mean([np.mean([v, m]) for v, m in diff.values()])*.66
                for p, (a, b) in diff.items():
                    if a >= th and b >= th:
                        P.append(p)
                        continue
                    if a >= th and not Pa:
                        Pa.append(p)
                    elif a >= th and arr[p] > arr[Pa[-1]] and np.where(pos==Pa[-1])[0]+1 == np.where(pos==p)[0]:
                        Pa[-1] = p
                    elif a >= th:
                        Pa.append(p)
                    if b >= th and not Pb:
                        Pb.append(p)
                    elif b >= th and arr[p] < arr[Pb[-1]] and np.where(pos==Pb[-1])[0]+1 == np.where(pos==p)[0]:
                        Pb[-1] = p
                    elif b >= th:
                        Pb.append(p)
                if len(pos)>1:
                    for i in range(1, len(pos)):
                        m, n = pos[i-1], pos[i]
                        if m in Pa and n in Pb:
                            P.append(m) if arr[m] > arr[n] else P.append(n)
                # if Pa and Pa[-1] == pos[-1]:
                #     P.append(Pa[-1])
                # if Pb and Pb[0] == pos[0]:
                #     P.append(Pb[0])
            else:
                P = pos
            return np.sort(P)
        pt, p_t = [], []
        for i in range(1, len(self.r_index)):
            q0, r0, s0 = self.q_index[i - 1], self.r_index[i - 1], self.s_index[i - 1]
            q1, r1, s1 = self.q_index[i], self.r_index[i], self.s_index[i]
            arr = self.ecg_signal[s0+7:q1-7]
            peaks = list(all_peaks_7(arr) + s0 + 7) 
            if peaks:
                pt.extend(peaks)
                p_t.append(peaks)
            else:
                p_t.append([0])
        return pt, p_t

    def find_pt(self):
        _, p_t1 = self.pt_detection_1()
        _, p_t2 = self.pt_detection_2()
        _, p_t3 = self.pt_detection_3()
        _, p_t4 = self.pt_detection_4() 
        pt = []
        p_t = []
        for i in range(len(p_t1)):
            _ = []
            for _pt in set(p_t1[i]+p_t2[i]+p_t3[i]+p_t4[i]):
                count = 0
                if any(val in p_t1[i] for val in range (_pt-2,_pt+3)):
                    count += 1
                if any(val in p_t2[i] for val in range (_pt-2,_pt+3)):
                    count += 1
                if any(val in p_t3[i] for val in range (_pt-2,_pt+3)):
                    count += 1
                if any(val in p_t4[i] for val in range (_pt-2,_pt+3)):
                    count += 1
                if count >= 3:
                    _.append(_pt)
                _.sort()
            if _:
                p_t.append(_)
            else:
                p_t.append([0])
        result = []
        for sublist in p_t:
            temp = [sublist[0]]
            for i in range(1, len(sublist)):
                if abs(sublist[i] - sublist[i-1]) > 5:
                    temp.append(sublist[i])
                else:
                    temp[-1] = sublist[i]  
            if temp:
                result.append(temp)
                pt.extend(temp)
            else:
                result.append([0])
        p_t = result
        return p_t, pt

    def segricate_p_t_pr_inerval(self):
        """
        threshold = 0.37 for JR and 0.5 for other diseases
        """
        diff_arr = ((np.diff(self.r_index)*self.thres)/self.fs).tolist()
        t_peaks_list, p_peaks_list, pr_interval, extra_peaks_list = [], [], [], []
        # threshold = (-0.0012 * len(r_index)) + 0.25
        for i in range(len(self.p_t)):
            p_dis = (self.r_index[i+1]-self.p_t[i][-1])/self.fs
            t_dis = (self.r_index[i+1]-self.p_t[i][0])/self.fs
            threshold = diff_arr[i]
            if t_dis > threshold and (self.p_t[i][0]>self.r_index[i]): 
                t_peaks_list.append(self.p_t[i][0])
            else:
                t_peaks_list.append(0)
            if p_dis <= threshold: 
                p_peaks_list.append(self.p_t[i][-1])
                pr_interval.append(p_dis*self.fs)
            else:
                p_peaks_list.append(0)
            if len(self.p_t[i])>0:
                if self.p_t[i][0] in t_peaks_list:
                    if self.p_t[i][-1] in p_peaks_list:
                        extra_peaks_list.extend(self.p_t[i][1:-1])
                    else:
                        extra_peaks_list.extend(self.p_t[i][1:])
                elif self.p_t[i][-1] in p_peaks_list:
                    extra_peaks_list.extend(self.p_t[i][:-1])
                else:
                    extra_peaks_list.extend(self.p_t[i])

        p_label, pr_label = "", ""
        if self.thres >= 0.5 and p_peaks_list and len(p_peaks_list)>2:
            pp_intervals = np.diff(p_peaks_list)
            pp_std = np.std(pp_intervals)
            pp_mean = np.mean(pp_intervals)
            threshold = 0.12 * pp_mean
            if pp_std <= threshold:
                p_label = "Constanat"
            else:
                p_label = "Not Constant"
            
            count=0
            for i in pr_interval:
                if round(np.mean(pr_interval)*0.75) <= i <= round(np.mean(pr_interval)*1.25):
                    count +=1
            if len(pr_interval) != 0: 
                per = count/len(pr_interval)
                pr_label = 'Not Constant' if per<=0.7 else 'Constant'
        data = {'T_Index':t_peaks_list, 
                'P_Index':p_peaks_list, 
                'PR_Interval':pr_interval, 
                'P_Label':p_label, 
                'PR_label':pr_label,
                'Extra_Peaks':extra_peaks_list}
        return data

    def find_inverted_t_peak(self):
        t_index = []
        for i in range(0, len(self.s_index)-1):
            t = self.ecg_signal[self.s_index[i]: self.q_index[i+1]]
            if t.any():
                check, _ = find_peaks(-t,  height=(0.21, 1), distance=70)
                peaks = check + self.s_index[i]
            else:
                peaks = np.array([])
            if peaks.any():
                t_index.extend(list(peaks))
        # t_label = 
        return t_index

    def get_data(self):
        
        self.r_index = self.find_r_peaks()
        rr_intervals = np.diff(self.r_index)
        rr_std = np.std(rr_intervals)
        rr_mean = np.mean(rr_intervals)
        threshold = self.rr_thres * rr_mean
        if rr_std <= threshold:
            self.r_label = "Regular"
        else:
            self.r_label = "Irregular"
        # if self.rr_thres == 0.15:
        #     self.ecg_signal = lowpass(self.ecg_signal,0.2)
        self.hr_ = self.hr_count()
        sd, qd = int(self.fs * 0.115), int(self.fs * 0.08)
        self.s_index = self.find_s_index(sd)
        # q_index = find_q_index(ecg_signal, r_index, qd)
        # s_index = find_new_s_index(ecg_signal,r_index,sd)
        self.q_index = self.find_new_q_index(qd)
        self.j_index = self.find_j_index()
        self.p_t, self.pt = self.find_pt()
        self.data_ = self.segricate_p_t_pr_inerval()
        self.inv_t_index = self.find_inverted_t_peak()
        data = {'R_Label':self.r_label, 
                'R_index':self.r_index, 
                'Q_Index':self.q_index, 
                'S_Index':self.s_index, 
                'J_Index':self.j_index, 
                'P_T List':self.p_t, 
                'PT PLot':self.pt, 
                'HR_Count':self.hr_, 
                'T_Index':self.data_['T_Index'], 
                'P_Index':self.data_['P_Index'],
                'Ex_Index':self.data_['Extra_Peaks'], 
                'PR_Interval':self.data_['PR_Interval'], 
                'P_Label':self.data_['P_Label'], 
                'PR_label':self.data_['PR_label'],
                'inv_t_index': self.inv_t_index}
        return data

# Low pass and baseline signal
class filter_signal:
    
    def __init__(self, ecg_signal, fs = 200):
        self.ecg_signal = ecg_signal
        self.fs = fs

    def baseline_construction_200(self, kernel_size=131):
        """Removes the baseline from an ECG signal using a median filter
        of a specified kernel size.

        Args:
            ecg_signal (array): The ECG signal
            kernel_size (int, optional): The kernel_size parameter is the size of the median filter 
        kernel used for baseline correction. Defaults to 101 (optional).

        Returns:
            array: The baseline-corrected ECG signal.
        """
        s_corrected = signal.detrend(self.ecg_signal)
        baseline_corrected = s_corrected - signal.medfilt(s_corrected, kernel_size)
        return baseline_corrected

    def baseline_als(self, file, lam, p, niter=10):
        L = len(file)
        D = sparse.csc_matrix(np.diff(np.eye(L), 2))
        w = np.ones(L)
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w*file)
            w = p * (file > z) + (1-p) * (file < z)
        return z

    def baseline_construction_250(self, kernel_size=131):
        als_baseline = self.baseline_als(self.ecg_signal, 16**5, 0.01) 
        s_als = self.ecg_signal - als_baseline
        s_corrected = signal.detrend(s_als)
        corrected_baseline = s_corrected - medfilt(s_corrected, kernel_size)
        return corrected_baseline

    def lowpass(self, cutoff=0.3):
        """A lowpass filter to a given file using the Butterworth filter.

        Args:
            signal (array): ECG Signal
            cutoff (float): 0.3 for PVC & 0.2 AFIB
        
        Returns:
            array: the low-pass filtered signal of the input file.
        """
        b, a = signal.butter(3, cutoff, btype='lowpass', analog=False)
        low_passed = signal.filtfilt(b, a, self.baseline_signal)
        return low_passed
    
    def get_data(self):

        if self.fs != 200:
            self.ecg_signal = MinMaxScaler(feature_range=(0,4)).fit_transform(self.ecg_signal.reshape(-1,1)).squeeze()
                    
        if self.fs == 200:
            self.baseline_signal = self.baseline_construction_200(kernel_size = 101)
            lowpass_signal = self.lowpass(cutoff = 0.3)
        elif self.fs == 250:
            self.baseline_signal = self.baseline_construction_250(kernel_size = 131)
            lowpass_signal = self.lowpass(cutoff = 0.25)
        elif self.fs == 360:
            self.baseline_signal = self.baseline_construction_200(kernel_size = 151)
            lowpass_signal = self.lowpass(cutoff = 0.2)
        elif self.fs == 1000:
            self.baseline_signal = self.baseline_construction_200(kernel_size = 399)
            lowpass_signal = self.lowpass(cutoff = 0.05)
        elif self.fs == 128:
            self.baseline_signal = self.baseline_construction_200(kernel_size = 101)
            lowpass_signal = self.lowpass(cutoff = 0.5)
            
        return self.baseline_signal, lowpass_signal

# PVC detection
class PVC_detection:
    def __init__(self, ecg_signal,fs=200):
        self.ecg_signal = ecg_signal
        self.fs = fs

    def lowpass(self, file):
        b, a = signal.butter(3, 0.3, btype='lowpass', analog=False)
        low_passed = signal.filtfilt(b, a, file)
        return low_passed
        
    def baseline_construction_200(self, kernel_Size=101):
        s_corrected = signal.detrend(self.ecg_signal)
        baseline_corrected = s_corrected - signal.medfilt(s_corrected, kernel_Size)
        return baseline_corrected

    def detect_beats(self,
		ecg,	# The raw ECG signal
		rate,	# Sampling rate in HZ
		# Window size in seconds to use for 
		ransac_window_size=3.0, #5.0
		# Low frequency of the band pass filter
		lowfreq=5.0,
		# High frequency of the band pass filter
		highfreq=7.0, #10.0
        ):
        ransac_window_size = int(ransac_window_size*rate)

        lowpass = scipy.signal.butter(1, highfreq/(rate/2.0), 'low')
        highpass = scipy.signal.butter(1, lowfreq/(rate/2.0), 'high')
        # TODO: Could use an actual bandpass filter
        ecg_low = scipy.signal.filtfilt(*lowpass, x=ecg)
        ecg_band = scipy.signal.filtfilt(*highpass, x=ecg_low)
        
        # Square (=signal power) of the first difference of the signal
        decg = np.diff(ecg_band)
        decg_power = decg**2
        
        # Robust threshold and normalizator estimation
        thresholds = []
        max_powers = []
        for i in range(int(len(decg_power)/ransac_window_size)):
            sample = slice(i*ransac_window_size, (i+1)*ransac_window_size)
            d = decg_power[sample]
            thresholds.append(0.5*np.std(d))
            max_powers.append(np.max(d))

        threshold = np.median(thresholds)
        max_power = np.median(max_powers)
        decg_power[decg_power < threshold] = 0

        decg_power /= max_power
        decg_power[decg_power > 1.0] = 1.0
        square_decg_power = decg_power**4
        #square_decg_power = decg_power**4

        shannon_energy = -square_decg_power*np.log(square_decg_power)
        shannon_energy[~np.isfinite(shannon_energy)] = 0.0

        mean_window_len = int(rate*0.125+1)
        lp_energy = np.convolve(shannon_energy, [1.0/mean_window_len]*mean_window_len, mode='same')
        #lp_energy = scipy.signal.filtfilt(*lowpass2, x=shannon_energy)
        
        lp_energy = scipy.ndimage.gaussian_filter1d(lp_energy, rate/16.0) #20.0
        #lp_energy = scipy.ndimage.gaussian_filter1d(lp_energy, rate/8.0)
        lp_energy_diff = np.diff(lp_energy)

        zero_crossings = (lp_energy_diff[:-1] > 0) & (lp_energy_diff[1:] < 0)
        zero_crossings = np.flatnonzero(zero_crossings)
        zero_crossings -= 1

        return zero_crossings
 
    def calculate_surface_area(self, ecg_signal, qrs_start_index, qrs_end_index, sampling_rate):
        if qrs_start_index==0 or qrs_end_index==0:
            surface_area=0
        else:
            qrs_complex = ecg_signal[qrs_start_index:qrs_end_index]
            absolute_qrs = np.abs(qrs_complex)
            time = np.arange(len(qrs_complex)) / sampling_rate
            surface_area = trapz(absolute_qrs, time)

        return surface_area
    
    def wide_qrs_find(self):
        wideQRS = []
        difference = []
        surface_area_list = []
        pvc = []
        above_r_peaks = []
        below_r_peaks = []
        for idx in self.r_index:
            if idx < len(self.low_pass_signal):
                if self.low_pass_signal[idx] >= 0:
                    above_r_peaks.append(idx)
                else:
                    below_r_peaks.append(idx)
        if self.hr_count <= 88:
            thresold = round(self.fs * 0.08) # 0.10
        else:
            thresold = round(self.fs * 0.12)
        for k in range(len(self.r_index)):
            diff = self.s_index[k] - self.q_index[k]
            if self.r_index[k] in above_r_peaks:
                surface_thres = 0.14
                wideqs_thres = 0.13
            elif self.r_index[k] in below_r_peaks:
                surface_thres = 0.05
                wideqs_thres = 0.10
            if diff > thresold:
                difference.append(diff)
                wideQRS.append(self.r_index[k])
                surface_area = self.calculate_surface_area(self.low_pass_signal, self.q_index[k], self.s_index[k], self.fs)  
                if (diff/200) >= wideqs_thres:
                    surface_area_list.append(round(surface_area, 3))
                    if surface_area >= surface_thres :
                        pvc.append(self.r_index[k])

        if len(difference) != 0:
            q_s_difference = [i/200 for i in difference]
        else:
            q_s_difference = np.array([])
        return np.array(wideQRS), q_s_difference, pvc
    
    def PVC_CLASSIFICATION(self, PVC_R_Peaks):
        vt_counter = 0
        couplet_counter = 0
        triplet_counter = 0
        bigeminy_counter = 0
        trigeminy_counter = 0
        quadrigeminy_counter = 0
        vt = 0
        i = 0
        while i < len(PVC_R_Peaks):
            count = 0
            ones_count = 0
            while i < len(PVC_R_Peaks) and PVC_R_Peaks[i] == 1:
                count += 1
                ones_count += 1
                i += 1
            
            if count >= 4:
                vt_counter += 1
                vt += ones_count
                count = 0
                ones_count = 0
            if count == 3:
                triplet_counter += 1
            elif count == 2:
                couplet_counter += 1

            i += 1
        j = 0
        while j < len(PVC_R_Peaks) - 1:
            if PVC_R_Peaks[j] == 1:
                k = j + 1
                spaces = 0
                while k < len(PVC_R_Peaks) and PVC_R_Peaks[k] == 0:
                    spaces += 1
                    k += 1

                if k < len(PVC_R_Peaks) and PVC_R_Peaks[k] == 1:
                    if spaces == 1:
                        bigeminy_counter += 1
                    elif spaces == 2:
                        trigeminy_counter += 1
                    elif spaces == 3:
                        quadrigeminy_counter += 1
                j = k

            else:
                j += 1
        
        total_one = (1*vt) + (couplet_counter*2)+ (triplet_counter*3)+ (bigeminy_counter*2)+ (trigeminy_counter*2)+ (quadrigeminy_counter*2)
        total = vt_counter + couplet_counter+ triplet_counter+ bigeminy_counter+ trigeminy_counter+ quadrigeminy_counter
        ones = PVC_R_Peaks.count(1)
        if total == 0:
            Isolated = ones
        else:
            Common = total-1
            Isolated = ones - (total_one - Common)
        if vt_counter>1:
            vt_counter=1
        return vt_counter, couplet_counter, triplet_counter, bigeminy_counter, trigeminy_counter, quadrigeminy_counter, Isolated, vt
    
    def VT_confirmation(self, ecg_signal, r_index):
        VTC = []
        pqrst_data = pqrst_detection(ecg_signal)
        for i in range(0, len(r_index)-1):
            aoi = ecg_signal[r_index[i]-5:r_index[i + 1]]
            low = pqrst_data.fir_lowpass_filter(aoi, 0.2, 30)
            if aoi.any():
                peaks, _ = find_peaks(low,prominence=0.2,width=(40))
                VTC.append(peaks)
        if round(len(VTC) / len(r_index))>=.7:
            label = 'VT'
        else:
            label = 'Abnormal'

        return label, len(VTC)
    
    # def prediction_model(self, load_model, image_path, target_shape=[224,224], class_name=True):
    #     # classes = ['Noise', 'Normal', 'PVC'] 
    #     classes = ['LBBB', 'NOISE','NORMAL','PVC','RBBB'] # pvc with lbbb, rbbb modal test
    #     image = tf.io.read_file(image_path)
    #     input_arr = tf.image.decode_image(image, channels=3)
    #     input_arr = tf.image.resize(input_arr, size = target_shape)
    #     input_arr = tf.expand_dims(input_arr, axis=0)
    #     if class_name:
    #         idx = np.argmax(load_model(input_arr)[0])
    #     return load_model(input_arr)[0],classes[idx]

    def prediction_model(self, image_path, target_shape=[224, 224], class_name=True):
        with results_lock:
           
            classes = ['LBBB', 'Noise', 'Normal', 'PVC', 'RBBB']
            image = tf.io.read_file(image_path)
            input_arr = tf.image.decode_jpeg(image, channels=3)
            input_arr = tf.image.resize(input_arr, size=target_shape, method=tf.image.ResizeMethod.BILINEAR)
            input_arr = tf.expand_dims(input_arr, axis=0)

            # Set the input tensor
            interpreter.set_tensor(input_details[0]['index'], input_arr)
            
            # Perform inference
            interpreter.invoke()
            # Get the output tensor
            output_data = interpreter.get_tensor(output_details[0]['index'])

        if class_name:
            idx = np.argmax(output_data[0])
            return output_data[0], classes[idx]
        else:
            return output_data[0]

    def model_r_detectron(self, e_signal, r_index, heart_rate, fs=200):
        pvc_0 = []
        lbbb, rbbb = [], []
        model_pred = []
        counter = 0
        detect_rpeaks = self.detect_beats(e_signal, self.fs)
        for r in detect_rpeaks:  
            # if r == detect_rpeaks[0]:
            #     aa = pd.DataFrame(e_signal[int(r)-16:int(r)+90]) #15
            # elif r == detect_rpeaks[-1]:
            #     aa = pd.DataFrame(e_signal[int(r)-50:int(r)+120]) #100
            # else:
            #     aa = pd.DataFrame(e_signal[int(r)-50:int(r)+80])

            if r == detect_rpeaks[0]:
                aa = pd.DataFrame(e_signal[int(r)-10:int(r)+100]) #15
            elif r == detect_rpeaks[-1]:
                aa = pd.DataFrame(e_signal[int(r)-50:int(r)+130]) #100
            else:
                if int(r) - 50>0:
                    st_win = int(r)-50
                else:
                    st_win = 0
                aa = pd.DataFrame(e_signal[st_win:int(r)+80])
            # if r == detect_rpeaks[0]:
            #     try:
            #         st_window = int(r) - 16
            #         st_window = max(int(r)-10,0)

            #     except:
            #         st_window = int(r) -20
            #     aa = pd.DataFrame(e_signal[st_window:int(r)+90]) #15
            # elif r == detect_rpeaks[1]:
            #     aa = pd.DataFrame(e_signal[int(r)-50:int(r)+120]) #100
            # else:
            #     aa = pd.DataFrame(e_signal[int(r)-50:int(r)+80])

            plt.plot(aa, "k")
            plt.axis("off")
            plt.savefig("r.jpg")
            # plt.savefig(f"{var}dt_{r}.jpg")
            aq = cv2.imread("r.jpg")
            aq = cv2.resize(aq,(360,720)) #290
            cv2.imwrite("r.jpg", aq)
            plt.close()

            predictions, model_label = self.prediction_model("r.jpg")
            if str(model_label) == 'PVC' and float(predictions[3]) > 0.78: # 0.75
                pvc_0.append(r)
            if str(model_label) == 'LBBB' and float(predictions[3]) > 0.78:
                lbbb.append(r)
            if str(model_label) == 'RBBB' and float(predictions[3]) > 0.78:
                rbbb.append(r)
            model_pred.append((r, (float(predictions[2]), model_label)))
            counter += 1
        return pvc_0,lbbb, rbbb,  model_pred, detect_rpeaks
    
    
    def get_pvc_data(self):
        self.baseline_signal = self.baseline_construction_200(kernel_Size = 101)
        self.low_pass_signal = self.lowpass(self.baseline_signal)

        pqrst_data = pqrst_detection(self.baseline_signal, fs=self.fs).get_data()
        self.r_index = pqrst_data['R_index']
        self.q_index = pqrst_data['Q_Index']
        self.s_index = pqrst_data['S_Index']
        self.hr_count = pqrst_data['HR_Count']
        self.p_t = pqrst_data['P_T List']
        self.ex_index = pqrst_data['Ex_Index'] 
        wide_qrs, q_s_difference, surface_index = self.wide_qrs_find()              
         
        # wide_qrs = np.array([])
        model_pred = model_pvc  = []
        lbbb_index , rbbb_index = [], []

        pvc_onehot = np.zeros(len(self.r_index)).tolist() # r_index
        
        if len(wide_qrs)>0:
            if self.fs == 200:
                model_pvc, lbbb_index, rbbb_index, model_pred, detect_rpeaks = self.model_r_detectron(self.low_pass_signal, wide_qrs, self.hr_count, fs=self.fs)
            else:
               
                model_pvc, lbbb_index, rbbb_index, model_pred, detect_rpeaks = self.model_r_detectron(self.ecg_signal, wide_qrs, self.hr_count, fs=self.fs)
            label = "PVC" if len(model_pvc)>0 else "Abnormal"
            pvc_onehot = [1 if r in model_pvc else 0 for r in detect_rpeaks]    
            if len(lbbb_index) > 0 or len(rbbb_index) > 0:
                if len(lbbb_index) / len(detect_rpeaks) > 0.3:
                    lbbb_rbbb_label = "LBBB"
                if len(rbbb_index) / len(detect_rpeaks) > 0.3:
                    lbbb_rbbb_label = "RBBB"
            else: 
                lbbb_rbbb_label = "Abnormal"
        else:
            label ="Abnormal"
            lbbb_rbbb_label = "Abnormal"
        
        pvc_count = pvc_onehot.count(1)
        vt_counter, couplet_counter, triplet_counter, bigeminy_counter, trigeminy_counter, quadrigeminy_counter, remaining_ones, v_bit_vt = self.PVC_CLASSIFICATION(pvc_onehot)
        conf_vt_count = 0
        if vt_counter>0:
            confirmation = self.VT_confirmation(self.low_pass_signal, detect_rpeaks)
            
            if self.hr_count > 100 and v_bit_vt > 12:
                conf_vt_count = 1
            if confirmation == "Abnormal":
                vt_counter = 0
            else:
                pass
        data = {'PVC-Label':label,
                'PVC-Count':pvc_count,
                'PVC-Index':model_pvc,
                'VT_counter': conf_vt_count, 
                'PVC-Couplet_counter':couplet_counter, 
                'PVC-Triplet_counter':triplet_counter, 
                'PVC-Bigeminy_counter':bigeminy_counter, 
                'PVC-Trigeminy_counter':trigeminy_counter, 
                'PVC-Quadrigeminy_counter':quadrigeminy_counter, 
                'PVC-Isolated_counter':remaining_ones,
                'PVC-wide_qrs':wide_qrs,
                'PVC-QRS_difference':q_s_difference,
                'PVC-model_pred':model_pred,
                "IVR_counter":0,
                "NSVT_counter":0,
                "lbbb_rbbb_label": lbbb_rbbb_label,
                "lbbb_index" : lbbb_index,
                "rbbb_index": rbbb_index
            }
        if vt_counter > 0:
            if 60 <= self.hr_count < 100:
                data['VT_counter'] = 0
                data["NSVT_counter"] = vt_counter
            elif self.hr_count < 60 and v_bit_vt > 3:
                data['VT_counter'] = 0
                data["IVR_counter"] = vt_counter
        # print(data)
        return data

# Bock detection
class BlockDetected:
    def __init__(self, ecg_signal, fs):
        self.ecg_signal = ecg_signal
        self.fs = fs
        self.block_processing()

    def block_processing(self):
        self.baseline_signal, self.lowpass_signal = filter_signal(self.ecg_signal, self.fs).get_data()
        pqrst_data = pqrst_detection(self.baseline_signal, fs=self.fs).get_data()
        self.r_index = pqrst_data["R_index"]
        self.q_index = pqrst_data["Q_Index"]
        self.s_index = pqrst_data["S_Index"]
        self.p_index = pqrst_data["P_Index"]
        self.hr_counts = pqrst_data["HR_Count"]
        self.p_t = pqrst_data["P_T List"]
        self.pr = pqrst_data["PR_Interval"]

    def third_degree_block_deetection(self):
        label= 'Abnormal'
        third_degree = []
        possible_mob_3rd = False
        if self.hr_counts <= 100 and len(self.p_t) != 0: # 60 70
            constant_2 = all(map(lambda innerlist: len(innerlist) == 2, self.p_t))
            cons_2_1 = all(len(inner_list) in {1, 2} for inner_list in self.p_t)
            ampli_val = list(map(lambda inner_list: sum(self.baseline_signal[i] > 0.05 for i in inner_list) / len(inner_list),self.p_t))
            count_above_threshold = sum(1 for value in ampli_val if value > 0.7)
            percentage_above_threshold = count_above_threshold / len(ampli_val)
            count = 0
            if percentage_above_threshold >= 0.7:
                inc_dec_count = 0
                for i in range(0, len(self.pr)):
                    if self.pr[i] > self.pr[i -1]:
                        inc_dec_count += 1
                if len(self.pr) != 0:
                    if round(inc_dec_count / (len(self.pr)), 2) >= 0.50: # if posibale to change more then 0.5
                        possible_mob_3rd = True
                # if cons_2_1 == False:
                #     for i in range(0, len(self.pr)):
                #         if self.pr[i] > self.pr[i-1]:
                #             count += 1
                #     if round(count/len(self.pr), 2) >= 0.5:
                #         possible_3rd = True
                for inner_list in self.p_t:
                    if len(inner_list) in [3, 4] :
                        ampli_val = [self.baseline_signal[i] for i in inner_list] 
                        if ampli_val  and (sum(value > 0.05 for value in ampli_val) / len(ampli_val)) > 0.7: 
                            differences = np.diff(inner_list).tolist()
                            diff_list = [x for x in differences if x >= 70]
                            if len(diff_list) != 0:
                                third_degree.append(1)
                            else:
                                third_degree.append(0)    
                    elif len(inner_list) in [3,4] and possible_mob_3rd==True and constant_2 == False:
                        differences = np.diff(inner_list).tolist()
                        if all(diff > 70 for diff in differences):
                            third_degree.append(1)
                        else:
                            third_degree.append(0)
                    else:
                        third_degree.append(0)
        if len(third_degree) != 0:
            if third_degree.count(1) /len(third_degree) >= 0.4 or possible_mob_3rd: # 0.5 0.4   
                label = "3rd Degree block"
        return label

    def second_degree_block_detection(self):
        label= 'Abnormal'
        constant_3_peak = []
        possible_mob_1 = False
        possible_mob_2 = False
        mob_count = 0
        if self.hr_counts <= 100: # 80
            if len(self.p_t) != 0:
                constant_2 = all(map(lambda innerlist: len(innerlist) == 2, self.p_t))
                rhythm_flag = all(len(inner_list) in {1, 2, 3} for inner_list in self.p_t)
                ampli_val = list(map(lambda inner_list: sum(self.baseline_signal[i] > 0.05 for i in inner_list) / len(inner_list), self.p_t))
                count_above_threshold = sum(1 for value in ampli_val if value > 0.7)
                percentage_above_threshold = count_above_threshold / len(ampli_val)
                if percentage_above_threshold >= 0.7:
                    if rhythm_flag and constant_2 == False:
                        pr_interval = []
                        for i, r_element in enumerate(self.r_index[1:], start=1):
                            if i <= len(self.p_t):
                                inner_list = self.p_t[i - 1]  
                                last_element = inner_list[-1] 
                                result = r_element - last_element 
                                pr_interval.append(result)

                        counts = {}
                        count_2 = 0
                        for i in range(0, len(pr_interval)):
                            counts[i] = 1
                            if i in counts:
                                counts[i] += 1
                            if pr_interval[i] > pr_interval[i -1]:
                                count_2 += 1
                        most_frequent = max(counts.values())
                        if round(count_2 / (len(pr_interval)), 2) >= 0.50: 
                            possible_mob_1 = True
                        elif round(most_frequent / len(pr_interval), 2) >= 0.4: 
                            possible_mob_2 = True

                        for inner_list in self.p_t:
                            if len(inner_list) == 3 :
                                differences = np.diff(inner_list).tolist()
                                if differences[0] <= 0.5 * differences[1] or differences[1] <= 0.5 * differences[0]:
                                    if possible_mob_1 or possible_mob_2:
                                        mob_count += 1
                                    else:
                                        constant_3_peak.append(1)
                            else:
                                constant_3_peak.append(0)
                    else:
                        for inner_list in self.p_t:
                            if len(inner_list) == 3 :
                                differences = np.diff(inner_list).tolist()
                                if differences[0] <= 0.5 * differences[1] or differences[1] <= 0.5 * differences[0]:
                                    constant_3_peak.append(1)
                                else:
                                    constant_3_peak.append(0)
                            else:
                                constant_3_peak.append(0)
        if len(constant_3_peak) != 0 and constant_3_peak.count(1) != 0:

            if constant_3_peak.count(1) /len(constant_3_peak) >= 0.4: # 0.4 0.5
                label = "Mobitz_II"
        elif possible_mob_1 and mob_count > 1: # 0 1 4
            label = "Mobitz_I"
        elif possible_mob_2 and mob_count > 1: # 0  4
            label = "Mobitz_II"
        return label

    def image_array_new(self, signal, scale=25):
        scales = np.arange(1, scale, 1)
        coef, freqs = pywt.cwt(signal, scales, 'gaus6')
        # coef, freqs = pywt.cwt(signal, scales, wavelet_name)
        abs_coef = np.abs(coef)
        y_scale = abs_coef.shape[0] / 224
        x_scale = abs_coef.shape[1] / 224
        x_indices = np.arange(224) * x_scale
        y_indices = np.arange(224) * y_scale
        x, y = np.meshgrid(x_indices, y_indices, indexing='ij')
        x = x.astype(int)
        y = y.astype(int)
        rescaled_coef = abs_coef[y, x]
        min_val = np.min(rescaled_coef)
        max_val = np.max(rescaled_coef)
        normalized_coef = (rescaled_coef - min_val) / (max_val - min_val)
        cmap_indices = (normalized_coef * 256).astype(np.uint8)
        cmap = colormaps.get_cmap('viridis')
        rgb_values = cmap(cmap_indices)
        image = rgb_values.reshape((224, 224, 4))[:, :, :3]
        denormalized_image = (image * 254) + 1
        rotated_image = np.rot90(denormalized_image, k=1, axes=(1, 0))
        return rotated_image.astype(np.uint8)

    def predict_tflite_model(self, model: tuple, input_data: tuple):
        with results_lock:
            interpreter, input_details, output_details = model
            for i in range(len(input_data)):
                interpreter.set_tensor(input_details[i]['index'], input_data[i])
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])

        return output

    def check_model(self, q_new, s_new, last_s, last_q):
        percent = {'ABNORMAL': 0, '1st_DEG': 0, '2nd_DEG': 0, '3rd_DEG': 0, 'NOISE': 0}
        total_data = len(self.s_index) - 1
        first_deg_data_index, second_deg_data_index, third_deg_data_index = [], [], []
        epoch_afib_index = []

        for q, s in zip(q_new, s_new):
            data = self.baseline_signal[q:s]
            if data.any():
                image_data = self.image_array_new(data)
                image_data = (tf.expand_dims(image_data.astype(np.float32), axis=0),)
                model_pred = self.predict_tflite_model(block_model, image_data)[0]

                model_idx = np.argmax(model_pred)
                if model_idx == 0:
                    if last_s and s > last_s[0]:
                        percent['ABNORMAL'] += last_s[1] / total_data
                    else:
                        percent['ABNORMAL'] += 4 / total_data
                elif model_idx == 1:
                    if last_s and s > last_s[0]:
                        percent['1st_DEG'] += last_s[1] / total_data
                        first_deg_data_index.append((last_q, s))
                    else:
                        percent['1st_DEG'] += 4 / total_data
                        first_deg_data_index.append((q, s))
                elif model_idx == 2:
                    if last_s and s > last_s[0]:
                        percent['2nd_DEG'] += last_s[1] / total_data
                        second_deg_data_index.append((last_q, s))
                    else:
                        percent['2nd_DEG'] += 4 / total_data
                        second_deg_data_index.append((q, s))
                elif model_idx == 3:
                    if last_s and s > last_s[0]:
                        percent['3rd_DEG'] += last_s[1] / total_data
                        third_deg_data_index.append((last_q, s))
                    else:
                        percent['3rd_DEG'] += 4 / total_data
                        third_deg_data_index.append((q, s))
                elif model_idx == 4:
                    if last_s and s > last_s[0]:
                        percent['NOISE'] += last_s[1] / total_data
                    else:
                        percent['NOISE'] += 4 / total_data

        return percent, first_deg_data_index, second_deg_data_index,third_deg_data_index

    def get_data(self):
        total_data = len(self.s_index) - 1
        last_s = None
        last_q = None
        check_2nd_lead = {'ABNORMAL': 0, '1st_DEG': 0, '2nd_DEG': 0, '3rd_DEG': 0, 'NOISE': 0}
        first_deg_data_index = second_deg_data_index = third_deg_data_index = []
        if len(self.q_index) > 4 and len(self.s_index) > 4:
            q_new = self.q_index[:-4:4].tolist()
            s_new = self.s_index[4::4].tolist()
            if s_new[-1] != self.s_index[-1]:
                temp_s = list(self.s_index).index(s_new[-1])
                fin_s = total_data - temp_s
                last_q = self.q_index[temp_s]
                last_s = (s_new[-1], fin_s)
                q_new.append(self.q_index[-5])
                s_new.append(self.s_index[-1])
            check_2nd_lead, first_deg_data_index, second_deg_data_index,third_deg_data_index = self.check_model(q_new, s_new,last_s, last_q)

        return check_2nd_lead, first_deg_data_index, second_deg_data_index,third_deg_data_index

def block_model_check(ecg_signal, frequency, abs_result):
    block_final_label = 'Abnormal'
    # abs_result = '3rd Degree block'
    first_deg_per = second_deg_per = third_deg_per = abanormal_per = noise_per = 0
    block_indexs = []
    block_percent,  first_deg_data_index, second_deg_data_index,third_deg_data_index = BlockDetected(ecg_signal, frequency).get_data()
    first_deg_per = int(block_percent['1st_DEG'] * 100)
    second_deg_per = int(block_percent['2nd_DEG'] * 100)
    third_deg_per = int(block_percent['3rd_DEG'] * 100)
    abanormal_per = int(block_percent['ABNORMAL'] * 100)
    noise_per = int(block_percent['NOISE'] * 100)
    model_label = 'Abnormal'

    block_indexs = {'first_deg_data_index':[], 'second_deg_data_index': [], 'third_deg_data_index':[]}
    if first_deg_per >= 55:
        model_label = '1st_DEG'
    if second_deg_per >= 55: 
        model_label = model_label +", "+  '2nd_DEG' if model_label != 'Abnormal' else '2nd_DEG'
    if third_deg_per >= 55: 
        model_label =  model_label +", "+  '3rd_DEG' if model_label != 'Abnormal' else '3rd_DEG'


    if '1st_DEG' in model_label and '1st deg. block' in abs_result:
        block_final_label = 'I_Degree'
        block_indexs['first_deg_data_index'] = first_deg_data_index
    if '2nd_DEG' in model_label and ('Mobitz_I' in abs_result or 'Mobitz_II' in abs_result):
        block_final_label = abs_result.upper()
        block_indexs['second_deg_data_index'] = second_deg_data_index
    if '3rd_DEG' in model_label and '3rd Degree block' in abs_result:
        block_final_label = 'III_Degree'
        block_indexs['third_deg_data_index'] = third_deg_data_index
    if '2nd_DEG' in model_label and '3rd Degree block' in abs_result:
        block_final_label = 'MOBITZ_I'
        block_indexs['second_deg_data_index'] = second_deg_data_index
    if '3rd_DEG' in model_label and ('Mobitz_I' in abs_result or  'Mobitz_II' in abs_result):
        block_final_label = 'III_Degree'
        block_indexs['third_deg_data_index'] = third_deg_data_index
    return block_final_label, block_percent, block_indexs

# For juctional S point detection
def find_s_newnew_index(ecg, R_index, d):
    end_index = len(ecg) - 1
    range_per = 0.03
    small_range_per = 0.01
    s = []
    for r in R_index:
        r_range = abs(ecg[r] * range_per)
        r_range__ = abs(ecg[r] * small_range_per)
        s_, sss = [], []
        if r == len(ecg):
            s.append(r)
            continue
        if ecg[r] > 0:
            c = r
            while c+1 <= end_index and ecg[c+1] < ecg[c] and abs(r-c) <= d:
                c += 1
                if (-(r_range) <= ecg[c] <= r_range):
                    sss.append(c)
            if (-(r_range) <= ecg[c] <= r_range)  or c == end_index or abs(r-c) > d:
                s_.append(c)
            # s_.append(c)
            while c+1 <= end_index and abs(r-c) <= d:
                while c+1 <= end_index and ecg[c+1] > ecg[c] and abs(r-c) <= d:
                    c += 1
                    if (-(r_range) <= ecg[c] <= r_range):
                        sss.append(c)
                while c+1 <= end_index and ecg[c+1] < ecg[c] and abs(r-c) <= d:
                    c += 1
                    if (-(r_range) <= ecg[c] <= r_range):
                        sss.append(c)
                if s_ and s_[-1] == c:
                    break
                s_.append(c)
                if abs(r-c) <= d and (-(r_range) <= ecg[c] <= r_range) or c == end_index:
                    break
            
        else:
            c = r
            while c+1 <= end_index and ecg[c+1] > ecg[c] and abs(r-c) <= d:
                c += 1
                if (-(r_range) <= ecg[c] <= r_range):
                    sss.append(c)
            if (-(r_range) <= ecg[c] <= r_range) or c == end_index or abs(r-c) > d:
                s_.append(c)
            # s_.append(c)
            while c <= end_index and abs(r-c) <= d:
                while c+1 <= end_index and ecg[c+1] > ecg[c] and abs(r-c) <= d:
                    c += 1
                    if (-(r_range) <= ecg[c] <= r_range):
                        sss.append(c)
                while c+1 <= end_index and ecg[c+1] < ecg[c] and abs(r-c) <= d:
                    c += 1
                    if (-(r_range) <= ecg[c] <= r_range):
                        sss.append(c)
                if s_ and s_[-1] == c:
                    break
                s_.append(c)
                if abs(r-c) <= d and (-(r_range) <= ecg[c] <= r_range)  or c == end_index:
                    break
        if s_ or sss:
            a = 0
            for _s in s_[::-1]:
                if (-(r_range__) <= ecg[_s] <= r_range__):
                    a = 1
                    s.append(_s)
                    break
            if a == 0:
                for _s in sss[::-1]:
                    if (-(r_range__) <= ecg[_s] <= r_range__):
                        a = 1
                        s.append(_s)
                        break
            if a == 0:
                if ecg[r] > 0:
                    for _s in s_[::-1]:
                        if ecg[_s] <= r_range:
                            a = 1
                            s.append(_s)
                            break
                    if a == 0:
                        for _s in sss[::-1]:
                            if ecg[_s] <= r_range__:
                                a = 1
                                s.append(_s)
                                break
                else:
                    for _s in s_[::-1]:
                        if -r_range <= ecg[_s] :
                            a = 1
                            s.append(_s)
                            break
                    if a == 0:
                        for _s in sss[::-1]:
                            if -r_range__ <= ecg[_s] :
                                a = 1
                                s.append(_s)
                                break
            if a == 0: 
                if r+d<=len(ecg):
                    s_array = ecg[r:r+d]
                else:
                    s_array = ecg[r:]
                if ecg[r] > 0:
                    s_index = r+np.where(s_array == min(s_array))[0][0]
                else:
                    s_index = r+np.where(s_array == max(s_array))[0][0]
                s.append(s_index)
    return np.sort(s)

# For juctional Q point detection
def find_q_newnew_index(ecg, R_index, d):
    q = []
    range_per = 0.03
    small_range_per = 0.01
    for r in R_index:
        r_range = abs(ecg[r] * range_per)
        r_range__ = abs(ecg[r] * small_range_per)
        q_, qqq = [], []
        if r == 0:
            q.append(r)
            continue
        if ecg[r] > 0:
            c = r
            while c > 0 and ecg[c-1] < ecg[c] and abs(r-c) <= d:
                c -= 1
                if (-(r_range) <= ecg[c] <= r_range):
                        qqq.append(c)
            if (-(r_range) <= ecg[c] <= r_range) or c == 0 or abs(r-c) > d:
                q_.append(c)
            # q_.append(c)
            while c > 0 and abs(r-c) <= d:
                while c > 0 and ecg[c-1] > ecg[c] and abs(r-c) <= d:
                    c -= 1
                    if (-(r_range) <= ecg[c] <= r_range):
                        qqq.append(c)
                while c > 0 and ecg[c-1] < ecg[c] and abs(r-c) <= d:
                    c -= 1
                    if (-(r_range) <= ecg[c] <= r_range):
                        qqq.append(c)
                if q_ and q_[-1] == c:
                    break
                q_.append(c)
                if abs(r-c) <= d and (-(r_range) <= ecg[c] <= r_range) or c == 0:
                    break
        else:
            c = r
            while c > 0 and ecg[c-1] > ecg[c] and abs(r-c) <= d:
                c -= 1
                if (-(r_range) <= ecg[c] <= r_range):
                    qqq.append(c)
            if (-(r_range) <= ecg[c] <= r_range) or c == 0 or abs(r-c) > d:
                q_.append(c)
            # q_.append(c)
            while c > 0 and abs(r-c) <= d:
                while c > 0 and ecg[c-1] < ecg[c] and abs(r-c) <= d:
                    c -= 1
                    if (-(r_range) <= ecg[c] <= r_range):
                        qqq.append(c)
                while c > 0 and ecg[c-1] > ecg[c] and abs(r-c) <= d:
                    c -= 1
                    if (-(r_range) <= ecg[c] <= r_range):
                        qqq.append(c)
                if q_ and q_[-1] == c:
                    break
                q_.append(c)
                if abs(r-c) <= d and (-(r_range) <= ecg[c] <= r_range) or c == 0:
                    break
        if q_ or qqq:
            a = 0
            for _q in q_[::-1]:
                if (-(r_range__) <= ecg[_q] <= r_range__):
                    a = 1
                    q.append(_q)
                    break
            if a == 0:
                for _q in qqq[::-1]:
                    if (-(r_range__) <= ecg[_q] <= r_range__):
                        a = 1
                        q.append(_q)
                        break
            if a == 0:
                if ecg[r] > 0:
                    for _q in q_[::-1]:
                        if ecg[_q] <= r_range:
                            a = 1
                            q.append(_q)
                            break
                    if a == 0:
                        for _q in qqq[::-1]:
                            if ecg[_q] <= r_range__:
                                a = 1
                                q.append(_q)
                                break
                else:
                    for _q in q_[::-1]:
                        if -r_range <= ecg[_q] :
                            a = 1
                            q.append(_q)
                            break
                    if a == 0:
                        for _q in qqq[::-1]:
                            if -r_range__ <= ecg[_q] :
                                a = 1
                                q.append(_q)
                                break
            if a == 0:
                if 0 <= r - d:
                    q_array = ecg[r - d:r]
                else:
                    q_array = ecg[:r]
                if ecg[r] > 0:
                    q_index = r - (len(q_array) - np.where(q_array == min(q_array))[0][0])
                else:
                    q_index = r - (len(q_array) - np.where(q_array == max(q_array))[0][0])
                q.append(q_index)
    return np.sort(q)

# Vfib & VFL detection
def resampled_ecg_data(ecg_signal, original_freq, desire_freq):
    original_time = np.arange(len(ecg_signal)) / original_freq
    new_time = np.linspace(original_time[0], original_time[-1], int(len(ecg_signal) * (desire_freq/original_freq)))
    interp_func = interp1d(original_time, ecg_signal, kind='linear')
    scaled_ecg_data = interp_func(new_time)
    return scaled_ecg_data

def image_array_vfib(signal):
    scales = np.arange(1, 50, 1)
    coef, freqs = pywt.cwt(signal, scales, 'mexh')
    abs_coef = np.abs(coef)
    y_scale = abs_coef.shape[0] / 224
    x_scale = abs_coef.shape[1] / 224
    x_indices = np.arange(224) * x_scale
    y_indices = np.arange(224) * y_scale
    x, y = np.meshgrid(x_indices, y_indices, indexing='ij')
    x = x.astype(int)
    y = y.astype(int)
    rescaled_coef = abs_coef[y, x]
    min_val = np.min(rescaled_coef)
    max_val = np.max(rescaled_coef)
    normalized_coef = (rescaled_coef - min_val) / (max_val - min_val)
    cmap_indices = (normalized_coef * 256).astype(np.uint8)
    cmap = colormaps.get_cmap('viridis')
    rgb_values = cmap(cmap_indices)
    image = rgb_values.reshape((224, 224, 4))[:, :, :3]
    denormalized_image = (image * 254) + 1
    rotated_image = np.rot90(denormalized_image, k=1, axes=(1, 0))
    return rotated_image.astype(np.uint8)

def vfib_predict_tflite_model(model:tuple, input_data:tuple):
    if type(model) != tuple and type(input_data) != tuple:
        raise TypeError
    interpreter, input_details, output_details = model
    for i in range(len(input_data)):
        interpreter.set_tensor(input_details[i]['index'], input_data[i])
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

def vfib_model_pred_tfite(raw_signal, model, fs):
    if fs == 200 and (np.max(raw_signal) > 4.1 or np.min(raw_signal) < 0):
        raw_signal = MinMaxScaler(feature_range=(0,4)).fit_transform(raw_signal.reshape(-1,1)).squeeze()
    seconds = 2.5
    steps_data = int(fs*seconds)
    total_data = raw_signal.shape[0]
    start = 0
    normal, vfib_vflutter, asys, noise = [], [], [], []
    percentage = {'NORMAL':0, 'VFIB-VFLUTTER':0, 'ASYS':0, 'NOISE':0}
    model_prediction = []
    while start < total_data:
        end = start+steps_data
        if end - start == steps_data and end < total_data:
            _raw_s_ = raw_signal[start:end]
            if _raw_s_.any() :
                raw = image_array_vfib(_raw_s_)
            else:
                raw = np.array([])
        else:
            _raw_s_ = raw_signal[start:end]
            if _raw_s_.any():
                _raw_s_ = raw_signal[-steps_data:total_data]
                raw = image_array_vfib(_raw_s_)
                end = total_data - 1
            else:
                raw = np.array([])
        if raw.any():
            raw = raw.astype(np.float32)/255
            rs_raw = resampled_ecg_data(_raw_s_, fs, 500/seconds)
            if rs_raw.shape[0] != 500:
                rs_raw = signal.resample(rs_raw, 500)
            image_data = (tf.expand_dims(raw, axis=0),) #tf.constant(rs_raw.reshape(1, -1, 1).astype(np.float32)))
            # image_data = (tf.cast(image_data[0],dtype=tf.float32), )
            model_pred = vfib_predict_tflite_model(model, image_data)[0]
            label = np.argmax(model_pred)
            model_prediction.append(f'{(start, end)}={model_pred}')
            if label == 0: normal.append(((start, end), model_pred)); percentage['NORMAL'] += (end-start)/total_data
            elif label == 1: vfib_vflutter.append(((start, end), model_pred)); percentage['VFIB-VFLUTTER'] += (end-start)/total_data
            elif label == 2: asys.append(((start, end), model_pred)); percentage['ASYS'] += (end-start)/total_data
            else: noise.append(((start, end), model_pred)); percentage['NOISE'] += (end-start)/total_data
        start = start+steps_data
    
    return normal, vfib_vflutter, asys, noise, model_prediction, percentage
   
def vfib_model_check(ecg_signal, baseline_signal, lowpass_signal, model, fs):
    normal, vfib_vflutter, asys, noise, model_prediction, percentage = vfib_model_pred_tfite(ecg_signal, model, fs)

    final_label_index = np.argmax([percentage['NORMAL'], percentage['VFIB-VFLUTTER'],
                             percentage['ASYS'], percentage['NOISE']])

    
    if final_label_index == 0 and percentage['NORMAL'] > .50:
        final_label = 'Normal'
        percentage = percentage['NORMAL']
    elif final_label_index == 0 and (percentage['VFIB-VFLUTTER'] < 0.3 and percentage['ASYS'] < 0.3 and percentage['NOISE'] < 0.3):
        final_label = 'Normal'
        percentage = percentage['NORMAL']
    else:
        final_label_index = np.argmax([percentage['VFIB-VFLUTTER'], percentage['ASYS'], percentage['NOISE']])
        if final_label_index == 0:
            final_label = 'VFIB'
            percentage = percentage['VFIB-VFLUTTER']
        elif final_label_index == 1:
            final_label = 'ASYSTOLE'
            percentage = percentage['ASYS']
        else:
            final_label = 'Normal'
            percentage = percentage['NOISE']

    
    # return final_label, percentage, (normal, vfib_vflutter, asys, noise, model_prediction)
    return final_label

# Pacemaker detection
def pacemake_detect(ecg_signal, fs=200):
    pqrst_data = pqrst_detection(ecg_signal, fs=fs, width=(3,50)).get_data()
    r_index = pqrst_data['R_index']
    q_index = pqrst_data['Q_Index']
    s_index = pqrst_data['S_Index']
    p_index = pqrst_data['P_Index']
    v_pacemaker = []
    a_pacemaker = []
    q_to_pace = []
   
    qd = int(fs * 0.08)
    percentage = 0
    for q in q_index:
        _q = q - qd
        aoi1 = ecg_signal[_q:q]
        if aoi1.any():
            peaks1 = np.where(np.min(aoi1) == aoi1)[0][0]
            peaks1 += _q
            q_peaks_distance = abs(q - peaks1)
            if q_peaks_distance < 11:
                q_to_pace.append(1)
            else:
                q_to_pace.append(0)

    if len(q_to_pace) != 0:
        percentage = (q_to_pace.count(1)/len(q_to_pace))

    for q in q_index:
        _q = q - qd
        aoi1 = ecg_signal[_q:q]
        if aoi1.any():
            peaks1 = np.where(np.min(aoi1) == aoi1)[0][0]
            peaks1 += _q
            if -0.6 <= ecg_signal[peaks1] <= -0.1 and ecg_signal[q] > ecg_signal[peaks1] and abs(ecg_signal[q] - ecg_signal[peaks1]) >= 0.15 and percentage > 0.5:
                if np.min(np.abs(r_index - peaks1)) > 14:
                    v_pacemaker.append(peaks1)

    for i in range (0,len(r_index)-1):
        aoi = ecg_signal[s_index[i]:q_index[i+1]]
        if aoi.any():
            check, _ = find_peaks(aoi, prominence=(0.2,0.3),distance=100,width = (1,6))
            peaks1 = check+s_index[i]
        else:
            peaks1 = np.array([])
        if peaks1.any():
            a_pacemaker.extend(list(peaks1))

    # Remove a_pacemaker if it falls within 20 data points of a v_pacemaker or Atrial_&_Ventricular_pacemaker
    for v_peak in v_pacemaker:
        for k in range(len(a_pacemaker) - 1, -1, -1):
            if abs(a_pacemaker[k] - v_peak) <= 20:
                a_pacemaker.pop(k)
                
    atrial_per = venti_per = 0
    if len(r_index) != 0:
        atrial_per = round((len(a_pacemaker)/ len(r_index)) * 100)
        venti_per = round((len(v_pacemaker)/ len(r_index)) * 100)

    if atrial_per > 30 and venti_per > 30:
        pacemaker = np.concatenate((v_pacemaker, a_pacemaker)).astype('int64').tolist()
        pacmaker_per = round((len(a_pacemaker)/ len(r_index)) * 100)
        label = "Atrial_&_Ventricular_pacemaker"
    elif atrial_per >= 50 and venti_per >= 50:
        if venti_per > atrial_per:
            label = "Ventricular_Pacemaker"
            pacemaker = v_pacemaker
        else:
            label = "Atrial_Pacemaker"
            pacemaker = a_pacemaker
    elif atrial_per >= 50:
        label = "Atrial_Pacemaker"
        pacemaker = a_pacemaker
    elif venti_per >= 50:
        label = "Ventricular_Pacemaker"
        pacemaker = v_pacemaker
    else:
        label = "False"
        pacemaker = np.array([])
    return label, pacemaker

def image_array_new(signal, scale=25):
    '''
    Other : scale=25, wavelet_name='gaus6'
    AFIB : scale=25, wavelet_name='morl'
    VFIB/VFlutter : scale=50, wavelet_name='mexh'
    '''
    scales = np.arange(1, scale, 1)
    coef, freqs = pywt.cwt(signal, scales, 'gaus6')
    # coef, freqs = pywt.cwt(signal, scales, wavelet_name)
    abs_coef = np.abs(coef)
    y_scale = abs_coef.shape[0] / 224
    x_scale = abs_coef.shape[1] / 224
    x_indices = np.arange(224) * x_scale
    y_indices = np.arange(224) * y_scale
    x, y = np.meshgrid(x_indices, y_indices, indexing='ij')
    x = x.astype(int)
    y = y.astype(int)
    rescaled_coef = abs_coef[y, x]
    min_val = np.min(rescaled_coef)
    max_val = np.max(rescaled_coef)
    normalized_coef = (rescaled_coef - min_val) / (max_val - min_val)
    cmap_indices = (normalized_coef * 256).astype(np.uint8)
    cmap = colormaps.get_cmap('viridis')
    rgb_values = cmap(cmap_indices)
    image = rgb_values.reshape((224, 224, 4))[:, :, :3]
    denormalized_image = (image * 254) + 1
    rotated_image = np.rot90(denormalized_image, k=1, axes=(1, 0))
    return rotated_image.astype(np.uint8)

# Afib & Flutter detection
class afib_flutter_detection:
    def __init__(self, ecg_signal, r_index, q_index, s_index, p_index, p_t, pr_interval, load_model):
        self.ecg_signal = ecg_signal
        self.r_index = r_index
        self.q_index = q_index
        self.s_index = s_index
        self.p_index = p_index
        self.p_t = p_t
        self.pr_inter = pr_interval
        self.load_model = load_model

    def image_array_new(self, signal, scale=25):
        scales = np.arange(1, scale, 1)
        coef, freqs = pywt.cwt(signal, scales, 'gaus6')
        # coef, freqs = pywt.cwt(signal, scales, wavelet_name)
        abs_coef = np.abs(coef)
        y_scale = abs_coef.shape[0] / 224
        x_scale = abs_coef.shape[1] / 224
        x_indices = np.arange(224) * x_scale
        y_indices = np.arange(224) * y_scale
        x, y = np.meshgrid(x_indices, y_indices, indexing='ij')
        x = x.astype(int)
        y = y.astype(int)
        rescaled_coef = abs_coef[y, x]
        min_val = np.min(rescaled_coef)
        max_val = np.max(rescaled_coef)
        normalized_coef = (rescaled_coef - min_val) / (max_val - min_val)
        cmap_indices = (normalized_coef * 256).astype(np.uint8)
        cmap = colormaps.get_cmap('viridis')
        rgb_values = cmap(cmap_indices)
        image = rgb_values.reshape((224, 224, 4))[:, :, :3]
        denormalized_image = (image * 254) + 1
        rotated_image = np.rot90(denormalized_image, k=1, axes=(1, 0))
        return rotated_image.astype(np.uint8)

    def abs_afib_flutter_check(self):
        check_afib_flutter = False
        rpeak_diff = np.diff(self.r_index)
        more_then_3_rhythm_per = len(list(filter(lambda x: len(x) >= 3, self.p_t)))/ len(self.r_index)
        inner_list_less_2 = len(list(filter(lambda x: len(x) < 2, self.p_t))) / len(self.r_index)
       
        zeros_count = self.p_index.count(0)
        list_per = zeros_count/len(self.p_index)
        pr_int = [round(num, 2) for num in self.pr_inter]

        constant_list = []
        if len(pr_int) >1:
            for i in range(len(pr_int) - 1):
                diff = abs(pr_int[i] - pr_int[i + 1])
                if diff == 0 or diff == 1:
                    constant_list.append(pr_int[i]) 
            
            if abs(pr_int[-1] - pr_int[-2]) == 0 or abs(pr_int[-1] - pr_int[-2]) == 1:
                constant_list.append(pr_int[-1])


        if more_then_3_rhythm_per >= 0.6:
            check_afib_flutter = True
        elif list_per >= 0.5:
            check_afib_flutter = True
        elif len(constant_list) != 0:
            if (len(constant_list) / len(pr_int) < 0.7):
                check_afib_flutter = True
        else:
            p_peak_diff = np.diff(self.p_index)
            percentage_diff = np.abs(np.diff(p_peak_diff)/ p_peak_diff[:-1])* 100
            
            mean_p = np.mean(percentage_diff)
            if mean_p != mean_p or mean_p == float('inf') or mean_p == float('-inf'):
                check_afib_flutter = True
            if (mean_p > 15 and more_then_3_rhythm_per >= 0.4) or (mean_p > 70 and inner_list_less_2 > 0.3):
                check_afib_flutter = True
            elif mean_p > 100 and inner_list_less_2 > 0.3:
                check_afib_flutter = True
            elif (mean_p > 20 and more_then_3_rhythm_per >= 0.1):
                check_afib_flutter = True
        return check_afib_flutter

    def predict_tflite_model(self, model:tuple, input_data:tuple):
        with results_lock:
            interpreter, input_details, output_details = model
            for i in range(len(input_data)):
                interpreter.set_tensor(input_details[i]['index'], input_data[i])
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
        
        return output

    def check_model(self, q_new, s_new, ecg_signal, last_s, last_q):
        percent = {'ABNORMAL': 0,'AFIB': 0, 'FLUTTER': 0, 'NOISE': 0, 'NORMAL': 0}
        total_data = len(self.s_index)-1
        afib_data_index, flutter_data_index = [], []
        for q, s in zip(q_new, s_new):
            data = ecg_signal[q:s]
            if data.any():
                image_data = self.image_array_new(data)
                image_data = (tf.expand_dims(image_data.astype(np.float32), axis=0),)
                model_pred = self.predict_tflite_model(self.load_model, image_data)[0]

                model_idx = np.argmax(model_pred)
                if model_idx == 0:
                    if last_s and s > last_s[0]:
                        percent['ABNORMAL'] += last_s[1] / total_data
                    else:
                        percent['ABNORMAL'] += 4 / total_data
                elif model_idx == 1:
                    if last_s and s > last_s[0]:
                        percent['AFIB'] += last_s[1] / total_data
                        afib_data_index.append((last_q, s))
                    else:
                        percent['AFIB'] += 4 / total_data
                        afib_data_index.append((q, s))
                elif model_idx == 2:
                    if last_s and s > last_s[0]:
                        percent['FLUTTER'] += last_s[1] / total_data
                        flutter_data_index.append((last_q, s))
                    else:
                        percent['FLUTTER'] += 4 / total_data
                        flutter_data_index.append((q, s))
                elif model_idx == 3:
                    if last_s and s > last_s[0]:
                        percent['NOISE'] += last_s[1] / total_data
                    else:
                        percent['NOISE'] += 4 / total_data
                elif model_idx == 4:
                    if last_s and s > last_s[0]:
                        percent['NORMAL'] += last_s[1] / total_data
                    else:
                        percent['NORMAL'] += 4 / total_data
        return percent, afib_data_index, flutter_data_index

    def get_data(self):
        total_data = len(self.s_index)-1
        last_s = None
        last_q = None
        check_2nd_lead = {'ABNORMAL': 0,'AFIB': 0, 'FLUTTER': 0, 'NOISE': 0, 'NORMAL': 0}
        afib_data_index, flutter_data_index = [],[]
        if len(self.q_index) > 4 and len(self.s_index) > 4:
            q_new  = self.q_index[:-4:4].tolist()
            s_new = self.s_index[4::4].tolist()
            if s_new[-1] != self.s_index[-1]:
                temp_s = list(self.s_index).index(s_new[-1])
                fin_s = total_data - temp_s
                last_q = self.q_index[temp_s]
                last_s = (s_new[-1], fin_s)
                q_new.append(self.q_index[-5])
                s_new.append(self.s_index[-1])
            check_2nd_lead, afib_data_index, flutter_data_index = self.check_model(q_new, s_new, self.ecg_signal, last_s, last_q)         
        return check_2nd_lead, afib_data_index, flutter_data_index

# Junctional detection
def jr_model_detection(baseline_signal, r_index, model, hr_counts, fs):
    model_prediction = []
    jr_pred = []
    jr_r = []
    # futures_jr = {}
    sd, qd = int(fs * 0.115), int(fs * 0.08)
    s_index = find_s_newnew_index(baseline_signal, r_index, sd)
    q_index = find_q_newnew_index(baseline_signal, r_index, qd)
    count_zeros = 0
    lowpass_signal = lowpass(baseline_signal, (fs * -0.00016964) + 0.20821429)
    for q, _s, s, q_ in zip(q_index[:-1], s_index[:-1], s_index[1:], q_index[1:]):
        data_with_r, data_without_r = lowpass_signal[q:s], lowpass_signal[_s:q_]
        if data_with_r.any() and data_without_r.any():
            image_data1, image_data2 = image_array_new(data_with_r), image_array_new(data_without_r)
            image_data = (tf.expand_dims(image_data1, axis=0), tf.expand_dims(image_data2, axis=0))
            image_data = (tf.cast(image_data[0], dtype=tf.float32), tf.cast(image_data[1], dtype=tf.float32))
            model_pred = predict_tflite_model(model, image_data)[0]
            # K.clear_session()
            label_jr = np.argmax(model_pred)
            model_prediction.append(f'{(_s, q_)}={model_pred}')
            if label_jr == 1:
                jr_pred.append((_s, q_))
                jr_r.append(1)
            else:
                jr_r.append(0)
                count_zeros += 1 
        else:
            jr_r.append(0)
            count_zeros += 1
    
    jr_model_percent = 0
    i = len(jr_r)
    if i != 0: jr_model_percent = (i -count_zeros)/i
    jr_label = "Abnormal"
    if jr_model_percent >= 0.75:
        jr_label = "Junctional_Rhythm" if hr_counts > 40 else "Junctional_Bradycardia"
    jr_data = {}
    jr_data['JR_Pred'] = jr_pred
    jr_data['JR_Model_Percent'] = jr_model_percent
    jr_data['Model_Pred'] = model_prediction
    jr_data['JR_Label'] = jr_label
    return jr_data

def jr_detection(baseline_signal, model, fs):
    pqrst_data = pqrst_detection(baseline_signal, fs=fs, thres=0.37, rr_thres = 0.15, lp_thres=0.1, JR=True).get_data()
    r_label = pqrst_data['R_Label']
    r_index = pqrst_data['R_index']
    p_t = pqrst_data['P_T List']
    hr_ = pqrst_data['HR_Count']
    t_index = pqrst_data['T_Index']
    p_index = pqrst_data['P_Index'] 
    ex_index = pqrst_data['Ex_Index']
    jr_label = 'Abnormal'

    count = 0
    jr_abstraction_per, combined_percent, jr_model_percent, jr_model_index = 0, 0, 0, []
    if r_label == "Regular" and hr_ <= 75:
        new_threshold = 0.065 if hr_ > 50 else 0.06
        for i in range(len(p_t)):
            dis = (r_index[i+1]-p_t[i][-1])/fs
            if dis <= new_threshold: count += 1
        counter = 0
        for i in p_index:
            if i != 0:
                counter += 1
        jr_abstraction_per = ((len(r_index)-1 - (counter-count))/(len(r_index)-1))
        if jr_abstraction_per >= 0.75:
            jr_model_data = jr_model_detection(baseline_signal, r_index, model, hr_, fs)
            jr_model_percent = jr_model_data['JR_Model_Percent']
            jr_model_index = jr_model_data['JR_Pred']
            combined_percent = (jr_model_percent *0.2) + (jr_abstraction_per *0.8)
            if combined_percent >= 0.75 and jr_model_percent >= 0.2:
                jr_label = "JN_RHY" if hr_ > 40 else "JN_BR"
        else:
            jr_label = "Abnormal"
    return jr_label

# Wide-qrs detection
def wide_qrs(q_index, r_index, s_index, hr, fs=200):
    label = 'Abnormal'
    wideQRS = []
    recheck_wide_qrs = []
    if hr <= 88:
        thresold = round(fs * 0.10)
    else:
        thresold = round(fs * 0.12)
    if len(r_index) != 0:
        
        for k in range(len(r_index)):
            diff = s_index[k] - q_index[k]
            if diff > thresold:
                wideQRS.append(r_index[k])

        if len(wideQRS)/ len(r_index) >= 0.50:
            final_thresh = round(fs * 0.18)
            for k in range(len(r_index)):
                if diff > final_thresh:
                    recheck_wide_qrs.append(r_index[k])
        
        # if hr <= 88:
        #     thresold = round(fs * 0.12)
        #     if len(wideQRS)/ len(r_index) > 0.9:
        #         for k in range(len(r_index)):
        #             diff = s_index[k] - q_index[k]
        #             if diff > thresold:
        #                 wide_qrs.append(r_index[k])
        # else:
        #     wide_qrs = wideQRS

        if len(recheck_wide_qrs)/ len(r_index) >= 0.50:
            label = 'WIDE_QRS'
    return label, wide_qrs

def wide_qrs_find_pac(q_index, r_index, s_index, hr_count, fs=200):
    max_indexs = 0
    if hr_count <= 88:
        ms = 0.18 # 0.10
    else:
        ms = 0.16 # 0.12
    max_indexs = int(fs * ms)
    pvc = []
    difference = []
    pvc_index = []
    wide_qs_diff = []
    for k in range(len(r_index)):
        diff = s_index[k] - q_index[k]
        difference.append(diff)
        if max_indexs != 0:
            if diff>=max_indexs:
                pvc.append(r_index[k])
    if hr_count <= 88 and len(r_index) != 0:
        wide_r_index_per = len(pvc)/ len(r_index)
        if wide_r_index_per < 0.8:
            pvc_index = np.array(pvc)
        else:
            ms = 0.12
            max_indexs = int(fs * ms)
            for k in range(len(r_index)):
                diff = s_index[k] - q_index[k]
                wide_qs_diff.append(diff)
                if max_indexs != 0:
                    if diff>=max_indexs:
                        pvc_index.append(r_index[k])
            difference = wide_qs_diff
    else:
        pvc_index = np.array(pvc) 
    q_s_difference = [i/fs for i in difference]
    return np.array(pvc_index), q_s_difference

def predict_tflite_model(model:tuple, input_data:tuple):
    with results_lock:
        interpreter, input_details, output_details = model
        for i in range(len(input_data)):
            interpreter.set_tensor(input_details[i]['index'], input_data[i])
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
    
    return output

# PAC detection
class PAC_detedction:
    def __init__(self, ecg_signal, fs, hr_counts):
        self.ecg_signal = ecg_signal
        self.fs = fs
        self.hr_counts = hr_counts

    def detect_beats_for_pac(self, ecg,  rate, ransac_window_size=3.0, lowfreq=5.0, highfreq=15.0):

        ransac_window_size = int(ransac_window_size * rate)

        # Designing low and high pass filters
        lowpass = scipy.signal.butter(1, highfreq / (rate / 2.0), 'low')
        highpass = scipy.signal.butter(1, lowfreq / (rate / 2.0), 'high')
        
        # Applying filters
        ecg_low = scipy.signal.filtfilt(*lowpass, x=ecg)
        ecg_band = scipy.signal.filtfilt(*highpass, x=ecg_low)
        
        # Square of the first difference of the signal
        decg = np.diff(ecg_band)
        decg_power = decg ** 2
        
        # Robust threshold and normalizator estimation
        thresholds = []
        max_powers = []
        for i in range(int(len(decg_power) / ransac_window_size)):
            sample = slice(i * ransac_window_size, (i + 1) * ransac_window_size)
            d = decg_power[sample]
            thresholds.append(0.5 * np.std(d))
            max_powers.append(np.max(d))

        threshold = np.median(thresholds)
        max_power = np.median(max_powers)
        decg_power[decg_power < threshold] = 0

        decg_power /= max_power
        decg_power[decg_power > 1.0] = 1.0
        square_decg_power = decg_power ** 4

        shannon_energy = -square_decg_power * np.log(square_decg_power)
        shannon_energy[~np.isfinite(shannon_energy)] = 0.0

        mean_window_len = int(rate * 0.125 + 1)
        lp_energy = np.convolve(shannon_energy, [1.0 / mean_window_len] * mean_window_len, mode='same')
        
        lp_energy = scipy.ndimage.gaussian_filter1d(lp_energy, rate / 14.0)
        lp_energy_diff = np.diff(lp_energy)

        zero_crossings = (lp_energy_diff[:-1] > 0) & (lp_energy_diff[1:] < 0)
        zero_crossings = np.flatnonzero(zero_crossings)
        zero_crossings -= 1
        
        return zero_crossings

    def PACcounter(self, PAC_R_Peaks, hr_counts):
        svt_counter = 0
        couplet_counter = 0
        triplet_counter = 0
        bigeminy_counter = 0
        trigeminy_counter = 0
        quadrigeminy_counter = 0
        at = 0
        i = 0
        while i < len(PAC_R_Peaks):
            count = 0
            ones_count = 0
            while i < len(PAC_R_Peaks) and PAC_R_Peaks[i] == 1:
                count += 1
                ones_count += 1
                i += 1

            if count >= 4:
                svt_counter += 1
                at += ones_count
                count = 0
                ones_count = 0
            if count == 3:
                triplet_counter += 1
            elif count == 2:
                couplet_counter += 1
            i += 1
        j = 0
        while j < len(PAC_R_Peaks) - 1:
            if PAC_R_Peaks[j] == 1:
                k = j + 1
                spaces = 0
                while k < len(PAC_R_Peaks) and PAC_R_Peaks[k] == 0:
                    spaces += 1
                    k += 1

                if k < len(PAC_R_Peaks) and PAC_R_Peaks[k] == 1:
                    if spaces == 1:
                        bigeminy_counter += 1
                    elif spaces == 2:
                        trigeminy_counter += 1
                    elif spaces == 3:
                        quadrigeminy_counter += 1
                j = k
            else:
                j += 1
        
        total_one = (1*at) + (couplet_counter*2)+ (triplet_counter*3)+ (bigeminy_counter*2)+ (trigeminy_counter*2)+ (quadrigeminy_counter*2)
        total = svt_counter + couplet_counter+ triplet_counter+ bigeminy_counter+ trigeminy_counter+ quadrigeminy_counter
        ones = PAC_R_Peaks.count(1)
        if total == 0:
            Isolated = ones
        else:
            Common = total-1
            Isolated = ones - (total_one - Common)
        if hr_counts > 100:
            if svt_counter != 0:
                triplet_counter = couplet_counter = quadrigeminy_counter = trigeminy_counter = bigeminy_counter = Isolated = 0
        if svt_counter>=1 and hr_counts > 100: # 190
            svt_counter=1
        else:
            svt_counter=0
        
        data = {"PAC-Isolated_counter":Isolated,
                "PAC-Bigem_counter":bigeminy_counter,
                "PAC-Trigem_counter":trigeminy_counter,
                "PAC-Quadrigem_counter":quadrigeminy_counter,
                "PAC-Couplet_counter":couplet_counter,
                "PAC-Triplet_counter":triplet_counter,
                "SVT_counter":svt_counter} # svt_counter
        return data

    def predict_pac_model(self, input_arr, target_shape=[224, 224], class_name=True):
        try:
            classes = ['Abnormal', 'Junctional', 'Normal', 'PAC']
            input_arr = tf.keras.preprocessing.image.img_to_array(input_arr)
            input_arr = tf.convert_to_tensor(input_arr, dtype=tf.float32)
            #input_arr = tf.cast(input_arr, dtype=tf.float32)
            #input_arr = tf.convert_to_tensor(input_arr, dtype=tf.float32)
            input_arr = tf.image.resize(input_arr, size=(224, 224), method=tf.image.ResizeMethod.BILINEAR)
            input_arr = (tf.expand_dims(input_arr, axis=0),)
            model_pred = predict_tflite_model(pac_model, input_arr)[0]
            idx = np.argmax(model_pred)
            if class_name:
                idx = np.argmax(model_pred)
                return model_pred, classes[idx]
            else:
                return model_pred
        except Exception as e:
            
            print("PAC ERROR",e)
            return [0,0,0,0],"Normal"
    def get_pac_data(self):
        baseline_signal = baseline_construction_200(self.ecg_signal, kernel_Size=101)
        lowpass_signal = lowpass(baseline_signal, cutoff=0.2)
        r_peaks = self.detect_beats_for_pac(lowpass_signal, self.fs)
        detected_list = []
        pac_index = []
        for i in range(len(r_peaks) - 1):
            fig, ax = plt.subplots(num=1, clear=True)
            segment = lowpass_signal[r_peaks[i]-16:r_peaks[i + 1]+20]
            ax.plot(segment,color='blue')
            ax.axis(False)
            #plt.show()
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            image = Image.fromarray(data)
            resized_image = image.resize((360, 720), Image.LANCZOS)
            plt.close(fig)
            #with tf.device('/GPU:0'):
                
            predictions,ids = self.predict_pac_model(resized_image)
            # print(predictions,ids)
            if str(ids) == "PAC" and float(predictions[3])>0.93: # 0.91
                detected_list.append(1)
                pac_index.append(int(r_peaks[i]))
                pac_index.append(int(r_peaks[i+1]))
                
                label = "PAC"
            else:
                detected_list.append(0)

        pac_data = self.PACcounter(detected_list, self.hr_counts)
        pac_data['PAC_Union'] = detected_list
        pac_data['PAC_Index'] = pac_index
        return pac_data

# long QT detection    
def detection_long_qt(ecg_signal, rpeaks,fs=200):
    print("*********long_qt_func")
    try:
        _, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=fs, method="peak")
        signal_dwt, waves_dwt = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=fs, method="dwt")

        Tpeaks = np.where(np.isnan(waves_peak['ECG_T_Peaks']), 0, waves_peak['ECG_T_Peaks']).astype('int64').tolist()
        Qpeaks = np.where(np.isnan(waves_peak['ECG_Q_Peaks']), 0, waves_peak['ECG_Q_Peaks']).astype('int64').tolist()
        QTint=[]
        finallist=[]

        for i in range(len(Qpeaks)-1):
            try:
                if Qpeaks[i]==0 or Tpeaks[i]==0:
                    QTint.append(0)
                else:
                    QT = abs(int(Qpeaks[i])-int(Tpeaks[i]))/200
                    QTint.append(QT)
                    if QT>0.5: finallist.append(QT)  #0.2
            except:
                QTint.append(0)

        label = "Abnormal"
        if len(finallist)>5:
            print("LONG QT PRESENT",QTint,"\n",finallist)
            label = "Long_QT_Syndrome"
        return label
    except Exception as r:
        return "Abnormal"

# First-deg block detection
def first_degree_detect(ecg_signal, fs=200):
    pqrst_data = pqrst_detection(ecg_signal, fs=fs,  width=(3, 50)).get_data()
    r_index = pqrst_data['R_index']
    q_index = pqrst_data['Q_Index']
    s_index = pqrst_data['S_Index']
    r_Label = pqrst_data['R_Label']
    hr_ = pqrst_data['HR_Count']
    block = []
    label = 'Abnormal'

    if r_Label == 'Regular' and hr_ <= 90:
        for i in range(len(r_index) - 1):
            aoi = ecg_signal[s_index[i]:q_index[i + 1]]
            if aoi.any():
                check, _ = find_peaks(aoi, width=(5, 80), height=(0.02, 0.70), distance=15)
                loc = check + s_index[i]

                if len(check) > 3:
                    peaks1 = np.array([])
                else:
                    if len(check) == 3:
                        sorted_indices = sorted(range(len(check)), key=lambda k: aoi[check[k]], reverse=True)
                        check = [check[sorted_indices[0]], check[sorted_indices[1]]]  # Keep only the top two peaks
                        loc = check + s_index[i]
                    check1 = sorted(loc)
                    if len(check) == 2:
                        dist_next_r_index = r_index[i + 1] - check1[1]
                        if dist_next_r_index >= 50: # 0.3 sec
                            peaks1 = check + s_index[i]
                        else:
                            peaks1 = np.array([])
                    else:
                        peaks1 = np.array([])
            else:
                peaks1 = np.array([])

            if peaks1.any():
                block.extend(list(peaks1))

        if len(r_index) != 0:
            block_per = round(((len(block)/2) / len(r_index)) * 100)
        else:
            block_per = np.array([])
        if block_per > 50:
            label = "1st deg. block"
        else:
            label = 'Abnormal'
    return label, block

# For RR regular or Irregular
def get_percentage_diff(previous, current):
    try:
        percentage = abs(previous - current) / max(previous, current) * 100
    except ZeroDivisionError:
        percentage = float('inf')
    return percentage

def Average(lst):
    return sum(lst) / len(lst)

def new_rr_check(r_index):
    variation = []
    r_label = "Regular"
    for i in range(len(r_index) - 1):
        variation.append(get_percentage_diff(r_index[i + 1], r_index[i]))
    if len(variation) != 0:
        if Average(variation) > 12:
            r_label = "Irregular"

    return r_label

def check_r_irregular(r_index):
    r_label = "Regular"
    mean_percentage_diff = irrgular_per_r = 0
    rpeak_diff = np.diff(r_index)
    if len(rpeak_diff) >= 3:
        percentage_diff = np.abs(np.diff(rpeak_diff) / rpeak_diff[:-1]) * 1003
        list_per_r = [value for value in percentage_diff if value > 14]
        irrgular_per_r = (len(list_per_r) / len(percentage_diff)) * 100
        mean_percentage_diff = np.mean(percentage_diff)

    if (mean_percentage_diff > 75) and (irrgular_per_r > 80):
        r_label = "Irregular"
    return r_label

# Long  & Short Puse detection
def SACompare(list1, val):
    l=[]
    for x in list1:
        if x>=val:
            l.append(1)
        else:
            l.append(0)
    if 1 in l:
        return True
    else:
        return False

def SACompareShort(list1, val1,val2):
    l=[]
    for x in list1:
        if x>=val1 and x<=val2:
            l.append(1)
        else:
            l.append(0)
    if 1 in l:
        return True
    else:
        return False

def check_long_short_pause(r_index):
    SAf= []
    # r_interval = np.diff(r_index)
    pause_label = 'Abnormal'
    if len(r_index) > 1:
        for i in range(len(r_index)-1):
            rr_peaks = abs(int(r_index[i])*5-int(r_index[i+1])*5)
            SAf.append(rr_peaks)

    if (SACompare(SAf, 3000)):
        l=[]
        for x in SAf:
            if x>=3000:
                l.append(1)
            else:
                l.append(0)
        if 1 in l:
            noofpause = l.count(1)
        else:
            noofpause = 0
        if noofpause != 0:
            pause_label = 'LONG_PAUSE'
        
        # "noOfPauseList":[a/1000 for a in SAf if a>3000]


    if SACompareShort(SAf,2000,2900):
        l=[]
        for x in SAf:
            if x>=2000 and x<=2900:
                l.append(1)
            else:
                l.append(0)
        if 1 in l:
            noofpause = l.count(1)
        else:
            noofpause = 0
        if noofpause != 0:
            pause_label = 'SHORT_PAUSE'
        # "noOfPauseList":[a/1000 for a in SAf if a>=2000 and a<=2900 ]
    return pause_label

# All arrhythmia and peak detection functions call
def combine(ecg_signal, is_lead, fs=200):
    baseline_signal, lowpass_signal = filter_signal(ecg_signal, fs).get_data()
    pace_label, pacemaker_index = pacemake_detect(baseline_signal, fs=fs)
        
    pac_data = {
            'PAC_Union': [],
            "PAC_Index":[],
            "PAC_Isolated":0,
            "PAC_Bigeminy":0,
            "PAC_Trigeminy":0,
            "PAC_Quadrigeminy":0,
            "PAC_Couplet":0,
            "PAC_Triplet":0,
            "PAC_SVT":0 }
    
    vfib_or_asystole_output  = vfib_model_check(ecg_signal, baseline_signal, lowpass_signal, vfib_model, fs)
   
    if vfib_or_asystole_output == "Normal":
        
        # PQRST Detection
        pqrst_data = pqrst_detection(baseline_signal, fs=fs).get_data()
        r_label = pqrst_data['R_Label'] 
        r_index = pqrst_data['R_index']
        q_index = pqrst_data['Q_Index']
        s_index = pqrst_data['S_Index']
        j_index = pqrst_data['J_Index']
        p_t = pqrst_data['P_T List']
        if pace_label != 'False':
            temp_list = pacemaker_index
            for sublist in p_t:
                for val in temp_list:
                    if val in sublist:
                        sublist.remove(val)
                        temp_list.remove(val)
                        
        pt = pqrst_data['PT PLot']
        hr_counts = pqrst_data['HR_Count']
        t_index = pqrst_data['T_Index'] 
        p_index = pqrst_data['P_Index']
        ex_index = pqrst_data['Ex_Index'] 
        pr_interval = pqrst_data['PR_Interval'] 
        p_label = pqrst_data['P_Label']
        pr_label = pqrst_data['PR_label']
        r_check_1 = new_rr_check(r_index)
        r_check_2 = check_r_irregular(r_index)
        r_label = "Regular"
        if r_check_1 == 'Irregular' and r_check_2 == 'Irregular':
            r_label = "Irregular"


        afib_label = jr_label = first_deg_block_label = second_deg_block = third_deg_block = aflutter_label = longqt_label = first_degree_block= PAC_label = abs_result = final_block_label = check_pause ='Abnormal'
        temp_index = wide_qrs_list = []
        pvc_class =  []
        pac_class = ['Abnormal']
        pvc_data = {'PVC-Index':[],"PVC-QRS_difference":[],"PVC-wide_qrs":np.array([]),'PVC-model_pred':[]}

        if len(r_index) != 0 or len(s_index) != 0 or len(q_index) != 0:
            if is_lead == 'II' or is_lead == 'III' or is_lead == "aVF" or is_lead == 'V5':
                pvc_data = PVC_detection(ecg_signal, fs).get_pvc_data()

                pvc_count = pvc_data['PVC-Count']

                temp_pvc = []
                for key, val in pvc_data.items():
                    if 'counter' in key and val > 0:
                        temp_pvc.append(key.split('_')[0])
                if len(temp_pvc) != 0:
                    pvc_class = [label.replace('-', '_') for label in temp_pvc]
                else:
                    pvc_class = temp_pvc

            wide_qrs_label, _ = wide_qrs(q_index, r_index, s_index, hr_counts, fs=fs) if len(pvc_class) == 0 else ("Abnormal", [])
            temp_index,wide_qrs_list = wide_qrs_find_pac(q_index, r_index, s_index, hr_counts, fs=fs)
        else:
            wide_qrs_label = 'Abnormal'
        
        if all(p not in ['VT', 'IVR', 'NSVT', 'PVC-Triplet', 'PVC-Couplet'] for p in pvc_class) and len(r_index)>0:
            if hr_counts <= 60:
                check_pause = check_long_short_pause(r_index)
            if r_label == 'Regular':
                if r_label == 'Regular' and (is_lead=='II' or is_lead=='III' or is_lead=="aVF" or is_lead == "V1" or is_lead == "V2" or is_lead =="V6"):
                    jr_label = jr_detection(baseline_signal, jr_model, fs=fs)
                if is_lead == "II" or is_lead == "III" or is_lead == "aVF" or is_lead == "V1": 
                    if all('PVC' not in p for p in pvc_class) and all('Abnormal' in l for l in [afib_label, aflutter_label]):
                        
                        if r_label == "Regular" and hr_counts >= 130:
                            pac_data = PAC_detedction(ecg_signal, fs, hr_counts).get_pac_data()
                            temp_pac = '; '.join([key.split('_')[0] for key, val in pac_data.items() if 'counter' in key and val > 0])

                            pac_class = temp_pac.replace('-', '_')
                        else:
                            pac_class = ""
                            pac_data  = {'Model_Check': [],
                                            'PAC_Union': [], 
                                            'PAC_Index':[]}
                
                if is_lead == "II" or is_lead == "III" or is_lead == "aVF" or is_lead == "V1" or is_lead == "V2":
                    if all('Abnormal' in l for l in [afib_label, aflutter_label]) and len(pac_class) == 0 and len(pvc_class) == 0:
                        lowpass_signal = lowpass(baseline_signal, 0.2)
                        first_deg_block_label, first_deg_block_index = first_degree_detect(lowpass_signal, fs)
                        abs_result = first_deg_block_label

                if hr_counts <= 80 and (is_lead == "II" or is_lead == "III" or is_lead == "aVF" or is_lead == "V1" or is_lead == "V2"):
                    if all('Abnormal' in l for l in [afib_label, aflutter_label, first_deg_block_label, jr_label]):  # and len(pac_class) == 0 and len(pvc_class) == 0
                        second_deg_block = BlockDetected(ecg_signal, fs).second_degree_block_detection()
                        if second_deg_block != 'Abnormal':
                            abs_result = second_deg_block
                    if all('Abnormal' in l for l in
                           [afib_label, aflutter_label, first_deg_block_label, second_deg_block,
                            jr_label]):  
                        third_deg_block = BlockDetected(ecg_signal, fs).third_degree_block_deetection()
                        if third_deg_block != 'Abnormal':
                            abs_result = third_deg_block
                    if abs_result != 'Abnormal':
                        final_block_label, block_percent, block_indexs = block_model_check(ecg_signal, fs, abs_result)
                if all('Abnormal' in l for l in [ afib_label, aflutter_label]) and len(pac_class) == 0 and len(pvc_class) == 0:
                    lowpass_signal = lowpass(baseline_signal, 0.2) 
                    longqt_label = detection_long_qt(lowpass_signal, r_index,fs)
            else:
                if is_lead == 'II' or is_lead == 'III' or is_lead == 'aVF':
                    afib_flutter_check = afib_flutter_detection(lowpass_signal, r_index, q_index, s_index, p_index, p_t, pr_interval,  afib_model)
                    is_afib_flutter = afib_flutter_check.abs_afib_flutter_check()
                    afib_model_per = flutter_model_per = 0
                    if is_afib_flutter:
                        afib_flutter_per, afib_indexs, flutter_indexs = afib_flutter_check.get_data()
                        afib_model_per = int(afib_flutter_per['AFIB']*100)         
                        flutter_model_per = int(afib_flutter_per['FLUTTER']*100) 
                    
                    if afib_model_per >= 60:
                        afib_label = 'AFIB'
                    
                    if afib_label != 'AFIB':
                        if flutter_model_per >= 60:
                            aflutter_label = 'AFL'
                if afib_label != 'AFIB':
                    if is_lead == 'II' or is_lead == 'III' or is_lead == "aVF" :
                        jr_label = jr_detection(baseline_signal, jr_model, fs=fs)
                    
                    if is_lead == "II" or is_lead == "III" or is_lead == "aVF" or is_lead == "V1":
                        if all('PVC' not in p for p in pvc_class) and all('Abnormal' in l for l in [afib_label, aflutter_label, check_pause]) and hr_counts <= 100:
                            pac_data = PAC_detedction(ecg_signal, fs, hr_counts).get_pac_data()
                            temp_pac = '; '.join([key.split('_')[0] for key, val in pac_data.items() if 'counter' in key and val > 0])
                            pac_class = temp_pac.replace('-', '_')
                    if hr_counts<=80 and (is_lead == "II" or is_lead == "III" or is_lead == "aVF" or is_lead == "V1" or is_lead == "V2"):
                        if all('Abnormal' in l for l in[afib_label, aflutter_label, first_deg_block_label,jr_label, check_pause]):  # and len(pac_class) == 0 and len(pvc_class) == 0
                            second_deg_block = BlockDetected(ecg_signal, fs).second_degree_block_detection()
                        if second_deg_block != 'Abnormal':
                            abs_result = second_deg_block
                        if all('Abnormal' in l for l in [afib_label, aflutter_label, first_deg_block_label,second_deg_block,jr_label, check_pause]): # and len(pac_class) == 0 and len(pvc_class) == 0
                            third_deg_block = BlockDetected(ecg_signal, fs).third_degree_block_deetection()
                        if third_deg_block != 'Abnormal':
                            abs_result = third_deg_block
                    if abs_result != 'Abnormal':
                        final_block_label, block_percent, block_indexs = block_model_check(ecg_signal, fs, abs_result)
                
                    
            pac_class = "Abnormal" if pac_class=='' else pac_class
            label = {'Afib_label': afib_label,
                    'Aflutter_label':aflutter_label,
                    'JR_label':jr_label,
                    'wide_qrs_label':wide_qrs_label,
                    'longqt_label': longqt_label,
                    'final_block_label': final_block_label,
                    'check_pause': check_pause,
                    'pac_class': pac_class}
            if pvc_class:
                c_label = "; ".join(pvc_class) + "; " + "; ".join([l for l in label.values() if 'Abnormal' not in l])
            else:
                c_label = "; ".join([l for l in label.values() if 'Abnormal' not in l])
        else:
            c_label = "; ".join(pvc_class)
            
        c_label = c_label + f"; {pace_label}" if pace_label != "False" else c_label
            
        if c_label in ["", "; "]: c_label = 'Normal' if r_label == 'Regular' else 'Sinus-arr'

        data = {'Input_Signal':ecg_signal, 
                'Baseline_Signal':baseline_signal, 
                'Lowpass_signal':lowpass_signal, 
                # 'Combine_Label':c_label.upper().replace("_","-"), 
                'Combine_Label':c_label, 
                'RR_Label':r_label, 
                'R_Index':r_index, 
                'Q_Index':q_index, 
                'S_Index':s_index, 
                'J_Index':j_index,
                'T_Index' : t_index,
                'P_Index' : p_index,
                'Ex_Index' : ex_index, 
                'P_T':pt, 
                'HR_Count':hr_counts, 
                'PVC_DATA':pvc_data,
                'PAC_DATA':pac_data,
                'PaceMaker':pace_label,}
    else:
        data = {'Input_Signal':ecg_signal, 
                'Baseline_Signal':baseline_signal, 
                'Lowpass_signal':lowpass(baseline_signal, 0.3), 
                'Combine_Label':vfib_or_asystole_output.upper(), 
                'RR_Label':'Not Defined', 
                'R_Index':np.array([]), 
                'Q_Index':[], 
                'S_Index':[], 
                'J_Index':[],
                'T_Index' : [],
                'P_Index' : [],
                'Ex_Index' : [], 
                'P_T':[], 
                'HR_Count':0, 
                'PVC_DATA':{'PVC-Index':[],"PVC-QRS_difference":[],"PVC-wide_qrs":np.array([]),'PVC-model_pred':[]},
                'PAC_DATA':pac_data,
                'PaceMaker':pace_label,}

    return data


# R peak detection using biosppy
class RPeakDetection:
    def __init__(self, baseline_signal, fs = 200):
        self.baseline_signal = baseline_signal
        self.fs = fs
        

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    def find_r_peak(self):
        lowcut = 0.5
        highcut = 50.0
        filtered_signal = self.butter_bandpass_filter(self.baseline_signal, lowcut, highcut, self.fs, order=6)
        # Step 3: R-Peak Detection
        out = hami.hamilton_segmenter(filtered_signal, sampling_rate=self.fs)
        rpeaks = hami.correct_rpeaks(filtered_signal, out[0], sampling_rate=self.fs, tol=0.1)
        r_peaks = rpeaks[0].tolist()
        return r_peaks

def find_new_q_index(ecg, R_index, d):
    q = []
    for i in R_index:
        q_ = []
        if i == 0:
            q.append(i)
            continue
        if ecg[i] > 0:
            c = i
            while c > 0 and ecg[c-1] < ecg[c]:
                c -= 1                  
            if ecg[i] * 0.01 > ecg[c] or ecg[c] < 0 or c == 0:
                if abs(i-c) <= d:
                    q.append(c)
                    continue
                else:
                    q_.append(c)
            while c > 0:
                while c > 0 and ecg[c-1] > ecg[c]:
                    c -= 1
                # q_.append(c)
                while c > 0 and ecg[c-1] < ecg[c]:
                    c -= 1
                if q_ and q_[-1] == c:
                    break
                q_.append(c)
                if ecg[i] * 0.01 > ecg[c] or ecg[c] < 0 or c == 0:
                    break
        else:
            c = i
            while c > 0 and ecg[c-1] > ecg[c]:
                c -= 1
            if ecg[i] * 0.01 < ecg[c] or ecg[c] > 0 or c == 0:
                if abs(i-c) <= d:
                    q.append(c)
                    continue
                else:
                    q_.append(c)
            while c > 0:
                while c > 0 and ecg[c-1] < ecg[c]:
                    c -= 1
                # q_.append(c)
                while c > 0 and ecg[c-1] > ecg[c]:
                    c -= 1
                if q_ and q_[-1] == c:
                    break
                q_.append(c)
                if ecg[i] * 0.01 < ecg[c] or ecg[c] > 0 or c == 0:
                    break
        if q_:
            a = 0
            for _q in q_[::-1]:
                if abs(i-_q) <= d:
                    a = 1
                    q.append(_q)
                    break
            if a == 0:
                q.append(q_[0])
    return np.sort(q)

def st_model_predict(input_arr):
    classes = ['Abnormal', 'Normal', 'Nstemi', 'Stemi', 'T_abnormal']
    input_arr = tf.cast(input_arr, dtype=tf.float32)
    input_arr = tf.image.resize(input_arr, size=(224, 224), method=tf.image.ResizeMethod.BILINEAR)
    input_arr = (tf.expand_dims(input_arr, axis=0),)
    model_pred = predict_tflite_model(st_t_abn_model, input_arr)[0]

    idx = np.argmax(model_pred)
    return classes[idx]

def st_t_abn_model_check(baseline_signal, lowpass_signal, fs):
    st_label = 'Abnormal'
    try:
        r_index = RPeakDetection(baseline_signal, fs).find_r_peak()
        qdiff = int(fs * 0.08)
        q_index = find_new_q_index(baseline_signal, r_index, qdiff)
        percentage = {'Abnormal': 0, 'Normal': 0, 'Nstemi': 0, 'Stemi': 0, 'T_abnormal':0}
        result , stemi_idx , nstemi_idx , t_abno_idx = [], [], [], []
        total_q_ind = len(q_index)-1
        st_label = 'Abnormal'
        for ind in range(len(q_index) -1):
            temp_data = pd.DataFrame(lowpass_signal[q_index[ind]-10: q_index[ind+1]-50])
            fig, ax = plt.subplots(num=1,clear=True,figsize=(2,2),dpi=56,layout ="constrained")
            ax.plot(temp_data, 'b')
            ax.axis(False)
            # fig.savefig("temp_st_beat.jpg")
            # r_img = cv2.imread("temp_st_beat.jpg")
            # cv2.imwrite("temp_st_beat.jpg", r_img)
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            data1 = data[:, :, ::-1]
            plt.close(fig)
            model_label = st_model_predict(data1)
            if model_label == 'Stemi':
                stemi_idx.append((q_index[ind]-10, q_index[ind+1]-50))
                percentage['Stemi'] += 1
            elif model_label == 'Nstemi':
                nstemi_idx.append((q_index[ind]-10, q_index[ind+1]-50))
                percentage['Nstemi'] += 1
            elif model_label == 'T_abnormal':
                t_abno_idx.append((q_index[ind]-10, q_index[ind+1]-50))
                percentage['T_abnormal'] += 1
            result.append(model_label)
        if int(percentage['Stemi']/ total_q_ind * 100) > 55:
            st_label = 'STEMI'
        elif int(percentage['Nstemi']/ total_q_ind * 100) > 55:
            st_label = 'NSTEMI'
        elif int(percentage['T_abnormal']/ total_q_ind * 100) > 55:
            st_label = 'T_wave_Abnormality'
        return st_label
    except:
        return st_label

def inf_lat_model_check(low_ecg_signal, r_index , lead): # input lowpass signal
    inf_lat_label = 'Abnormal'
    try:
        percent = {'ABNORMAL': 0, 'Inferior': 0, 'Lateral': 0}
        inferior_leads = ['II', 'III', 'aVF']
        lateral_leads = ['I', 'aVL', 'v5']

        inferior_data_index, lateral_data_index = [], []
        total_r_index = len(r_index)-1
        inf_lat_label = 'Abnormal'
        if len(r_index) > 1:
            for idx in range(len(r_index)-1):
                data = low_ecg_signal[r_index[idx]-15: r_index[idx]+170]
                if data.any():
                    image_data = image_array_new(data)
                    image_data = (tf.expand_dims(image_data.astype(np.float32), axis=0),)
                    model_pred = predict_tflite_model(let_inf_moedel, image_data)[0]

                    model_idx = np.argmax(model_pred)
                    if model_idx == 0:
                        percent['ABNORMAL'] += 1
                    elif model_idx == 1:
                        percent['Inferior'] += 1
                        inferior_data_index.append((r_index[idx]-15,r_index[idx]+170))
                    elif model_idx == 2:
                        percent['Lateral'] += 1
                        lateral_data_index.append((r_index[idx]-15,r_index[idx]+170))
        if int(percent['Inferior'] / total_r_index* 100) > 55 and lead in inferior_leads:
            inf_lat_label = 'Inferior_MI'
        elif int(percent['Lateral'] / total_r_index* 100) > 55 and lead in lateral_leads:
            inf_lat_label = 'Lateral_MI'
        return inf_lat_label #, inferior_data_index, lateral_data_index
    except:
        return inf_lat_label

def find_ecg_info(ecg_signal):
    fa = 200
    rpeaks = detect_beats(ecg_signal, float(fa))
    rr_interval = []
    #data_dic = {}
    data_dic = {"rr_interval": 0,
                "PRInterval": 0,
                "QTInterval": 0,
                "QRSComplex": 0,
                "STseg": 0,
                "PRseg": 0,
                "QTc": 0}
    if len(rpeaks) > 1:
        try:
            # print()
            _, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=fa, method="peak")
            signal_dwt, waves_dwt = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=fa, method="dwt")
            for i in range(len(rpeaks)):
                try:
                    RRpeaks = abs(int(rpeaks[i])*5-int(rpeaks[i+1])*5)
                    rr_interval.append(RRpeaks)
                except:
                    rr_interval.append(0)
                    RRpeaks="0"
            data_dic['rr_interval'] = rr_interval[0]
            try:
                Ppeak = waves_peak['ECG_P_Peaks'][2]
                Rpeak = rpeaks[2]
                Ppeak = int(Ppeak)*5
                Rpeak = int(Rpeak)*5
                PRpeaks = abs(Rpeak-Ppeak)
            except:
                PRpeaks = "0"
            data_dic['PRInterval'] = PRpeaks
            try:
                Tpeak = waves_peak['ECG_T_Peaks'][2]
                Qpeak = waves_peak['ECG_Q_Peaks'][2]
                Tpeak = int(Tpeak)*5
                Qpeak = int(Qpeak)*5
                QTpeaks = abs(Tpeak-Qpeak)
            except:
                QTpeaks="0"
            data_dic['QTInterval'] =  QTpeaks
            try:
                Speak = waves_peak['ECG_S_Peaks'][2]
                Qpeak = waves_peak['ECG_Q_Peaks'][2]
                Speak = int(Speak)*5
                Qpeak = int(Qpeak)*5
                SQpeaks = abs(Speak-Qpeak)
            except:
                SQpeaks = "0"
            data_dic['QRSComplex'] =  SQpeaks
            try:
                Spa = waves_peak['ECG_S_Peaks'][2]
                Ton = waves_dwt['ECG_T_Onsets'][2]
                Spa = int(Spa)*5
                Ton = int(Ton)*5
                STseg = abs(Ton-Spa)
            except:
                STseg = "0"
            data_dic['STseg'] =  STseg    
            try:
                PP = waves_dwt['ECG_P_Offsets']
                RRO = waves_dwt['ECG_R_Onsets']
                if math.isnan(PP[2]) or math.isnan(RRO[2]):
                    PRseg = "0"
                else:
                    PPIn = int(PP[2])*5
                    RRon = int(RRO[2])*5
                    PRseg =  abs(PPIn - RRon)                                    
            except:
                PRseg = "0"
            data_dic['PRseg'] = PRseg

            Tpeaks = np.where(np.isnan(waves_peak['ECG_T_Peaks']), 0, waves_peak['ECG_T_Peaks']).astype('int64').tolist()
            Qpeaks = np.where(np.isnan(waves_peak['ECG_Q_Peaks']), 0, waves_peak['ECG_Q_Peaks']).astype('int64').tolist()
            QTint=[]
            finallist=[]

            for i in range(len(Qpeaks)-1):
                try:
                    if Qpeaks[i]==0 or Tpeaks[i]==0:
                        QTint.append(0)
                    else:
                        QT = abs(int(Qpeaks[i])-int(Tpeaks[i]))/200
                        QTint.append(QT*1000)
                        # if QT>0.5: finallist.append(QT)  #0.2
                except:
                    QTint.append(0)
            data_dic['QTc'] = QTint[0]
            return data_dic
        except Exception as e:
            # print(e,": error for find_ecg_info")
            print('\033[93m'+f"Error : {e} on line_no:  {e.__traceback__.tb_lineno}"+ '\033[0m')
            return data_dic

def detect_beats(
        baseline_signal,  # The raw ECG signal
        fs,  # Sampling fs in HZ
        # Window size in seconds to use for
        ransac_window_size=3.0, #5.0
        # Low frequency of the band pass filter
        lowfreq=5.0,
        # High frequency of the band pass filter
        highfreq=7.0,
    ):
    ransac_window_size = int(ransac_window_size * fs)
 
    lowpass = scipy.signal.butter(1, highfreq / (fs / 2.0), 'low')
    highpass = scipy.signal.butter(1, lowfreq / (fs / 2.0), 'high')
    # TODO: Could use an actual bandpass filter
    ecg_low = scipy.signal.filtfilt(*lowpass, x=baseline_signal)
    ecg_band = scipy.signal.filtfilt(*highpass, x=ecg_low)
 
    # Square (=signal power) of the first difference of the signal
    decg = np.diff(ecg_band)
    decg_power = decg ** 2
 
    # Robust threshold and normalizator estimation
    thresholds = []
    max_powers = []
    for i in range(int(len(decg_power) / ransac_window_size)):
        sample = slice(i * ransac_window_size, (i + 1) * ransac_window_size)
        d = decg_power[sample]
        thresholds.append(0.5 * np.std(d))
        max_powers.append(np.max(d))
 
    threshold = np.median(thresholds)
    max_power = np.median(max_powers)
    decg_power[decg_power < threshold] = 0
 
    decg_power /= max_power
    decg_power[decg_power > 1.0] = 1.0
    square_decg_power = decg_power ** 4
    # square_decg_power = decg_power**4
 
    shannon_energy = -square_decg_power * np.log(square_decg_power)
    shannon_energy[~np.isfinite(shannon_energy)] = 0.0
 
    mean_window_len = int(fs * 0.125 + 1)
    lp_energy = np.convolve(shannon_energy, [1.0 / mean_window_len] * mean_window_len, mode='same')
    # lp_energy = scipy.signal.filtfilt(*lowpass2, x=shannon_energy)
 
    lp_energy = scipy.ndimage.gaussian_filter1d(lp_energy, fs / 14.0) # 20.0
    # lp_energy = scipy.ndimage.gaussian_filter1d(lp_energy, fs/8.0)
    lp_energy_diff = np.diff(lp_energy)
 
    zero_crossings = (lp_energy_diff[:-1] > 0) & (lp_energy_diff[1:] < 0)
    zero_crossings = np.flatnonzero(zero_crossings)
    zero_crossings -= 1
    return zero_crossings

class arrhythmia_detection:
    def __init__(self, pd_data:pd.DataFrame, fs:int, name:str):
        self.all_leads_data = pd_data
        self.fs = fs
        self.f_name = name

    def find_repeated_elements(self, nested_list, test_for='Arrhythmia'):
        flat_list = []
        if test_for == 'Arrhythmia':
            threshold= 3
        else:
            threshold= 3
        for element in nested_list:
            if isinstance(element, list):
                flat_list.extend(element)
            else:
                flat_list.append(element)
        
        counts = Counter(flat_list)
        repeated_elements = [item for item, count in counts.items() if count > threshold]
        return repeated_elements

    def ecg_signal_processing(self):
        self.leads_pqrst_data = {"avg_hr" : 0,
            "arr_final_result": 'Normal',
            "mi_final_result": 'Abnormal',
            "beats": 0,
            "RRInterval": 0,
            "PRInterval": 0,
            "QTInterval": 0,
            "QRSComplex": 0,
            "STseg": 0,
            "PRseg": 0,
            "QTc": 0,
            "pvcQrs": [],
            "pacQrs": [],
            "Vbeat": 0,
            "Abeat": 0,
            }
        arr_final_result = mi_final_result = 'Abnormal'
        try:
            for lead in self.all_leads_data.columns:
                lead_data = {}
                let_inf_label = 'Abnormal'
                st_t_abn_label = 'Abnormal'
                mi_data = {}
                ecg_signal = self.all_leads_data[lead].values
                if ecg_signal.any():
                    arrhythmia_result = combine(ecg_signal, lead, self.fs)
                    baseline_signal = arrhythmia_result['Baseline_Signal']
                    lowpass_signal = arrhythmia_result['Lowpass_signal']
                    r_index = arrhythmia_result['R_Index']
                    st_t_abn_label = st_t_abn_model_check(baseline_signal, lowpass_signal, self.fs)
                    if lead in ['II', 'III', 'aVF', 'I', 'aVL', 'v5'] and (st_t_abn_label == 'NSTEMI' or st_t_abn_label == 'STEMI'):
                        let_inf_label = inf_lat_model_check(lowpass_signal, r_index, lead)
                    print(f"{lead} : {arrhythmia_result['Combine_Label']}")
                    lead_data['arrhythmia_data'] = arrhythmia_result

                    lbbb_rbbb_data = PVC_detection(ecg_signal, self.fs).get_pvc_data()
                    if lbbb_rbbb_data['lbbb_rbbb_label'] != 'Abnormal':
                        mi_data['lbbb_rbbb_label'] = lbbb_rbbb_data['lbbb_rbbb_label']
                        lead_data['mi_data'] = mi_data
                    # if st_t_abn_label != 'Abnormal':
                    #     mi_data['st_t_abn_label'] = st_t_abn_label
                    #     lead_data['mi_data'] = mi_data
                    if let_inf_label != 'Abnormal':
                        mi_data['let_inf_label'] = let_inf_label
                        lead_data['mi_data'] = mi_data
                    self.leads_pqrst_data[lead] = lead_data
            if (self.leads_pqrst_data) != 0:
                st_t_abn_mi_label, let_inf_mi_label, lb_rb_mi_label, comm_arrhy_label, all_lead_hr = [], [], [], [], []
                for lead in self.all_leads_data.keys():
                    comm_arrhy_label.append(self.leads_pqrst_data[lead]['arrhythmia_data']['Combine_Label'].split(' '))
                    if self.leads_pqrst_data[lead]['arrhythmia_data']['HR_Count'] > 40:
                        all_lead_hr.append(self.leads_pqrst_data[lead]['arrhythmia_data']['HR_Count'])
                
                if len(all_lead_hr) != 0:
                    total_hr = int(sum(all_lead_hr)/len(all_lead_hr))
                else:
                    temp_lead = next(iter(self.leads_pqrst_data))
                    total_hr = self.leads_pqrst_data[temp_lead]['arrhythmia_data']['HR_Count']
                
                mod_comm_arrhy = [[item.replace(';', '') for item in sublist if item] for sublist in comm_arrhy_label]
                all_arrhy_result = self.find_repeated_elements(mod_comm_arrhy, test_for="Arrhythmia")
                if len(all_arrhy_result) > 1 and 'Normal' in all_arrhy_result:
                    all_arrhy_result.remove('Normal')

                arr_final_result = ' '.join(all_arrhy_result)
                
                for lead in self.all_leads_data.keys():
                    # if 'mi_data' in self.leads_pqrst_data[lead]:
                    #     st_t_abn_mi_label.append(self.leads_pqrst_data[lead]['mi_data']['st_t_abn_label'])
                    if 'mi_data' in self.leads_pqrst_data[lead] and 'let_inf_label' in self.leads_pqrst_data[lead]['mi_data']:
                        let_inf_mi_label.append(self.leads_pqrst_data[lead]['mi_data']['let_inf_label'])
                    if 'mi_data' in self.leads_pqrst_data[lead] and 'lbbb_rbbb_label' in self.leads_pqrst_data[lead]['mi_data']:
                        lb_rb_mi_label.append(self.leads_pqrst_data[lead]['mi_data']['lbbb_rbbb_label'])
                
                if len(lb_rb_mi_label) != 0:
                    check_lb_rb_label = self.find_repeated_elements(lb_rb_mi_label, test_for="mi")
                    if len(check_lb_rb_label) != 0:
                        mi_final_result = ' '.join(check_lb_rb_label)
                    
                # if len(st_t_abn_mi_label) != 0: 
                #     check_st_label = self.find_repeated_elements(st_t_abn_mi_label, test_for="mi")
                #     if len(check_st_label) != 0:
                #         mi_final_result = ' '.join(check_st_label)
                if len(let_inf_mi_label) != 0:
                    check_inf_label = self.find_repeated_elements(let_inf_mi_label, test_for="mi")
                    if check_inf_label != 'Abnormal':
                        temp_list = check_inf_label
                        mi_final_result = ' '.join(temp_list)
            if len(arr_final_result) == 0:
                if self.leads_pqrst_data['II']['arrhythmia_data']['RR_Label'] == 'Regular':
                    arr_final_result = 'Normal'
                    if total_hr<60:
                        arr_final_result="BR"
                    if total_hr>100:
                        arr_final_result="TC"
                        
                else:
                    arr_final_result = 'SINUS_ARR'
            if "II" in self.all_leads_data.keys():
                get_temp_lead = 'II'
                get_pro_lead = self.all_leads_data["II"]
            else:
                get_temp_lead = next(iter(self.all_leads_data))
                get_pro_lead = self.all_leads_data[get_temp_lead]

            detections = []
            if len(all_arrhy_result) > 1:
                for lab in all_arrhy_result:
                    detections.append({"detect": lab, "detectType": "Arrhythmia", "confidence":100})
            else:
                detections.append({"detect": arr_final_result, "detectType": "Arrhythmia", "confidence":100})

            if mi_final_result != 'Abnormal':
                detections.append({"detect": mi_final_result, "detectType": "MI", "confidence":100})

            total_pac = self.leads_pqrst_data[get_temp_lead]['arrhythmia_data']['PAC_DATA']['PAC_Union']
            lead_info_data = find_ecg_info(get_pro_lead)
            self.leads_pqrst_data['avg_hr'] = total_hr
            self.leads_pqrst_data['arr_final_result'] = arr_final_result 
            self.leads_pqrst_data['mi_final_result'] = mi_final_result
            self.leads_pqrst_data['detections'] = detections
            self.leads_pqrst_data['beats'] = len(self.leads_pqrst_data[get_temp_lead]['arrhythmia_data']['R_Index'])
            self.leads_pqrst_data['RRInterval'] =lead_info_data['rr_interval']
            self.leads_pqrst_data['PRInterval'] =lead_info_data['PRInterval']
            self.leads_pqrst_data['QTInterval'] =lead_info_data['QTInterval']
            self.leads_pqrst_data['QRSComplex'] =lead_info_data['QRSComplex']
            self.leads_pqrst_data['STseg'] =lead_info_data['STseg']
            self.leads_pqrst_data['PRseg'] =lead_info_data['PRseg']
            self.leads_pqrst_data['QTc'] =lead_info_data['QTc']
            self.leads_pqrst_data['pvcQrs'] = self.leads_pqrst_data[get_temp_lead]['arrhythmia_data']['PVC_DATA']['PVC-Index']
            self.leads_pqrst_data['Vbeat'] = self.leads_pqrst_data[get_temp_lead]['arrhythmia_data']['PVC_DATA']['PVC-Count']
            self.leads_pqrst_data['pacQrs'] = self.leads_pqrst_data[get_temp_lead]['arrhythmia_data']['PAC_DATA']['PAC_Index']
            self.leads_pqrst_data['Abeat'] = total_pac.count(1) if len(total_pac) != 0 else 0
            return self.leads_pqrst_data
        except Exception as e:
            print('\033[93m'+f"Arrhythmia detection Error : {e} on line_no:  {e.__traceback__.tb_lineno}"+ '\033[0m')
            return self.leads_pqrst_data

def check_noise(all_leads_data, fs):
    noise_result = []
    final_result = 'Normal'
    for lead in all_leads_data.columns:
        lead_data = {}
        ecg_signal = all_leads_data[lead].values
        get_noise = NoiseDetection(ecg_signal, frequency=fs).noise_model_check()
        noise_result.append(get_noise)
    noise_cou = noise_result.count('ARTIFACTS')
    if noise_cou > 5:
        final_result = 'ARTIFACTS'
    return final_result
 

# Function to recognize lead text
def recognize_lead(text):
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    text = text.strip().upper().replace(' ', '')
    for lead in lead_names:
        if lead.upper().replace(' ', '') == text:
            return lead
    return None

def extract_ecg_signal(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_img, (5, 5), 0)
    _, binary_image = cv2.threshold(blurred_image, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary_image)
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    signal = cv2.bitwise_and(gray_img, gray_img, mask=mask)

    voltage_values = [np.mean(np.where(col > 0)[0]) if np.sum(col) > 0 else 0 for col in signal.T]
    voltage_values = np.max(voltage_values) - voltage_values
    voltage_normalized = (voltage_values - np.min(voltage_values)) / (np.max(voltage_values) - np.min(voltage_values))
    
    ecg_signal = MinMaxScaler(feature_range=(0,4)).fit_transform(voltage_normalized.reshape(-1,1)).squeeze()
    return ecg_signal

def resting_ecg_extractor(image_path):
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    left_side_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF']
    right_side_leads = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']


    fixed_left_positions = {
        'I': (100, 240),
        'II': (100, 350),
        'III': (100, 460),
        'aVR': (100, 570),
        'aVL': (100, 680),
        'aVF': (100, 790)
    }

    fixed_right_positions = {
        'V1': (850, 240),
        'V2': (850, 350),
        'V3': (850, 460),
        'V4': (850, 570),
        'V5': (850, 680),
        'V6': (850, 790)
    }
    ecg_data = {}
    try:
        # Read the image
        image = cv2.imread(image_path)
        all_lead_data = {}
        if image is None:
            print(f"Error: Unable to load image at {image_path}")
            return
        else:
            crop_start_y = 350  # Top crop value
            crop_end_y = -200   # Bottom crop value
            cropped_image = image[crop_start_y:crop_end_y, :]

            # Resize the cropped image to 1800x1000 pixels
            resized_image = cv2.resize(cropped_image, (1800, 1000))

            # Convert the image to grayscale
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

            # Invert the colors (white to black and black to white)
            inverted_image = cv2.bitwise_not(gray_image)

            num_sections = 20
            section_height = resized_image.shape[0] // num_sections
            section_width = resized_image.shape[1] // num_sections

            recognized_leads = set()
            lead_positions = {}

            # Loop through each section
            for row in range(num_sections):
                for col in range(num_sections):
                    # Define the coordinates of the section
                    y1 = row * section_height
                    y2 = y1 + section_height
                    x1 = col * section_width
                    x2 = x1 + section_width

                    # Extract the section from the inverted image
                    section = inverted_image[y1:y2, x1:x2]

                    # Perform OCR on the section using EasyOCR
                    easyocr_result = reader.readtext(section)

                    # Perform OCR on the section using Pytesseract
                    pytesseract_result = pytesseract.image_to_data(section, output_type=pytesseract.Output.DICT)

                    # Combine results from EasyOCR and Pytesseract
                    for bbox, text, _ in easyocr_result:
                        lead_name = recognize_lead(text)
                        if lead_name and lead_name not in recognized_leads and lead_name != 'I':
                            bbox_abs = [(int(bbox[0][0] + x1), int(bbox[0][1] + y1)), (int(bbox[1][0] + x1), int(bbox[1][1] + y1))]
                            cv2.rectangle(resized_image, bbox_abs[0], bbox_abs[1], (0, 255, 0), 2)
                            cv2.putText(resized_image, lead_name, (bbox_abs[0][0], bbox_abs[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            box_top_left = (int(bbox_abs[0][0]), int(bbox_abs[0][1]))
                            box_bottom_right = (int(bbox_abs[0][0] + 750), int(bbox_abs[0][1] + 110))
                            cv2.rectangle(resized_image, box_top_left, box_bottom_right, (0, 255, 0), 2)
                            lead_data = resized_image[box_top_left[1]:box_bottom_right[1], box_top_left[0]:box_bottom_right[0]]
                            if np.mean(lead_data) > 10:
                                lead_voltage_data = extract_ecg_signal(lead_data)
                                all_lead_data[lead_name] = lead_voltage_data

                            recognized_leads.add(lead_name)
                            lead_positions[lead_name] = bbox_abs[0]  # Store the position of each detected lead

                    # Process results from Pytesseract
                    for i in range(len(pytesseract_result['text'])):
                        if int(pytesseract_result['conf'][i]) > 0:  # Only consider confident results
                            text = pytesseract_result['text'][i]
                            lead_name = recognize_lead(text)
                            if lead_name and lead_name not in recognized_leads and lead_name != 'I':
                                bbox = [(pytesseract_result['left'][i] + x1, pytesseract_result['top'][i] + y1),
                                        (pytesseract_result['left'][i] + pytesseract_result['width'][i] + x1,
                                        pytesseract_result['top'][i] + pytesseract_result['height'][i] + y1)]
                                cv2.rectangle(resized_image, bbox[0], bbox[1], (0, 255, 0), 2)
                                cv2.putText(resized_image, lead_name, (bbox[0][0], bbox[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                box_top_left = (int(bbox[0][0]), int(bbox[0][1]))
                                box_bottom_right = (int(bbox[0][0] + 750), int(bbox[0][1] + 110))
                                cv2.rectangle(resized_image, box_top_left, box_bottom_right, (0, 255, 0), 2)
                                lead_data = resized_image[box_top_left[1]:box_bottom_right[1], box_top_left[0]:box_bottom_right[0]]
                                if np.mean(lead_data) > 10:
                                    lead_voltage_data = extract_ecg_signal(lead_data)
                                    all_lead_data[lead_name] = lead_voltage_data 
                                recognized_leads.add(lead_name)
                                lead_positions[lead_name] = bbox[0]  # Store the position of each detected lead

            # Print detected leads and their positions
            for lead in recognized_leads:
                side = "Left" if lead in left_side_leads else "Right"
                print(f"Detected {lead} on {side} side at position {lead_positions[lead]}")

            # Estimate and annotate missing left-side leads
            detected_left_leads = sorted((lead for lead in recognized_leads if lead in left_side_leads and lead != 'I'), key=left_side_leads.index)
            if not detected_left_leads:
                print("No left-side leads detected. Using fixed positions.")
                for lead, position in fixed_left_positions.items():
                    (text_width, text_height), _ = cv2.getTextSize(lead, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    margin = 10
                    bbox = [(position[0] - margin, position[1] - text_height - margin),
                            (position[0] + text_width + margin, position[1] + margin)]
                    cv2.rectangle(resized_image, bbox[0], bbox[1], (255, 0, 0), 2)
                    cv2.putText(resized_image, lead, (position[0], position[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    box_top_left = (int(position[0]), int(position[1]))
                    box_bottom_right = (int(position[0] + 750), int(position[1] + 110))
                    cv2.rectangle(resized_image, box_top_left, box_bottom_right, (0, 255, 0), 2)
                    lead_data = resized_image[box_top_left[1]:box_bottom_right[1], box_top_left[0]:box_bottom_right[0]]
                    if np.mean(lead_data) > 10:
                        lead_voltage_data = extract_ecg_signal(lead_data)
                        all_lead_data[lead] = lead_voltage_data
                    lead_positions[lead] = position
                    recognized_leads.add(lead)
            else:
                reference_lead = detected_left_leads[0]
                reference_position = lead_positions[reference_lead]
                reference_index = left_side_leads.index(reference_lead)

                for index, lead in enumerate(left_side_leads):
                    if lead == 'I' and 'II' in recognized_leads:
                        # Position lead I above lead II
                        lead_II_position = lead_positions['II']
                        estimated_position = (lead_II_position[0], lead_II_position[1] - 110)
                        (text_width, text_height), _ = cv2.getTextSize('I', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        margin = 10
                        bbox = [(estimated_position[0] - margin, estimated_position[1] - text_height - margin),
                                (estimated_position[0] + text_width + margin, estimated_position[1] + margin)]
                        cv2.rectangle(resized_image, bbox[0], bbox[1], (255, 0, 0), 2)
                        cv2.putText(resized_image, 'I', (estimated_position[0], estimated_position[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        box_top_left = (int(estimated_position[0]), int(estimated_position[1]))
                        box_bottom_right = (int(estimated_position[0] + 750), int(estimated_position[1] + 110))
                        cv2.rectangle(resized_image, box_top_left, box_bottom_right, (0, 255, 0), 2)
                        lead_data = resized_image[box_top_left[1]:box_bottom_right[1], box_top_left[0]:box_bottom_right[0]]
                        if np.mean(lead_data) > 10:
                            lead_voltage_data = extract_ecg_signal(lead_data)
                            all_lead_data['I'] = lead_voltage_data
                        lead_positions['I'] = estimated_position
                        recognized_leads.add('I')
                    elif lead not in recognized_leads:
                        position_difference = (index - reference_index) * 110
                        estimated_position = (reference_position[0], reference_position[1] + position_difference)
                        (text_width, text_height), _ = cv2.getTextSize(lead, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        margin = 10
                        bbox = [(estimated_position[0] - margin, estimated_position[1] - text_height - margin),
                                (estimated_position[0] + text_width + margin, estimated_position[1] + margin)]
                        cv2.rectangle(resized_image, bbox[0], bbox[1], (255, 0, 0), 2)
                        cv2.putText(resized_image, lead, (estimated_position[0], estimated_position[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        box_top_left = (int(estimated_position[0]), int(estimated_position[1]))
                        box_bottom_right = (int(estimated_position[0] + 750), int(estimated_position[1] + 110))
                        cv2.rectangle(resized_image, box_top_left, box_bottom_right, (0, 255, 0), 2)
                        lead_data = resized_image[box_top_left[1]:box_bottom_right[1], box_top_left[0]:box_bottom_right[0]]
                        if np.mean(lead_data) > 10:
                            lead_voltage_data = extract_ecg_signal(lead_data)
                            all_lead_data[lead] = lead_voltage_data
                        
                        lead_positions[lead] = estimated_position
                        recognized_leads.add(lead)

            # Estimate and annotate missing right-side leads
            detected_right_leads = sorted((lead for lead in recognized_leads if lead in right_side_leads), key=right_side_leads.index)
            if not detected_right_leads:
                print("No right-side leads detected. Using fixed positions.")
                for lead, position in fixed_right_positions.items():
                    (text_width, text_height), _ = cv2.getTextSize(lead, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    margin = 10
                    bbox = [(position[0] - margin, position[1] - text_height - margin),
                            (position[0] + text_width + margin, position[1] + margin)]
                    cv2.rectangle(resized_image, bbox[0], bbox[1], (255, 0, 0), 2)
                    cv2.putText(resized_image, lead, (position[0], position[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    box_top_left = (int(position[0]), int(position[1]))
                    box_bottom_right = (int(position[0] + 750), int(position[1] + 110))
                    cv2.rectangle(resized_image, box_top_left, box_bottom_right, (0, 255, 0), 2)
                    lead_data = resized_image[box_top_left[1]:box_bottom_right[1], box_top_left[0]:box_bottom_right[0]]
                    if np.mean(lead_data) > 10:
                        lead_voltage_data = extract_ecg_signal(lead_data)
                        all_lead_data[lead] = lead_voltage_data
                    lead_positions[lead] = position
                    recognized_leads.add(lead)
            else:
                reference_lead = detected_right_leads[0]
                reference_position = lead_positions[reference_lead]
                reference_index = right_side_leads.index(reference_lead)

                for index, lead in enumerate(right_side_leads):
                    if lead not in recognized_leads:
                        position_difference = (index - reference_index) * 110
                        estimated_position = (reference_position[0], reference_position[1] + position_difference)
                        (text_width, text_height), _ = cv2.getTextSize(lead, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        margin = 10
                        bbox = [(estimated_position[0] - margin, estimated_position[1] - text_height - margin),
                                (estimated_position[0] + text_width + margin, estimated_position[1] + margin)]
                        cv2.rectangle(resized_image, bbox[0], bbox[1], (255, 0, 0), 2)
                        cv2.putText(resized_image, lead, (estimated_position[0], estimated_position[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        box_top_left = (int(estimated_position[0]), int(estimated_position[1]))
                        box_bottom_right = (int(estimated_position[0] + 750), int(estimated_position[1] + 110))
                        cv2.rectangle(resized_image, box_top_left, box_bottom_right, (0, 255, 0), 2)
                        lead_data = resized_image[box_top_left[1]:box_bottom_right[1], box_top_left[0]:box_bottom_right[0]]
                        if np.mean(lead_data) > 10:
                            lead_voltage_data = extract_ecg_signal(lead_data)
                            all_lead_data[lead] = lead_voltage_data
                        
                        lead_positions[lead] = estimated_position
                        recognized_leads.add(lead)

            desired_keys = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            new_final_voltage_dic = {}
            for key in  desired_keys:
                if key in all_lead_data:
                    new_final_voltage_dic[key] = all_lead_data[key]
                    
            # Find the maximum length of the arrays
            print(new_final_voltage_dic.keys(),"=============keysss")
            new_final_voltage_dic = {key: arr for key, arr in new_final_voltage_dic.items() if not (np.array_equal(arr, np.nan) or (isinstance(arr, np.ndarray) and np.all(np.isnan(arr))))}
            if len(new_final_voltage_dic) != 0:
                max_len = max(len(arr) for arr in new_final_voltage_dic.values())
                
            
                # Pad the arrays with NaN values
                for key, arr in new_final_voltage_dic.items():
                    if len(arr) < max_len:
                        new_final_voltage_dic[key] = np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)
                
                
                # Create a DataFrame and save to CSV
                ecg_data = pd.DataFrame(new_final_voltage_dic)
                # ecg_data.to_csv(f"D:\\OneDrive - KALI MEDTECH PRIVATE LIMITED\\Avani\\Image_to_csv\\06.07.2024\\.csv")
                return ecg_data
            return ecg_data   
    except Exception as e:
        print('\033[93m'+f"ECG signal not detected  : {e} on line_no:  {e.__traceback__.tb_lineno}"+ '\033[0m')
        return ecg_data

def img_to_ecg_detect(image_path):
    # image, binary = preprocess_image(image_path)
    # filtered_boxes = extract_contours(binary)
    # ecg_voltage_dic = save_and_display_graphs(image, filtered_boxes)
    ecg_voltage_dic = resting_ecg_extractor(image_path)
    return ecg_voltage_dic
 
##@app.route('/upload', methods=['POST'])
##def upload_file():
##    data = request.json
##    # Check if all required fields are present
##    if '_id' not in data or 'path' not in data or 'image' not in data:
##        return jsonify({'error': 'Invalid input'}), 400
## 
##    _id = data['_id']
##    path = data['path']
##    ecgimage = data['image']
## 
##    # Process the data (for example, save it to the database)
##    # For demonstration purposes, we'll just return the received data
##    get_response = {
##        '_id': _id,
##        'path': path,
##        'image': ecgimage
##    }
## 
##    print(get_response)
##    url = 'https://oomcardiodev.projectkmt.com/oea/api/v1/uploads/images/'+ecgimage
    #print(url)
##    # Make the GET request
##    response_server = requests.get(url)
    #print(response)
    # Check if the request was successful
##    if response_server.status_code == 200:
##        image_write = "newimages/"+ ecgimage
##        # Print the content of the response
##        with open(image_write, 'wb') as file:
##            file.write(response_server.content)
            #print(response.content)
img_path ="newimages/"
#print(img_path)
# class_name = prediction_model_ECGnoECG(img_path)
# if class_name=="ECG":
##    try:
ecg_all_lead_data = img_to_ecg_detect(img_path)
if len(ecg_all_lead_data) != 0:
    new_df = ecg_all_lead_data.fillna(0)

    noise_label = check_noise(new_df, 200)
    print("noise_label : ", noise_label)

    arrhythmia_result = arrhythmia_detection(new_df,fs=200, name = f"resting_ecg").ecg_signal_processing()
    ##    print(final_label)
    Total_leads = new_df.keys()
    try:
        # if arrhythmia_result['mi_final_result']=="Abnormal":        
        data = {
            "_id": _id,
            "status":"success",
            "processData": {
                "voltage":{
                    "i":new_df['I'].tolist() if 'I' in new_df else [],
                    "ii":new_df['II'].tolist() if 'II' in new_df else [],
                    "iii":new_df['III'].tolist() if 'III' in new_df else [],
                    "avr":new_df['aVR'].tolist() if 'aVR' in new_df else [],
                    "avl":new_df['aVL'].tolist() if 'aVL' in new_df else [],
                    "avf":new_df['aVF'].tolist() if 'aVF'in new_df else [],
                    "v1":new_df['V1'].tolist() if 'V1' in new_df else [],
                    "v2":new_df['V2'].tolist() if 'V2' in new_df else [],
                    "v3":new_df['V3'].tolist() if 'V3'in new_df else [],
                    "v4":new_df['V4'].tolist() if 'V4' in new_df else [],
                    "v5":new_df['V5'].tolist() if 'V5' in new_df else [],
                    "v6":new_df['V6'].tolist() if 'V6' in new_df else [],
                },
                "HR": arrhythmia_result['avg_hr'],
                "detections": arrhythmia_result['detections'],#[{"detect": arrhythmia_result["arr_final_result"],"detectType": "Arrhythmia","confidence":100}],
                "beats": arrhythmia_result['beats'],
                "RRInterval": arrhythmia_result['RRInterval'],
                "PRInterval": arrhythmia_result['PRInterval'],
                "QTInterval": arrhythmia_result['QTInterval'],
                "QRSComplex": arrhythmia_result['QRSComplex'],
                "STseg": arrhythmia_result['STseg'],
                "PRseg": arrhythmia_result['PRseg'],
                "QTc": arrhythmia_result['QTc'],
                "pvcQrs": arrhythmia_result['pvcQrs'],
                "pacQrs": arrhythmia_result['pacQrs'],
                "Vbeat": arrhythmia_result['Vbeat'],
                "Abeat": arrhythmia_result['Abeat'],
                "AL":False
                
            }
        }
    
        
    #break
    except Exception as e:
        data = {
                "_id": _id,
                "message": "something went wrong",
                "status":"fail"
            }
        print(e)
else:
    print('\033[41m'+f"No ECG found in image something wrong"+ '\033[0m')
    data = {
        # "_id": _id,
        "message": "No ECG image in something wrong",
        "status":"fail"
    }  
print(data)
##            except Exception as e:
##                print("ERROR:",e)
##                data = {
####                "_id": _id,
##                "message": "ECG Found with Poor ECG data Quality",
##                "status":"fail"
##            }

       
        
##    else:
##        # Define the JSON data
##        data = {
##    ##                "_id": _id,
##            "message": "No ECG or Something else uploaded",
##            "status":"fail"
##        }

# Convert the data to JSON format
json_data = json.dumps(data)
print(data)
##    # Make the POST request
##    url = 'https://oomcardiodev.projectkmt.com/oea/api/v1/uploads/processData'
##    response = requests.post(url, headers={'Content-Type': 'application/json'}, data=json_data)
##    print("Successfully Analysed")



return '', 200


## 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1600, debug=True)



