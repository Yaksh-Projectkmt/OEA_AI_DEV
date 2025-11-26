import base64
import math
import time
import random
import uuid
import pandas as pd
import numpy as np
import tensorflow as tf
import shutil
import easyocr
from scipy.ndimage import gaussian_filter1d
from skimage import io
from skimage.transform import rotate, hough_line, hough_line_peaks
from scipy import signal
from scipy import sparse
from numpy import trapz
from scipy.sparse.linalg import spsolve
from scipy.signal import (find_peaks, firwin, medfilt, butter, filtfilt)
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
import scipy
import os
from PIL import Image
from collections import Counter
import torch, json, gc
import re,redis, requests
import io

# Ignore specific FutureWarnings
warnings.filterwarnings("ignore")
results_lock = threading.RLock()

redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

img_interpreter, img_input_details, img_output_details = load_tflite_model("Model/restingecgModel_autoGrid_20.tflite")

# For grid in lead detection
Lead_list = ["I", "II", "III", "V1", "V2", "V3", "V4", "V5", "V6", "aVF", "aVL", "aVR"]
# register_coco_instances("my_dataset", {}, '20-09_PM_updated_annotations.coco.json', "New PM Cardio Train")
# MetadataCatalog.get("my_dataset").set(thing_classes=Lead_list)



# MODEL_PATHS = {
#     "6_2": "Model/model_final_29_01_R_101_FPN_3x.pth",
#     "3_4": "Model/model_final_3X4_08_07_25.pth",
#     "12_1": "Model/model_final_12X1_19_05_25.pth"
# }

# def load_object_detection_model(grid_type, img):
#     cfg = get_cfg()
#     if grid_type in ["3_4", "12_1", "6_2"]:
#         cfg.merge_from_file(r"D:\Extract_Signals_from_image1\detectron2\configs\COCO-Detection\faster_rcnn_R_101_FPN_3x.yaml")
#     cfg.MODEL.WEIGHTS = MODEL_PATHS[grid_type]
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(Lead_list)
#     cfg.DATASETS.TEST = ("my_dataset",)
#     cfg.MODEL.DEVICE = "cpu"
#     MetadataCatalog.get("my_dataset").set(thing_classes=Lead_list)
#     return DefaultPredictor(cfg)


def lowpass(file, cutoff=0.4):
    b, a = signal.butter(3, cutoff, btype='lowpass', analog=False)
    low_passed = signal.filtfilt(b, a, file)
    return low_passed


def baseline_construction_200(ecg_signal, kernel_size=101):
    s_corrected = signal.detrend(ecg_signal)
    baseline_corrected = s_corrected - signal.medfilt(s_corrected, kernel_size)
    return baseline_corrected


def force_remove_folder(folder_path):
    """Forcefully remove a folder, even if files are read-only."""

    def onerror(func, path, exc_info):
        # Change the permission and retry
        os.chmod(path, 0o777)  # Grant full permissions
        func(path)

    shutil.rmtree(folder_path, onerror=onerror)

class NoiseDetection:
    def __init__(self, raw_data, class_name, frequency=200):
        self.frequency = frequency
        self.raw_data = raw_data
        self.class_name = class_name

    def prediction_model(self, input_arr):
        classes = ['Noise', 'Normal']
        input_arr = tf.cast(input_arr, dtype=tf.float32)
        input_arr = tf.image.resize(input_arr, size=(224, 224), method=tf.image.ResizeMethod.BILINEAR)
        input_arr = (tf.expand_dims(input_arr, axis=0),)
        model_pred = predict_tflite_model(noise_model, input_arr)[0]
        idx = np.argmax(model_pred)
        return classes[idx]

    def plot_to_imagearray(self, ecg_signal):
        # Ensure ecg_signal is a 1D array
        ecg_signal = np.asarray(ecg_signal).ravel()

        # Create the plot
        fig, ax = plt.subplots(num=1, clear=True)
        ax.plot(ecg_signal, color='black')  # Plot the flattened array
        ax.axis(False)  # Hide axes

        # Convert plot to image array
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)
        return data[:, :, ::-1]

    def noise_model_check(self):
        # Noise detection logic for individual lead
        if self.class_name == '12_1':
            steps_data = int(self.frequency * 5)
        else:
            steps_data = int(self.frequency * 2.5)
        total_data = self.raw_data.shape[0]
        start_data = 0
        normal_index, noise_index = [], []
        percentage = {'Normal': 0, 'Noise': 0, 'total_slice': 0}

        while start_data < total_data:
            end_data = start_data + steps_data

            if end_data - start_data == steps_data and end_data < total_data:
                img_data = pd.DataFrame(self.raw_data[start_data:end_data])
            else:
                img_data = pd.DataFrame(self.raw_data[-steps_data:total_data])
            end_data = total_data - 1

            # Assuming the noise detection model uses image input
            data1 = self.plot_to_imagearray(img_data)

            # plt.plot(img_data);plt.show()

            # Get noise model result for the image
            model_result = self.prediction_model(data1)
            percentage['total_slice'] += 1

            if model_result == 'Normal':
                normal_index.append((start_data, end_data))
                # percentage['Normal'] += (end_data - start_data) / total_data
                percentage['Normal'] += 1
            else:
                noise_index.append((start_data, end_data))
                # percentage['Noise'] += (end_data - start_data) / total_data
                percentage['Noise'] += 1
            start_data += steps_data

        # If the percentage of noise is high, return 'ARTIFACTS'
        noise_label = 'Normal'

        # if int(percentage['Noise'] * 100) >= 60:
        #     noise_label = 'ARTIFACTS'

        if percentage['total_slice'] != 0:
            if percentage['Noise'] == percentage['total_slice']:
                noise_label = 'ARTIFACTS'
            elif percentage['Noise'] / percentage['total_slice'] >= 0.6:
                noise_label = 'ARTIFACTS'

        return noise_label

# Peak detection
class pqrst_detection:
    def __init__(self, ecg_signal, class_name='6_2', fs=200, thres=0.5, lp_thres=0.2, rr_thres=0.12, width=(5, 50),
                 JR=False):
        self.ecg_signal = ecg_signal
        self.fs = fs
        self.thres = thres
        self.lp_thres = lp_thres
        self.rr_thres = rr_thres
        self.width = width
        self.JR = JR
        self.class_name = class_name

    def hamilton_segmenter(self):

        # check inputs
        if self.ecg_signal is None:
            print("Please specify an input signal.")
        ##            raise TypeError("Please specify an input signal.")

        sampling_rate = float(self.fs)
        length = len(self.ecg_signal)
        dur = length / sampling_rate

        # algorithm parameters
        v1s = int(1.0 * sampling_rate)
        v100ms = int(0.1 * sampling_rate)
        TH_elapsed = np.ceil(0.36 * sampling_rate)
        sm_size = int(0.08 * sampling_rate)
        init_ecg = 10  # seconds for initialization
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
                    diff_now = np.diff(self.ecg_signal[0: f + diff_nr])
                elif f + diff_nr >= len(self.ecg_signal):
                    diff_now = np.diff(self.ecg_signal[f - diff_nr: len(dx)])
                else:
                    diff_now = np.diff(self.ecg_signal[f - diff_nr: f + diff_nr])
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
                            diff_prev = np.diff(self.ecg_signal[0: prev_rpeak + diff_nr])
                        elif prev_rpeak + diff_nr >= len(self.ecg_signal):
                            diff_prev = np.diff(self.ecg_signal[prev_rpeak - diff_nr: len(dx)])
                        else:
                            diff_prev = np.diff(
                                self.ecg_signal[prev_rpeak - diff_nr: prev_rpeak + diff_nr]
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
                window = self.ecg_signal[0: i + lim]
                add = 0
            elif i + lim >= length:
                window = self.ecg_signal[i - lim: length]
                add = i - lim
            else:
                window = self.ecg_signal[i - lim: i + lim]
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

    def hr_count(self, class_name):
        cal_sec = 5
        if class_name == '6_2':
            cal_sec = 5
        elif class_name == '12_1':
            cal_sec = 10
        elif class_name == '3_4':
            cal_sec = 2.5
        if cal_sec != 0:
            hr = round(self.r_index.shape[0] * 60 / cal_sec)
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
        increment = int(self.fs * 0.05)
        for z in range(0, len(self.s_index)):
            data = []
            j_index = self.ecg_signal[self.s_index[z]:self.s_index[z] + increment]
            for k in range(0, len(j_index)):
                data.append(j_index[k])
            max_d = max(data)
            max_id = data.index(max_d)
            j.append(self.s_index[z] + max_id)
        return j

    def find_s_index(self, d):
        d = int(d) + 1
        s = []
        for i in self.r_index:
            if i == len(self.ecg_signal):
                s.append(i)
                continue
            elif i + d <= len(self.ecg_signal):
                s_array = self.ecg_signal[i:i + d]
            else:
                s_array = self.ecg_signal[i:]
            if self.ecg_signal[i] > 0:
                s_index = i + np.where(s_array == min(s_array))[0][0]
            else:
                s_index = i + np.where(s_array == max(s_array))[0][0]
                if abs(s_index - i) < d / 2:
                    s_index_ = i + np.where(s_array == min(s_array))[0][0]
                    if abs(s_index_ - i) > d / 2:
                        s_index = s_index_
            s.append(s_index)
        return np.sort(s)

    # def find_q_index(self, d):
    #     """The Q wave index in an ECG signal given the R wave index and a specified
    #     distance.
    #
    #     Args:
    #         ecg (array): ECG signal values
    #         R_index (array/list): R peak indices in the ECG signal
    #         d (int): The maximum distance (in samples) between the R peak and the Q wave onset that we want to find.
    #
    #     Returns:
    #         list: Q-wave indices for each R-wave index in the ECG signal.
    #     """
    #     d = int(d) + 1
    #     q = []
    #     for i in self.r_index:
    #         if i == 0:
    #             q.append(i)
    #             continue
    #         elif 0 <= i - d:
    #             q_array = self.ecg_signal[i - d:i]
    #         else:
    #             q_array = self.ecg_signal[:i]
    #         if self.ecg_signal[i] > 0:
    #             q_index = i - (len(q_array) - np.where(q_array == min(q_array))[0][0])
    #         else:
    #             q_index = i - (len(q_array) - np.where(q_array == max(q_array))[0][0])
    #         q.append(q_index)
    #     return np.sort(q)

    def find_new_q_index(self, d):
        q = []
        for i in self.r_index:
            q_ = []
            if i == 0:
                q.append(i)
                continue
            if self.ecg_signal[i] > 0:
                c = i
                while c > 0 and self.ecg_signal[c - 1] < self.ecg_signal[c]:
                    c -= 1
                if self.ecg_signal[i] * 0.01 > self.ecg_signal[c] or self.ecg_signal[c] < 0 or c == 0:
                    if abs(i - c) <= d:
                        q.append(c)
                        continue
                    else:
                        q_.append(c)
                while c > 0:
                    while c > 0 and self.ecg_signal[c - 1] > self.ecg_signal[c]:
                        c -= 1
                    # q_.append(c)
                    while c > 0 and self.ecg_signal[c - 1] < self.ecg_signal[c]:
                        c -= 1
                    if q_ and q_[-1] == c:
                        break
                    q_.append(c)
                    if self.ecg_signal[i] * 0.01 > self.ecg_signal[c] or self.ecg_signal[c] < 0 or c == 0:
                        break
            else:
                c = i
                while c > 0 and self.ecg_signal[c - 1] > self.ecg_signal[c]:
                    c -= 1
                if self.ecg_signal[i] * 0.01 < self.ecg_signal[c] or self.ecg_signal[c] > 0 or c == 0:
                    if abs(i - c) <= d:
                        q.append(c)
                        continue
                    else:
                        q_.append(c)
                while c > 0:
                    while c > 0 and self.ecg_signal[c - 1] < self.ecg_signal[c]:
                        c -= 1
                    # q_.append(c)
                    while c > 0 and self.ecg_signal[c - 1] > self.ecg_signal[c]:
                        c -= 1
                    if q_ and q_[-1] == c:
                        break
                    q_.append(c)
                    if self.ecg_signal[i] * 0.01 < self.ecg_signal[c] or self.ecg_signal[c] > 0 or c == 0:
                        break
            if q_:
                a = 0
                for _q in q_[::-1]:
                    if abs(i - _q) <= d:
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
                while c + 1 < end_index and self.ecg_signal[c + 1] < self.ecg_signal[c]:
                    c += 1
                if self.ecg_signal[i] * 0.01 > self.ecg_signal[c] or self.ecg_signal[c] < 0 or c == end_index - 1:
                    if abs(i - c) <= d:
                        s.append(c)
                        continue
                    else:
                        s_.append(c)
                while c + 1 < end_index:
                    while c + 1 < end_index and self.ecg_signal[c + 1] > self.ecg_signal[c]:
                        c += 1
                    while c + 1 < end_index and self.ecg_signal[c + 1] < self.ecg_signal[c]:
                        c += 1
                    if s_ and s_[-1] == c:
                        break
                    s_.append(c)
                    if self.ecg_signal[i] * 0.01 > self.ecg_signal[c] or self.ecg_signal[c] < 0 or c == end_index - 1:
                        break
            else:
                c = i
                while c + 1 < end_index and self.ecg_signal[c + 1] > self.ecg_signal[c]:
                    c += 1
                if self.ecg_signal[i] * 0.01 < self.ecg_signal[c] or self.ecg_signal[c] > 0 or c == end_index - 1:
                    if abs(i - c) <= d:
                        s.append(c)
                        continue
                    else:
                        s_.append(c)
                while c < end_index:
                    while c + 1 < end_index and self.ecg_signal[c + 1] > self.ecg_signal[c]:
                        c += 1
                    while c + 1 < end_index and self.ecg_signal[c + 1] < self.ecg_signal[c]:
                        c += 1
                    if s_ and s_[-1] == c:
                        break
                    s_.append(c)
                    if self.ecg_signal[i] * 0.01 < self.ecg_signal[c] or self.ecg_signal[c] > 0 or c == end_index - 1:
                        break
            if s_:
                a = 0
                for _s in s_[::-1]:
                    if abs(i - _s) <= d:
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
        heart_rate = self.hr_count(self.class_name)
        if self.JR:  # ---------------------
            diff_indexs = abs(round((self.fs * 0.4492537) + (heart_rate * -1.05518351) + 40.40601032654332))
        else:  # ---------------------
            diff_indexs = abs(round((self.fs * 0.4492537) + (heart_rate * -1.009) + 58.40601032654332))

        for r in self.r_index:
            if r - diff_indexs >= 0 and len(self.ecg_signal) >= r + diff_indexs:
                data = self.ecg_signal[r - diff_indexs:r + diff_indexs]
                abs_data = np.abs(data)
                r_.append(np.where(abs_data == max(abs_data))[0][0] + r - diff_indexs)
            else:
                r_.append(r)

        new_r = np.unique(r_) if r_ else self.r_index
        fs_diff = int((25 * self.fs) / 200)
        final_r = []
        if new_r.any(): final_r = [new_r[0]] + [new_r[j + 1] for j, i in enumerate(np.diff(new_r)) if i >= fs_diff]
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
        max_signal = max(self.ecg_signal) / 100
        pt = []
        p_t = []
        for i in range(0, len(self.r_index) - 1):
            aoi = self.ecg_signal[self.s_index[i]:self.q_index[i + 1]]
            max_signal = max(self.ecg_signal) / 100
            low = self.fir_lowpass_filter(aoi, self.lp_thres, 30)
            if self.ecg_signal[self.r_index[i]] < 0:
                max_signal = 0.05
            else:
                max_signal = max_signal
            if aoi.any():
                peaks, _ = find_peaks(low, height=max_signal, width=self.width)
                peaks1 = peaks + (self.s_index[i])
            else:
                peaks1 = [0]
            p_t.append(list(peaks1))
            pt.extend(list(peaks1))
            for i in range(len(p_t)):
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
        for i in range(0, len(self.r_index) - 1):
            aoi = self.ecg_signal[self.s_index[i]:self.q_index[i + 1]]
            if aoi.any():
                low = self.fir_lowpass_filter(aoi, self.lp_thres, 30)
                if self.ecg_signal[self.r_index[i]] < 0:
                    max_signal = 0.05
                else:
                    max_signal = max(low) * 0.2
                if aoi.any():
                    peaks, _ = find_peaks(low, height=max_signal, width=self.width)
                    peaks1 = peaks + (self.s_index[i])
                else:
                    peaks1 = [0]
                p_t.append(list(peaks1))
                pt.extend(list(peaks1))
                for i in range(len(p_t)):
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
        for i in range(0, len(self.r_index) - 1):
            aoi = self.ecg_signal[self.s_index[i]:self.q_index[i + 1]]
            low = self.fir_lowpass_filter(aoi, self.lp_thres, 30)
            if aoi.any():
                peaks, _ = find_peaks(low, prominence=0.05, width=self.width)
                peaks1 = peaks + (self.s_index[i])
            else:
                peaks1 = [0]
            p_t.append(list(peaks1))
            pt.extend(list(peaks1))
            for i in range(len(p_t)):
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
                        m, n, o = arr[0], arr[al[index]], arr[al[index + 1]]
                    elif index == len(al) - 1:
                        m, n, o = arr[al[index - 1]], arr[al[index]], arr[-1]
                    else:
                        m, n, o = arr[al[index - 1]], arr[al[index]], arr[al[index + 1]]
                    diff[p] = [abs(n - m), abs(n - o)]
                th = np.mean([np.mean([v, m]) for v, m in diff.values()]) * .66
                for p, (a, b) in diff.items():
                    if a >= th and b >= th:
                        P.append(p)
                        continue
                    if a >= th and not Pa:
                        Pa.append(p)
                    elif a >= th and arr[p] > arr[Pa[-1]] and np.where(pos == Pa[-1])[0] + 1 == np.where(pos == p)[0]:
                        Pa[-1] = p
                    elif a >= th:
                        Pa.append(p)
                    if b >= th and not Pb:
                        Pb.append(p)
                    elif b >= th and arr[p] < arr[Pb[-1]] and np.where(pos == Pb[-1])[0] + 1 == np.where(pos == p)[0]:
                        Pb[-1] = p
                    elif b >= th:
                        Pb.append(p)
                if len(pos) > 1:
                    for i in range(1, len(pos)):
                        m, n = pos[i - 1], pos[i]
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
            arr = self.ecg_signal[s0 + 7:q1 - 7]
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
            for _pt in set(p_t1[i] + p_t2[i] + p_t3[i] + p_t4[i]):
                count = 0
                if any(val in p_t1[i] for val in range(_pt - 2, _pt + 3)):
                    count += 1
                if any(val in p_t2[i] for val in range(_pt - 2, _pt + 3)):
                    count += 1
                if any(val in p_t3[i] for val in range(_pt - 2, _pt + 3)):
                    count += 1
                if any(val in p_t4[i] for val in range(_pt - 2, _pt + 3)):
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
                if abs(sublist[i] - sublist[i - 1]) > 5:
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
        diff_arr = ((np.diff(self.r_index) * self.thres) / self.fs).tolist()
        t_peaks_list, p_peaks_list, pr_interval, extra_peaks_list = [], [], [], []
        # threshold = (-0.0012 * len(r_index)) + 0.25
        for i in range(len(self.p_t)):
            p_dis = (self.r_index[i + 1] - self.p_t[i][-1]) / self.fs
            t_dis = (self.r_index[i + 1] - self.p_t[i][0]) / self.fs
            threshold = diff_arr[i]
            if t_dis > threshold and (self.p_t[i][0] > self.r_index[i]):
                t_peaks_list.append(self.p_t[i][0])
            else:
                t_peaks_list.append(0)
            if p_dis <= threshold:
                p_peaks_list.append(self.p_t[i][-1])
                pr_interval.append(p_dis * self.fs)
            else:
                p_peaks_list.append(0)
            if len(self.p_t[i]) > 0:
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
        if self.thres >= 0.5 and p_peaks_list and len(p_peaks_list) > 2:
            pp_intervals = np.diff(p_peaks_list)
            pp_std = np.std(pp_intervals)
            pp_mean = np.mean(pp_intervals)
            threshold = 0.12 * pp_mean
            if pp_std <= threshold:
                p_label = "Constanat"
            else:
                p_label = "Not Constant"

            count = 0
            for i in pr_interval:
                if round(np.mean(pr_interval) * 0.75) <= i <= round(np.mean(pr_interval) * 1.25):
                    count += 1
            if len(pr_interval) != 0:
                per = count / len(pr_interval)
                pr_label = 'Not Constant' if per <= 0.7 else 'Constant'
        data = {'T_Index': t_peaks_list,
                'P_Index': p_peaks_list,
                'PR_Interval': pr_interval,
                'P_Label': p_label,
                'PR_label': pr_label,
                'Extra_Peaks': extra_peaks_list}
        return data

    def find_inverted_t_peak(self):
        t_index = []
        for i in range(0, len(self.s_index) - 1):
            t = self.ecg_signal[self.s_index[i]: self.q_index[i + 1]]
            if t.any():
                check, _ = find_peaks(-t, height=(0.21, 1), distance=70)
                peaks = check + self.s_index[i]
            else:
                peaks = np.array([])
            if peaks.any():
                t_index.extend(list(peaks))
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
        self.hr_ = self.hr_count(self.class_name)
        sd, qd = int(self.fs * 0.115), int(self.fs * 0.08)
        self.s_index = self.find_s_index(sd)
        # q_index = find_q_index(ecg_signal, r_index, qd)
        # s_index = find_new_s_index(ecg_signal,r_index,sd)
        self.q_index = self.find_new_q_index(qd)
        self.j_index = self.find_j_index()
        self.p_t, self.pt = self.find_pt()
        self.data_ = self.segricate_p_t_pr_inerval()
        self.inv_t_index = self.find_inverted_t_peak()
        data = {'R_Label': self.r_label,
                'R_index': self.r_index,
                'Q_Index': self.q_index,
                'S_Index': self.s_index,
                'J_Index': self.j_index,
                'P_T List': self.p_t,
                'PT PLot': self.pt,
                'HR_Count': self.hr_,
                'T_Index': self.data_['T_Index'],
                'P_Index': self.data_['P_Index'],
                'Ex_Index': self.data_['Extra_Peaks'],
                'PR_Interval': self.data_['PR_Interval'],
                'P_Label': self.data_['P_Label'],
                'PR_label': self.data_['PR_label'],
                'inv_t_index': self.inv_t_index}
        # print(rr_intervals)
        return data

# Low pass and baseline signal
class filter_signal:

    def __init__(self, ecg_signal, fs=200):
        self.ecg_signal = ecg_signal
        self.fs = fs
        self.baseline_signal = None

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
            z = spsolve(Z, w * file)
            w = p * (file > z) + (1 - p) * (file < z)
        return z

    def baseline_construction_250(self, kernel_size=131):
        als_baseline = self.baseline_als(self.ecg_signal, 16 ** 5, 0.01)
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

        lowpass_signal = None

        if self.fs != 200:
            self.ecg_signal = MinMaxScaler(feature_range=(0, 4)).fit_transform(self.ecg_signal.reshape(-1, 1)).squeeze()

        if self.fs == 200:
            self.baseline_signal = self.baseline_construction_200(kernel_size=101)
            lowpass_signal = self.lowpass(cutoff=0.3)
        elif self.fs == 250:
            self.baseline_signal = self.baseline_construction_250(kernel_size=131)
            lowpass_signal = self.lowpass(cutoff=0.25)
        elif self.fs == 360:
            self.baseline_signal = self.baseline_construction_200(kernel_size=151)
            lowpass_signal = self.lowpass(cutoff=0.2)
        elif self.fs == 1000:
            self.baseline_signal = self.baseline_construction_200(kernel_size=399)
            lowpass_signal = self.lowpass(cutoff=0.05)
        elif self.fs == 128:
            self.baseline_signal = self.baseline_construction_200(kernel_size=101)
            lowpass_signal = self.lowpass(cutoff=0.5)
        else:
            self.baseline_signal = self.baseline_construction_200(kernel_size=101)
            lowpass_signal = self.lowpass(cutoff=0.5)
            # raise ValueError(f"Unsupported sampling frequency: {self.fs}")

        return self.baseline_signal, lowpass_signal

# PVC detection
class PVC_detection:
    def __init__(self, ecg_signal, r_id, fs=100):  # 200
        self.ecg_signal = ecg_signal
        self.fs = fs
        self.r_id = r_id

    def lowpass(self, file):
        b, a = signal.butter(3, 0.3, btype='lowpass', analog=False)
        low_passed = signal.filtfilt(b, a, file)
        return low_passed

    def baseline_construction_200(self, kernel_size=101):
        s_corrected = signal.detrend(self.ecg_signal)
        baseline_corrected = s_corrected - signal.medfilt(s_corrected, kernel_size)
        return baseline_corrected

    def detect_beats(self, ecg, rate, ransac_window_size=3.35, lowfreq=5.0, highfreq=15.0):
        ransac_window_size = int(ransac_window_size * rate)
        lowpass = scipy.signal.butter(1, highfreq / (rate / 2.0), 'low')
        highpass = scipy.signal.butter(1, lowfreq / (rate / 2.0), 'high')
        ecg_low = scipy.signal.filtfilt(*lowpass, x=ecg)
        ecg_band = scipy.signal.filtfilt(*highpass, x=ecg_low)
        decg = np.diff(ecg_band)
        decg_power = decg ** 2
        thresholds, max_powers = [], []
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
        lp_energy = gaussian_filter1d(lp_energy, rate / 14.0)
        lp_energy_diff = np.diff(lp_energy)

        zero_crossings = (lp_energy_diff[:-1] > 0) & (lp_energy_diff[1:] < 0)
        zero_crossings = np.flatnonzero(zero_crossings)
        zero_crossings -= 1

        rpeaks = []
        for idx in zero_crossings:
            search_window = slice(max(0, idx - int(rate * 0.2)), min(len(ecg), idx + int(rate * 0.1)))
            local_signal = ecg[search_window]
            max_amplitude = np.max(local_signal)
            min_amplitude = np.min(local_signal)
            if abs(max_amplitude) > abs(min_amplitude):
                rpeak = np.argmax(local_signal) + search_window.start
            elif abs(max_amplitude + 0.11) < abs(min_amplitude):
                rpeak = np.argmin(local_signal) + search_window.start
            else:
                if max_amplitude >= 0:
                    rpeak = np.argmax(local_signal) + search_window.start
                else:
                    rpeak = np.argmin(local_signal) + search_window.start

            rpeaks.append(rpeak)

        return np.array(rpeaks)

    def calculate_surface_area(self, ecg_signal, qrs_start_index, qrs_end_index, sampling_rate):
        if qrs_start_index == 0 or qrs_end_index == 0:
            surface_area = 0
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
        # if self.hr_count <= 88:
        #     thresold = round(self.fs * 0.08)  # 0.10
        # else:
        thresold = round(self.fs * 0.12) 
        for k in range(len(self.r_index)):
            diff = self.s_index[k] - self.q_index[k]
            if self.r_index[k] in above_r_peaks:
                surface_thres = 0.02  
                wideqs_thres = 0.01  
            elif self.r_index[k] in below_r_peaks:
                surface_thres = 0.02  
                wideqs_thres = 0.02 
            if diff > thresold:
                difference.append(diff)
                wideQRS.append(self.r_index[k])
                surface_area = self.calculate_surface_area(self.low_pass_signal, self.q_index[k], self.s_index[k],
                                                           self.fs)
                if (diff / 100) >= wideqs_thres:
                    surface_area_list.append(round(surface_area, 3))
                    if surface_area >= surface_thres:
                        pvc.append(self.r_index[k])

        if len(difference) != 0:
            q_s_difference = [i / 100 for i in difference]  # 200
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

        total_one = (1 * vt) + (couplet_counter * 2) + (triplet_counter * 3) + (bigeminy_counter * 2) + (
                trigeminy_counter * 2) + (quadrigeminy_counter * 2)
        total = vt_counter + couplet_counter + triplet_counter + bigeminy_counter + trigeminy_counter + quadrigeminy_counter
        ones = PVC_R_Peaks.count(1)
        if total == 0:
            Isolated = ones
        else:
            Common = total - 1
            Isolated = ones - (total_one - Common)
        if vt_counter > 1:
            vt_counter = 1
        return vt_counter, couplet_counter, triplet_counter, bigeminy_counter, trigeminy_counter, quadrigeminy_counter, Isolated, vt

    def VT_confirmation(self, ecg_signal, r_index):
        VTC = []
        pqrst_data = pqrst_detection(ecg_signal)
        for i in range(0, len(r_index) - 1):
            aoi = ecg_signal[r_index[i] - 5:r_index[i + 1]]
            low = pqrst_data.fir_lowpass_filter(aoi, 0.2, 30)
            if aoi.any():
                peaks, _ = find_peaks(low, prominence=0.2, width=(40))
                VTC.append(peaks)
        if round(len(VTC) / len(r_index)) >= .7:
            label = 'VT'
        else:
            label = 'Abnormal'

        return label, len(VTC)

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

    def model_r_detectron(self, e_signal, r_index, heart_rate, fs=100):
        pvc_0 = []
        lbbb, rbbb = [], []
        mod_pre_lbb, mod_pre_rbbb = [], []
        model_pred = []
        counter = 0
        detect_rpeaks = r_index
        try:
            if not os.path.exists(f'temp_pvc_img/' + self.r_id):
                os.mkdir('temp_pvc_img/' + self.r_id)
        except Exception as e:
            print(e)
        for r in detect_rpeaks:
            if int(r) - 50 > 0:
                windo_start = int(r) - 50
            else:
                windo_start = 0
            windo_end = int(r) + 80
            aa = pd.DataFrame(e_signal[windo_start:windo_end])
            plt.plot(aa, color='blue')
            plt.axis("off")
            plt.savefig(f"temp_pvc_img/{self.r_id}/{r}.jpg")
            aq = cv2.imread(f"temp_pvc_img/{self.r_id}/{r}.jpg")
            aq = cv2.resize(aq, (360, 720))
            cv2.imwrite(f"temp_pvc_img/{self.r_id}/{r}.jpg", aq)
            plt.close()

        files = sorted(glob.glob(f"temp_pvc_img/{self.r_id}/*.jpg"), key=len)
        for p_files in files:
            predictions, model_label = self.prediction_model(p_files)
            # print(predictions, model_label)
            r_peak = int(p_files.split("/")[-1].split(".")[0])
            if str(model_label) == 'PVC' and float(predictions[3]) > 0.78:  # 0.75
                pvc_0.append(r_peak)
                model_pred.append(int(float(predictions[3]) * 100))
            if str(model_label) == 'LBBB' and float(predictions[0]) > 0.78:
                lbbb.append(r_peak)
                mod_pre_lbb.append(int(float(predictions[0])*100))
            if str(model_label) == 'RBBB' and float(predictions[4]) > 0.78:
                rbbb.append(r_peak)
                mod_pre_rbbb.append(int(float(predictions[4])*100))
        for i in glob.glob(f"temp_pvc_img/{self.r_id}/*.jpg"):
            os.remove(i)
        force_remove_folder(os.path.join("temp_pvc_img/" + self.r_id))
        return pvc_0, lbbb, rbbb, mod_pre_lbb, mod_pre_rbbb, model_pred, detect_rpeaks

    def get_pvc_data(self):
        self.baseline_signal = self.baseline_construction_200(kernel_size=131)  # 101
        self.low_pass_signal = self.lowpass(self.baseline_signal)
        lbbb_rbbb_label = "Abnormal"
        pqrst_data = pqrst_detection(self.baseline_signal, fs=self.fs).get_data()
        self.r_index = pqrst_data['R_index']
        self.q_index = pqrst_data['Q_Index']
        self.s_index = pqrst_data['S_Index']
        self.hr_count = pqrst_data['HR_Count']
        self.p_t = pqrst_data['P_T List']
        self.ex_index = pqrst_data['Ex_Index']
        wide_qrs, q_s_difference, surface_index = self.wide_qrs_find()
        # wide_qrs = np.array([])
        model_pred = model_pvc = []
        lbbb_index, rbbb_index = [], []

        pvc_onehot = np.zeros(len(self.r_index)).tolist()  # r_index
        lbbb_rbbb_per = 0
        if len(wide_qrs) > 0:
            if self.fs == 200:
                model_pvc, lbbb_index, rbbb_index, mod_pre_lbb, mod_pre_rbbb, model_pred, detect_rpeaks = self.model_r_detectron(
                    self.low_pass_signal, wide_qrs, self.hr_count, fs=self.fs)
            else:

                model_pvc, lbbb_index, rbbb_index, mod_pre_lbb, mod_pre_rbbb, model_pred, detect_rpeaks = self.model_r_detectron(self.ecg_signal,
                                                                                                      wide_qrs,
                                                                                                      self.hr_count,
                                                                                                      fs=self.fs)
            label = "PVC" if len(model_pvc) > 0 else "Abnormal"
            pvc_onehot = [1 if r in model_pvc else 0 for r in detect_rpeaks]
            pvc_per = int(sum(model_pred) / len(model_pred)) if len(model_pred) > 0 else 0
            if detect_rpeaks.any():
                if len(lbbb_index) > 0 or len(rbbb_index) > 0:
                    if len(lbbb_index) / len(detect_rpeaks) > 0.3:
                        lbbb_rbbb_label = "LBBB"
                        lbbb_rbbb_per = int(sum(mod_pre_lbb)/len(mod_pre_lbb))
                    elif len(rbbb_index) / len(detect_rpeaks) > 0.3:
                        lbbb_rbbb_label = "RBBB"
                        lbbb_rbbb_per = int(sum(mod_pre_rbbb)/len(mod_pre_rbbb))
                else:
                    lbbb_rbbb_label = "Abnormal"
            else:
                lbbb_rbbb_label = "Abnormal"
        else:
            label = "Abnormal"
            lbbb_rbbb_label = "Abnormal"
            lbbb_rbbb_per = 0
            pvc_per = 0

        pvc_count = pvc_onehot.count(1)
        vt_counter, couplet_counter, triplet_counter, bigeminy_counter, trigeminy_counter, quadrigeminy_counter, remaining_ones, v_bit_vt = self.PVC_CLASSIFICATION(
            pvc_onehot)
        conf_vt_count = 0
        if vt_counter > 0:
            confirmation = self.VT_confirmation(self.low_pass_signal, detect_rpeaks)

            if self.hr_count > 100 and v_bit_vt > 12:
                conf_vt_count = 1
            if confirmation == "Abnormal":
                vt_counter = 0
            else:
                pass
        data = {'PVC-Label': label,
                'PVC-Count': pvc_count,
                'PVC-Index': model_pvc,
                'VT_counter': conf_vt_count,
                'PVC-Couplet_counter': couplet_counter,
                'PVC-Triplet_counter': triplet_counter,
                'PVC-Bigeminy_counter': bigeminy_counter,
                'PVC-Trigeminy_counter': trigeminy_counter,
                'PVC-Quadrigeminy_counter': quadrigeminy_counter,
                'PVC-Isolated_counter': remaining_ones,
                'PVC-wide_qrs': wide_qrs,
                'PVC-QRS_difference': q_s_difference,
                'pvc_pred': pvc_per,
                "IVR_counter": 0,
                "NSVT_counter": 0,
                "lbbb_rbbb_label": lbbb_rbbb_label,
                "lbbb_index": lbbb_index,
                "lbbb_rbbb_per": lbbb_rbbb_per
                }
        if vt_counter > 0:
            if 60 <= self.hr_count < 100:
                data['VT_counter'] = 0
                data["NSVT_counter"] = vt_counter
            elif self.hr_count < 60 and v_bit_vt > 3:
                data['VT_counter'] = 0
                data["IVR_counter"] = vt_counter
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
        label = 'Abnormal'
        third_degree = []
        possible_mob_3rd = False
        if self.hr_counts <= 100 and len(self.p_t) != 0:  # 60 70
            constant_2 = all(map(lambda innerlist: len(innerlist) == 2, self.p_t))
            cons_2_1 = all(len(inner_list) in {1, 2} for inner_list in self.p_t)
            ampli_val = list(
                map(lambda inner_list: sum(self.baseline_signal[i] > 0.05 for i in inner_list) / len(inner_list),
                    self.p_t))
            count_above_threshold = sum(1 for value in ampli_val if value > 0.7)
            percentage_above_threshold = count_above_threshold / len(ampli_val)
            count = 0
            if percentage_above_threshold >= 0.7:
                inc_dec_count = 0
                for i in range(0, len(self.pr)):
                    if self.pr[i] > self.pr[i - 1]:
                        inc_dec_count += 1
                if len(self.pr) != 0:
                    if round(inc_dec_count / (len(self.pr)), 2) >= 0.50:  # if posibale to change more then 0.5
                        possible_mob_3rd = True
                # if cons_2_1 == False:
                #     for i in range(0, len(self.pr)):
                #         if self.pr[i] > self.pr[i-1]:
                #             count += 1
                #     if round(count/len(self.pr), 2) >= 0.5:
                #         possible_3rd = True
                for inner_list in self.p_t:
                    if len(inner_list) in [3, 4]:
                        ampli_val = [self.baseline_signal[i] for i in inner_list]
                        if ampli_val and (sum(value > 0.05 for value in ampli_val) / len(ampli_val)) > 0.7:
                            differences = np.diff(inner_list).tolist()
                            diff_list = [x for x in differences if x >= 70]
                            if len(diff_list) != 0:
                                third_degree.append(1)
                            else:
                                third_degree.append(0)
                    elif len(inner_list) in [3, 4] and possible_mob_3rd == True and constant_2 == False:
                        differences = np.diff(inner_list).tolist()
                        if all(diff > 70 for diff in differences):
                            third_degree.append(1)
                        else:
                            third_degree.append(0)
                    else:
                        third_degree.append(0)
        if len(third_degree) != 0:
            if third_degree.count(1) / len(third_degree) >= 0.4 or possible_mob_3rd:  # 0.5 0.4
                label = "3rd Degree block"
        return label

    def second_degree_block_detection(self):
        label = 'Abnormal'
        constant_3_peak = []
        possible_mob_1 = False
        possible_mob_2 = False
        mob_count = 0
        if self.hr_counts <= 100:  # 80
            if len(self.p_t) != 0:
                constant_2 = all(map(lambda innerlist: len(innerlist) == 2, self.p_t))
                rhythm_flag = all(len(inner_list) in {1, 2, 3} for inner_list in self.p_t)
                ampli_val = list(
                    map(lambda inner_list: sum(self.baseline_signal[i] > 0.05 for i in inner_list) / len(inner_list),
                        self.p_t))
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
                            if pr_interval[i] > pr_interval[i - 1]:
                                count_2 += 1
                        most_frequent = max(counts.values())
                        if round(count_2 / (len(pr_interval)), 2) >= 0.50:
                            possible_mob_1 = True
                        elif round(most_frequent / len(pr_interval), 2) >= 0.4:
                            possible_mob_2 = True

                        for inner_list in self.p_t:
                            if len(inner_list) == 3:
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
                            if len(inner_list) == 3:
                                differences = np.diff(inner_list).tolist()
                                if differences[0] <= 0.5 * differences[1] or differences[1] <= 0.5 * differences[0]:
                                    constant_3_peak.append(1)
                                else:
                                    constant_3_peak.append(0)
                            else:
                                constant_3_peak.append(0)
        if len(constant_3_peak) != 0 and constant_3_peak.count(1) != 0:

            if constant_3_peak.count(1) / len(constant_3_peak) >= 0.4:  # 0.4 0.5
                label = "Mobitz_II"
        elif possible_mob_1 and mob_count > 1:  # 0 1 4
            label = "Mobitz_I"
        elif possible_mob_2 and mob_count > 1:  # 0  4
            label = "Mobitz_II"
        return label

    # Block new trans model for added
    def prediction_model_block(self, input_arr):
        classes = ['1st_deg', '2nd_deg', '3rd_deg', 'abnormal', 'normal']
        input_arr = tf.io.decode_jpeg(tf.io.read_file(input_arr), channels=3)
        input_arr = tf.image.resize(input_arr, size=(224, 224), method=tf.image.ResizeMethod.BILINEAR)
        input_arr = (tf.expand_dims(input_arr, axis=0),)
        model_pred = predict_tflite_model(block_model, input_arr)[0]
        # print(model_pred)
        idx = np.argmax(model_pred)
        return model_pred, classes[idx]

    def check_block_model(self, low_ecg_signal):
        label = 'Abnormal'
        for i in glob.glob('temp_block_img' + "/*.jpg"):
            os.remove(i)

        randome_number = random.randint(200000, 1000000)
        temp_img = low_ecg_signal
        plt.figure()
        plt.plot(temp_img)
        plt.axis("off")
        plt.savefig(f"temp_block_img/p_{randome_number}.jpg")
        aq = cv2.imread(f"temp_block_img/p_{randome_number}.jpg")
        aq = cv2.resize(aq, (2400, 360), interpolation=cv2.INTER_LANCZOS4)
        aq = Image.fromarray(cv2.cvtColor(aq, cv2.COLOR_BGR2RGB))
        aq.save(f"temp_block_img/p_{randome_number}.jpg", dpi=(2000, 700))
        plt.close()
        ei_ti_label = []
        files = sorted(glob.glob("temp_block_img/*.jpg"), key=extract_number)
        for pvcfilename in files:
            predictions, ids = self.prediction_model_block(pvcfilename)
            label = "Abnormal"  # "Normal"
            if str(ids) == "3rd_deg" and float(predictions[2]) > 0.8:
                label = "3rd degree"
            if str(ids) == "2nd_deg" and float(predictions[1]) > 0.8:
                label = "2nd degree"
            if str(ids) == "1st_deg" and float(predictions[0]) > 0.8:
                label = "1st degree"

            if 0.40 < float(predictions[1]) < 0.70:
                ei_ti_label.append('2nd degree')
            if 0.40 < float(predictions[0]) < 0.70:
                ei_ti_label.append('1st degree')
            if 0.40 < float(predictions[2]) < 0.70:
                ei_ti_label.append('3rd degree')
        return label, ei_ti_label, predictions

def block_model_check(ecg_signal, frequency, abs_result):
    model_label = 'Abnormal'
    ei_ti_block = []
    lowpass_signal = lowpass(ecg_signal, 0.3)
    baseline_signal = baseline_construction_200(lowpass_signal, 131)
    get_block = BlockDetected(ecg_signal, frequency)
    block_result, ei_ti_label, model_pre = get_block.check_block_model(baseline_signal)
    model_prediction = 0
    if block_result == '1st degree' and abs_result != 'Abnormal':
        model_label = 'I_Degree'
        model_prediction = int(float(model_pre[0]) * 100)
    if block_result == '2nd degree' and (abs_result == 'Mobitz II' or abs_result == 'Mobitz I'):
        if abs_result == "Mobitz I":
            model_label = 'MOBITZ_I'
            model_prediction = int(float(model_pre[1]) * 100)
        if abs_result == "Mobitz II":
            model_label = 'MOBITZ_II'
            model_prediction = int(float(model_pre[1]) * 100)
    if block_result == '3rd degree' and abs_result != "Abnormal":
        model_label = 'III_Degree'
        model_prediction = int(float(model_pre[2]) * 100)
    if abs_result in ['1st deg. block', "3rd Degree block", 'Mobitz II', 'Mobitz I']:
        if block_result == '2nd degree':
            model_label = 'MOBITZ_I'
            model_prediction = int(float(model_pre[1]) * 100)
        elif block_result == '3rd degree':
            model_label = 'III_Degree'
            model_prediction = int(float(model_pre[2]) * 100)
    if ei_ti_label:
        if '1st degree' in ei_ti_label and abs_result != "Abnormal":
            model_label = 'I_Degree'
            ei_ti_block.append({"Arrhythmia": "I_Degree", "percentage": model_pre[0] * 100})
        if '2nd degree' in ei_ti_label and (abs_result == 'Mobitz I' or abs_result == 'Mobitz II'):
            if abs_result == "Mobitz I":
                model_label = 'MOBITZ_I'
                ei_ti_block.append({"Arrhythmia": "MOBITZ_I", "percentage": model_pre[1] * 100})
            if abs_result == "Mobitz II":
                model_label = 'MOBITZ_II'
                ei_ti_block.append({"Arrhythmia": "MOBITZ_II", "percentage": model_pre[1] * 100})
        if '3rd degree' in ei_ti_label and abs_result != "Abnormal":
            model_label = 'III_Degree'
            ei_ti_block.append({"Arrhythmia": "III_Degree", "percentage": model_pre[2] * 100})
    return model_label, ei_ti_block, model_prediction

# Vfib & VFL detection
def resampled_ecg_data(ecg_signal, original_freq, desire_freq):
    original_time = np.arange(len(ecg_signal)) / original_freq
    new_time = np.linspace(original_time[0], original_time[-1], int(len(ecg_signal) * (desire_freq / original_freq)))
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

def vfib_predict_tflite_model(model: tuple, input_data: tuple):
    with results_lock:
        if type(model) != tuple and type(input_data) != tuple:
            print("Error")
        ##        raise TypeError
        interpreter, input_details, output_details = model
        for i in range(len(input_data)):
            interpreter.set_tensor(input_details[i]['index'], input_data[i])
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        return output

def vfib_model_pred_tfite(raw_signal, model, fs):
    if fs == 200 and (np.max(raw_signal) > 4.1 or np.min(raw_signal) < 0):
        raw_signal = MinMaxScaler(feature_range=(0, 4)).fit_transform(raw_signal.reshape(-1, 1)).squeeze()
    seconds = 2.5
    steps_data = int(fs * seconds)
    total_data = raw_signal.shape[0]
    start = 0
    normal, vfib_vflutter, asys, noise = [], [], [], []
    percentage = {'NORMAL': 0, 'VFIB-VFLUTTER': 0, 'ASYS': 0, 'NOISE': 0}
    model_prediction = []
    while start < total_data:
        end = start + steps_data
        if end - start == steps_data and end < total_data:
            _raw_s_ = raw_signal[start:end]
            if _raw_s_.any():
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
            raw = raw.astype(np.float32) / 255
            rs_raw = resampled_ecg_data(_raw_s_, fs, 500 / seconds)
            if rs_raw.shape[0] != 500:
                rs_raw = signal.resample(rs_raw, 500)
            image_data = (tf.expand_dims(raw, axis=0),)  # tf.constant(rs_raw.reshape(1, -1, 1).astype(np.float32)))
            # image_data = (tf.cast(image_data[0],dtype=tf.float32), )
            model_pred = vfib_predict_tflite_model(model, image_data)[0]
            label = np.argmax(model_pred)
            model_prediction.append(f'{(start, end)}={model_pred}')
            if label == 0:
                normal.append(((start, end), model_pred));
                percentage['NORMAL'] += (end - start) / total_data
            elif label == 1:
                vfib_vflutter.append(((start, end), model_pred));
                percentage['VFIB-VFLUTTER'] += (
                                                       end - start) / total_data
            elif label == 2:
                asys.append(((start, end), model_pred));
                percentage['ASYS'] += (end - start) / total_data
            else:
                noise.append(((start, end), model_pred));
                percentage['NOISE'] += (end - start) / total_data
        start = start + steps_data

    return normal, vfib_vflutter, asys, noise, model_prediction, percentage

def vfib_model_check(ecg_signal, baseline_signal, lowpass_signal, model, fs):
    normal, vfib_vflutter, asys, noise, model_prediction, percentage = vfib_model_pred_tfite(ecg_signal, model, fs)

    final_label_index = np.argmax([percentage['NORMAL'], percentage['VFIB-VFLUTTER'],
                                   percentage['ASYS'], percentage['NOISE']])
    final_label = "NORMAL"
    return final_label

def prediction_model_vfib_vfl(input_arr, vfib_vfl_model):
    classes = ['VFIB', 'asystole', 'noise', 'normal']
    input_arr = tf.io.decode_jpeg(tf.io.read_file(input_arr), channels=3)
    input_arr = tf.image.resize(input_arr, size=(224, 224), method=tf.image.ResizeMethod.BILINEAR)
    input_arr = (tf.expand_dims(input_arr, axis=0),)
    model_pred = predict_tflite_model(vfib_vfl_model, input_arr)[0]
    idx = np.argmax(model_pred)
    return model_pred, classes[idx]

def check_vfib_vfl_model(ecg_signal, vfib_vfl_model):
    baseline_signal = baseline_construction_200(ecg_signal)
    low_ecg_signal = lowpass(baseline_signal, cutoff=0.2)
    label = 'Abnormal'
    temp_uuid = str(uuid.uuid1())
    folder_path = os.path.join("vflutter_img/", temp_uuid)
    os.makedirs(folder_path)

    plt.figure()
    plt.plot(low_ecg_signal)
    plt.axis('off')
    plt.savefig(f'{folder_path}/temp_img.jpg')
    aq = cv2.imread(f'{folder_path}/temp_img.jpg')
    aq = cv2.resize(aq,(1080,460))
    cv2.imwrite(f'{folder_path}/temp_img.jpg',aq)
    plt.close()

    combine_result = []
    label = 'Abnormal'

    files = sorted(glob.glob(f"{folder_path}/*.jpg"), key=extract_number)
    for vfib_file in files:
        with tf.device("CPU"):
            predictions, ids = prediction_model_vfib_vfl(vfib_file, vfib_vfl_model)
        # print(predictions, ids)
        label = "Abnormal"  # "Normal"
        if str(ids) == "VFIB" and float(predictions[0]) > 0.75:
            label = "VFIB/Vflutter"
            combine_result.append(label)
        if str(ids) == "asystole" and float(predictions[1]) > 0.75:
            label = "ASYS"
            combine_result.append(label)

        if str(ids) == "noise" and float(predictions[2]) > 0.75:
            label = "Noise"
            combine_result.append(label)

        if str(ids) == "normal" and float(predictions[3]) > 0.75:
            label = "Normal"
            combine_result.append(label)
    for img_path in glob.glob(f'{folder_path}/*.jpg'):
        os.remove(img_path)

    force_remove_folder(folder_path)
    temp_label = list(set(combine_result))
    if temp_label:
        if len(temp_label) > 1:

            label = 'Abnormal'
            if 'ASYS' in temp_label:
                label = 'ASYS'
            elif 'Noise' in temp_label:
                label = 'Noise'
        else:
            label = temp_label[0]

    return label

# Pacemaker detection
def pacemake_detect(ecg_signal, fs=200):
    pqrst_data = pqrst_detection(ecg_signal, fs=fs, width=(3, 50)).get_data()
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
        percentage = (q_to_pace.count(1) / len(q_to_pace))

    for q in q_index:
        _q = q - qd
        aoi1 = ecg_signal[_q:q]
        if aoi1.any():
            peaks1 = np.where(np.min(aoi1) == aoi1)[0][0]
            peaks1 += _q
            if -0.6 <= ecg_signal[peaks1] <= -0.1 and ecg_signal[q] > ecg_signal[peaks1] and abs(
                    ecg_signal[q] - ecg_signal[peaks1]) >= 0.15 and percentage > 0.5:
                if np.min(np.abs(r_index - peaks1)) > 14:
                    v_pacemaker.append(peaks1)

    for i in range(0, len(r_index) - 1):
        aoi = ecg_signal[s_index[i]:q_index[i + 1]]
        if aoi.any():
            check, _ = find_peaks(aoi, prominence=(0.2, 0.3), distance=100, width=(1, 6))
            peaks1 = check + s_index[i]
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
        atrial_per = round((len(a_pacemaker) / len(r_index)) * 100)
        venti_per = round((len(v_pacemaker) / len(r_index)) * 100)

    if atrial_per > 70 and venti_per > 70:
        pacemaker = np.concatenate((v_pacemaker, a_pacemaker)).astype('int64').tolist()
        pacmaker_per = round((len(a_pacemaker) / len(r_index)) * 100)
        label = "Atrial_&_Ventricular_pacemaker"
    elif atrial_per >= 80 and venti_per >= 80:
        if venti_per > atrial_per:
            label = "Ventricular_Pacemaker"
            pacemaker = v_pacemaker
        else:
            label = "Atrial_Pacemaker"
            pacemaker = a_pacemaker
    elif atrial_per >= 80:
        label = "Atrial_Pacemaker"
        pacemaker = a_pacemaker
    elif venti_per >= 80:
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
        list_per = 0
        more_then_3_rhythm_per = 0
        inner_list_less_2 = 0
        rpeak_diff = np.diff(self.r_index)
        if self.r_index.any():
            more_then_3_rhythm_per = len(list(filter(lambda x: len(x) >= 3, self.p_t))) / len(self.r_index)
            inner_list_less_2 = len(list(filter(lambda x: len(x) < 2, self.p_t))) / len(self.r_index)

        zeros_count = self.p_index.count(0)
        if self.p_index:
            list_per = zeros_count / len(self.p_index)
        pr_int = [round(num, 2) for num in self.pr_inter]

        constant_list = []
        if len(pr_int) > 1:
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
            percentage_diff = np.abs(np.diff(p_peak_diff) / p_peak_diff[:-1]) * 100

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

    def predict_tflite_model(self, model: tuple, input_data: tuple):
        with results_lock:
            interpreter, input_details, output_details = model
            for i in range(len(input_data)):
                interpreter.set_tensor(input_details[i]['index'], input_data[i])
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])

        return output

    def check_model(self, q_new, s_new, ecg_signal, last_s, last_q):
        percent = {'ABNORMAL': 0, 'AFIB': 0, 'FLUTTER': 0, 'NOISE': 0, 'NORMAL': 0, 'total_slice': 0}
        total_data = len(self.s_index) - 1
        afib_data_index, flutter_data_index = [], []
        afib_predictions, flutter_predictions = [], []
        for q, s in zip(q_new, s_new):
            data = ecg_signal[q:s]
            if data.any():
                image_data = self.image_array_new(data)
                image_data = (tf.expand_dims(image_data.astype(np.float32), axis=0),)
                model_pred = self.predict_tflite_model(self.load_model, image_data)[0]

                model_idx = np.argmax(model_pred)
                percent['total_slice'] += 1
                if model_idx == 0:
                    percent['ABNORMAL'] += 1
                elif model_idx == 1:
                    percent['AFIB'] += 1
                    afib_data_index.append((q, s))
                    afib_predictions.append(int(float(model_pred[1])*100))
                elif model_idx == 2:
                    percent['FLUTTER'] += 1
                    flutter_data_index.append((q, s))
                    flutter_predictions.append(int(float(model_pred[2])*100))
                elif model_idx == 3:
                    percent['NOISE'] += 1
                elif model_idx == 4:
                    percent['NORMAL'] += 1
        return percent, afib_data_index, flutter_data_index, afib_predictions, flutter_predictions

    def get_data(self):
        total_data = len(self.s_index) - 1
        last_s = None
        last_q = None
        check_2nd_lead = {'ABNORMAL': 0, 'AFIB': 0, 'FLUTTER': 0, 'NOISE': 0, 'NORMAL': 0, 'total_slice': 0}
        afib_data_index, flutter_data_index = [], []
        afib_predict, flutter_predict = 0, 0
        afib_predictions, flutter_predictions = [], []
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
            check_2nd_lead, afib_data_index, flutter_data_index, afib_predictions, flutter_predictions = self.check_model(q_new, s_new, self.ecg_signal,
                                                                                   last_s, last_q)
        else:
            if len(self.q_index) > 0 and len(self.s_index) > 0:
                q_new = [0]
                s_new = [len(self.ecg_signal)]

                last_q = q_new
                last_s = s_new

                check_2nd_lead, afib_data_index, flutter_data_index, afib_predictions, flutter_predictions = self.check_model(q_new, s_new, self.ecg_signal,
                                                                                       last_s, last_q)
        afib_predict = int(sum(afib_predictions)/ len(afib_predictions)) if afib_predictions else 0
        flutter_predict = int(sum(flutter_predictions)/ len(flutter_predictions)) if flutter_predictions else 0
        return check_2nd_lead, afib_data_index, flutter_data_index, afib_predict, flutter_predict


# Wide-qrs detection
def wide_qrs(q_index, r_index, s_index, hr, fs=100):
    label = 'Abnormal'
    wideQRS = []
    recheck_wide_qrs = []

    thresold = round(fs * 0.12)  # 0.10
    if len(r_index) != 0:

        for k in range(len(r_index)):
            diff = s_index[k] - q_index[k]
            if diff > thresold:
                wideQRS.append(r_index[k])
        if len(wideQRS) / len(r_index) >= 0.90:  # .50
            final_thresh = round(fs * 0.20)  # 0.18
            for k in range(len(r_index)):
                if diff > final_thresh:
                    recheck_wide_qrs.append(r_index[k])

        if len(recheck_wide_qrs) / len(r_index) >= 2.5:
            label = 'WIDE_QRS'
    return label, wide_qrs

def wide_qrs_find_pac(q_index, r_index, s_index, hr_count, fs=200):
    max_indexs = 0
    if hr_count <= 88:
        ms = 0.18  # 0.10
    else:
        ms = 0.16  # 0.12
    max_indexs = int(fs * ms)
    pvc = []
    difference = []
    pvc_index = []
    wide_qs_diff = []
    for k in range(len(r_index)):
        diff = s_index[k] - q_index[k]
        difference.append(diff)
        if max_indexs != 0:
            if diff >= max_indexs:
                pvc.append(r_index[k])
    if hr_count <= 88 and len(r_index) != 0:
        wide_r_index_per = len(pvc) / len(r_index)
        if wide_r_index_per < 0.8:
            pvc_index = np.array(pvc)
        else:
            ms = 0.12
            max_indexs = int(fs * ms)
            for k in range(len(r_index)):
                diff = s_index[k] - q_index[k]
                wide_qs_diff.append(diff)
                if max_indexs != 0:
                    if diff >= max_indexs:
                        pvc_index.append(r_index[k])
            difference = wide_qs_diff
    else:
        pvc_index = np.array(pvc)
    q_s_difference = [i / fs for i in difference]
    return np.array(pvc_index), q_s_difference

def predict_tflite_model(model: tuple, input_data: tuple):
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

    def detect_beats_for_pac(self, ecg, rate, ransac_window_size=3.35, lowfreq=5.0, highfreq=15.0):
        ransac_window_size = int(ransac_window_size * rate)
        lowpass = scipy.signal.butter(1, highfreq / (rate / 2.0), 'low')
        highpass = scipy.signal.butter(1, lowfreq / (rate / 2.0), 'high')
        ecg_low = scipy.signal.filtfilt(*lowpass, x=ecg)
        ecg_band = scipy.signal.filtfilt(*highpass, x=ecg_low)
        decg = np.diff(ecg_band)
        decg_power = decg ** 2
        thresholds, max_powers = [], []
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
        lp_energy = gaussian_filter1d(lp_energy, rate / 14.0)
        lp_energy_diff = np.diff(lp_energy)

        zero_crossings = (lp_energy_diff[:-1] > 0) & (lp_energy_diff[1:] < 0)
        zero_crossings = np.flatnonzero(zero_crossings)
        zero_crossings -= 1

        rpeaks = []
        for idx in zero_crossings:
            search_window = slice(max(0, idx - int(rate * 0.2)), min(len(ecg), idx + int(rate * 0.1)))
            local_signal = ecg[search_window]
            max_amplitude = np.max(local_signal)
            min_amplitude = np.min(local_signal)

            if abs(max_amplitude) > abs(min_amplitude):
                rpeak = np.argmax(local_signal) + search_window.start
            elif abs(max_amplitude + 0.11) < abs(min_amplitude):
                rpeak = np.argmin(local_signal) + search_window.start
            else:
                if max_amplitude >= 0:

                    rpeak = np.argmax(local_signal) + search_window.start
                else:
                    rpeak = np.argmin(local_signal) + search_window.start

            rpeaks.append(rpeak)

        return np.array(rpeaks)

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

        total_one = (1 * at) + (couplet_counter * 2) + (triplet_counter * 3) + (bigeminy_counter * 2) + (
                trigeminy_counter * 2) + (quadrigeminy_counter * 2)
        total = svt_counter + couplet_counter + triplet_counter + bigeminy_counter + trigeminy_counter + quadrigeminy_counter
        ones = PAC_R_Peaks.count(1)
        if total == 0:
            Isolated = ones
        else:
            Common = total - 1
            Isolated = ones - (total_one - Common)
        if hr_counts > 100:
            if svt_counter != 0:
                triplet_counter = couplet_counter = quadrigeminy_counter = trigeminy_counter = bigeminy_counter = Isolated = 0
        if svt_counter >= 1 and hr_counts > 100:  # 190
            svt_counter = 1
        else:
            svt_counter = 0

        data = {"PAC-Isolated_counter": Isolated,
                "PAC-Bigem_counter": bigeminy_counter,
                "PAC-Trigem_counter": trigeminy_counter,
                "PAC-Quadrigem_counter": quadrigeminy_counter,
                "PAC-Couplet_counter": couplet_counter,
                "PAC-Triplet_counter": triplet_counter,
                "SVT_counter": svt_counter}  # svt_counter
        return data

    def predict_pac_model(self, input_arr, target_shape=[224, 224], class_name=True):
        try:
            classes = ['Abnormal', 'Junctional', 'Normal', 'PAC']
            input_arr = tf.keras.preprocessing.image.img_to_array(input_arr)
            input_arr = tf.convert_to_tensor(input_arr, dtype=tf.float32)
            # input_arr = tf.cast(input_arr, dtype=tf.float32)
            # input_arr = tf.convert_to_tensor(input_arr, dtype=tf.float32)
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

            print("PAC ERROR", e)
            return [0, 0, 0, 0], "Normal"

    def get_pac_data(self):
        baseline_signal = baseline_construction_200(self.ecg_signal, kernel_size=131)  # 101
        lowpass_signal = lowpass(baseline_signal, cutoff=0.2)
        r_peaks = self.detect_beats_for_pac(lowpass_signal, self.fs)
        pqrst_data = pqrst_detection(baseline_signal, fs=200, thres=0.37, lp_thres=0.1, rr_thres=0.15).get_data()
        junc_r_label = pqrst_data['R_Label']
        p_index = pqrst_data['P_Index']
        p_t = pqrst_data['P_T List']
        updated_union, junc_union, pac_list = [], [], []
        pac_detect, junc_index = [], []
        pac_label, jr_label = "Abnormal", "Abnormal"
        pac_predictions, junctional_predictions = [], []
        for i in range(len(r_peaks) - 1):
            try:
                # time.sleep(0.1)
                with results_lock:
                    fig, ax = plt.subplots(num=1, clear=True)
                    segment = lowpass_signal[r_peaks[i] - 25:r_peaks[i + 1] + 20]  # 16,20
                    ax.plot(segment, color='blue')
                    ax.axis(False)
                    fig.canvas.draw()
                    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    image = Image.fromarray(data)
                    resized_image = image.resize((360, 720), Image.LANCZOS)

                    plt.close(fig)
                    # with tf.device('/GPU:0'):

                    predictions, ids = self.predict_pac_model(resized_image)
                    if str(ids) == "PAC" and float(predictions[3]) > 0.93:  # 0.91
                        updated_union.append(1)
                        junc_union.append(0)
                        pac_list.append(int(r_peaks[i]))
                        pac_list.append(int(r_peaks[i + 1]))
                        pac_predictions.append(int(float(predictions[3]) * 100))
                        pac_detect.append((int(r_peaks[i]), int(r_peaks[i + 1])))
                    elif str(ids) == "Junctional" and float(predictions[1]) > 0.90:
                        junc_union.append(1)
                        updated_union.append(0)
                        junctional_predictions.append(int(float(predictions[1]) * 100))
                        junc_index.append((int(r_peaks[i]), int(r_peaks[i + 1])))
                    else:
                        updated_union.append(0)
                        junc_union.append(0)
            except Exception as e:
                print(e)
        if junc_r_label == "Regular" and self.hr_counts <= 60:
            if len(r_peaks) != 0:
                junc_count = junc_union.count(1)
                if junc_count / len(r_peaks) >= 0.5:
                    jr_label = "JN_RHY" if self.hr_counts > 40 else "JN_BR"

        pac_data = self.PACcounter(updated_union, self.hr_counts)
        pac_data['pac_plot'] = pac_list
        pac_data['PAC_Union'] = updated_union
        pac_data['PAC_Index'] = pac_detect
        pac_data['jr_label'] = jr_label
        pac_data['pac_label'] = pac_label
        pac_data['pac_predict'] = int(sum(pac_predictions) / len(pac_predictions)) if pac_predictions else 0
        pac_data["junctional_predict"] = int(sum(junctional_predictions) / len(junctional_predictions)) if junctional_predictions else 0
        return pac_data

# long QT detection
def detection_long_qt(ecg_signal, rpeaks, fs=200):
    try:
        _, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=fs, method="peak")
        signal_dwt, waves_dwt = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=fs, method="dwt")

        Tpeaks = np.where(np.isnan(waves_peak['ECG_T_Peaks']), 0, waves_peak['ECG_T_Peaks']).astype('int64').tolist()
        Qpeaks = np.where(np.isnan(waves_peak['ECG_Q_Peaks']), 0, waves_peak['ECG_Q_Peaks']).astype('int64').tolist()
        QTint = []
        finallist = []

        for i in range(len(Qpeaks) - 1):
            try:
                if Qpeaks[i] == 0 or Tpeaks[i] == 0:
                    QTint.append(0)
                else:
                    QT = abs(int(Qpeaks[i]) - int(Tpeaks[i])) / 200
                    QTint.append(QT)
                    if QT > 0.5: finallist.append(QT)  # 0.2
            except:
                QTint.append(0)

        label = "Abnormal"
        if len(finallist) > 5:
            label = "Long_QT_Syndrome"
        return label
    except Exception as r:
        return "Abnormal"

# First-deg block detection
def first_degree_detect(ecg_signal, fs=200):
    pqrst_data = pqrst_detection(ecg_signal, fs=fs, width=(3, 50)).get_data()
    r_index = pqrst_data['R_index']
    q_index = pqrst_data['Q_Index']
    s_index = pqrst_data['S_Index']
    r_Label = pqrst_data['R_Label']
    hr_ = pqrst_data['HR_Count']
    block = []
    label = 'Abnormal'

    # if r_Label == 'Regular' and hr_ <= 90:
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
                    if dist_next_r_index >= 50:  # 0.3 sec
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
        block_per = round(((len(block) / 2) / len(r_index)) * 100)
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
    l = []
    for x in list1:
        if x >= val:
            l.append(1)
        else:
            l.append(0)
    if 1 in l:
        return True
    else:
        return False

def SACompareShort(list1, val1, val2):
    l = []
    for x in list1:
        if x >= val1 and x <= val2:
            l.append(1)
        else:
            l.append(0)
    if 1 in l:
        return True
    else:
        return False

def check_long_short_pause(r_index):
    SAf = []
    # r_interval = np.diff(r_index)
    pause_label = 'Abnormal'
    if len(r_index) > 1:
        for i in range(len(r_index) - 1):
            rr_peaks = abs(int(r_index[i]) * 5 - int(r_index[i + 1]) * 5)
            SAf.append(rr_peaks)

    if (SACompare(SAf, 4500)):
        l = []
        for x in SAf:
            if x >= 4500:
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

    if SACompareShort(SAf, 3500, 4000):
        l = []
        for x in SAf:
            if x >= 3500 and x <= 4000:
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

def combine(ecg_signal, is_lead, class_name, r_id, fs=200, skip_afib_flutter=False):
    baseline_signal, lowpass_signal = filter_signal(ecg_signal, fs).get_data()
    pace_label, pacemaker_index = pacemake_detect(baseline_signal, fs=fs)

    pac_data = {
        'PAC_Union': [],
        "PAC_Index": [],
        "PAC_Isolated": 0,
        "PAC_Bigeminy": 0,
        "PAC_Trigeminy": 0,
        "PAC_Quadrigeminy": 0,
        "PAC_Couplet": 0,
        "PAC_Triplet": 0,
        "PAC_SVT": 0,
        "jr_label": "Abnormal",
        "pac_predict": 0,
        "junctional_predict": 0}
    
    pvc_data = {'PVC-Index': [], "PVC-QRS_difference": [], "PVC-wide_qrs": np.array([]), 'pvc_pred': 0,
                    'lbbb_rbbb_label': 'Abnormal', "lbbb_rbbb_per":0}
    afib_predict, flutter_predict, block_model_prediction = 0, 0, 0


    #    vfib_or_asystole_output = check_vfib_vfl_model(ecg_signal, vfib_vfl_model)
    vfib_or_asystole_output = vfib_model_check(ecg_signal, baseline_signal, lowpass_signal, vfib_model, fs)

    if vfib_or_asystole_output == "Abnormal" or vfib_or_asystole_output == "NORMAL":

        pqrst_data = pqrst_detection(baseline_signal, class_name=class_name, fs=fs).get_data()
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

        afib_label = jr_label = first_deg_block_label = second_deg_block = third_deg_block = aflutter_label = longqt_label = first_degree_block = PAC_label = abs_result = final_block_label = check_pause = 'Abnormal'
        temp_index = wide_qrs_list = []
        pvc_class = []
        pac_class = ['Abnormal']
        

        if len(r_index) != 0 or len(s_index) != 0 or len(q_index) != 0:
            if (is_lead == 'II' or is_lead == 'III' or is_lead == "I" or is_lead == 'V1'
                    or is_lead == 'V2' or is_lead == 'V5' or is_lead == 'V6'):
                pvc_data = PVC_detection(ecg_signal, r_id, fs).get_pvc_data()

                pvc_count = pvc_data['PVC-Count']

                temp_pvc = []
                for key, val in pvc_data.items():
                    if 'counter' in key and val > 0:
                        temp_pvc.append(key.split('_')[0])
                if len(temp_pvc) != 0:
                    pvc_class = [label.replace('-', '_') for label in temp_pvc]
                else:
                    pvc_class = temp_pvc

            wide_qrs_label, _ = wide_qrs(q_index, r_index, s_index, hr_counts, fs=fs) if len(pvc_class) == 0 else (
                "Abnormal", [])
            temp_index, wide_qrs_list = wide_qrs_find_pac(q_index, r_index, s_index, hr_counts, fs=fs)
        else:
            wide_qrs_label = 'Abnormal'
            
            
        if not skip_afib_flutter and class_name == '3_4':
            if is_lead == 'II' or is_lead == 'III' or is_lead == 'I' or is_lead == 'V5' or is_lead == 'V6':
                afib_flutter_check = afib_flutter_detection(lowpass_signal, r_index, q_index, s_index, p_index,
                                                            p_t,
                                                            pr_interval, afib_model)

                is_afib_flutter = afib_flutter_check.abs_afib_flutter_check()
                print('check is_afib_flutter:', is_afib_flutter)
                afib_model_per = flutter_model_per = 0
                if is_afib_flutter:
                    afib_flutter_per, afib_indexs, flutter_indexs, afib_predict, flutter_predict = afib_flutter_check.get_data()
                    if afib_flutter_per['total_slice']>0:
                        afib_model_per = int((afib_flutter_per['AFIB'] / afib_flutter_per['total_slice']) * 100)
                        flutter_model_per = int((afib_flutter_per['FLUTTER'] / afib_flutter_per['total_slice']) * 100)
                if afib_model_per >= 40:
                    afib_label = 'AFIB'

                if afib_label != 'AFIB':
                    if flutter_model_per >= 60:
                        aflutter_label = 'AFL'

        if all(p not in ['VT', 'IVR', 'NSVT', 'PVC-Triplet', 'PVC-Couplet'] for p in pvc_class) and len(
                r_index) > 0:  # 'PVC-Triplet', 'PVC-Couplet'
            if hr_counts <= 60:
                check_pause = check_long_short_pause(r_index)
            if r_label == 'Regular':
                if aflutter_label != "AFL" and afib_label != "AFIB":
                    if is_lead == 'II' or is_lead == 'III' or is_lead == "I" or is_lead == "V1" or is_lead == "V2" or is_lead == "V5" or is_lead == "V6":
                        pac_data = PAC_detedction(ecg_signal, fs, hr_counts).get_pac_data()
                        if r_label == 'Regular':
                            jr_label = pac_data['jr_label']
                        if is_lead == "II" or is_lead == "III" or is_lead == "I" or is_lead == "V1":
                            if all('PVC' not in p for p in pvc_class) and all(
                                    'Abnormal' in l for l in [afib_label, aflutter_label]):
                                if hr_counts >= 55:
                                    temp_pac = '; '.join(
                                        [key.split('_')[0] for key, val in pac_data.items() if
                                         'counter' in key and val > 0])
                                    pac_class = temp_pac.replace('-', '_')
                                else:
                                    pac_class = ""
                                    pac_data['PAC_Union'] = []
                                    pac_data['PAC_Index'] = []

                if is_lead == "II" or is_lead == "III" or is_lead == "I" or is_lead == "V1" or is_lead == "V2" or is_lead == "V4" or is_lead == 'V5':
                    if all('Abnormal' in l for l in
                           [afib_label, aflutter_label]):  # and len(pac_class) == 0 and len(pvc_class) == 0
                        lowpass_signal = lowpass(baseline_signal, 0.3)
                        first_deg_block_label, first_deg_block_index = first_degree_detect(lowpass_signal, fs)
                        abs_result = first_deg_block_label
                    if hr_counts <= 80:
                        if all('Abnormal' in l for l in [afib_label, aflutter_label, first_deg_block_label,
                                                         jr_label]):  # and len(pac_class) == 0 and len(pvc_class) == 0
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
                        final_block_label, block_ei_ti, block_model_prediction = block_model_check(ecg_signal, fs, abs_result)
                if all('Abnormal' in l for l in [afib_label, aflutter_label]) and len(pac_class) == 0 and len(
                        pvc_class) == 0:
                    lowpass_signal = lowpass(baseline_signal, 0.3)
                    longqt_label = detection_long_qt(lowpass_signal, r_index, fs)
            else:
#                if not skip_afib_flutter:
#                    if is_lead == 'II' or is_lead == 'III' or is_lead == 'I' or is_lead == 'V5' or is_lead == 'V6':
#                        afib_flutter_check = afib_flutter_detection(lowpass_signal, r_index, q_index, s_index, p_index,
#                                                                    p_t,
#                                                                    pr_interval, afib_model)
#                        is_afib_flutter = afib_flutter_check.abs_afib_flutter_check()
#                        afib_model_per = flutter_model_per = 0
#                        if is_afib_flutter:
#                            afib_flutter_per, afib_indexs, flutter_indexs = afib_flutter_check.get_data()
#                            afib_model_per = int(afib_flutter_per['AFIB'] * 100)
#                            flutter_model_per = int(afib_flutter_per['FLUTTER'] * 100)
#                        if afib_model_per >= 40:
#                            afib_label = 'AFIB'
#
#                        if afib_label != 'AFIB':
#                            if flutter_model_per >= 60:
#                                aflutter_label = 'AFL'
                if afib_label != 'AFIB':
                    if is_lead == "II" or is_lead == "III" or is_lead == "I" or is_lead == "V1" or is_lead == 'V2' or is_lead == 'V5' or is_lead == 'V6':
                        pac_data = PAC_detedction(ecg_signal, fs, hr_counts).get_pac_data()
                        if is_lead == 'II' or is_lead == 'III' or is_lead == "aVF":
                            jr_label = pac_data['jr_label']
                        if all('PVC' not in p for p in pvc_class) and all('Abnormal' in l for l in
                                                                          [afib_label, aflutter_label,
                                                                           check_pause]) and hr_counts <= 100:
                            temp_pac = '; '.join(
                                [key.split('_')[0] for key, val in pac_data.items() if 'counter' in key and val > 0])
                            pac_class = temp_pac.replace('-', '_')
                    if is_lead == "II" or is_lead == "III" or is_lead == "I" or is_lead == "V1" or is_lead == "V2" or is_lead == "V4" or is_lead == 'V5':
                        if all('Abnormal' in l for l in
                               [afib_label, aflutter_label]):  # and len(pac_class) == 0 and len(pvc_class) == 0
                            lowpass_signal = lowpass(baseline_signal, 0.3)
                            first_deg_block_label, first_deg_block_index = first_degree_detect(lowpass_signal, fs)
                            abs_result = first_deg_block_label

                        if hr_counts <= 80:
                            if all('Abnormal' in l for l in
                                   [afib_label, aflutter_label, first_deg_block_label, jr_label,
                                    check_pause]):  # and len(pac_class) == 0 and len(pvc_class) == 0
                                second_deg_block = BlockDetected(ecg_signal, fs).second_degree_block_detection()
                            if second_deg_block != 'Abnormal':
                                abs_result = second_deg_block
                            if all('Abnormal' in l for l in
                                   [afib_label, aflutter_label, first_deg_block_label, second_deg_block, jr_label,
                                    check_pause]):  # and len(pac_class) == 0 and len(pvc_class) == 0
                                third_deg_block = BlockDetected(ecg_signal, fs).third_degree_block_deetection()
                            if third_deg_block != 'Abnormal':
                                abs_result = third_deg_block
                        if abs_result != 'Abnormal':
                            final_block_label, block_ei_ti, block_model_prediction = block_model_check(ecg_signal, fs, abs_result)

            pac_class = "Abnormal" if pac_class == '' else pac_class
            label = {'Afib_label': afib_label,
                     'Aflutter_label': aflutter_label,
                     'JR_label': jr_label,
                     'wide_qrs_label': wide_qrs_label,
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

        if c_label in ["", "; "]: c_label = 'NORMAL'

        data = {'Input_Signal': ecg_signal,
                'Baseline_Signal': baseline_signal,
                'Lowpass_signal': lowpass_signal,
                # 'Combine_Label':c_label.upper().replace("_","-"),
                'Combine_Label': c_label,
                'RR_Label': r_label,
                'R_Index': r_index,
                'Q_Index': q_index,
                'S_Index': s_index,
                'J_Index': j_index,
                'T_Index': t_index,
                'P_Index': p_index,
                'Ex_Index': ex_index,
                'P_T': pt,
                'HR_Count': hr_counts,
                'PVC_DATA': pvc_data,
                'PAC_DATA': pac_data,
                'PaceMaker': pace_label,
                "afib_predict": afib_predict, 
                "flutter_predict": flutter_predict, 
                "block_model_prediction": block_model_prediction 
                }
    else:
        data = {'Input_Signal': ecg_signal,
                'Baseline_Signal': baseline_signal,
                'Lowpass_signal': lowpass(baseline_signal, 0.3),
                'Combine_Label': vfib_or_asystole_output.upper(),
                'RR_Label': 'Not Defined',
                'R_Index': np.array([]),
                'Q_Index': [],
                'S_Index': [],
                'J_Index': [],
                'T_Index': [],
                'P_Index': [],
                'Ex_Index': [],
                'P_T': [],
                'HR_Count': 0,
                'PVC_DATA': pvc_data,
                'PAC_DATA': pac_data,
                'PaceMaker': pace_label, 
                "afib_predict": afib_predict, 
                "flutter_predict": flutter_predict,
                "block_model_prediction": block_model_prediction}

    return data

# R peak detection using biosppy
class RPeakDetection:
    def __init__(self, baseline_signal, fs=200):
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
            while c > 0 and ecg[c - 1] < ecg[c]:
                c -= 1
            if ecg[i] * 0.01 > ecg[c] or ecg[c] < 0 or c == 0:
                if abs(i - c) <= d:
                    q.append(c)
                    continue
                else:
                    q_.append(c)
            while c > 0:
                while c > 0 and ecg[c - 1] > ecg[c]:
                    c -= 1
                # q_.append(c)
                while c > 0 and ecg[c - 1] < ecg[c]:
                    c -= 1
                if q_ and q_[-1] == c:
                    break
                q_.append(c)
                if ecg[i] * 0.01 > ecg[c] or ecg[c] < 0 or c == 0:
                    break
        else:
            c = i
            while c > 0 and ecg[c - 1] > ecg[c]:
                c -= 1
            if ecg[i] * 0.01 < ecg[c] or ecg[c] > 0 or c == 0:
                if abs(i - c) <= d:
                    q.append(c)
                    continue
                else:
                    q_.append(c)
            while c > 0:
                while c > 0 and ecg[c - 1] < ecg[c]:
                    c -= 1
                # q_.append(c)
                while c > 0 and ecg[c - 1] > ecg[c]:
                    c -= 1
                if q_ and q_[-1] == c:
                    break
                q_.append(c)
                if ecg[i] * 0.01 < ecg[c] or ecg[c] > 0 or c == 0:
                    break
        if q_:
            a = 0
            for _q in q_[::-1]:
                if abs(i - _q) <= d:
                    a = 1
                    q.append(_q)
                    break
            if a == 0:
                q.append(q_[0])
    return np.sort(q)

def extract_number(filename):
    match = re.search(r'(\d+)', os.path.basename(filename))
    return int(match.group(1)) if match else float('inf')

def prediction_model_mi(input_arr):
    classes = ['Abnormal', 'stdep', 'stele', 't_abnormal', 't_invert']
    input_arr = tf.io.decode_jpeg(tf.io.read_file(input_arr), channels=3)
    input_arr = tf.image.resize(input_arr, size=(150,400), method=tf.image.ResizeMethod.BILINEAR) # (224, 224)
    input_arr = (tf.expand_dims(input_arr, axis=0),)
    model_pred = predict_tflite_model(let_inf_moedel, input_arr )[0]
    idx = np.argmax(model_pred)
    return model_pred, classes[idx]

def check_st_model(ecg_data, fs, _id):
    label, ei_ti_index = 'Abnormal', 'Abnormal'
    mi_temp_img = os.path.join('STsegimages/' + _id)
    if not os.path.exists(mi_temp_img):
        os.mkdir('STsegimages/' + _id)
    
    i = 0
    if ecg_data.shape[0] <= 2500:
        steps = ecg_data.shape[0]
    else:
        steps = round(fs * 10)
    result_dic = {'label':"Normal", 'model_per':0.0}
    result_count = {'t_invert':0, 'stdep':0, 'stele':0, 't_abnormal':0, 'total_data':0}
    model_pre = []
    while i < ecg_data.shape[0]:
        ecg_signal = ecg_data[i : i+steps]
        baseline_signal = baseline_construction_200(ecg_signal, kernel_size=101)
        low_ecg_signal = lowpass(baseline_signal, cutoff=0.3)
        for rm_file in glob.glob(mi_temp_img+"/*.jpg"):
            os.remove(rm_file)
        randome_number = random.randint(200000, 1000000)
        if len(low_ecg_signal)<700:
            plt.figure()
        else:
            plt.figure(layout="constrained")
        plt.plot(low_ecg_signal, color='blue', linewidth=1.5)
        plt.axis("off")
        plt.savefig(f"{mi_temp_img}/P_{randome_number}.jpg")
        aq = cv2.imread(f"{mi_temp_img}/P_{randome_number}.jpg")
        aq = cv2.resize(aq, (1200, 290))
        cv2.imwrite(f"{mi_temp_img}/P_{randome_number}.jpg", aq)
        plt.close()
        predictions, ids = prediction_model_mi(f"{mi_temp_img}/P_{randome_number}.jpg")
        result_count['total_data'] += 1
        if str(ids) == "t_invert" and float(predictions[4]) > 0.70:
            model_pre.append(int(float(predictions[4])*100))
            result_count['t_invert'] += 1
        elif str(ids) == "stdep" and float(predictions[1]) > 0.70:
            model_pre.append(int(float(predictions[1])*100))
            result_count['stdep'] += 1
        elif str(ids) == "stele" and float(predictions[2]) > 0.70:
            model_pre.append(int(float(predictions[2])*100))
            result_count['stele'] += 1
        elif str(ids) == "t_abnormal" and float(predictions[3]) > 0.70:
            model_pre.append(int(float(predictions[3])*100))
            result_count['t_abnormal'] += 1
        i += steps

    if result_count['total_data'] != 0:
        t_inver_per = result_count['t_invert'] / result_count['total_data']
        stdep_per = result_count['stdep'] / result_count['total_data']
        stele_per = result_count['stele'] / result_count['total_data']
        t_abn_per = result_count['t_abnormal'] / result_count['total_data']
        if t_inver_per > 0.6:
            label = "TAB"
            result_dic['label'] = label
        elif stdep_per > 0.6:
            label = "STDEP"
            result_dic['label'] = label
        elif stele_per > 0.6:
            label = "STELE"
            result_dic['label'] = label
        elif t_abn_per > 0.6:
            label = "TAB" 
            result_dic['label'] = label
        else:
            label = "Normal"

        if result_dic['label'] != "Normal":
            result_dic['model_per'] = int(sum(model_pre) / len(model_pre))
    for rm_file in glob.glob(mi_temp_img +"/*.jpg"):
        os.remove(rm_file)
    force_remove_folder(mi_temp_img)
    return result_dic

# Define a function to validate 'digits / digits ms' pattern
def is_single_format(value):
    return re.match(r'^\d+\s?[\|/]\s?\d+\s?ms$', value.strip(), re.IGNORECASE)

# Function to validate HR values
def validate_hr(hr_values):
    validated_hr = []
    for hr in hr_values:
        # Remove non-digit characters
        hr_cleaned = re.sub(r'\D', '', hr)
        # Check if the cleaned HR is a 2 or 3 digit number
        if hr_cleaned.isdigit() and 10 <= int(hr_cleaned) <= 250:
            validated_hr.append(int(hr_cleaned))
        else:
            validated_hr.append(None)
    return validated_hr

# Define a function to classify MI results
def classify_mi_result(input_list):
    keywords = ['lateral', 'inferior', 'abnormality']
    mi_result = [item for item in input_list if any(keyword in item.lower() for keyword in keywords)]
    result_classification = []
    for item in mi_result:
        if 'abnormality' in item.lower():
            result_classification.append('T_wave_Abnormality')
        if ' lateral' in item.lower() or item.lower().startswith('lateral'):
            result_classification.append('Lateral_MI')
        if 'inferior' in item.lower():
            result_classification.append('Inferior_MI')
    result_classification = list(set(result_classification))
    return result_classification

def classify_arrhythmia(input_list):
    afib_afl = []
    afib_keywords = ['atrial fibrillation', 'atrial fib', 'fibrillation']
    aflutter_keywords = ['atrial flutter', 'flutter']
    # Search for AFIB keywords
    if any(keyword in " ".join(input_list).lower() for keyword in afib_keywords):
        afib_afl.append('AFIB')
    # Search for AFLUTTER keywords
    if any(keyword in " ".join(input_list).lower() for keyword in aflutter_keywords):
        afib_afl.append('AFL')
    return afib_afl

# Function to classify hypertrophy correctly
def classify_hypertrophy(input_list):
    hypertrophy = []
    # Normalize and join all text
    input_text = " ".join(input_list).lower()
    # Replace underscores, slashes, and hyphens with spaces
    input_text = re.sub(r'[_/-]', ' ', input_text)
    input_text = re.sub(r'\s+', ' ', input_text)  # Collapse multiple spaces

    # Check for RVH
    if ('right ventricular hypertrophy' in input_text) or ('rvh' in input_text):
        hypertrophy.append('RVH')
    # Check for LVH
    if ('left ventricular hypertrophy' in input_text) or ('lvh' in input_text):
        hypertrophy.append('LVH')
    # Check for generic "ventricular hypertrophy"
    if 'ventricular hypertrophy' in input_text:
        # Use regex to extract a window of 3–4 words around the match
        match = re.search(r'(.{0,30}ventricular hypertrophy.{0,30})', input_text)
        if match:
            context = match.group(1)
            if 'left' not in context and 'right' not in context and 'lvh' not in context and 'rvh' not in context:
                hypertrophy.append('LVH')

    return list(set(hypertrophy))  # Remove duplicates

# Function to validate QT/QTcBaz and RR/PP values with duplicate and format check
def validate_intervals(interval_list):
    validated_intervals = []
    # Updated regex pattern to match 'xxx / xxx' or 'xxxx / xxxx' with or without 'ms'
    pattern = re.compile(r'(\d{3,4})\s*[/|]\s*(\d{3,4})', re.IGNORECASE)
    for interval in interval_list:
        match = pattern.search(interval)
        if match:
            # Format as 'xxx / xxx' or 'xxxx / xxxx' (removes 'ms' dependency)
            formatted_interval = f"{match.group(1)} / {match.group(2)}"
            validated_intervals.append(formatted_interval)
        else:
            validated_intervals.append(None)

    # Remove duplicates
    return list(dict.fromkeys(validated_intervals))

# Function to remove 'ms', split values, and remove square brackets
def process_and_split(values):
    processed_dict = {"Part1": [], "Part2": []}
    for value in values:
        if value:
            # Remove 'ms', clean value, and split by '/'
            cleaned_value = re.sub(r'\s?ms$', '', value).strip()
            parts = cleaned_value.split('/')
            # Add split parts to respective keys, trimming extra spaces
            if len(parts) == 2:
                processed_dict["Part1"].append(parts[0].strip())
                processed_dict["Part2"].append(parts[1].strip())
    return processed_dict

def text_detection(img_path):
    img = cv2.imread(img_path)
    result = reader.readtext(img)
    HR = []
    QT_QTcBaz = []
    RR_PP = []
    QRS_values = []
    PR_values = []
    output_dict = {}
    extracted_text = [detection[1] for detection in result]

    # print('Extracted text:', extracted_text)
    for i, text in enumerate(extracted_text):
        # print(f"Index {i}: {text}")
        # Detect HR (Heart Rate)

        match = re.search(r'(\d+)\s?(BPM|bpm|opm|bprn|bpr)', text)  # Match '137bpm' or '137 bpm'
        if match:
            HR.append(match.group(1))  # Extract only the numeric part
            print(f"Debug: Found HR -> {match.group(1)}")
        elif any(keyword in text for keyword in ['BPM', 'bpm', 'opm', 'bprn', 'bpr']) and i > 0:
            prev_value = extracted_text[i - 1].strip()
            if prev_value.isdigit():
                HR.append(prev_value)
                print(f"Debug: Found HR -> {prev_value}")

        # Detect 'QRS' (case-insensitive) and validate its next value
        if text.lower() in ['qrs', 'qrs duration'] and i + 1 < len(extracted_text):
            next_value = extracted_text[i + 1]
            # Check if the next value is in the format 'xx ms' or 'xxx ms'
            if re.match(r'^\d{2,3}\s?ms$', next_value.strip(), re.IGNORECASE):
                QRS_values.append(next_value.strip())
                print(f"Debug: Found QRS -> {next_value.strip()}")
            elif next_value.isdigit():
                QRS_values.append(next_value)
                print(f"Debug: Found QRS (without ms) -> {next_value}")

        # Detect 'PR' (case-insensitive) and validate its next value
        if text.lower() in ['pr', 'pr interval'] and i + 1 < len(extracted_text):
            next_value = extracted_text[i + 1]
            # Check if the next value is in the format 'xx ms' or 'xxx ms'
            if re.match(r'^\d{2,3}\s?ms$', next_value.strip(), re.IGNORECASE):
                PR_values.append(next_value.strip())
                print(f"Debug: Found PR -> {next_value.strip()}")
            elif next_value.isdigit():
                PR_values.append(next_value)
                print(f"Debug: Found PR (without ms) -> {next_value}")

        if (
                'QT / QTcBaz' in text or 'QT/ QTcB' in text or 'QT / QTcB' in text or 'QT/QTcB' in text or 'QTI/ QTcBaz' in text
                or 'QTIQTc-Baz' in text or 'QTQTc-Baz' in text or (
                text == 'QT' and i + 1 < len(extracted_text) and extracted_text[i + 1] == 'QTcBaz')):
            # If 'QT' and 'QTcBaz' are separate strings
            # print('----',text)
            if text == 'QT' and i + 1 < len(extracted_text) and extracted_text[i + 1] == 'QTcBaz':
                qt_index = i + 1
            else:
                qt_index = i

            if qt_index + 1 < len(extracted_text) and is_single_format(extracted_text[qt_index + 1]):
                QT_QTcBaz.append(extracted_text[qt_index + 1])
            elif qt_index + 2 < len(extracted_text):
                value = f"{extracted_text[qt_index + 1]} / {extracted_text[qt_index + 2]}"
                QT_QTcBaz.append(value)

        if text == 'QT:' and i + 2 < len(extracted_text) and extracted_text[i + 1] == 'QTcBaz':
            next_value = extracted_text[i + 2]
            if is_single_format(next_value):
                QT_QTcBaz.append(next_value)

        if 'RR / PP' in text or 'RR/PP' in text or 'RR/ PP' in text or 'RRIPP' in text or 'PP' in text or 'RR / Pp' in text:
            if i + 1 < len(extracted_text):
                next_value = extracted_text[i + 1]
                if is_single_format(next_value):
                    RR_PP.append(next_value)
                elif re.match(r'^\d+:\s*/\s*\d+\.ms$', next_value):
                    match = re.search(r'^(\d+):\s*/\s*(\d+)\.ms$', next_value)
                    if match:
                        normalized_value = f"{match.group(1)} / {match.group(2)} ms"
                        RR_PP.append(normalized_value)
                elif i + 2 < len(extracted_text):
                    first_value = extracted_text[i + 1]
                    second_value = extracted_text[i + 2]
                    if second_value.lower().endswith('ms'):
                        value = f"{first_value} / {second_value}"
                        RR_PP.append(value)
                    elif first_value.isdigit() and second_value.isdigit():
                        value = f"{first_value} / {second_value} ms"
                        RR_PP.append(value)

        if text == 'RR' and i + 2 < len(extracted_text) and extracted_text[i + 1] == 'PP':
            next_value = extracted_text[i + 2]
            if is_single_format(next_value):
                RR_PP.append(next_value)
            elif i + 3 < len(extracted_text):
                first_value = extracted_text[i + 2]
                second_value = extracted_text[i + 3]
                if second_value.lower().endswith('ms'):
                    value = f"{first_value} / {second_value}"
                    RR_PP.append(value)
                elif first_value.isdigit() and second_value.isdigit():
                    value = f"{first_value} / {second_value} ms"
                    RR_PP.append(value)

        if text == 'RR:' and i + 2 < len(extracted_text) and extracted_text[i + 1] == 'PP':
            next_value = extracted_text[i + 2]
            if is_single_format(next_value):
                RR_PP.append(next_value)
            elif i + 3 < len(extracted_text):
                first_value = extracted_text[i + 2]
                second_value = extracted_text[i + 3]
                if second_value.lower().endswith('ms'):
                    value = f"{first_value} / {second_value}"
                    RR_PP.append(value)

        if text == 'RR /: PP' and i + 1 < len(extracted_text):
            next_value = extracted_text[i + 1]
            if is_single_format(next_value):
                RR_PP.append(next_value)

    validated_hr_list = validate_hr(HR)
    validated_qt_qtc_list = validate_intervals(QT_QTcBaz)
    validated_rr_pp_list = validate_intervals(RR_PP)
    classification = classify_mi_result(extracted_text)
    classify_arrhy = classify_arrhythmia(extracted_text)
    hypertrophy = classify_hypertrophy(extracted_text)

    if validated_hr_list:
        output_dict["HR"] = validated_hr_list[0]
    if validated_qt_qtc_list or validated_rr_pp_list:
        try:
            qt_split = process_and_split(validated_qt_qtc_list)
            rr_split = process_and_split(validated_rr_pp_list)
            output_dict["QT"] = int(qt_split["Part1"][0]) if qt_split['Part1'] else 0
            output_dict['QTcBaz'] = int(qt_split["Part2"][0]) if qt_split['Part2'] else 0
            output_dict["RR"] = int(rr_split["Part1"][0]) if rr_split['Part1'] else 0
            output_dict["PP"] = int(rr_split["Part2"][0]) if rr_split['Part2'] else 0
        except Exception as e:
            print(e)

    if classification:
        output_dict["MI"] = classification

    if classify_arrhy:
        output_dict["Arrhythmia"] = list(set(classify_arrhy))

    if hypertrophy:
        output_dict['Hypertrophy'] = hypertrophy

    if QRS_values:
        try:
            output_dict["QRS"] = int(QRS_values[0].split(' ')[0])
        except:
            output_dict["QRS"] = int(QRS_values[0].split('ms')[0])
    if PR_values:
        try:
            output_dict["PR"] = int(PR_values[0].split(' ')[0])
        except:
            output_dict["PR"] = int(PR_values[0].split('ms')[0])
    return output_dict

def find_ecg_info(ecg_signal, img_type, image_path):
    if img_type == '12_1':
        fa = 130
    elif img_type == '3_4':
        fa = 60
    else:
        fa = 110
    ocr_results = {}
    # Only run OCR if image_path is a valid file
    # if img_type == '6_2' and image_path and isinstance(image_path, str) and os.path.exists(image_path):
    #     ocr_results = text_detection(image_path)
    # print(ocr_results, "========ocr_results")
    rpeaks = detect_beats(ecg_signal, float(fa))
    rr_interval = []
    data_dic = {"rr_interval": 0,
                "PRInterval": 0,
                "QTInterval": 0,
                "QRSComplex": 0,
                "STseg": 0,
                "PRseg": 0,
                "QTc": 0,
                "QTcBaz":0,
                "QT":0,
                "RR":0,
                "QRS":0
                }
    try:
        _, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=fa, method="peaks")
        signal_dwt, waves_dwt = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=fa, method="dwt")
        Tpeaks = np.where(np.isnan(waves_peak['ECG_T_Peaks']), 0, waves_peak['ECG_T_Peaks']).astype('int64').tolist()
        Qpeaks = np.where(np.isnan(waves_peak['ECG_Q_Peaks']), 0, waves_peak['ECG_Q_Peaks']).astype('int64').tolist()
    except Exception as e:
        print('Nurokit error:', e)
    for i in range(len(rpeaks) - 1):
        try:
            RRpeaks = abs(int(rpeaks[i]) * 3 - int(rpeaks[i + 1]) * 3)
            rr_interval.append(RRpeaks)
        except:
            rr_interval.append(0)
            RRpeaks = "0"
    try:
        data_dic['rr_interval'] = rr_interval[0]
    except:
        data_dic['rr_interval'] = "100"
    try:
        Ppeak = waves_peak['ECG_P_Peaks'][1]
        Rpeak = rpeaks[1]
        Ppeak = int(Ppeak) * 3
        Rpeak = int(Rpeak) * 3
        PRpeaks = abs(Rpeak - Ppeak)
    except:
        PRpeaks = "0"
    data_dic['PRInterval'] = PRpeaks
    try:
        Tpeak = waves_peak['ECG_T_Peaks'][1]
        Qpeak = waves_peak['ECG_Q_Peaks'][1]
        Tpeak = int(Tpeak) * 3
        Qpeak = int(Qpeak) * 3
        QTpeaks = abs(Tpeak - Qpeak)
    except:
        QTpeaks = "0"
    data_dic['QTInterval'] = QTpeaks
    try:
        Speak = waves_peak['ECG_S_Peaks'][1]
        Qpeak = waves_peak['ECG_Q_Peaks'][1]
        Speak = int(Speak) * 3
        Qpeak = int(Qpeak) * 3
        SQpeaks = abs(Speak - Qpeak)
    except:
        SQpeaks = "0"
    data_dic['QRSComplex'] = SQpeaks
    try:
        Spa = waves_peak['ECG_S_Peaks'][1]
        Ton = waves_dwt['ECG_T_Onsets'][1]
        Spa = int(Spa) * 3
        Ton = int(Ton) * 3
        STseg = abs(Ton - Spa)
    except:
        STseg = "0"
    data_dic['STseg'] = STseg
    try:
        PP = waves_dwt['ECG_P_Offsets']
        RRO = waves_dwt['ECG_R_Onsets']
        if math.isnan(PP[2]) or math.isnan(RRO[2]):
            PRseg = "0"
        else:
            PPIn = int(PP[1]) * 3
            RRon = int(RRO[1]) * 3
            PRseg = abs(PPIn - RRon)
    except:
        PRseg = "0"
    data_dic['PRseg'] = PRseg

    QTint = []
    finallist = []
    try:
        for i in range(len(Qpeaks) - 1):
            try:
                if Qpeaks[i] == 0 or Tpeaks[i] == 0:
                    QTint.append(0)
                else:
                    QT = abs(int(Qpeaks[i]) - int(Tpeaks[i])) / 200
                    QTint.append(QT * 1000)
                    # if QT>0.5: finallist.append(QT)  #0.2
            except:
                QTint.append(0)
    except:
        QTint.append(0)
    data_dic['QTc'] = QTint[0]
    if ocr_results:
        if 'QTcBaz' in ocr_results and ocr_results['QTcBaz'] != 0:
            data_dic['QTc'] = ocr_results['QTcBaz']

        if 'QT' in ocr_results and ocr_results['QT'] != 0:
            data_dic['QTInterval'] = ocr_results['QT']

        if 'RR' in ocr_results and ocr_results['RR'] != 0:
            data_dic['rr_interval'] = ocr_results['RR']

        if 'QRS' in ocr_results and ocr_results['QRS'] != 0:
            data_dic['QRSComplex'] = ocr_results['QRS']

        if 'HR' in ocr_results and ocr_results['HR'] != 0:
            data_dic['HR'] = ocr_results['HR']

        if 'PR' in ocr_results and ocr_results['PR'] != 0:
            data_dic['PRInterval'] = ocr_results['PR']
        if 'MI' in ocr_results:
            data_dic['MI'] = ocr_results['MI']
        if 'Hypertrophy' in ocr_results:
            data_dic['Hypertrophy'] = ocr_results['Hypertrophy']
        if 'Arrhythmia' in ocr_results:
            data_dic['Arrhythmia'] = list(set(ocr_results['Arrhythmia']))
    return data_dic

def detect_beats(ecg, rate, ransac_window_size=3.35, lowfreq=5.0, highfreq=9.0):
    ransac_window_size = int(ransac_window_size * rate)
    lowpass = scipy.signal.butter(1, highfreq / (rate / 2.0), 'low')
    highpass = scipy.signal.butter(1, lowfreq / (rate / 2.0), 'high')
    ecg_low = scipy.signal.filtfilt(*lowpass, x=ecg)
    ecg_band = scipy.signal.filtfilt(*highpass, x=ecg_low)
    decg = np.diff(ecg_band)
    decg_power = decg ** 2
    thresholds, max_powers = [], []
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
    lp_energy = gaussian_filter1d(lp_energy, rate / 14.0)
    lp_energy_diff = np.diff(lp_energy)

    zero_crossings = (lp_energy_diff[:-1] > 0) & (lp_energy_diff[1:] < 0)
    zero_crossings = np.flatnonzero(zero_crossings)
    zero_crossings -= 1
    # print(zero_crossings)

    rpeaks = []
    for idx in zero_crossings:
        search_window = slice(max(0, idx - int(rate * 0.2)), min(len(ecg), idx + int(rate * 0.1)))
        local_signal = ecg[search_window]
        max_amplitude = np.max(local_signal)
        min_amplitude = np.min(local_signal)

        if abs(max_amplitude) > abs(min_amplitude):
            rpeak = np.argmax(local_signal) + search_window.start
        elif abs(max_amplitude + 0.11) < abs(min_amplitude):
            rpeak = np.argmin(local_signal) + search_window.start
        else:
            if max_amplitude >= 0:
                rpeak = np.argmax(local_signal) + search_window.start
            else:
                rpeak = np.argmin(local_signal) + search_window.start

        rpeaks.append(rpeak)

    return np.array(rpeaks)

def lad_rad_detect_beats(ecg,rate,ransac_window_size=3.0,lowfreq=5.0,highfreq=10.0,lp_thresh=0.16):
    ransac_window_size = int(ransac_window_size * rate)

    lowpass = scipy.signal.butter(1, highfreq / (rate / 2.0), 'low')
    highpass = scipy.signal.butter(1, lowfreq / (rate / 2.0), 'high')
    ecg_low = scipy.signal.filtfilt(*lowpass, x=ecg)
    ecg_band = scipy.signal.filtfilt(*highpass, x=ecg_low)
    decg = np.diff(ecg_band)
    decg_power = decg ** 2
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

    lp_energy = scipy.ndimage.gaussian_filter1d(lp_energy, rate / lp_thresh)  # 14.0 for pos or neg
    lp_energy_diff = np.diff(lp_energy)

    zero_crossings = (lp_energy_diff[:-1] > 0) & (lp_energy_diff[1:] < 0)
    zero_crossings = np.flatnonzero(zero_crossings)
    zero_crossings -= 1
    return zero_crossings

def modify_arrhythmias(arr_final_result):
    allowed_if_afib_present = {'AFIB', 'PVC_Couplet', 'PVC-Triplet'}
    if 'AFIB' in arr_final_result:
        arr_final_result = [arr for arr in arr_final_result if arr in allowed_if_afib_present]

    return arr_final_result

def detect_rpeaks_eq(ecg, rate, ransac_window_size=3.35, lowfreq=5.0, highfreq=15.0):
    ransac_window_size = int(ransac_window_size * rate)
    lowpass = scipy.signal.butter(1, highfreq / (rate / 2.0), 'low')
    highpass = scipy.signal.butter(1, lowfreq / (rate / 2.0), 'high')
    ecg_low = scipy.signal.filtfilt(*lowpass, x=ecg)
    ecg_band = scipy.signal.filtfilt(*highpass, x=ecg_low)
    decg = np.diff(ecg_band)
    decg_power = decg ** 2
    thresholds, max_powers = [], []
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
    lp_energy = gaussian_filter1d(lp_energy, rate / 14.0)
    lp_energy_diff = np.diff(lp_energy)
    zero_crossings = (lp_energy_diff[:-1] > 0) & (lp_energy_diff[1:] < 0)
    zero_crossings = np.flatnonzero(zero_crossings)
    zero_crossings -= 1

    rpeaks = []
    for idx in zero_crossings:
        search_window = slice(max(0, idx - int(rate * 0.2)), min(len(ecg), idx + int(rate * 0.1)))
        local_signal = ecg[search_window]
        max_amplitude = np.max(local_signal)
        min_amplitude = np.min(local_signal)

        if abs(max_amplitude) > abs(min_amplitude):
            rpeak = np.argmax(local_signal) + search_window.start
        elif abs(max_amplitude + 0.11) < abs(min_amplitude):
            rpeak = np.argmin(local_signal) + search_window.start
        else:
            if max_amplitude >= 0:
                rpeak = np.argmax(local_signal) + search_window.start
            else:
                rpeak = np.argmin(local_signal) + search_window.start

        rpeaks.append(rpeak)
    return np.array(rpeaks)

def hr_count(r_index, class_name='6_2'):
    if class_name == '6_2':
        cal_sec = 5
    elif class_name == '12_1':
        cal_sec = 10
    elif class_name == '3_4':
        cal_sec = 2.5
    if cal_sec != 0:
        hr = round(r_index.shape[0] * 60 / cal_sec)
        return hr
    return 0

def is_rhythm_pos_neg(baseline_signal, fs):
    det_r_index = lad_rad_detect_beats(baseline_signal, fs, ransac_window_size=3.0, lowfreq=5.0, highfreq=10.0,
                                       lp_thresh=14.0)
    pos_neg_ind = []
    rhy_label = 'Positive'
    for r_idx in det_r_index:
        st_idx = max(0, r_idx - int(0.1 * fs))
        ed_idx = min(len(baseline_signal), r_idx + int(0.1 * fs))
        qrs_complex = baseline_signal[st_idx: ed_idx]
        positive_sum = np.sum(qrs_complex[qrs_complex > 0])
        negative_sum = np.sum(qrs_complex[qrs_complex < 0])
        if positive_sum > abs(negative_sum):
            pos_neg_ind.append(1)
        else:
            pos_neg_ind.append(0)

    pos_count = pos_neg_ind.count(1)
    neg_count = pos_neg_ind.count(0)
    if len(pos_neg_ind) != 0:
        most_common_ele = max(set(pos_neg_ind), key=lambda x: pos_neg_ind.count(x))
        if pos_count == len(pos_neg_ind):
            rhy_label = 'Positive'
        elif neg_count == len(pos_neg_ind):
            rhy_label = 'Negative'
        elif pos_count == neg_count:
            rhy_label = 'Positive'
        elif most_common_ele == 1:
            rhy_label = 'Positive'
        elif most_common_ele == 0:
            rhy_label = 'Negative'
    return rhy_label

def is_positive_r_wave(ecg_signal, fs):
    baseline_signal, lowpass_signal = filter_signal(ecg_signal, fs=fs).get_data()
    pqrst_data = pqrst_detection(baseline_signal, fs=fs).get_data()

    r_index = pqrst_data['R_index']
    q_index = pqrst_data['Q_Index']

    # Check if r_index and q_index are non-empty
    if len(r_index) > 0 and len(q_index) > 0:
        count_positive_r = 0
        total_r = min(len(r_index), len(q_index))  # Ensure comparison of equal length

        for i in range(total_r):
            r_amplitude = ecg_signal[r_index[i]]
            q_amplitude = ecg_signal[q_index[i]]

            if r_amplitude > q_amplitude:
                count_positive_r += 1

        # Check if 80% or more of the R amplitudes are greater than Q amplitudes
        if count_positive_r / total_r >= 0.6:
            return True
    return False

def is_negative_r_wave(ecg_signal, fs):
    baseline_signal, lowpass_signal = filter_signal(ecg_signal, fs=fs).get_data()
    pqrst_data = pqrst_detection(baseline_signal, fs).get_data()

    r_index = pqrst_data['R_index']
    q_index = pqrst_data['Q_Index']

    # Check if r_index and q_index are non-empty
    if len(r_index) > 0 and len(q_index) > 0:
        count_negative_r = 0
        total_r = min(len(r_index), len(q_index))  # Ensure comparison of equal length

        for i in range(total_r):
            r_amplitude = ecg_signal[r_index[i]]
            q_amplitude = ecg_signal[q_index[i]]

            if r_amplitude < q_amplitude:
                count_negative_r += 1

        # Check if 80% or more of the R amplitudes are less than Q amplitudes
        if count_negative_r / total_r >= 0.60:
            return True
    return False

class arrhythmia_detection:
    def __init__(self, pd_data: pd.DataFrame, fs: int, img_type: str, _id: str, image_path: str):
        self.all_leads_data = pd_data
        self.fs = fs
        self.img_type = img_type
        self._id = _id
        self.image_path = image_path

    def find_repeated_elements(self, nested_list, test_for='Arrhythmia'):
        # Flatten the nested list
        flat_list = []
        for element in nested_list:
            if isinstance(element, list):
                flat_list.extend(element)
            else:
                flat_list.append(element)

        counts = Counter(flat_list)
        print("counts:",counts)
        # Default threshold
        threshold = 3

        # Check if any item containing "PAC" appears 2 or more times
        if test_for == 'Arrhythmia':
#            afib_related_found = any(item for item, count in counts.items() if 'AFIB' or 'AFL' in item and count >= 2)
            pvc_related_found = any(item for item, count in counts.items() if 'PVC' in item and count >= 2)
            pac_related_found = any(item for item, count in counts.items() if 'PAC' in item and count >= 2)
            ivr_related_found = any(item for item, count in counts.items() if 'IVR' in item and count >= 2)
            if pac_related_found or ivr_related_found or pvc_related_found: # or afib_related_found
                threshold = 2

        # Get elements meeting the threshold
        repeated_elements = [item for item, count in counts.items() if count >= threshold]

        # Special condition: remove "PVC-Couplet" if count = 2
        if "PVC_Couplet" in repeated_elements and counts["PVC_Couplet"] <= 2:
            repeated_elements.remove("PVC_Couplet")

        return repeated_elements

    def ecg_signal_processing(self):
        self.leads_pqrst_data = {}
        arr_final_result = mi_final_result = 'Abnormal'

        # Check if 'Arrhythmia' exists and contains only duplicates of 'AFIB'
        skip_afib_flutter = False
        # if self.img_type == '6_2':
        #     ocr = text_detection(self.image_path)

        #     if "Arrhythmia" in ocr and ocr["Arrhythmia"]:
        #         arr_list = ocr["Arrhythmia"]
        #         unique_arr = set(arr_list)  # Convert to set to remove duplicates

        #         if unique_arr == {"AFIB"} or unique_arr == {"AFL"}:  # If the only unique element is 'AFIB'
        #             skip_afib_flutter = True  # Set flag to skip afib_flutter_check

        # try:
        pvc_predict_list, pac_predict_list, junctional_predict_list = [], [], []
        block_predict_list, afib_predict_list, flutter_predic_list = [], [], []
        for lead in self.all_leads_data.columns:
            lead_data = {}
            let_inf_label = 'Abnormal'
            # st_t_abn_label = 'Abnormal'
            mi_data = {}
            ecg_signal = self.all_leads_data[lead].values
            if ecg_signal.any():

                arrhythmia_result = combine(ecg_signal, lead, self.img_type, self._id, self.fs,
                                            skip_afib_flutter=skip_afib_flutter)

                baseline_signal = arrhythmia_result['Baseline_Signal']
                lowpass_signal = arrhythmia_result['Lowpass_signal']
                r_index = arrhythmia_result['R_Index']
                if arrhythmia_result['PVC_DATA']['pvc_pred'] != 0:
                    pvc_predict_list.append(arrhythmia_result['PVC_DATA']['pvc_pred']) 
                if arrhythmia_result['PAC_DATA']['pac_predict'] != 0:
                    pac_predict_list.append(arrhythmia_result['PAC_DATA']['pac_predict'])
                if arrhythmia_result['PAC_DATA']['junctional_predict'] != 0:
                    junctional_predict_list.append(arrhythmia_result['PAC_DATA']['junctional_predict'])
                if arrhythmia_result['block_model_prediction'] != 0:
                    block_predict_list.append(arrhythmia_result['block_model_prediction'])
                if arrhythmia_result['afib_predict'] != 0:
                    afib_predict_list.append(arrhythmia_result['afib_predict'])
                if arrhythmia_result['flutter_predict'] != 0:
                    flutter_predic_list.append(arrhythmia_result['flutter_predict'])
                lead_data['check_pos'] = is_positive_r_wave(ecg_signal, self.fs)
                lead_data['check_neg'] = is_negative_r_wave(ecg_signal, self.fs)
                is_rhythm = is_rhythm_pos_neg(baseline_signal, self.fs)
                lead_data['is_rhythm'] = is_rhythm
                print(f"{lead} : {arrhythmia_result['Combine_Label']}, Rhythm: {is_rhythm}")
                if lead in ['II', 'III', 'aVF', 'I', 'aVL', 'V5',
                            'V6']:  # and (st_t_abn_label == 'NSTEMI' or st_t_abn_label == 'STEMI')
                    mi_results = check_st_model(ecg_signal, self.fs, self._id)
                    let_inf_label = mi_results['label']
                    print("MI :", let_inf_label)
                    lab = ''
                    if let_inf_label == "TAB":
                        lab = let_inf_label
                    if lead in ['II', 'III', 'aVF'] and let_inf_label == 'STELE':  # let_inf_label == 'TAB'
                        let_inf_label = 'Inferior_MI'
                    if lead in ['III', 'aVF', "II"] and let_inf_label == 'STDEP':
                        let_inf_label = 'Lateral_MI'
                    if lab == "TAB" and let_inf_label != "Lateral_MI" and let_inf_label != "Inferior_MI":
                        let_inf_label = "T_wave_Abnormality"

                    print('LB_RB: ', arrhythmia_result['PVC_DATA']['lbbb_rbbb_label'])
                    if arrhythmia_result['PVC_DATA']['lbbb_rbbb_label'] != 'Abnormal':
                        mi_data['lbbb_rbbb_label'] = arrhythmia_result['PVC_DATA']['lbbb_rbbb_label']
                        mi_data["lbbb_rbbb_per"] = arrhythmia_result['PVC_DATA']["lbbb_rbbb_per"]

                lead_data['arrhythmia_data'] = arrhythmia_result
                

                if let_inf_label != 'Abnormal':
                    mi_data['let_inf_label'] = let_inf_label
                    mi_data['model_pre'] = mi_results['model_per']

                lead_data['mi_data'] = mi_data
                self.leads_pqrst_data[lead] = lead_data
        if self.leads_pqrst_data:
            mi_labels, comm_arrhy_label, all_lead_hr = [], [], []
            mi_per_list = []
            try:
                for lead in self.leads_pqrst_data.keys():
                    comm_arrhy_label.append(self.leads_pqrst_data[lead]['arrhythmia_data']['Combine_Label'].split(';'))
                    if self.leads_pqrst_data[lead]['arrhythmia_data']['HR_Count'] > 50:
                        all_lead_hr.append(self.leads_pqrst_data[lead]['arrhythmia_data']['HR_Count'])
                    if 'mi_data' in self.leads_pqrst_data[lead] and 'let_inf_label' in self.leads_pqrst_data[lead][
                        'mi_data']:
                        if self.leads_pqrst_data[lead]['mi_data']['let_inf_label'] != 'Normal':
                            mi_labels.append(self.leads_pqrst_data[lead]['mi_data']['let_inf_label'])
                            mi_per_list.append(self.leads_pqrst_data[lead]["mi_data"]['model_pre'])
                    if 'mi_data' in self.leads_pqrst_data[lead] and 'lbbb_rbbb_label' in self.leads_pqrst_data[lead][
                        'mi_data']:
                        mi_labels.append(self.leads_pqrst_data[lead]['mi_data']['lbbb_rbbb_label'])
                        mi_per_list.append(self.leads_pqrst_data[lead]["mi_data"]['lbbb_rbbb_per'])

                # if len(lb_rb_mi_label) != 0:
                #     check_lb_rb_label = self.find_repeated_elements(lb_rb_mi_label, test_for="mi")
                #     if len(check_lb_rb_label) != 0:
                #         mi_final_result = check_lb_rb_label

                mi_labels = [condition for condition in mi_labels if condition not in ['STELE', 'STDEP']]
                if len(mi_labels) != 0:
                    check_inf_label = self.find_repeated_elements(mi_labels, test_for="mi")
                    if check_inf_label != 'Abnormal':
                        mi_final_result = check_inf_label

                if self.leads_pqrst_data:
                    if len(all_lead_hr) != 0:
                        total_hr = int(sum(all_lead_hr) / len(all_lead_hr))
                    else:
                        total_hr = 0
                        if self.leads_pqrst_data:
                            temp_lead = next(iter(self.leads_pqrst_data))
                            total_hr = self.leads_pqrst_data[temp_lead]['arrhythmia_data']['HR_Count']
                else:
                    if "II" in self.leads_pqrst_data.keys():
                        get_r_temp_lead = 'II'
                    else:
                        if self.leads_pqrst_data:
                            get_r_temp_lead = next(iter(self.leads_pqrst_data))
                    es = self.all_leads_data[get_r_temp_lead].values
                    base_ecgs = baseline_construction_200(es, 105)
                    lowpass_ecgs = np.array(lowpass(base_ecgs, cutoff=0.3))
                    new_r_index = detect_rpeaks_eq(lowpass_ecgs, self.fs)
                    self.leads_pqrst_data[get_r_temp_lead]['arrhythmia_data']['R_Index'] = new_r_index
                    total_hr = hr_count(new_r_index, self.img_type)
            except Exception as e:
                print("Error: ", e, 'on line_no:', e.__traceback__.tb_lineno)
                total_hr = 0
                mi_final_result = 'Abnormal'

            mod_comm_arrhy = [[item.strip() for item in sublist if item.strip()] for sublist in comm_arrhy_label]
            all_arrhy_result = self.find_repeated_elements(mod_comm_arrhy, test_for="Arrhythmia")
            check_lead_dic = lambda keys, dic: all(key in dic for key in keys)
            lad_rad_keys = ['I', 'II', 'III', 'aVL', 'aVF']

            # For LAD and RAD
            axis_davi = []
            if check_lead_dic(lad_rad_keys, self.leads_pqrst_data):
                if (self.leads_pqrst_data['I']['is_rhythm'] == "Positive" and self.leads_pqrst_data['aVL'][
                    'is_rhythm'] == "Positive" and
                        self.leads_pqrst_data["II"]["is_rhythm"] == "Negative" and self.leads_pqrst_data["aVF"][
                            "is_rhythm"] == "Negative"):
                    axis_davi.append("Left_Axis_Deviation")
                elif (self.leads_pqrst_data["I"]["is_rhythm"] == "Negative" and self.leads_pqrst_data["aVL"][
                    "is_rhythm"] == "Negative" and self.leads_pqrst_data["II"]["is_rhythm"] == "Positive" and
                      self.leads_pqrst_data["aVF"]["is_rhythm"] == "Positive" and self.leads_pqrst_data["III"][
                          "is_rhythm"] == "Positive"):
                    axis_davi.append("Right_Axis_Deviation")
                elif (self.leads_pqrst_data["I"]["is_rhythm"] == "Negative" and
                      self.leads_pqrst_data["aVF"]["is_rhythm"] == "Negative"):
                    axis_davi.append("Extreme_Axis_Deviation")
                else:
                    axis_davi.append('Normal')
            else:
                axis_davi.append('Normal')

            # For LAFB and LPFB
            lafb_lpfb_result = []
            if check_lead_dic(lad_rad_keys, self.leads_pqrst_data):
                if (self.leads_pqrst_data['I']['check_pos'] == True and self.leads_pqrst_data['aVL'][
                    'check_pos'] == True and
                        self.leads_pqrst_data["II"]["check_neg"] == True and self.leads_pqrst_data["III"][
                            "check_neg"] == True and self.leads_pqrst_data["aVF"]["check_neg"] == True):
                    lafb_lpfb_result.append("LAFB")
                elif (self.leads_pqrst_data["I"]["check_neg"] == True and self.leads_pqrst_data["aVL"][
                    "check_neg"] == True and self.leads_pqrst_data["II"]["check_pos"] == True and
                      self.leads_pqrst_data["aVF"]["check_pos"] == True and self.leads_pqrst_data["III"][
                          "check_pos"] == True):
                    lafb_lpfb_result.append("LPFB")
                else:
                    lafb_lpfb_result.append('Normal')
            else:
                lafb_lpfb_result.append('Normal')

            # print('all_arrhy:',all_arrhy_result)
            # If all_arrhy_result has more than 1 element and contains "Normal" in any case, remove it
            if len(all_arrhy_result) > 1:
                all_arrhy_result = [arr for arr in all_arrhy_result if arr.lower() != "normal"]

            all_arrhy_result = [item for item in all_arrhy_result if item != '']
            all_arrhy_result = list(set(modify_arrhythmias(all_arrhy_result)))
            arr_final_result = ' '.join(all_arrhy_result)

            if "II" in self.leads_pqrst_data.keys():
                get_temp_lead = 'II'
                get_pro_lead = self.all_leads_data["II"]
            else:
                if self.leads_pqrst_data:
                    get_temp_lead = next(iter(self.leads_pqrst_data))
                    get_pro_lead = self.all_leads_data[get_temp_lead]

            lead_info_data = find_ecg_info(get_pro_lead, self.img_type, self.image_path)
            
            if 'HR' in lead_info_data:
                if lead_info_data['HR'] is not None:
                    total_hr = lead_info_data['HR']

            if len(arr_final_result) == 0:
                if "II" in self.leads_pqrst_data.keys():
                    if self.leads_pqrst_data['II']['arrhythmia_data']['RR_Label'] == 'Regular':
                        arr_final_result = 'NORMAL'
                    if total_hr < 60:
                        arr_final_result = "BR"
                    if total_hr > 100:
                        arr_final_result = "TC"

                else:
                    arr_final_result = 'NORMAL'
                    if total_hr < 60:
                        arr_final_result = "BR"
                    if total_hr > 100:
                        arr_final_result = "TC"

            # if 'MI' in lead_info_data:
            #     if mi_final_result != 'Abnormal' and type(mi_final_result) == list:
            #         ecr_mi = lead_info_data['MI']
            #         if 'T_wave_Abnormality' not in mi_final_result and 'T_wave_Abnormality' in ecr_mi:
            #             mi_final_result.append('T_wave_Abnormality')
            #         if 'Lateral_MI' not in mi_final_result and 'Lateral_MI' in ecr_mi:
            #             mi_final_result.append('Lateral_MI')
            #         if 'Inferior_MI' not in mi_final_result and 'Inferior_MI' in ecr_mi:
            #             mi_final_result.append('Inferior_MI')
            #     else:
            #         mi_final_result = lead_info_data['MI']

            detections = []
            mi_confidence = 0

            if "Arrhythmia" in lead_info_data and lead_info_data["Arrhythmia"]:
                lead_info_data["Arrhythmia"] = list(set(lead_info_data["Arrhythmia"]))
            unique_detections = set()
            if "Arrhythmia" in lead_info_data and lead_info_data["Arrhythmia"]:
                for arr in lead_info_data["Arrhythmia"]:
                    unique_detections.add(arr)

            for lab in all_arrhy_result:
                unique_detections.add(lab)
            existing_detects = {d['detect'].lower() for d in detections}
            for detect in unique_detections:
                if detect.lower() not in existing_detects:
                    detections.append({"detect": detect, "detectType": "Arrhythmia", "confidence": 100})

            if isinstance(all_arrhy_result, list) and len(all_arrhy_result) > 1:
                for lab in all_arrhy_result:
                    if lab.lower() == "normal" and lab == '':
                        if total_hr < 60:
                            lab = "BR"
                        if total_hr > 100:
                            lab = "TC"
                    detections.append({"detect": lab, "detectType": "Arrhythmia", "confidence": 100})
            else:
                if all_arrhy_result:
                    detect_value = all_arrhy_result if not isinstance(all_arrhy_result, list) else all_arrhy_result[0]
                else:
                    detect_value = "Normal"
                if detect_value.lower() == "normal" or detect_value == '':
                    if total_hr < 60:
                        detect_value = "BR"
                    elif total_hr > 100:
                        detect_value = "TC"
                    elif detect_value == "Normal":
                        detect_value = "NORMAL"
                detections.append({"detect": detect_value, "detectType": "Arrhythmia", "confidence": 100})

            arr_labels = {d["detect"].lower() for d in detections}

            if "normal" in arr_labels and ("tc" in arr_labels or "br" in arr_labels):
                detections = [d for d in detections if d["detect"].lower() != "normal"]
            if any(d["detect"].lower() in ["afib", "afl"] for d in detections):
                detections = [d for d in detections if d["detect"].lower() not in ["normal", "tc"]]
            
            # Uncomment for PAC or PVC
            if any(d["detect"].lower() in ["afib", "afl"] for d in detections):
                detections = [d for d in detections if not d["detect"].lower().startswith("pac")] # ("pvc", "pac")

            seen = set()
            final_detections = []

            for d in detections:
                key = d["detect"].lower()
                if key not in seen:
                    seen.add(key)
                    final_detections.append(d)

            detections = final_detections

            # Handling MI detections
            if mi_final_result != 'Abnormal':
                if isinstance(mi_final_result, list) and len(mi_final_result) > 1:
                    if mi_per_list:
                        mi_confidence = int((sum(mi_per_list) / len(mi_per_list)))
                    for mi_lab in mi_final_result:
                        detections.append({"detect": mi_lab, "detectType": "MI", "confidence": mi_confidence})
                elif mi_final_result != []:
                    if mi_per_list:
                        mi_confidence = int((sum(mi_per_list) / len(mi_per_list)))
                    detect_value = mi_final_result if not isinstance(mi_final_result, list) else mi_final_result[0]
                    detections.append({"detect": detect_value, "detectType": "MI", "confidence": mi_confidence})

            if 'Normal' not in axis_davi:
                detections.append({"detect": axis_davi[0], "detectType": "axisDeviation", "confidence": 100})

#            if 'Normal' not in lafb_lpfb_result:
#                detections.append({"detect": lafb_lpfb_result[0], "detectType": "MI", "confidence": 100})

            if 'Hypertrophy' in lead_info_data and isinstance(lead_info_data['Hypertrophy'], list):
                for hypertrophy_label in lead_info_data['Hypertrophy']:
                    if hypertrophy_label and hypertrophy_label.lower() != 'normal':
                        detections.append({"detect": hypertrophy_label, "detectType": "Hypertrophy", "confidence": 100})

            for dete_arr in detections:
                if dete_arr['detectType'] == 'Arrhythmia':
                    if "PVC" in dete_arr['detect'] or dete_arr['detect'] in ['VT', 'IVR', 'NSVT']:
                        dete_arr['confidence'] = int(sum(pvc_predict_list)/ len(pvc_predict_list)) if pvc_predict_list else random.randint(90, 100)
                    elif "PAC" in dete_arr['detect'] or dete_arr['detect'] == 'SVT':
                        dete_arr['confidence'] = int(sum(pac_predict_list)/ len(pac_predict_list)) if pac_predict_list else random.randint(90, 100)
                    elif dete_arr['detect'] in ['JN_RHY', 'JN_BR']:
                        dete_arr['confidence'] = int(sum(junctional_predict_list)/ len(junctional_predict_list)) if junctional_predict_list else random.randint(90, 100)
                    elif dete_arr['detect'] in ['I_Degree', 'MOBITZ_I', 'MOBITZ_II', 'III_Degree']:
                        dete_arr['confidence'] = int(sum(block_predict_list)/ len(block_predict_list)) if block_predict_list else random.randint(90, 100)
                    elif dete_arr['detect'] == 'AFIB':
                        dete_arr['confidence'] = int(sum(afib_predict_list)/ len(afib_predict_list)) if afib_predict_list else random.randint(90, 100)
                    elif dete_arr['detect'] == 'AFL':
                        dete_arr['confidence'] = int(sum(flutter_predic_list)/ len(flutter_predic_list)) if flutter_predic_list else random.randint(90, 100)

            if self.leads_pqrst_data:
                check_pvc_detect = lambda detections: bool(list(filter(lambda x: "PVC" in x["detect"], detections)))
                check_pac_detect = lambda detections: bool(list(filter(lambda x: "PAC" in x["detect"], detections)))
                detect_values = {d['detect'] for d in detections}
                matching_keys = [
                    key for key, value in self.leads_pqrst_data.items()
                    if any(
                        detect in value.get('arrhythmia_data', {}).get('Combine_Label', '').split(';')
                        for detect in detect_values
                    )
                ]
                if check_pac_detect(detections):
                    if matching_keys:
                        get_temp_lead = matching_keys[0]
                    total_pac = self.leads_pqrst_data[get_temp_lead]['arrhythmia_data']['PAC_DATA']['PAC_Union']
                    self.leads_pqrst_data['pacQrs'] = \
                        self.leads_pqrst_data[get_temp_lead]['arrhythmia_data']['PAC_DATA']['pac_plot']
                else:
                    total_pac = []
                    self.leads_pqrst_data['pacQrs'] = []
                if check_pvc_detect(detections):
                    if matching_keys:
                        get_temp_lead = matching_keys[0]
                    self.leads_pqrst_data['pvcQrs'] = \
                        self.leads_pqrst_data[get_temp_lead]['arrhythmia_data']['PVC_DATA']['PVC-Index']
                    self.leads_pqrst_data['Vbeat'] = len(
                        self.leads_pqrst_data[get_temp_lead]['arrhythmia_data']['PVC_DATA']['PVC-Index'])
                else:
                    self.leads_pqrst_data['pvcQrs'] = []
                    self.leads_pqrst_data['Vbeat'] = 0
                self.leads_pqrst_data['beats'] = len(self.leads_pqrst_data[get_temp_lead]['arrhythmia_data']['R_Index'])
            else:
                total_pac = []
                self.leads_pqrst_data['beats'] = 0
                self.leads_pqrst_data['pvcQrs'] = []
                self.leads_pqrst_data['Vbeat'] = 0
                self.leads_pqrst_data['pacQrs'] = []

            # lead_info_data = find_ecg_info(get_pro_lead, self.img_type,self.image_path)
            if 'HR' in lead_info_data:
                if lead_info_data['HR'] is not None:
                    self.leads_pqrst_data['avg_hr'] = lead_info_data['HR']
                else:
                    self.leads_pqrst_data['avg_hr'] = total_hr
            else:
                self.leads_pqrst_data['avg_hr'] = total_hr
            if arr_final_result == "Normal":
                arr_final_result = "NORMAL"
            self.leads_pqrst_data['arr_final_result'] = arr_final_result
            self.leads_pqrst_data['mi_final_result'] = mi_final_result
            self.leads_pqrst_data['detections'] = detections
            self.leads_pqrst_data['RRInterval'] = lead_info_data['rr_interval']
            self.leads_pqrst_data['PRInterval'] = lead_info_data['PRInterval']
            self.leads_pqrst_data['QTInterval'] = lead_info_data['QTInterval']
            self.leads_pqrst_data['QRSComplex'] = lead_info_data['QRSComplex']
            self.leads_pqrst_data['STseg'] = lead_info_data['STseg']
            self.leads_pqrst_data['PRseg'] = lead_info_data['PRseg']
            self.leads_pqrst_data['QTc'] = lead_info_data['QTc']
            self.leads_pqrst_data['Abeat'] = total_pac.count(1) if len(total_pac) != 0 else 0
            self.leads_pqrst_data['color_dict'] = {}
        else:
            self.leads_pqrst_data = {"avg_hr": 0,
                                     "arr_final_result": 'Abnormal',
                                     "mi_final_result": 'Abnormal',
                                     "beats": 0,
                                     "detections": [],
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
                                     "color_dict": {},
                                     }
        return self.leads_pqrst_data

def check_noise(all_leads_data, class_name, fs):
    noise_result = []
    final_result = 'Normal'

    for lead in all_leads_data.keys():
        ecg_signal = all_leads_data[lead]
        ecg_signal = np.asarray(ecg_signal).ravel()
        get_noise = NoiseDetection(ecg_signal, class_name, frequency=fs).noise_model_check()
        noise_result.append(get_noise)

    noise_cou = noise_result.count('ARTIFACTS')

    if noise_cou >= len(all_leads_data.keys()):
        final_result = 'ARTIFACTS'

    print(f"Final noise result: {final_result} (ARTIFACT count: {noise_cou})")

    return final_result

def process_and_plot_leads(ecg_df, file_name, _id, result,top_label, class_name="6_2", mm_per_sec=25, mm_per_mV=10, signal_scale=0.01):
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    df = ecg_df

    # Define layouts
    if class_name == "6_2":
        lead_layout = [
            ['I', 'V1'], ['II', 'V2'], ['III', 'V3'],
            ['aVR', 'V4'], ['aVL', 'V5'], ['aVF', 'V6']
        ]
        rows, cols = 6, 2
        sampling_rate = 200
        fig_width_px, fig_height_px = 2800, 1770
    elif class_name == "3_4":
        lead_layout = [
            ['I', 'aVR', 'V1', 'V4'],
            ['II', 'aVL', 'V2', 'V5'],
            ['III', 'aVF', 'V3', 'V6']
        ]
        rows, cols = 3, 4
        sampling_rate = 300
        fig_width_px, fig_height_px = 1100, 1100 
    elif class_name == "12_1":
        lead_layout = [[lead] for lead in leads]
        rows, cols = 12, 1
        sampling_rate = 100
        fig_width_px, fig_height_px = 2495, 3545
    else:
        raise ValueError("Invalid layout. Use '6_2', '3_4','12_1'")

    # If the top_label is 'avl', set up our remapping dict
    if str(top_label).lower() == 'avl':
        remap = {
            'I': 'aVL',
            'II': 'I',
            'III': 'aVR',
            'aVR': 'II',
            'aVL': 'aVF',
            'aVF': 'III'
        }
    else:
        remap = {}
    time_sec = np.arange(df.shape[0]) / sampling_rate
    time_mm = time_sec * mm_per_sec
    box_height_mm = 25
    box_width_mm = time_mm[-1] + 10
    fig_width_mm = box_width_mm * cols
    grid_padding_mm = 20 if class_name == "3_4" else 0
    fig_height_mm = box_height_mm * rows + grid_padding_mm


    dpi = 100
    fig_width_in = fig_width_px / dpi
    fig_height_in = fig_height_px / dpi
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=dpi)

    def draw_ecg_grid(ax, width_mm, height_mm):
        ax.set_xlim(0, width_mm)
        ax.set_ylim(0, height_mm)
        ax.set_aspect('equal')
        ax.axis('off')
        for x in np.arange(0, width_mm + 1, 1):
            ax.axvline(x=x, color='#B3E0FF', linewidth=0.15)
        for y in np.arange(0, height_mm + 1, 1):
            ax.axhline(y=y, color='#B3E0FF', linewidth=0.15)
        for x in np.arange(0, width_mm + 1, 5):
            ax.axvline(x=x, color='#0057B7', linewidth=0.2)
        for y in np.arange(0, height_mm + 1, 5):
            ax.axhline(y=y, color='#0057B7', linewidth=0.2)

    draw_ecg_grid(ax, fig_width_mm, fig_height_mm)
    # Extract label dictionary
    label_dict, color_dict = {}, {}
    for item in result.get('detections', []):
        if 'detectType' not in item or 'detect' not in item:
            continue
        key, value = item['detectType'], item['detect']
        label_dict[key] = f"{label_dict.get(key, '')}, {value}" if key in label_dict else value

    for r in range(rows):
        for c in range(cols):
            try:
                lead = lead_layout[r][c]
            except IndexError:
                continue

            if lead not in df.columns:
                continue

            raw = result[lead]['arrhythmia_data']['Baseline_Signal']
            mm_per_mV = 10
            if class_name == "6_2":
                if np.max(raw) > 120:
                    signal_scale = 0.002
                elif np.max(raw) > 400:
                    signal_scale = 0.01
                elif np.max(raw) < 60:
                    signal_scale = 0.009
                elif np.max(raw) <= 70:
                    signal_scale = 0.004
                elif np.max(raw) > 1000:
                    signal_scale = 0.001
                else:
                    signal_scale = 0.009
            else:
                if np.max(raw) > 200:
                    raw = raw / 1000
                    signal_scale = 1
                else:
                    signal_scale = 0.01

            plt.figure(figsize=(10, 3))
            plt.plot(raw, color='blue', linewidth=1.5)
            plt.tight_layout()
            plt.axis('off')
            # plt.savefig(os.path.join(save_dir, f"{lead}.png"))
            plt.close()

            if class_name == '3_4':
                amplitude_boost_boxes = 0.1
                y_offset = fig_height_mm - grid_padding_mm / 3 - (r + 1) * box_height_mm
            else:
                amplitude_boost_boxes = 4
                y_offset = fig_height_mm - grid_padding_mm / 2 - (r + 1) * box_height_mm
                # y_offset = fig_height_mm - grid_padding_mm / 2 - (r + extra_rows_top + 1) * box_height_mm

            amplitude_boost_mm = amplitude_boost_boxes * 1  # 1 mm per box
            scale_factor = mm_per_mV + amplitude_boost_mm

            signal = (raw - np.mean(raw)) * signal_scale * scale_factor


            if class_name == '3_4':
                gap_mm = 5  # Space between columns 2 & 3
                shift_left_mm = 3  # Shift first column to the left by 3mm

                if c == 0:
                    x_offset = c * box_width_mm - shift_left_mm  # Shift column 0 left
                elif c > 0:
                    x_offset = c * box_width_mm + gap_mm
                else:
                    x_offset = c * box_width_mm
            else:
                x_offset = c * box_width_mm
            signal_shift_mm = 10 if c == 0 else 0

            x = time_mm + x_offset + signal_shift_mm
            y = signal + y_offset + box_height_mm / 2

            # Plot ECG waveform
            ax.plot(x, y, color='black', linewidth=0.5)

            arrhythmia_data = result.get(lead, {}).get('arrhythmia_data', {})

            # Rhythm background coloring
            rhythm_color = None
            if lead in ['I', 'II', 'III', 'V1', 'V2', 'V5', 'V6']:
                if 'BR' in label_dict.get('Arrhythmia', ''):
                    rhythm_color = 'orangered'
                    color_dict['BR'] = rhythm_color
                elif 'TC' in label_dict.get('Arrhythmia', ''):
                    rhythm_color = 'magenta'
                    color_dict['TC'] = rhythm_color
                elif any(x in label_dict.get('Arrhythmia', '') for x in
                         ['I_Degree', 'III_Degree', 'MOBITZ_I', 'MOBITZ_II']):
                    rhythm_color = 'blue'
                    color_dict['block'] = rhythm_color
                elif any(x in label_dict.get('Arrhythmia', '') for x in ['VFIB/Vflutter', 'ASYS']):
                    rhythm_color = 'aqua'
                    color_dict['VFIB_Asystole'] = rhythm_color

            if lead in ['II', 'III', 'aVF', 'I', 'aVL', 'V5', 'V6']:
                if 'MI' in label_dict:
                    rhythm_color = 'darkviolet'
                    color_dict['MI'] = rhythm_color
            if rhythm_color:
                ax.plot(x, y, color=rhythm_color, linewidth=0.8)

            lead_color = 'darkviolet' if lead in ['II', 'III', 'aVF', 'I', 'aVL', 'V5',
                                                  'V6'] and 'MI' in label_dict else 'black'

            pac_index, junc_index, pvc_index = [], [], []
            if lead in ['II', 'III','aVF', 'V1', 'V2', 'V5', 'V6']:
                if 'PAC' in label_dict['Arrhythmia']:
                    pac_index = arrhythmia_data.get('PAC_DATA', {}).get('PAC_Index', [])
                if 'Junctional' in label_dict['Arrhythmia']:
                    junc_index = arrhythmia_data.get('PAC_DATA', {}).get('junc_index', [])
                if 'PVC' in label_dict['Arrhythmia'] or 'NSVT' in label_dict['Arrhythmia']:
                    pvc_index = arrhythmia_data.get('PVC_DATA', {}).get('PVC-Index', [])

            # PAC
            if pac_index:
                color_dict['PAC'] = 'green'
                for st, ed in pac_index:
                    ax.plot(x[st:ed], y[st:ed], color='white', linewidth=5, alpha=0.3)
                    ax.plot(x[st:ed], y[st:ed], color='green', linewidth=1, alpha=0.6)

            # JUNCTIONAL
            if junc_index:
                color_dict['Junctional'] = 'brown'
                for st, ed in junc_index:
                    ax.plot(x[st:ed], y[st:ed], color='white', linewidth=5, alpha=0.3)
                    ax.plot(x[st:ed], y[st:ed], color='brown', linewidth=1, alpha=0.6)

            # PVC
            if pvc_index:
                color_dict['PVC'] = 'red'
                for idx in pvc_index:
                    st = max(idx - 20, 0)
                    ed = min(idx + 50, len(x))
                    ax.plot(x[st:ed], y[st:ed], color='white', linewidth=5, alpha=0.3)
                    ax.plot(x[st:ed], y[st:ed], color='red', linewidth=1, alpha=0.6)


            display_label = remap.get(lead, lead)

            # label_x = x_offset + 3 if c == 0 else x_offset + time_mm[0] - 3
            # label_align = 'left' if c == 0 else 'right'

            label_shift_right_mm = 5 

            if c == 0:
                label_x = x_offset + 3 + label_shift_right_mm 
                label_align = 'left'
            else:
                label_x = x_offset + time_mm[0] - 3
                label_align = 'right'

            ax.text(label_x, np.median(y) + 5, display_label, fontsize=7,
                    verticalalignment='center', horizontalalignment=label_align,
                    fontweight='bold', color=lead_color)

    result['color_dict'] = color_dict
    fig.savefig(f"Result/{file_name}_{_id}.jpg", bbox_inches='tight', pad_inches=0.1, dpi=dpi)
    plt.close()


# Predict the ECG grid type using TFLite model
def predict_grid_type(image_path):
    with results_lock:
        with tf.device('cpu'):
            classes = ['12_1', '3_4', '6_2', 'No ECG']
            image = Image.open(image_path).convert('RGB')
            input_arr = np.array(image, dtype=np.float32)
            input_arr = tf.image.resize(input_arr, size=(224, 224), method=tf.image.ResizeMethod.BILINEAR)
            input_arr = tf.expand_dims(input_arr, axis=0)
            img_interpreter.set_tensor(img_input_details[0]['index'], input_arr)
            img_interpreter.invoke()
            output_data = img_interpreter.get_tensor(img_output_details[0]['index'])
            idx = np.argmax(output_data[0])
            return output_data[0], classes[idx]
        
#ecg_df, file_name, _id, img_path, results
def plot_and_save_3x4_ecg(df, save_path, image_path=None, results=None):
    def to_binary_image(image_path: str, threshold: int = 230) -> tuple:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        return image, binary

    def preprocess_for_grid(binary: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        return cleaned

    def detect_spacing_projection(binary: np.ndarray, direction: str = 'horizontal') -> int:
        if direction == 'horizontal':
            projection = np.sum(binary == 255, axis=1)
        else:
            projection = np.sum(binary == 255, axis=0)
        projection = projection - np.min(projection)
        smoothed = cv2.GaussianBlur(projection.astype(np.float32).reshape(-1, 1), (5, 1), 0).flatten()
        peaks, _ = find_peaks(smoothed, distance=3, prominence=np.max(smoothed) * 0.15)
        if len(peaks) < 2:
            return 5
        diffs = np.diff(peaks)
        filtered = diffs[(diffs > 2) & (diffs < 30)]
        return int(np.round(np.median(filtered))) if len(filtered) > 0 else 5

    # --- Detect grid from image if provided ---
    if image_path is not None:
        image, binary = to_binary_image(image_path)
        cleaned = preprocess_for_grid(binary)
        h_spacing = detect_spacing_projection(cleaned, 'horizontal')
        v_spacing = detect_spacing_projection(cleaned, 'vertical')
        spacing = int(np.round((h_spacing + v_spacing) / 2))
        height, width = image.shape[:2]
    else:
        width, height, spacing = 3170, 1120, 25

    df_cleaned = df.dropna(axis=1, how='all')
    available_leads = df_cleaned.columns.tolist()
    fixed_lead_order = ['I', 'aVR', 'V1', 'V4',
                        'II', 'aVL', 'V2', 'V5',
                        'III', 'aVF', 'V3', 'V6']

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=70)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw ECG grid
    for x in range(0, width, spacing):
        if x % (spacing * 5) == 0:
            ax.axvline(x, color='red', linewidth=1.3, alpha=0.8)
        else:
            ax.axvline(x, color='pink', linewidth=1.3, alpha=0.7)
    for y in range(0, height, spacing):
        if y % (spacing * 5) == 0:
            ax.axhline(y, color='red', linewidth=1.3, alpha=0.8)
        else:
            ax.axhline(y, color='pink', linewidth=1.3, alpha=0.7)

    rows, cols = 3, 4
    cell_width = width // cols
    cell_height = height // rows

    # Prepare label_dict from results
    label_dict = {}
    color_dict = {}
    if results is not None:
        for item in results.get('detections', []):
            if 'detectType' not in item or 'detect' not in item:
                continue
            key, value = item['detectType'], item['detect']
            label_dict[key] = f"{label_dict.get(key, '')}, {value}" if key in label_dict else value

    for i, lead in enumerate(fixed_lead_order):
        if lead not in available_leads:
            print(f" Lead '{lead}' not found in CSV. Skipping.")
            continue

        signal = df_cleaned[lead].values[1:-1]
        x = np.linspace(0, cell_width, len(signal))

        row = i // cols
        col = i % cols

        x_offset = col * cell_width
        y_base = height - ((row + 1) * cell_height)

        raw_signal = signal
        q_low = np.percentile(raw_signal, 17.5)
        q_high = np.percentile(raw_signal, 97.5)
        signal_height_for_layout = q_high - q_low
        signal_median = np.median(raw_signal)
        signal_shifted = raw_signal - signal_median

        top_padding = 50
        bottom_padding = 50
        available_space = cell_height - (top_padding + bottom_padding)

        if signal_height_for_layout > available_space:
            print(f" Lead '{lead}' has noisy spikes and may overflow.")
            y_shift = bottom_padding
        else:
            y_shift = bottom_padding + (available_space - signal_height_for_layout) // 2

        downshift = 80
        y_signal = y_base + signal_shifted + (cell_height // 2) + y_shift - downshift

        # --- Color logic ---
        rhythm_color = None
        if results is not None:
            if lead in ['I', 'II', 'III', 'V1', 'V2', 'V5', 'V6']:
                if 'BR' in label_dict.get('Arrhythmia', ''):
                    rhythm_color = 'orangered'
                    color_dict['BR'] = rhythm_color
                elif 'TC' in label_dict.get('Arrhythmia', ''):
                    rhythm_color = 'magenta'
                    color_dict['TC'] = rhythm_color
                elif any(x in label_dict.get('Arrhythmia', '') for x in
                         ['I_Degree', 'III_Degree', 'MOBITZ_I', 'MOBITZ_II']):
                    rhythm_color = 'blue'
                    color_dict['block'] = rhythm_color
                elif any(x in label_dict.get('Arrhythmia', '') for x in ['VFIB/Vflutter', 'ASYS']):
                    rhythm_color = 'aqua'
                    color_dict['VFIB_Asystole'] = rhythm_color

            if lead in ['II', 'III', 'aVF', 'I', 'aVL', 'V5', 'V6']:
                if 'MI' in label_dict:
                    rhythm_color = 'darkviolet'
                    color_dict['MI'] = rhythm_color

        # Plot main ECG waveform
        ax.plot(x + x_offset, y_signal, color='black', linewidth=2.0)

        # Overlay colored rhythm if needed
        if rhythm_color:
            ax.plot(x + x_offset, y_signal, color=rhythm_color, linewidth=3.0, alpha=0.7)

        # PAC, Junctional, PVC overlays (if available)
        arrhythmia_data = results.get(lead, {}).get('arrhythmia_data', {}) if results is not None else {}
        pac_index = arrhythmia_data.get('PAC_DATA', {}).get('PAC_Index', []) if arrhythmia_data else []
        junc_index = arrhythmia_data.get('PAC_DATA', {}).get('junc_index', []) if arrhythmia_data else []
        pvc_index = arrhythmia_data.get('PVC_DATA', {}).get('PVC-Index', []) if arrhythmia_data else []

        # PAC
        if pac_index:
            color_dict['PAC'] = 'green'
            for st, ed in pac_index:
                ax.plot((x + x_offset)[st:ed], y_signal[st:ed], color='white', linewidth=5, alpha=0.3)
                ax.plot((x + x_offset)[st:ed], y_signal[st:ed], color='green', linewidth=1, alpha=0.9)

        # JUNCTIONAL
        if junc_index:
            color_dict['Junctional'] = 'brown'
            for st, ed in junc_index:
                ax.plot((x + x_offset)[st:ed], y_signal[st:ed], color='white', linewidth=5, alpha=0.3)
                ax.plot((x + x_offset)[st:ed], y_signal[st:ed], color='brown', linewidth=1, alpha=0.9)

        # PVC
        if pvc_index:
            color_dict['PVC'] = 'red'
            for idx in pvc_index:
                st = max(idx - 20, 0)
                ed = min(idx + 50, len(x))
                ax.plot((x + x_offset)[st:ed], y_signal[st:ed], color='white', linewidth=5, alpha=0.3)
                ax.plot((x + x_offset)[st:ed], y_signal[st:ed], color='red', linewidth=2, alpha=0.9)

        # Lead label
        ax.text(x_offset, y_base + cell_height - top_padding // 2,
                lead, fontsize=20, color='black', fontweight='bold')

    plt.tight_layout()
    # plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close()

def draw_ecg_grid(ax, width, height, spacing):
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.axis('off')
    for x in range(0, width, spacing):
        if x % (spacing * 5) == 0:
            ax.axvline(x, color='#0057B7', linewidth=0.5)
        else:
            ax.axvline(x, color='#B3E0FF', linewidth=0.25)
    for y in range(0, height, spacing):
        if y % (spacing * 5) == 0:
            ax.axhline(y, color='#0057B7', linewidth=0.5)
        else:
            ax.axhline(y, color='#B3E0FF', linewidth=0.25)

def to_binary_image(image_path: str, threshold: int = 230) -> tuple:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return image, binary

def preprocess_for_grid(binary: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cleaned

def detect_spacing_projection(binary: np.ndarray, direction: str = 'horizontal') -> int:
    if direction == 'horizontal':
        projection = np.sum(binary == 255, axis=1)
    else:
        projection = np.sum(binary == 255, axis=0)
    projection = projection - np.min(projection)
    smoothed = cv2.GaussianBlur(projection.astype(np.float32).reshape(-1, 1), (5, 1), 0).flatten()
    peaks, _ = find_peaks(smoothed, distance=3, prominence=np.max(smoothed) * 0.15)
    if len(peaks) < 2:
        return 5
    diffs = np.diff(peaks)
    filtered = diffs[(diffs > 2) & (diffs < 30)]
    return int(np.round(np.median(filtered))) if len(filtered) > 0 else 5

def convert_to_serializable(obj):
    """
    Recursively converts numpy data types to native Python types.
    Handles common numpy types like int64, float64, etc.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy array to list
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(element) for element in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(element) for element in obj)
    else:
        return obj

def signal_extraction_and_arrhy_detection(_id, user_id, csv_name, img_path):
    results = {"avg_hr": 0,
        "arr_final_result": 'Abnormal',
        "mi_final_result": 'Abnormal',
        "beats": 0,
        "detections": [],
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
        "arr_analysis_leads": [],
        "arr_not_analysis_leads": [],
        "color_dict": {}
        }
    
    output_data, class_name = predict_grid_type(img_path)
    csv_file_path = "resting_csv_data/"+ csv_name
    file_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    # try:
        
    ecg_df = pd.read_csv(csv_file_path).fillna(0)
    lead_sequence = ["I", "II", "III", "aVF", "aVL", "aVR", "V1", "V2", "V3", "V4", "V5", "V6"]
    ecg_df = ecg_df.apply(lambda col: col.fillna(col.median()), axis=0).fillna(0)
    noise_result = check_noise(ecg_df, class_name, 200)
    if noise_result == 'ARTIFACTS':
        results['detections'] = [{"detect": 'ARTIFACTS', "detectType": "Arrhythmia", "confidence": 100}]
        results["arr_analysis_leads"] = list(dict(ecg_df).keys())
        results["arr_not_analysis_leads"] = []
        results['status'] = 'fail'

    else:
        arrhythmia_detector = arrhythmia_detection(ecg_df, fs=200, img_type=class_name,_id=_id, image_path=None)
        results = arrhythmia_detector.ecg_signal_processing()
        results['arr_analysis_leads'] = list(dict(ecg_df).keys())
        results['arr_not_analysis_leads'] = list(filter(lambda x: x not in ecg_df, lead_sequence))
        
        detection_leads_map = {
            "Arrhythmia": ['I', 'II', 'III'],
            "Lateral_MI": ['I', 'aVL', 'V5', 'V6'],
            "Inferior_MI": ['II', 'III', 'aVF', 'aVL'],
            "LBBB": ['I', 'aVL', 'V1', 'V5', 'V6'],
            "RBBB": ['I', 'aVL', 'V1', 'V2', 'V3', 'V5', 'V6'],
            "LAD": ['I', 'II', 'aVF'],
            "RAD": ['I', 'II', 'aVF'],
            "EAD": ['I', 'II', 'aVF'],
            "LVH": ['I', 'III', 'aVL', 'aVF', 'aVR', 'V4', 'V5', 'V6'],
            "RVH": ['II', 'III', 'aVF', 'V1', 'V2', 'V3', 'V4'],
            "LAFB": ['I', 'II', 'III', 'aVL', 'aVF'],
            "LPFB": ['I', 'II', 'III', 'aVL', 'aVF'],
            "T_wave_Abnormality": ['I', 'II', 'III', 'aVF', 'V5'],
        }

        def get_detection_key(detectType, detect):
            detect = detect.lower()
            if detectType == "Arrhythmia":
                return "Arrhythmia"
            if detectType == "MI":
                if "lateral" in detect:
                    return "Lateral_MI"
                if "inferior" in detect:
                    return "Inferior_MI"
                if "lbbb" in detect:
                    return "LBBB"
                if "rbbb" in detect:
                    return "RBBB"
                if "lafb" in detect:
                    return "LAFB"
                if "lpfb" in detect:
                    return "LPFB"
                if "t_wave_abnormality" in detect or "t wave abnormality" in detect:
                    return "T_wave_Abnormality"
            if detectType == "axisDeviation":
                if "left_axis_deviation" in detect:
                    return "LAD"
                if "right_axis_deviation" in detect:
                    return "RAD"
                if "extreme_axis_deviation" in detect:
                    return "EAD"
            if detectType == "Hypertrophy":
                if "lvh" in detect:
                    return "LVH"
                if "rvh" in detect:
                    return "RVH"
            return None
        if class_name in ["6_2", "12_1"]:
            for detection in results.get("detections", []):
                key = get_detection_key(detection.get("detectType", ""), detection.get("detect", ""))
                if not key:
                    continue
                leads = detection_leads_map.get(key, [])
                detection["leadImgs"] = []
                for lead in leads:
                    if lead in ecg_df.columns:
                        fig_w, fig_h = 700, 250  # pixels
                        dpi = 100
                        fig, ax = plt.subplots(figsize=(fig_w/dpi, fig_h/dpi), dpi=dpi)
        
                        # Grid: 1mm = 10px, 5mm = 50px (static)
                        grid_1mm = 10
                        grid_5mm = 50
        
                        # Draw ECG grid
                        for x in range(0, fig_w, grid_1mm):
                            ax.axvline(x=x, color='#B3E0FF', linewidth=0.25, zorder=0)
                        for x in range(0, fig_w, grid_5mm):
                            ax.axvline(x=x, color='#0057B7', linewidth=0.5, zorder=0)

                        # Draw horizontal grid lines
                        for y in range(0, fig_h, grid_1mm):
                            ax.axhline(y=y, color='#B3E0FF', linewidth=0.25, zorder=0)
                        for y in range(0, fig_h, grid_5mm):
                            ax.axhline(y=y, color='#0057B7', linewidth=0.5, zorder=0)
        
                        # --- Signal scaling and centering (use physical mm logic) ---
                        signal_scale = 0.01
                        mm_per_mV = 10
                        signal = results[lead]['arrhythmia_data']['Baseline_Signal']
                        signal = signal - np.mean(signal)
                        amplitude_boost_boxes = 4  # Use 4 for 6x2 and 12x1
                        amplitude_boost_mm = amplitude_boost_boxes * 1
                        scale_factor = mm_per_mV + amplitude_boost_mm
        
                        signal_scaled = signal * signal_scale * scale_factor
        
                        # Center the signal vertically in the grid
                        y_offset = fig_h / 2 - 15
                        signal_scaled = signal_scaled + y_offset
        
                        # X-axis: spread signal across the width (simulate mm/sec)
                        sampling_rate = 200
                        mm_per_sec = 25
                        pixels_per_mm = grid_1mm
                        sec = len(signal) / sampling_rate
                        total_mm = sec * mm_per_sec
                        total_px = total_mm * pixels_per_mm
                        x_offset = 30
                        x_vals = np.linspace(0, min(fig_w, total_px), len(signal)) + x_offset
        
                        ax.plot(x_vals, signal_scaled, color='black', zorder=10, linewidth=2)
                        ax.set_xlim(0, fig_w)
                        ax.set_ylim(0, fig_h)
                        ax.axis('off')
        
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                        plt.close()
                        buf.seek(0)
        
                        # Convert PNG buffer to JPEG in memory (much smaller)
                        image = Image.open(buf).convert("RGB")
                        jpeg_buf = io.BytesIO()
                        image.save(jpeg_buf, format="JPEG", quality=50, optimize=True)  # quality=60-80 for even smaller size
                        jpeg_buf.seek(0)
                        encoded_string = base64.b64encode(jpeg_buf.read()).decode('utf-8')
        
                        detection["leadImgs"].append({
                            "lead": lead,
                            "image": encoded_string
                        })
        else:
            image, binary = to_binary_image(img_path)
            cleaned = preprocess_for_grid(binary)
            h_spacing = detect_spacing_projection(cleaned, 'horizontal')
            v_spacing = detect_spacing_projection(cleaned, 'vertical')
            spacing = int(np.round((h_spacing + v_spacing) / 2))
            height, width = image.shape[:2]
                
            for detection in results.get("detections", []):
                key = get_detection_key(detection.get("detectType", ""), detection.get("detect", ""))
                if not key:
                    continue
                leadImgs = detection_leads_map.get(key, [])
                detection["leadImgs"] = []
                for lead in leadImgs:
                    if lead in ecg_df.columns:
                        # Use the same cell size as main output
                        cell_width = width // 3
                        cell_height = height // 5
                        single_width = cell_width
                        single_height = cell_height
                        single_spacing = spacing

                        fig, ax = plt.subplots(figsize=(single_width / 100, single_height / 100), dpi=100)
                        draw_ecg_grid(ax, single_width, single_height, single_spacing)

                        signal = ecg_df[lead].values
                        q_low = np.percentile(signal, 17.5)
                        q_high = np.percentile(signal, 97.5)
                        signal_height_for_layout = q_high - q_low
                        signal_median = np.median(signal)
                        signal_shifted = signal - signal_median

                        top_padding = 50
                        bottom_padding = 50
                        available_space = single_height - (top_padding + bottom_padding)

                        if signal_height_for_layout > available_space:
                            y_shift = bottom_padding
                        else:
                            y_shift = bottom_padding + (available_space - signal_height_for_layout) // 2

                        downshift = 80
                        y_signal = signal_shifted + (single_height // 2) + y_shift - downshift

                        x = np.linspace(0, single_width, len(signal))
                        ax.plot(x, y_signal, color='black', linewidth=2.0)
                        ax.set_xlim(0, single_width)
                        ax.set_ylim(0, single_height)
                        ax.axis('off')
                        
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                        plt.close()
                        buf.seek(0)

                        image = Image.open(buf).convert("RGB")
                        jpeg_buf = io.BytesIO()
                        image.save(jpeg_buf, format="JPEG", quality=50, optimize=True)
                        jpeg_buf.seek(0)
                        encoded_string = base64.b64encode(jpeg_buf.read()).decode('utf-8')
                        detection["leadImgs"].append({
                            "lead": lead,
                            "image": encoded_string
                        })
        
        # if class_name == "3_4":
        #     plot_and_save_3x4_ecg(ecg_df, file_name, image_path=img_path, results=results)
        # else:
        process_and_plot_leads(ecg_df, file_name, _id, results, top_label='', class_name=class_name, mm_per_sec=25,
                                mm_per_mV=10, signal_scale=0.01)


    # except Exception as e:
    #     print("Signal extraction and arrhythmia detection error : ", e, e.__traceback__.tb_lineno)
   
    arrhythmia_result = results

    data = {
        "_id": _id,
        "status": results['status']  if "status" in results else "success",
        "processData": {
            "HR": arrhythmia_result['avg_hr'],
            "detections": arrhythmia_result['detections'],
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
            "Analysis_leads": arrhythmia_result['arr_analysis_leads'],
            "Not_analysis_leads": arrhythmia_result['arr_not_analysis_leads'],
            "AL": False,
            "imageName": f"{file_name}_{_id}.jpg",  # image_path.split("/")[1],
            "ecgFormat": class_name,
            #                "ArrhythmiaColor": arrhythmia_result['color_dict']

        }
    }
    data = convert_to_serializable(data)
    with open("output_2.txt", "w") as f:
        f.write(json.dumps(data, indent=4))
    if os.path.isfile(f"Result/{file_name}_{_id}.jpg"):
        image_path = f"Result/{file_name}_{_id}.jpg"
        files = {
            'data': (None, json.dumps(data)),  # JSON as part of form-data
            'image': open(image_path, 'rb')  # Open the image file in binary mode
        }
    else:
        files = {
            'data': (None, json.dumps(data)),  # JSON as part of form-data
        }
    url = 'https://oeadev.projectkmt.com/oea/api/v1/uploads/processDataWithImage'
    response = requests.post(url, files=files)
    
    print("Successfully Analysed",response)


def another_call():
    tf.keras.backend.clear_session()
    gc.collect()

    # Get users in round-robin order (oldest timestamp first)
    user_ids = redis_client.zrange("user_priority_zset_csv", 0, -1)

    datas = None
    selected_user_id = None

    for user_id in user_ids:
        user_id_str = user_id.decode() if isinstance(user_id, bytes) else user_id
        queue_key = f"user_csv_queue:{user_id_str}"

        datas = redis_client.lpop(queue_key)

        if datas:
            selected_user_id = user_id_str

            # Move this user to the end of priority queue by updating timestamp
            redis_client.zadd("user_priority_zset_csv", {selected_user_id: time.time()})

            # If queue is now empty, remove from zset
            if redis_client.llen(queue_key) == 0:
                redis_client.zrem("user_priority_zset_csv", selected_user_id)
            break
        else:
            # If empty queue, remove from zset just in case
            redis_client.zrem("user_priority_zset_csv", user_id_str)

    if not datas:
        #        print("|PROCESS_OEA_dev_1| No users in priority queue")
        #        time.sleep(3)
        return '', 200

    get_response = json.loads(datas)
    print(get_response)



    if get_response:
        _id = str(get_response["file_id"])
        userId = get_response["user_id"]
        csv_name = get_response["csv_file_name"]
        img_name = get_response["img_name"]
        print(_id,"******",userId, "*******",csv_name, "*******",img_name)  
        img_path = "newimages/" + img_name

        if os.path.exists(img_path):
            signal_extraction_and_arrhy_detection(_id, user_id, csv_name, img_path)
        else:
            print('\033[41m' + f"No ECG found in image, something wrong" + '\033[0m')
            data = {
                "message": "The ECG is challenging to interpret, possibly due to noise or other irregularities.",
                "status": "fail"
            }

            data = convert_to_serializable(data)
            json_data = json.dumps(data)
            print(data)

            url = 'https://oeadev.projectkmt.com/oea/api/v1/uploads/processDataWithImage'
            response = requests.post(url, data=json_data)
            
            print("File not exist error........")
    else:
        print('\033[41m' + f"No ECG found in image, something wrong" + '\033[0m')
        data = {
            "message": "The ECG is challenging to interpret, possibly due to noise or other irregularities.",
            "status": "fail"
        }

        data = convert_to_serializable(data)
        json_data = json.dumps(data)
        print(data)

        url = 'https://oeadev.projectkmt.com/oea/api/v1/uploads/processDataWithImage'
        response = requests.post(url, data=json_data)
        
        print("Redis queue is not data.......")
    return '', 200




if __name__ == '__main__':
    with tf.device('/CPU:0'):
        interpreter = tf.lite.Interpreter(model_path="Model/PVC_Trans_mob_84_test_tiny_iter1_OEA_minimal.tflite")
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        pac_model = load_tflite_model("Model/PAC_TRANS_GRU_mob_47_OEA_Minimal.tflite")
        afib_model = load_tflite_model("Model/oea_afib_flutter_26_6.tflite")
        vfib_model = load_tflite_model("Model/VFIB_Model_07JUN2024_1038.tflite")
        # vfib_vfl_model = load_tflite_model("Model/vfib_trans_mob_5_enhanced.tflite")
        block_model = load_tflite_model("Model/Block_Trans_mob_29_super_new_minimal.tflite")
        noise_model = load_tflite_model('Model/NOISE_20_GPT.tflite')
        let_inf_moedel = load_tflite_model("Model/MI_15_7_25_oea.tflite")
        reader = easyocr.Reader(['en'])

    while True:
        another_call()
        time.sleep(1)


