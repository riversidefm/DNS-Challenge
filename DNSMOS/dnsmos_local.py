# Usage:
# python dnsmos_local.py -t c:\temp\DNSChallenge4_Blindset -o DNSCh4_Blind.csv -p
#

from abc import ABC, abstractmethod
import concurrent.futures
import glob
from multiprocessing.pool import ThreadPool
import os
import threading
import time
from typing import List, Optional, Tuple

import librosa
import numpy as np
import onnxruntime as ort
import pandas as pd
import soundfile as sf
from tqdm import tqdm

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01

DEFAULT_ORT_PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")


class ComputeScore:
    def __init__(
        self,
        session_options: Optional[ort.SessionOptions] = None,
        providers: Tuple[str, ...] = DEFAULT_ORT_PROVIDERS,
    ) -> None:
        # Note on the execution providers: we can get them from onnxruntime.get_available_providers(),
        #  which returns this: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        #  However, when provided in the InferenceSession "providers" parameters, onnxruntime complains
        #  that the TensorRT libraries are not installed. So I'm manually setting the providers to
        #  'CUDAExecutionProvider' and 'CPUExecutionProvider'.

        curdir = os.path.dirname(os.path.abspath(__file__))
        self._primary_model_path = os.path.join(curdir, "DNSMOS", "sig_bak_ovr.onnx")
        self._p808_model_path = os.path.join(curdir, "DNSMOS", "model_v8.onnx")
        self._providers = providers
        self._session_options = session_options
        self._p808_onnx_sess = None
        self._onnx_sess = None

        # pre-load the primary model
        self.onnx_sess

    @property
    def p808_onnx_sess(self) -> ort.InferenceSession:
        if self._p808_onnx_sess is None:
            self._p808_onnx_sess = ort.InferenceSession(
                self._p808_model_path,
                providers=self._providers,
                sess_options=self._session_options,
            )
        return self._p808_onnx_sess

    @property
    def onnx_sess(self) -> ort.InferenceSession:
        if self._onnx_sess is None:
            self._onnx_sess = ort.InferenceSession(
                self._primary_model_path,
                providers=self._providers,
                sess_options=self._session_options,
            )
        return self._onnx_sess

    def audio_melspec(
        self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True
    ):
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length, n_mels=n_mels
        )
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(
        self,
        audio: np.ndarray,
        sampling_rate: int,
        is_personalized_MOS=False,
        return_p808=True,
    ) -> dict:
        actual_audio_len = len(audio)
        len_samples = int(INPUT_LENGTH * sampling_rate)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / sampling_rate) - INPUT_LENGTH) + 1
        hop_len_samples = sampling_rate
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[
                int(idx * hop_len_samples) : int((idx + INPUT_LENGTH) * hop_len_samples)
            ]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype("float32")[np.newaxis, :]
            oi = {"input_1": input_features}
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(
                mos_sig_raw, mos_bak_raw, mos_ovr_raw, is_personalized_MOS
            )
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            if return_p808:
                p808_input_features = np.array(
                    self.audio_melspec(audio=audio_seg[:-160])
                ).astype("float32")[np.newaxis, :, :]
                p808_oi = {"input_1": p808_input_features}
                p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
                predicted_p808_mos.append(p808_mos)

        clip_dict = {
            "len_in_sec": actual_audio_len / sampling_rate,
            "sr": sampling_rate,
        }
        clip_dict["num_hops"] = num_hops
        clip_dict["OVRL_raw"] = np.mean(predicted_mos_ovr_seg_raw)
        clip_dict["SIG_raw"] = np.mean(predicted_mos_sig_seg_raw)
        clip_dict["BAK_raw"] = np.mean(predicted_mos_bak_seg_raw)
        clip_dict["OVRL"] = np.mean(predicted_mos_ovr_seg)
        clip_dict["SIG"] = np.mean(predicted_mos_sig_seg)
        clip_dict["BAK"] = np.mean(predicted_mos_bak_seg)
        if return_p808:
            clip_dict["P808_MOS"] = np.mean(predicted_p808_mos)
        return clip_dict


def main(args):
    models = glob.glob(os.path.join(args.testset_dir, "*"))
    audio_clips_list = []
    p808_model_path = os.path.join("DNSMOS", "model_v8.onnx")

    if args.personalized_MOS:
        primary_model_path = os.path.join("pDNSMOS", "sig_bak_ovr.onnx")
    else:
        primary_model_path = os.path.join("DNSMOS", "sig_bak_ovr.onnx")

    compute_score = ComputeScore(primary_model_path, p808_model_path)

    rows = []
    clips = []
    clips = glob.glob(os.path.join(args.testset_dir, "*.wav"))
    is_personalized_eval = args.personalized_MOS
    desired_fs = SAMPLING_RATE
    for m in tqdm(models):
        max_recursion_depth = 10
        audio_path = os.path.join(args.testset_dir, m)
        audio_clips_list = glob.glob(os.path.join(audio_path, "*.wav"))
        while len(audio_clips_list) == 0 and max_recursion_depth > 0:
            audio_path = os.path.join(audio_path, "**")
            audio_clips_list = glob.glob(os.path.join(audio_path, "*.wav"))
            max_recursion_depth -= 1
        clips.extend(audio_clips_list)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_url = {
            executor.submit(compute_score, clip, desired_fs, is_personalized_eval): clip
            for clip in clips
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_url)):
            clip = future_to_url[future]
            try:
                data = future.result()
            except Exception as exc:
                print("%r generated an exception: %s" % (clip, exc))
            else:
                rows.append(data)

    df = pd.DataFrame(rows)
    if args.csv_path:
        csv_path = args.csv_path
        df.to_csv(csv_path)
    else:
        print(df.describe())


# if __name__=="__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-t', "--testset_dir", default='.',
#                         help='Path to the dir containing audio clips in .wav to be evaluated')
#     parser.add_argument('-o', "--csv_path", default=None, help='Dir to the csv that saves the results')
#     parser.add_argument('-p', "--personalized_MOS", action='store_true',
#                         help='Flag to indicate if personalized MOS score is needed or regular')

#     args = parser.parse_args()

#     main(args)


class TestCaseBase(ABC):
    @abstractmethod
    def run(self, audios: List[np.ndarray]) -> List[dict]:
        pass


class TestCase1(TestCaseBase):
    def __init__(self) -> None:
        self.dnsmos = ComputeScore()

    def run(self, audios: List[np.ndarray]) -> List[dict]:
        res = []
        for audio in audios:
            score = self.dnsmos(
                audio=audio, sampling_rate=SAMPLING_RATE, return_p808=False
            )
            res.append(score)
        return res

class TestCase6(TestCase1):
    def __init__(self) -> None:
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        self.dnsmos = ComputeScore(session_options=options)


class TestCase2(TestCaseBase):
    def __init__(self, num_threads: int) -> None:
        self.dnsmos = ComputeScore()
        self.pool = ThreadPool(num_threads)

    def run(self, audios: List[np.ndarray]) -> List[dict]:
        res = []
        self.inputs = [(audio, SAMPLING_RATE, False) for audio in audios]
        res = self.pool.starmap(self.dnsmos, self.inputs)
        return res


class TestCase3(TestCaseBase):
    def __init__(self, num_intra_threads: int) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = num_intra_threads
        self.dnsmos = ComputeScore(session_options=options)

    def run(self, audios: List[np.ndarray]) -> List[dict]:
        res = []
        self.inputs = [(audio, SAMPLING_RATE, False) for audio in audios]
        for audio in audios:
            score = self.dnsmos(
                audio=audio, sampling_rate=SAMPLING_RATE, return_p808=False
            )
            res.append(score)
        return res


class TestCase4(TestCase1):
    def __init__(self):
        self.dnsmos = ComputeScore(
            providers=(("CUDAExecutionProvider", {"enable_cuda_graph": True}),)
        )


_thread_local_data = threading.local()


class TestCase5(TestCase1):
    def __init__(self, num_threads: int) -> None:
        self.thread_pool_executor = concurrent.futures.ThreadPoolExecutor(num_threads)

    def run(self, audios: List[np.ndarray]) -> List[dict]:
        res = []
        self.inputs = [(audio, SAMPLING_RATE, False) for audio in audios]
        res = list(self.thread_pool_executor.map(self._run, self.inputs))
        return res

    def _run(self, inps):
        audio, sampling_rate, return_p808 = inps
        dnsmos = self.dnsmos
        return dnsmos(audio=audio, sampling_rate=sampling_rate, return_p808=return_p808)

    @property
    def dnsmos(self):
        dnsmos = getattr(_thread_local_data, "dnsmos", None)
        if dnsmos is None:
            options = ort.SessionOptions()
            dnsmos = ComputeScore(session_options=options)
            setattr(_thread_local_data, "dnsmos", dnsmos)
        return dnsmos


if __name__ == "__main__":
    audio, _ = librosa.load(
        "/home/eliran/repos/ai-research/original.wav", sr=SAMPLING_RATE
    )
    num_audios = 1000
    max_nthreads = 4
    # audios = [audio for _ in range(num_audios)]
    # case 1
    named_cases = {
        "loop": TestCase1(),
        # "thread": TestCase2(4),
        # "thread_pool_executor_with_local_data": TestCase5(max_nthreads),
        "ort_parallel": TestCase6(),
    }
    for case_name, case in named_cases.items():
        audios = [audio.copy() for _ in range(num_audios)]  # fail when moving it outside the loop
        case.run(
            audios[:max_nthreads]
        )  # warmup - as stated in https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
        t0 = time.perf_counter()
        score = case.run(audios)
        t1 = time.perf_counter()
        print(f"{case_name}: {t1-t0:.3f} s")
    print("done")
