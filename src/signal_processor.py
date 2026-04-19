import numpy as np
import librosa
from typing import Tuple, List
from .layer2_models import AudioFeatures


class SignalProcessor:
    def __init__(self, sample_rate: int = 16000, n_mfcc: int = 13):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.hop_length = 512
        self.n_fft = 2048
    
    def extract_features(self, audio_path: str) -> AudioFeatures:
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        mfcc_mean, mfcc_std = self._extract_mfcc(y)
        pitch_mean, pitch_std, pitch_var = self._extract_pitch(y)
        energy_mean, energy_std = self._extract_energy(y)
        zcr = self._extract_zero_crossing_rate(y)
        spectral_centroid = self._extract_spectral_centroid(y)
        spectral_rolloff = self._extract_spectral_rolloff(y)
        pause_count, pause_mean, speech_dur, pause_dur = self._extract_pause_patterns(y)
        speech_rate = self._calculate_speech_rate(y, speech_dur)
        
        return AudioFeatures(
            mfcc_mean=mfcc_mean,
            mfcc_std=mfcc_std,
            pitch_mean=pitch_mean,
            pitch_std=pitch_std,
            pitch_variance=pitch_var,
            energy_mean=energy_mean,
            energy_std=energy_std,
            zero_crossing_rate=zcr,
            spectral_centroid_mean=spectral_centroid,
            spectral_rolloff_mean=spectral_rolloff,
            pause_count=pause_count,
            pause_duration_mean=pause_mean,
            speech_rate=speech_rate,
            total_speech_duration=speech_dur,
            total_pause_duration=pause_dur
        )
    
    def _extract_mfcc(self, y: np.ndarray) -> Tuple[List[float], List[float]]:
        mfccs = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=self.n_mfcc, hop_length=self.hop_length)
        return np.mean(mfccs, axis=1).tolist(), np.std(mfccs, axis=1).tolist()
    
    def _extract_pitch(self, y: np.ndarray) -> Tuple[float, float, float]:
        pitches, magnitudes = librosa.piptrack(y=y, sr=self.sample_rate, hop_length=self.hop_length, n_fft=self.n_fft)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) > 0:
            return float(np.mean(pitch_values)), float(np.std(pitch_values)), float(np.var(pitch_values))
        return 0.0, 0.0, 0.0
    
    def _extract_energy(self, y: np.ndarray) -> Tuple[float, float]:
        energy = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        return float(np.mean(energy)), float(np.std(energy))
    
    def _extract_zero_crossing_rate(self, y: np.ndarray) -> float:
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)[0]
        return float(np.mean(zcr))
    
    def _extract_spectral_centroid(self, y: np.ndarray) -> float:
        centroid = librosa.feature.spectral_centroid(y=y, sr=self.sample_rate, hop_length=self.hop_length)[0]
        return float(np.mean(centroid))
    
    def _extract_spectral_rolloff(self, y: np.ndarray) -> float:
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sample_rate, hop_length=self.hop_length)[0]
        return float(np.mean(rolloff))
    
    def _extract_pause_patterns(self, y: np.ndarray) -> Tuple[int, float, float, float]:
        energy = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        threshold = np.mean(energy) * 0.3
        is_speech = energy > threshold
        
        pauses = []
        speech_segments = []
        current_pause_start = None
        current_speech_start = None
        frame_duration = self.hop_length / self.sample_rate
        
        for i, speaking in enumerate(is_speech):
            if not speaking:
                if current_speech_start is not None:
                    speech_segments.append((current_speech_start, i * frame_duration))
                    current_speech_start = None
                if current_pause_start is None:
                    current_pause_start = i * frame_duration
            else:
                if current_pause_start is not None:
                    pauses.append((current_pause_start, i * frame_duration))
                    current_pause_start = None
                if current_speech_start is None:
                    current_speech_start = i * frame_duration
        
        pause_count = len(pauses)
        pause_durations = [end - start for start, end in pauses]
        pause_mean = float(np.mean(pause_durations)) if pause_durations else 0.0
        total_pause_duration = float(sum(pause_durations))
        speech_durations = [end - start for start, end in speech_segments]
        total_speech_duration = float(sum(speech_durations))
        
        return pause_count, pause_mean, total_speech_duration, total_pause_duration
    
    def _calculate_speech_rate(self, y: np.ndarray, speech_duration: float) -> float:
        if speech_duration == 0:
            return 0.0
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sample_rate)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sample_rate)
        syllable_count = len(onsets)
        return float(syllable_count / speech_duration if speech_duration > 0 else 0.0)
