//! MFCC frontend. Must match training/features.py exactly -- that file is the
//! spec; any change there must be mirrored here. The boot self-test compares
//! this implementation against features computed by the Python side on an
//! embedded PCM clip.
//!
//! Spec: 480-sample frames (30 ms) hopped by 320 (20 ms), periodic Hann
//! window, 512-point real FFT, power spectrum, 40 HTK-mel triangular filters
//! over 20..4000 Hz (bins 0..=128), ln(mel + 1e-6), orthonormal DCT-II, first
//! 10 coefficients.

pub const FRAME_LEN: usize = 480;
pub const FRAME_HOP: usize = 320;
pub const NUM_FRAMES: usize = 49;
pub const NUM_MFCC: usize = 10;

const FFT_LEN: usize = 512;
const NUM_MEL_BINS: usize = 129; // k*31.25 Hz, k = 0..=128 covers 0..4000 Hz
const MEL_FILTERS: usize = 40;
const MEL_LOW_HZ: f32 = 20.0;
const MEL_HIGH_HZ: f32 = 4000.0;
const LOG_FLOOR: f32 = 1e-6;

/// Precomputed tables (~22 KB). Lives in a zeroed static; call `init` once.
pub struct Tables {
    window: [f32; FRAME_LEN],
    mel_w: [[f32; NUM_MEL_BINS]; MEL_FILTERS],
    dct: [[f32; MEL_FILTERS]; NUM_MFCC],
}

fn mel(hz: f32) -> f32 {
    2595.0 * libm::log10f(1.0 + hz / 700.0)
}

fn mel_to_hz(m: f32) -> f32 {
    700.0 * (libm::powf(10.0, m / 2595.0) - 1.0)
}

impl Tables {
    pub const ZEROED: Tables = Tables {
        window: [0.0; FRAME_LEN],
        mel_w: [[0.0; NUM_MEL_BINS]; MEL_FILTERS],
        dct: [[0.0; MEL_FILTERS]; NUM_MFCC],
    };

    pub fn init(&mut self) {
        const PI: f32 = core::f32::consts::PI;

        for i in 0..FRAME_LEN {
            self.window[i] = 0.5 - 0.5 * libm::cosf(2.0 * PI * i as f32 / FRAME_LEN as f32);
        }

        // 42 equally spaced points on the mel axis; triangle m spans points
        // m..m+2, evaluated at bin center frequencies k * 16000/512.
        let mel_lo = mel(MEL_LOW_HZ);
        let mel_hi = mel(MEL_HIGH_HZ);
        let step = (mel_hi - mel_lo) / (MEL_FILTERS + 1) as f32;
        for m in 0..MEL_FILTERS {
            let lo = mel_to_hz(mel_lo + step * m as f32);
            let ctr = mel_to_hz(mel_lo + step * (m + 1) as f32);
            let hi = mel_to_hz(mel_lo + step * (m + 2) as f32);
            for k in 0..NUM_MEL_BINS {
                let f = k as f32 * 16000.0 / FFT_LEN as f32;
                let rising = (f - lo) / (ctr - lo);
                let falling = (hi - f) / (hi - ctr);
                let w = if rising < falling { rising } else { falling };
                self.mel_w[m][k] = if w > 0.0 { w } else { 0.0 };
            }
        }

        let scale0 = libm::sqrtf(1.0 / MEL_FILTERS as f32);
        let scale = libm::sqrtf(2.0 / MEL_FILTERS as f32);
        for j in 0..NUM_MFCC {
            for m in 0..MEL_FILTERS {
                let basis =
                    libm::cosf(PI * j as f32 * (2.0 * m as f32 + 1.0) / (2.0 * MEL_FILTERS as f32));
                self.dct[j][m] = if j == 0 { scale0 } else { scale } * basis;
            }
        }
    }

    /// One 30 ms frame of i16 PCM -> 10 MFCCs.
    pub fn process(&self, frame: &[i16; FRAME_LEN]) -> [f32; NUM_MFCC] {
        let mut buf = [0.0f32; FFT_LEN];
        for i in 0..FRAME_LEN {
            buf[i] = frame[i] as f32 / 32768.0 * self.window[i];
        }
        let spec = microfft::real::rfft_512(&mut buf);
        // microfft packs the (real) Nyquist bin into spec[0].im; numpy's rfft
        // has a pure-real DC bin there. The mel range stops at bin 128 anyway.
        spec[0].im = 0.0;

        let mut power = [0.0f32; NUM_MEL_BINS];
        for k in 0..NUM_MEL_BINS {
            let c = spec[k];
            power[k] = c.re * c.re + c.im * c.im;
        }

        let mut logmel = [0.0f32; MEL_FILTERS];
        for m in 0..MEL_FILTERS {
            let mut acc = 0.0f32;
            let w = &self.mel_w[m];
            for k in 0..NUM_MEL_BINS {
                acc += w[k] * power[k];
            }
            logmel[m] = libm::logf(acc + LOG_FLOOR);
        }

        let mut out = [0.0f32; NUM_MFCC];
        for j in 0..NUM_MFCC {
            let mut acc = 0.0f32;
            for m in 0..MEL_FILTERS {
                acc += self.dct[j][m] * logmel[m];
            }
            out[j] = acc;
        }
        out
    }
}
