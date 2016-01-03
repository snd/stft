use std::str::FromStr;

extern crate num;
use num::complex::Complex;
use num::traits::{Float, Signed, FromPrimitive, Zero};

extern crate apodize;
use apodize::CanRepresentPi;

extern crate strider;
use strider::{SliceRing, SliceRingImpl};

extern crate rustfft;
use rustfft::FFT;

/// returns zero if `log10(value).is_negative()`.
/// otherwise returns `log10(value)`.
/// `log10` turns values in domain `0..1` into values
/// in range `-inf..0`.
/// `log10_positive` turns values in domain `0..1` into `0`.
/// this sets very small values to zero which may not be
/// what you want depending on your application.
#[inline]
pub fn log10_positive<T: Float + Signed + Zero>(value: T) -> T {
    // Float.log10
    // Signed.is_negative
    // Zero.zero
    let log = value.log10();
    if log.is_negative() {
        T::zero()
    } else {
        log
    }
}

/// the type of apodization window to use
#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Debug, Hash)]
pub enum WindowType {
    Hanning,
    Hamming,
    Blackman,
    Nuttall,
    None,
}

impl FromStr for WindowType {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.to_lowercase();
        match &lower[..] {
            "hanning" => Ok(WindowType::Hanning),
            "hann" => Ok(WindowType::Hanning),
            "hamming" => Ok(WindowType::Hamming),
            "blackman" => Ok(WindowType::Blackman),
            "nuttall" => Ok(WindowType::Nuttall),
            "none" => Ok(WindowType::None),
            _ => Err("no match"),
        }
    }
}

// this also implements ToString::to_string
impl std::fmt::Display for WindowType {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "{:?}", self)
    }
}

// TODO write a macro that does this automatically for any enum
static WINDOW_TYPES: [WindowType; 5] = [
    WindowType::Hanning,
    WindowType::Hamming,
    WindowType::Blackman,
    WindowType::Nuttall,
    WindowType::None];

impl WindowType {
    pub fn values() -> [WindowType; 5] {
        WINDOW_TYPES
    }
}

pub struct STFT<T> {
    pub window_size: usize,
    pub step_size: usize,
    pub fft: FFT<T>,
    pub window: Option<Vec<T>>,
    pub sample_ring: SliceRingImpl<T>,
    pub real_input: Vec<T>,
    pub complex_input: Vec<Complex<T>>,
    pub complex_output: Vec<Complex<T>>,
}

impl<T: Float + Signed + Zero + FromPrimitive + CanRepresentPi> STFT<T> {
    pub fn window_type_to_window_vec(window_type: WindowType,
                                     window_size: usize)
        -> Option<Vec<T>> {
            match window_type {
                WindowType::Hanning => Some(apodize::hanning_iter(window_size).collect::<Vec<T>>()),
                WindowType::Hamming => Some(apodize::hamming_iter(window_size).collect::<Vec<T>>()),
                WindowType::Blackman => Some(apodize::blackman_iter(window_size).collect::<Vec<T>>()),
                WindowType::Nuttall => Some(apodize::nuttall_iter(window_size).collect::<Vec<T>>()),
                WindowType::None => None,
            }
        }

    pub fn new(window_type: WindowType,
               window_size: usize,
               step_size: usize)
        -> STFT<T> {
            let window = STFT::window_type_to_window_vec(window_type, window_size);
            STFT::<T>::new_with_window(window, window_size, step_size)
        }

    // TODO this should ideally take an iterator and not a vec
    pub fn new_with_window(window: Option<Vec<T>>,
                           window_size: usize,
                           step_size: usize)
        -> STFT<T> {
            // TODO more assertions:
            // window_size is power of two
            // step_size > 0
            assert!(step_size <= window_size);
            let inverse = false;
            STFT {
                window_size: window_size,
                step_size: step_size,
                fft: FFT::new(window_size, inverse),
                sample_ring: SliceRingImpl::new(),
                window: window,
                real_input: std::iter::repeat(T::zero())
                    .take(window_size)
                    .collect(),
                complex_input: std::iter::repeat(Complex::<T>::zero())
                    .take(window_size)
                    .collect(),
                complex_output: std::iter::repeat(Complex::<T>::zero())
                    .take(window_size)
                    .collect(),
            }
        }

    #[inline]
    pub fn output_size(&self) -> usize {
        self.window_size / 2
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.sample_ring.len()
    }

    pub fn feed(&mut self, input: &[T]) {
        self.sample_ring.push_many_back(input);
    }

    #[inline]
    pub fn can_compute(&self) -> bool {
        self.window_size <= self.sample_ring.len()
    }

    fn compute_into_complex_output(&mut self) {
        assert!(self.can_compute());

        // read into real_input
        self.sample_ring.read_many_front(&mut self.real_input[..]);

        // multiply real_input with window
        if let Some(ref window) = self.window {
            for (dst, src) in self.real_input.iter_mut().zip(window.iter()) {
                *dst = *dst * *src;
            }
        }

        // copy windowed real_input as real parts into complex_input
        for (dst, src) in self.complex_input.iter_mut().zip(self.real_input.iter()) {
            dst.re = src.clone();
        }

        // compute fft
        self.fft.process(&self.complex_input, &mut self.complex_output);
    }

    /// only half of it
    /// # Panics
    /// panics unless `self.output_size() == output.len()`
    pub fn compute_complex(&mut self, output: &mut [Complex<T>]) {
        assert_eq!(self.output_size(), output.len());

        self.compute_into_complex_output();

        for (dst, src) in output.iter_mut().zip(self.complex_output.iter()) {
            *dst = src.clone();
        }
    }

    /// # Panics
    /// panics unless `self.output_size() == output.len()`
    pub fn compute_magnitude(&mut self, output: &mut [T]) {
        assert_eq!(self.output_size(), output.len());

        self.compute_into_complex_output();

        for (dst, src) in output.iter_mut().zip(self.complex_output.iter()) {
            *dst = src.norm();
        }
    }

    /// # Panics
    /// panics unless `self.output_size() == output.len()`
    pub fn compute(&mut self, output: &mut [T]) {
        assert_eq!(self.output_size(), output.len());

        self.compute_into_complex_output();

        for (dst, src) in output.iter_mut().zip(self.complex_output.iter()) {
            *dst = log10_positive(src.norm());
        }
    }

    /// make a step
    /// drops `self.step_size` samples from the internal buffer `self.sample_ring`.
    pub fn step(&mut self) {
        self.sample_ring.drop_many_front(self.step_size);
    }
}
