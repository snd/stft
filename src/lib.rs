extern crate apodize;
use apodize::CanRepresentPi;

extern crate num;
use num::complex::Complex;
use num::traits::{Float, Signed, FromPrimitive, Zero};

extern crate strider;
use strider::{SliceRing, SliceRingImpl};

extern crate rustfft;
use rustfft::FFT;

pub enum WindowType {
    Hanning,
    Hamming,
    Blackman,
    Nuttall,
    None,
}

pub struct STFT<T> {
    pub window_size: usize,
    pub step_size: usize,
    pub fft: FFT<T>,
    pub window: Option<Vec<T>>,
    pub sample_ring: SliceRingImpl<T>,
    pub complex_fft_input: Vec<Complex<T>>,
    pub complex_fft_output: Vec<Complex<T>>,
}

/// returns log10 of `value` or zero if the log10 of `value` is negative
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

impl<T: Float + Signed + FromPrimitive + CanRepresentPi> STFT<T> {
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
               step_size: usize,
               zero_padding: usize)
               -> STFT<T> {
        let window = STFT::window_type_to_window_vec(window_type, window_size);
        STFT::<T>::new_with_window(window, window_size, step_size, zero_padding)
    }

    // TODO this should ideally take an iterator and not a vec
    pub fn new_with_window(window: Option<Vec<T>>,
                           window_size: usize,
                           step_size: usize,
                           zero_padding: usize)
                           -> STFT<T> {
        // TODO more assertions:
        // window_size is power of two
        // step_size > 0
        assert!(step_size <= window_size);
        let inverse = false;
        let complex_zero = Complex::new(T::from(0.).unwrap(), T::from(0.).unwrap());
        STFT {
            window_size: window_size,
            step_size: step_size,
            fft: FFT::new(window_size, inverse),
            sample_ring: SliceRingImpl::new(),
            window: window,
            complex_fft_input: std::iter::repeat(complex_zero)
                                   .take(window_size)
                                   .collect(),
            complex_fft_output: std::iter::repeat(complex_zero)
                                    .take(window_size)
                                    .collect(),
        }
    }

    pub fn output_size(&self) -> usize {
        self.window_size / 2
    }

    pub fn feed(&mut self, input: &[T]) {
        self.sample_ring.push_many_back(input);
    }

    pub fn can_compute(&self) -> bool {
        self.window_size <= self.sample_ring.len()
    }

    // only half of it
    /// # Panics
    /// panics unless `self.output_size() == output.len()`
    pub fn compute_complex(&mut self, output: &mut [Complex<T>]) {
        assert_eq!(self.output_size(), output.len());

        self.fft.process(&self.complex_fft_input, &mut self.complex_fft_output);

        // using iterators to omit bounds checking
        for (dst, src) in output.iter_mut().zip(self.complex_fft_output.iter()) {
            *dst = src.clone();
        }
    }

    /// # Panics
    /// panics unless `self.output_size() == output.len()`
    pub fn compute_magnitude(&mut self, output: &mut [T]) {
        assert_eq!(self.output_size(), output.len());

        self.fft.process(&self.complex_fft_input, &mut self.complex_fft_output);

        for (dst, src) in output.iter_mut().zip(self.complex_fft_output.iter()) {
            *dst = src.norm();
        }
    }

    /// # Panics
    /// panics unless `self.output_size() == output.len()`
    pub fn compute(&mut self, output: &mut [T]) {
        assert_eq!(self.output_size(), output.len());

        self.fft.process(&self.complex_fft_input, &mut self.complex_fft_output);

        // using iterators to omit bounds checking
        for (dst, src) in output.iter_mut().zip(self.complex_fft_output.iter()) {
            *dst = log10_positive(src.norm());
        }
    }

    /// make a step
    /// drops `self.step_size` samples from the internal buffer `self.sample_ring`.
    pub fn step(&mut self) {
        self.sample_ring.drop_many_front(self.step_size);
    }
}
