extern crate apodize;

extern crate num;
use num::complex::Complex;

extern crate strider;
use strider::SliceRingImpl;

extern crate rustfft;
use rustfft::FFT;

enum Window {
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
    pub window: Vec<T>,
    pub deque: SliceRingImpl<T>,
    pub complex_fft_input: Vec<Complex<T>>,
    pub complex_fft_output: Vec<Complex<T>>,
}

impl<T> STFT<T> {
    pub fn new(
        window_size: usize,
        step_size: usize,
        window: Window
    ) -> Self {
        let window = 
    }
    pub fn new_with_window_slice(
        window_size: usize,
        step_size: usize,
        window: &[T],
    ) -> Self {
        let inverse = false;
        STFT {
            window_size: window_size,
            step_size: step_size,
            fft: FFT::new(window_size, inverse),
            deque: SliceRingImpl::new(),
            window: window,
            complex_fft_input: repeat(Complex::new(0., 0.))
                .take(window_size)
                .collect(),
            complex_fft_output: repeat(Complex::new(0., 0.))
                .take(window_size)
                .collect()
        }
    }

    pub fn output_size(&self) {

    }

    pub fn append(&mut self, input: &[T]) {
        self.deque.push_many_back(input);
    }

    // only half of it
    // # Panics
    // panics unless `self.output_size() == output.len()`
    pub fn compute_complex(&self, output: &mut [Complex<T>]) {
        assert_eq!(self.output_size(), output.len());

        self.fft.process(&self.complex_fft_input, &mut self.complex_fft_output);

        // using iterators to eschew bounds checking
        for (dst, src) in output.iter_mut().zip(self.complex_fft_output.iter()) {
            dst = src;
        }
    }

    // # Panics
    // panics unless `self.output_size() == output.len()`
    pub fn compute_magnitude(&self, output: &mut [T]) {
        assert_eq!(self.output_size(), output.len());

        self.fft.process(&self.complex_fft_input, &mut self.complex_fft_output);

        for (dst, src) in output.iter_mut().zip(self.complex_fft_output.iter()) {
            dst = src.norm();
        }
    }

    // # Panics
    // panics unless `self.output_size() == output.len()`
    pub fn compute(&mut self, output: &mut [T]) {
        assert_eq!(self.output_size(), output.len());

        self.fft.process(&self.complex_fft_input, &mut self.complex_fft_output);

        for (dst, src) in output.iter_mut().zip(self.complex_fft_output.iter()) {
            dst = src.norm().log10();
            if dst.is_sign_negative() {
                dst = 0.;
            }
        }
    }

    // TODO step, next
    pub fn next_window(&mut self) {
        self.buf.next();
    )
}
