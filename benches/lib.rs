#![feature(test)]
extern crate test;

extern crate num;
use num::complex::Complex;

extern crate rustfft;
use rustfft::FFT;

extern crate stft;
use stft::{STFT, WindowType};

macro_rules! bench_fft_process {
    ($bencher:expr, $window_size:expr, $float:ty) => {{
        let inverse = false;
        let window_size = $window_size;
        let mut fft = FFT::<$float>::new(window_size, inverse);
        let input = std::iter::repeat(Complex::new(0., 0.))
            .take(window_size)
            .collect::<Vec<Complex<$float>>>();
        let mut output = std::iter::repeat(Complex::new(0., 0.))
            .take(window_size)
            .collect::<Vec<Complex<$float>>>();
        $bencher.iter(|| {
            fft.process(&input[..], &mut output[..])
        });

    }}
}

#[bench]
fn bench_fft_process_1024_f32(bencher: &mut test::Bencher) {
    bench_fft_process!(bencher, 1024, f32);
}

#[bench]
fn bench_fft_process_1024_f64(bencher: &mut test::Bencher) {
    bench_fft_process!(bencher, 1024, f64);
}

macro_rules! bench_stft_compute {
    ($bencher:expr, $window_size:expr, $float:ty) => {{
        let mut stft = STFT::<$float>::new(WindowType::Hanning, $window_size, 0, 0);
        let input = std::iter::repeat(1.).take($window_size).collect::<Vec<$float>>();
        let mut output = std::iter::repeat(0.).take(stft.output_size()).collect::<Vec<$float>>();
        stft.feed(&input[..]);
        $bencher.iter(|| {
            stft.compute(&mut output[..])
        });
    }}
}

#[bench]
fn bench_stft_compute_1024_f64(bencher: &mut test::Bencher) {
    bench_stft_compute!(bencher, 1024, f64);
}

#[bench]
fn bench_stft_compute_1024_f32(bencher: &mut test::Bencher) {
    bench_stft_compute!(bencher, 1024, f32);
}

// #[bench]
// fn bench_stft_1_minute_audio(bencher: &mut test::Bencher) {
//     $bencher.iter(|| {
//         fft.process(&input[..], &mut output[..])
//     });
// }
