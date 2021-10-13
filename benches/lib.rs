use criterion::{criterion_group, criterion_main, Criterion};
use num::complex::Complex;
use rustfft::FFTplanner;
use stft::{WindowType, STFT};

macro_rules! bench_fft_process {
    ($c:expr, $window_size:expr, $float:ty) => {{
        let inverse = false;
        let mut planner = FFTplanner::new(inverse);
        let fft = planner.plan_fft($window_size);
        let mut input = std::iter::repeat(Complex::new(0., 0.))
            .take($window_size)
            .collect::<Vec<Complex<$float>>>();
        let mut output = std::iter::repeat(Complex::new(0., 0.))
            .take($window_size)
            .collect::<Vec<Complex<$float>>>();
        $c.bench_function(
            concat!(
                "bench_fft_process_",
                stringify!($window_size),
                "_",
                stringify!($float)
            ),
            |b| b.iter(|| fft.process(&mut input[..], &mut output[..])),
        );
    }};
}

fn bench_fft_process_1024_f32(c: &mut Criterion) {
    bench_fft_process!(c, 1024, f32);
}

fn bench_fft_process_1024_f64(c: &mut Criterion) {
    bench_fft_process!(c, 1024, f64);
}

criterion_group!(
    benches_fft_process,
    bench_fft_process_1024_f32,
    bench_fft_process_1024_f64
);

macro_rules! bench_stft_compute {
    ($c:expr, $window_size:expr, $float:ty) => {{
        let mut stft = STFT::<$float>::new(WindowType::Hanning, $window_size, 0);
        let input = std::iter::repeat(1.)
            .take($window_size)
            .collect::<Vec<$float>>();
        let mut output = std::iter::repeat(0.)
            .take(stft.output_size())
            .collect::<Vec<$float>>();
        stft.append_samples(&input[..]);
        $c.bench_function(
            concat!(
                "bench_stft_compute_",
                stringify!($window_size),
                "_",
                stringify!($float)
            ),
            |b| b.iter(|| stft.compute_column(&mut output[..])),
        );
    }};
}

fn bench_stft_compute_1024_f32(c: &mut Criterion) {
    bench_stft_compute!(c, 1024, f32);
}

fn bench_stft_compute_1024_f64(c: &mut Criterion) {
    bench_stft_compute!(c, 1024, f64);
}

criterion_group!(
    benches_stft_compute,
    bench_stft_compute_1024_f32,
    bench_stft_compute_1024_f64
);

macro_rules! bench_stft_audio {
    ($c:expr, $seconds:expr, $float:ty) => {{
        // let's generate some fake audio
        let sample_rate: usize = 44100;
        let seconds: usize = $seconds;
        let sample_count = sample_rate * seconds;
        let all_samples = (0..sample_count)
            .map(|x| x as $float)
            .collect::<Vec<$float>>();
        $c.bench_function(
            concat!(
                "bench_stft_audio_",
                stringify!($windowsize),
                "_",
                stringify!($float)
            ),
            |b| {
                b.iter(|| {
                    // let's initialize our short-time fourier transform
                    let window_type: WindowType = WindowType::Hanning;
                    let window_size: usize = 1024;
                    let step_size: usize = 512;
                    let mut stft = STFT::<$float>::new(window_type, window_size, step_size);
                    // we need a buffer to hold a computed column of the spectrogram
                    let mut spectrogram_column: Vec<$float> =
                        std::iter::repeat(0.).take(stft.output_size()).collect();
                    for some_samples in (&all_samples[..]).chunks(3000) {
                        stft.append_samples(some_samples);
                        while stft.contains_enough_to_compute() {
                            stft.compute_column(&mut spectrogram_column[..]);
                            stft.move_to_next_column();
                        }
                    }
                })
            },
        );
    }};
}

fn bench_stft_10_seconds_audio_f32(c: &mut Criterion) {
    bench_stft_audio!(c, 10, f32);
}

fn bench_stft_10_seconds_audio_f64(c: &mut Criterion) {
    bench_stft_audio!(c, 10, f64);
}

criterion_group!(
    benches_stft_audio,
    bench_stft_10_seconds_audio_f32,
    bench_stft_10_seconds_audio_f64
);

criterion_main!(
    benches_fft_process,
    benches_stft_compute,
    benches_stft_audio
);
