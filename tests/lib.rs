extern crate stft;
use stft::{STFT, WindowType};

#[test]
fn test_stft() {
    let stft = STFT::<f64>::new(WindowType::Hanning, 1024, 512, 0);
    assert!(!stft.can_compute());
    assert_eq!(stft.output_size(), 512);
}
