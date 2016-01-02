extern crate stft;
use stft::{STFT, WindowType};

#[test]
fn test_stft() {
    let mut stft = STFT::<f64>::new(WindowType::Hanning, 8, 4);
    assert!(!stft.can_compute());
    assert_eq!(stft.output_size(), 512);
}
