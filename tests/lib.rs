extern crate stft;
use stft::{STFT, WindowType};

#[test]
fn test_stft() {
    let mut stft = STFT::<f64>::new(WindowType::Hanning, 8, 4);
    assert!(!stft.can_compute());
    assert_eq!(stft.output_size(), 4);
    assert_eq!(stft.len(), 0);
    stft.feed(&vec![500., 0., 100.][..]);
    assert_eq!(stft.len(), 3);
    assert!(!stft.can_compute());
    let mut output: Vec<f64> = vec![0.; 4];
    stft.compute(&mut output[..]);
    println!("{:?}", output);
}
