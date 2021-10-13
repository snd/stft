use std::str::FromStr;
use stft::{WindowType, STFT};

#[test]
fn test_window_type_from_string() {
    assert_eq!(
        WindowType::from_str("Hanning").unwrap(),
        WindowType::Hanning
    );
    assert_eq!(
        WindowType::from_str("hanning").unwrap(),
        WindowType::Hanning
    );
    assert_eq!(WindowType::from_str("hann").unwrap(), WindowType::Hanning);
    assert_eq!(
        WindowType::from_str("blackman").unwrap(),
        WindowType::Blackman
    );
}

#[test]
fn test_window_type_to_string() {
    assert_eq!(WindowType::Hanning.to_string(), "Hanning");
}

#[test]
fn test_window_types_to_strings() {
    assert_eq!(
        vec!["Hanning", "Hamming", "Blackman", "Nuttall", "None"],
        WindowType::values()
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
    );
}

#[test]
fn test_log10_positive() {
    assert!(stft::log10_positive(-1. as f64).is_nan());
    assert_eq!(stft::log10_positive(0.), 0.);
    assert_eq!(stft::log10_positive(1.), 0.);
    assert_eq!(stft::log10_positive(10.), 1.);
    assert_eq!(stft::log10_positive(100.), 2.);
    assert_eq!(stft::log10_positive(1000.), 3.);
}

#[test]
fn test_stft() {
    let mut stft = STFT::new(WindowType::Hanning, 8, 4);
    assert!(!stft.contains_enough_to_compute());
    assert_eq!(stft.output_size(), 4);
    assert_eq!(stft.len(), 0);
    stft.append_samples(&vec![500., 0., 100.][..]);
    assert_eq!(stft.len(), 3);
    assert!(!stft.contains_enough_to_compute());
    stft.append_samples(&vec![500., 0., 100., 0.][..]);
    assert_eq!(stft.len(), 7);
    assert!(!stft.contains_enough_to_compute());

    stft.append_samples(&vec![500.][..]);
    assert!(stft.contains_enough_to_compute());

    let mut output: Vec<f64> = vec![0.; 4];
    stft.compute_column(&mut output[..]);
    println!("{:?}", output);
}
