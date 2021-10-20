use approx::assert_ulps_eq;
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
    stft.append_samples(&[500., 0., 100.]);
    assert_eq!(stft.len(), 3);
    assert!(!stft.contains_enough_to_compute());
    stft.append_samples(&[500., 0., 100., 0.]);
    assert_eq!(stft.len(), 7);
    assert!(!stft.contains_enough_to_compute());

    stft.append_samples(&[500.]);
    assert!(stft.contains_enough_to_compute());

    let mut output: Vec<f64> = vec![0.; 4];
    stft.compute_column(&mut output);
    println!("{:?}", output);

    let expected = vec![
        2.7763337740785166,
        2.7149781042402594,
        2.6218024907053796,
        2.647816050270838,
    ];
    assert_ulps_eq!(output.as_slice(), expected.as_slice(), max_ulps = 10);

    // repeat the calculation to ensure results are independent of the internal buffer
    let mut output2: Vec<f64> = vec![0.; 4];
    stft.compute_column(&mut output2);
    assert_ulps_eq!(output.as_slice(), output2.as_slice(), max_ulps = 10);
}

#[test]
fn test_stft_padded() {
    let mut stft = STFT::new_with_zero_padding(WindowType::Hanning, 8, 32, 4);
    assert!(!stft.contains_enough_to_compute());
    assert_eq!(stft.output_size(), 16);
    assert_eq!(stft.len(), 0);
    stft.append_samples(&[500., 0., 100.]);
    assert_eq!(stft.len(), 3);
    assert!(!stft.contains_enough_to_compute());
    stft.append_samples(&[500., 0., 100., 0.]);
    assert_eq!(stft.len(), 7);
    assert!(!stft.contains_enough_to_compute());

    stft.append_samples(&[500.]);
    assert!(stft.contains_enough_to_compute());

    let mut output: Vec<f64> = vec![0.; 16];
    stft.compute_column(&mut output);
    println!("{:?}", output);

    let expected = vec![
        2.7763337740785166,
        2.772158781619449,
        2.7598791705720664,
        2.740299218211912,
        2.7149781042402594,
        2.686495897766628,
        2.6585877421915676,
        2.635728083951981,
        2.6218024907053796,
        2.6183544930578027,
        2.6238833073831658,
        2.634925941918913,
        2.647816050270838,
        2.65977332745612,
        2.6691025866822033,
        2.6749381613735683,
    ];
    assert_ulps_eq!(output.as_slice(), expected.as_slice(), max_ulps = 10);

    // repeat the calculation to ensure results are independent of the internal buffer
    let mut output2: Vec<f64> = vec![0.; 16];
    stft.compute_column(&mut output2);
    assert_ulps_eq!(output.as_slice(), output2.as_slice(), max_ulps = 10);
}
