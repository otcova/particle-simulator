use egui::{Color32, FontId, RichText, TextFormat, WidgetText};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumFormat {
    Dashed,
    Scientific,
    Metric,
}

impl NumFormat {
    pub const fn name(self) -> &'static str {
        match self {
            NumFormat::Dashed => "Dashed",
            NumFormat::Scientific => "Scientific",
            NumFormat::Metric => "Metric",
        }
    }
}

#[derive(Clone, Copy)]
pub struct NumFormatter {
    pub format: NumFormat,
    pub figures: u32,
    pub rgb: [u8; 3],
}

const METRIC: &[(f32, &str)] = &[
    (1e9, "G"),
    (1e6, "M"),
    (1e3, "k"),
    (1.0, ""),
    (1e-3, "m"),
    (1e-6, "µ"),
    (1e-9, "n"),
    (1e-12, "p"),
    (1e-15, "f"),
];

impl NumFormatter {
    pub fn raw_string(&self, n: f32, unit: &'static str) -> String {
        let sign = if n < 0. { "-" } else { " " };
        match self.format {
            NumFormat::Dashed => {
                let decs = decimals_for_figures(n, self.figures);
                format_with_underscores(n, decs) + " " + unit
            }
            NumFormat::Scientific => {
                let mut exp = n.log10();
                if exp == f32::NEG_INFINITY {
                    exp = 0.;
                }
                let exp = exp.floor() as i32;
                let mantissa = n / 10_f32.powi(exp);
                let text = format!("{sign}{:.*}·10", self.figures as usize - 1, mantissa);
                format!("{text}^{} {unit}", exp)
            }
            NumFormat::Metric => {
                let mut metric = *METRIC.last().unwrap();
                if n == 0. || !n.is_finite() {
                    metric = (1.0, "");
                } else {
                    for &(divisor, suffix) in METRIC {
                        let threshold = divisor;
                        if n >= threshold {
                            metric = (divisor, suffix);
                            break;
                        }
                    }
                };

                let scaled = n / metric.0;
                let decs = decimals_for_figures(scaled, self.figures);
                format!("{sign}{:.*} {}{unit}", decs, scaled, metric.1)
            }
        }
    }

    pub fn fmt(&self, n: f32, unit: &'static str) -> WidgetText {
        let sign = if n < 0. { "-" } else { " " };
        let n = n.abs();

        let figs = self.figures as usize;
        let color = Color32::from_rgb(self.rgb[0], self.rgb[1], self.rgb[2]);
        let font_id = FontId::monospace(12.);
        let exp_font = FontId::monospace(10.);

        match self.format {
            NumFormat::Dashed => {
                let decs = decimals_for_figures(n, self.figures);
                RichText::new(format_with_underscores(n, decs) + " " + unit)
                    .color(color)
                    .font(font_id)
                    .into()
            }
            NumFormat::Scientific => {
                let mut exp = n.log10();
                if exp == f32::NEG_INFINITY {
                    exp = 0.;
                }
                let exp = exp.floor() as i32;
                let mantissa = n / 10_f32.powi(exp);

                let mut text = egui::text::LayoutJob::default();
                text.append(
                    &format!("{sign}{:.*}·10", figs - 1, mantissa),
                    0.,
                    TextFormat {
                        color,
                        font_id: font_id.clone(),
                        ..Default::default()
                    },
                );
                text.append(
                    &format!("{} ", exp),
                    0.,
                    TextFormat {
                        color,
                        valign: egui::Align::TOP,
                        font_id: exp_font,
                        ..Default::default()
                    },
                );
                text.append(
                    unit,
                    0.,
                    TextFormat {
                        color,
                        font_id,
                        ..Default::default()
                    },
                );
                text.into()
            }
            NumFormat::Metric => {
                let mut metric = *METRIC.last().unwrap();
                if n == 0. || !n.is_finite() {
                    metric = (1.0, "");
                } else {
                    for &(divisor, suffix) in METRIC {
                        let threshold = divisor;
                        if n >= threshold {
                            metric = (divisor, suffix);
                            break;
                        }
                    }
                };

                let scaled = n / metric.0;
                let decs = decimals_for_figures(scaled, self.figures);
                RichText::new(format!("{sign}{:.*} {}{unit}", decs, scaled, metric.1))
                    .color(color)
                    .font(font_id)
                    .into()
            }
        }
    }
}

fn decimals_for_figures(n: f32, sign_figures: u32) -> usize {
    let exp10 = n.abs().log10();
    if exp10 == f32::NEG_INFINITY {
        return 0;
    }
    let digits = exp10.floor() as isize + 1;
    (sign_figures as isize - digits).max(0) as usize
}

/// Formats an f32 with underscores as grouping separators (every 3 digits)
/// in both integer and fractional parts, similar to Rust's numeric literals.
///
/// Examples:
///   1234567.8912.123456789 → "1_234_567_8912.123_456_789"
///   1000.5               → "1_000.5"
///   0.000123             → "0.000_123"
pub fn format_with_underscores(value: f32, decimals: usize) -> String {
    // Handle special cases
    if value.is_nan() {
        return " NaN".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 {
            " inf".to_string()
        } else {
            "-inf".to_string()
        };
    }

    // Format with high precision, then clean up
    let s = format!("{:.*}", decimals, value);

    let is_negative = s.starts_with('-');
    let abs_part = if is_negative { &s[1..] } else { &s };

    let (int_part, frac_part) = abs_part.split_once('.').unwrap_or((abs_part, ""));

    let mut result = String::with_capacity(s.len() + s.len() / 3);

    if is_negative {
        result.push('-');
    } else {
        result.push(' ');
    }

    // Group integer part from the right
    let int_len = int_part.len();
    for (i, ch) in int_part.chars().enumerate() {
        if i > 0 && (int_len - i) % 3 == 0 {
            result.push('_');
        }
        result.push(ch);
    }

    // Group fractional part from the left
    if !frac_part.is_empty() {
        result.push('.');
        for (i, ch) in frac_part.chars().enumerate() {
            if i > 0 && i % 3 == 0 {
                result.push('_');
            }
            result.push(ch);
        }
    }

    // Fix -0
    if result == "-0" {
        " 0".to_string()
    } else {
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_with_underscores() {
        assert_eq!(format_with_underscores(-1000.0, 0), "-1_000");
        assert_eq!(format_with_underscores(12.3456, 6), " 12.345_600");
    }
}
