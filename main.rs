// Função para calcular a média
pub fn calcular_media(valores: &[f64]) -> f64 {
    valores.iter().sum::<f64>() / valores.len() as f64
}

// Função para calcular coeficientes da regressão linear
pub fn calcular_coeficientes(x: &[f64], y: &[f64]) -> (f64, f64) {
    let media_x = calcular_media(x);
    let media_y = calcular_media(y);

    let mut numerador = 0.0;
    let mut denominador = 0.0;

    for i in 0..x.len() {
        numerador += (x[i] - media_x) * (y[i] - media_y);
        denominador += (x[i] - media_x).powi(2);
    }

    let slope = numerador / denominador;
    let intercept = media_y - slope * media_x;
    (slope, intercept)
}

// Função para calcular R² e MSE
pub fn calcular_r2_e_mse(x: &[f64], y: &[f64], slope: f64, intercept: f64) -> (f64, f64) {
    let media_y = calcular_media(y);
    let mut ss_total = 0.0;
    let mut ss_residual = 0.0;

    for i in 0..x.len() {
        let y_pred = slope * x[i] + intercept;
        ss_total += (y[i] - media_y).powi(2);
        ss_residual += (y[i] - y_pred).powi(2);
    }

    let r2 = 1.0 - (ss_residual / ss_total);
    let mse = ss_residual / x.len() as f64;
    (r2, mse)
}

// Função principal para rodar o modelo
fn main() {
    let tempos = vec![1.0, 2.0, 3.0, 4.0];
    let valores = vec![3.0, 5.0, 7.0, 9.0];

    let (slope, intercept) = calcular_coeficientes(&tempos, &valores);
    println!("Inclinação (slope): {:.4}", slope);
    println!("Intercepto: {:.4}", intercept);

    let (r2, mse) = calcular_r2_e_mse(&tempos, &valores, slope, intercept);
    println!("R²: {:.4}", r2);
    println!("MSE: {:.4}", mse);

    let proximo_tempo = 5.0;
    let previsao = slope * proximo_tempo + intercept;
    println!("Previsão para o tempo {}: {:.2}", proximo_tempo, previsao);
}

// Testes unitários (simples e práticos)
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calcular_media() {
        let dados = vec![10.0, 20.0, 30.0];
        assert_eq!(calcular_media(&dados), 20.0);
    }

    #[test]
    fn test_coeficientes_regressao() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![3.0, 5.0, 7.0];
        let (slope, intercept) = calcular_coeficientes(&x, &y);
        assert!((slope - 2.0).abs() < 1e-6);
        assert!((intercept - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_r2_e_mse() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![3.0, 5.0, 7.0];
        let (slope, intercept) = calcular_coeficientes(&x, &y);
        let (r2, mse) = calcular_r2_e_mse(&x, &y, slope, intercept);
        assert!((r2 - 1.0).abs() < 1e-6);
        assert!((mse - 0.0).abs() < 1e-6);
    }
}


    