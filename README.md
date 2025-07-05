# Stock Price Forecasting with Ensemble Learning

## 1. Projenin Amacı

Bu projenin temel amacı, hisse senedi fiyat tahmini alanında farklı öğrenme modellerinin ve modellerin birleştirilmesiyle oluşan yöntemlerin performansını karşılaştırmaktır. Özellikle **LightGBM**, **LSTM** ve **SVR** gibi popüler algoritmaların tek başına ve çeşitli ensemble stratejileriyle (basit ortalama, hata bazlı ağırlıklandırma ve lineer regresyon) birleştirildiğinde tahmin doğruluğundaki değişimi incelemek hedeflenmiştir.  
Proje kapsamında **Toyota Motor Corporation (TM)** hissesinin geçmiş verileri kullanılarak en iyi tahmin performansına ulaşmak için farklı model ve yöntemler test edilmiştir.  
Bu projede *“Weight-Training Ensemble Model for Stock Price Forecast”* adlı bilimsel makalenin metodolojisi uygulanmıştır.

---

## 2. Metodoloji

### 2.1. Veri Seti ve Ön İşleme

- **Veri Kaynağı:** Kaggle üzerinden alınan Toyota Motor Corporation (TM) hisse senedi verileri (2001–2019).
- **Özellikler (Features):** Open, High, Low, Close, Volume
- **Hedef (Target):** Close (Kapanış fiyatı)
- **Veri Bölümü:**
  - Eğitim: 2001–2016
  - Doğrulama: 2017–2018
  - Test: 2019
- **Ölçekleme:** MinMaxScaler ile 0–1 aralığına çekildi.
- **Window:** 60 günlük geçmiş veri kullanılarak bir sonraki günün kapanış fiyatı tahmin edildi.

---

### 2.2. Modeller ve Yöntemler

- **Temel Modeller**
  - LightGBM
  - LSTM
  - SVR
- **Ensemble Yöntemleri**
  - **Basit Ortalama:** Üç modelin tahminlerinin aritmetik ortalaması
  - **Hata Bazlı Ağırlıklandırma:** Doğrulama setindeki hata oranlarına göre ağırlıklandırma
  - **Lineer Regresyon Ensemble:** Üç modelin tahminleriyle lineer regresyon modeli eğitimi

---

### 2.3. Hata Metrikleri

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

---

## 3. Deneysel Sonuçlar

| Model                    | RMSE   | MAE   |
|--------------------------|--------|-------|
| Linear Regression Ensemble | 1.1090 | 0.8683 |
| LightGBM                 | 1.6145 | 1.2962 |
| LSTM                     | 1.6199 | 1.3582 |
| Weighted Ensemble        | 1.6726 | 1.4041 |
| Simple Ensemble          | 1.7103 | 1.4375 |
| SVR                      | 2.2719 | 1.8815 |

---

## Çıkarımlar

- **Linear Regression Ensemble** yöntemi, tüm yöntemler arasında en düşük RMSE (1.1090) ve MAE (0.8683) değerleri ile en iyi sonucu vermiştir.
- Tek başına kullanılan modeller arasında **LightGBM** en düşük hata değerlerine sahiptir.
- Hata bazlı ensemble, basit ortalama ensemble’dan daha başarılıdır; ancak lineer regresyon ensemble kadar etkili değildir.
- **SVR**, tek başına en yüksek hata değerine sahip modeldir.
- Tüm ensemble yöntemleri, en kötü temel modelden (SVR) daha iyi performans göstermiştir.
- Sonuçlar, makalede de vurgulandığı gibi, farklı modellerin uygun şekilde birleştirilmesinin tahmin doğruluğunu artırdığını göstermektedir.

---

## 4. Grafik

![image](https://github.com/user-attachments/assets/f78578d6-ebe8-4e4c-8921-5007f82df0ba)
