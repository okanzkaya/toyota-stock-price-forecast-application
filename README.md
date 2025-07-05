Hisse Senedi Fiyat Tahmini için Ensemble Modellerinin Karşılaştırılması
Bu proje, "Weight-Training Ensemble Model for Stock Price Forecast" adlı bilimsel makalenin metodolojisini uygulayarak hisse senedi fiyat tahmini alanında farklı makine öğrenmesi modellerinin ve ensemble yöntemlerinin performansını karşılaştırmaktadır.

1. Projenin Amacı
Bu projenin temel amacı, hisse senedi fiyat tahmini alanında farklı öğrenme modellerinin ve modellerin birleştirilmesiyle oluşan yöntemlerin performansını karşılaştırmaktır. Özellikle LightGBM, LSTM ve SVR gibi popüler algoritmaların tek başına ve çeşitli ensemble stratejileriyle (basit ortalama, hataya dayalı ağırlıklandırma ve lineer regresyon ile) birleştirildiğinde tahmin doğruluğunda nasıl bir değişim olduğunu incelemek hedeflenmiştir. Proje kapsamında, Toyota Motor Corporation (TM) hissesinin geçmiş verileri kullanılarak, en iyi tahmin performansına ulaşmak için farklı model ve yöntemler test edilmiştir.

2. Metodoloji
2.1. Veri Seti ve Ön İşleme (Preprocessing)
Veri Kaynağı: Kaggle üzerinden alınan Toyota Motor Corporation (TM) hisse senedi verileri (2001-2019).

Öznitelikler (Features): Open, High, Low, Close, Volume.

Hedef Değişken (Target): Close (Kapanış fiyatı).

Veri Bölümlemesi:

Eğitim (Training): 2001-2016 yılları arası

Doğrulama (Validation): 2017-2018 yılları arası

Test: 2019 yılı

Normalizasyon: Veri, MinMaxScaler ile 0-1 aralığına ölçeklendi.

Pencere Yapısı (Windowing): 60 günlük geçmiş veri ile bir sonraki günün kapanış fiyatı tahmin edildi. (Makalede net bir pencere aralığı belirtilmemiştir.)

2.2. Modeller ve Yöntemler
Temel Modeller:

LightGBM

LSTM

SVR

Ensemble Yöntemleri:

Basit Ortalama (Simple Averaging Ensemble): Üç modelin tahminlerinin aritmetik ortalaması.

Hataya Dayalı Ensemble (Error-based Ensemble): Doğrulama setindeki hata oranlarına göre ağırlıklandırılmış birleşik model.

Lineer Regresyon Ensemble (Stacking): Üç modelin tahminleriyle yeni bir lineer regresyon modeli eğitilerek oluşturulan birleşik model.

2.3. Hata Metrikleri
RMSE (Root Mean Squared Error - Kök Ortalama Kare Hata)

MAE (Mean Absolute Error - Ortalama Mutlak Hata)

4. Deneysel Sonuçlar
Aşağıdaki tablo, test seti üzerindeki modellerin performansını göstermektedir.

Model

RMSE

MAE

Linear Regression Ensemble

1.1090

0.8683

LightGBM

1.6145

1.2962

LSTM

1.6199

1.3582

Weighted Ensemble

1.6726

1.4041

Simple Ensemble

1.7103

1.4375

SVR

2.2719

1.8815


Export to Sheets
Çıkarımlar
Lineer Regresyon Ensemble (Stacking) yöntemi, tüm diğer yöntemlerden daha düşük RMSE (1.1090) ve MAE (0.8683) değerleri ile en iyi sonucu vermiştir.

Tek başına kullanılan modeller (LightGBM, LSTM, SVR) arasında LightGBM en düşük hata değerlerine sahiptir.

Ağırlıklı ensemble (doğrulama setindeki hatalara göre ağırlıklandırma), basit ortalama ensemble’dan daha iyi sonuç verirken, Lineer Regresyon Ensemble yönteminden daha başarılı değildir.

SVR modeli tek başına en yüksek hata değerlerine sahiptir.

Ensemble yöntemlerinin tamamı, en kötü temel modelden (SVR) daha iyi performans göstermiştir.

Sonuçlar, makalede de vurgulandığı gibi, farklı tahmin modellerinin birleştirilmesinin (özellikle uygun ağırlıklandırma ve lineer regresyon ile) tahmin doğruluğunu önemli ölçüde artırdığını göstermektedir.

Grafik: Gerçek Değerler vs. Linear Regression Ensemble Tahminleri
Python

import matplotlib.pyplot as plt

# Bu kodun çalışması için y_test_real ve stacking_pred değişkenlerinin tanımlı olması gerekir.
plt.figure(figsize=(12, 6))
plt.plot(y_test_real, label='Gerçek Değerler')
plt.plot(stacking_pred, label='Linear Regression Ensemble Tahmini') # Grafikteki 'lgbm_pred_real' 'stacking_pred' olarak düzeltildi
plt.legend()
plt.title('Linear Regression Ensemble vs Gerçek Değerler')
plt.xlabel('Zaman')
plt.ylabel('Hisse Fiyatı')
plt.show()
Model Performans Metrikleri
Aşağıdaki kod parçacıkları, hata metriklerinin nasıl hesaplandığını göstermektedir.

Python

# Hata hesaplama fonksiyonu (Örnek)
def calculate_errors(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae

# Test seti için hataların hesaplanması
rmse_lgbm_test, mae_lgbm_test = calculate_errors(y_test_real, lgbm_test_pred_real)
rmse_svr_test, mae_svr_test = calculate_errors(y_test_real, svr_test_pred_real)
rmse_lstm_test, mae_lstm_test = calculate_errors(y_test_real, lstm_test_pred_real)
rmse_simple_ensemble, mae_simple_ensemble = calculate_errors(y_test_real, simple_ensemble)
rmse_weighted_ensemble, mae_weighted_ensemble = calculate_errors(y_test_real, ensemble_pred)
rmse_stacking_ensemble, mae_stacking_ensemble = calculate_errors(y_test_real, stacking_pred)
Aşağıdaki tablo, modellerin doğrulama (validation) ve test setlerindeki RMSE değerlerini karşılaştırmaktadır.
| Model | Validation RMSE | Test RMSE |
| :--- | :--- | :--- |
| LightGBM | 0.015323 | 1.614474 |
| SVR | 0.020868 | 2.271901 |
| LSTM | 0.016755 | 1.619854 |

5. Sonuç
Bu projede, hisse senedi fiyat tahmini için farklı makine öğrenmesi modelleri ve ensemble yöntemleri karşılaştırılmıştır. Sonuçlar, ensemble yöntemlerinin tekil modellere göre daha iyi performans gösterebileceğini, özellikle de lineer regresyon ile yapılan ensemble (stacking) yönteminin hata oranlarını önemli ölçüde düşürdüğünü ortaya koymuştur. Bu çalışma, finansal zaman serisi tahminlerinde model çeşitliliği ve birleşik model kullanımının önemini vurgulamaktadır. Projede vardığımız nokta, gelişmiş ensemble yöntemlerinin hem tekil modellerden hem de basit ensemble yöntemlerinden daha iyi performans gösterdiği ve bu bulguların referans alınan makaledeki tezi kanıtlar nitelikte olduğudur.

6. Kaynakça
Veri Seti: Kaggle - Toyota Motors Stock Data (1980-2024)

Referans Makale: J. Zhao, A. Takai and E. Kita, "Weight-Training Ensemble Model for Stock Price Forecast," 2022 IEEE International Conference on Data Mining Workshops (ICDMW), Orlando, FL, USA, 2022, pp. 1-6, doi: 10.1109/ICDMW58026.2022.00024.
