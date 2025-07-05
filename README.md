1. Projenin Amacı
Bu projenin temel amacı, hisse senedi fiyat tahmini alanında farklı öğrenme modellerinin ve modellerin birleştirilmesiyle oluşan yöntemlerin performansını karşılaştırmaktır. Özellikle LightGBM, LSTM ve SVR gibi popüler algoritmaların tek başına ve çeşitli ensemble stratejileriyle (basit ortalama, error based weighting ve lineer regresyon ile) birleştirildiğinde tahmin doğruluğunda nasıl bir değişim olduğunu incelemek hedeflenmiştir. Proje kapsamında, Toyota Motor Corporation (TM) hissesinin geçmiş verileri kullanılarak, en iyi tahmin performansına ulaşmak için farklı model ve yöntemler test edilmiştir. Bu projede “Weight-Training Ensemble Model for Stock Price Forecast” adlı bilimsel makalenin metodolojisi uygulanmıştır.

2. Metodoloji
2.1. Veri Seti ve Preprocessing
    • Veri Kaynağı: Kaggle üzerinden alınan Toyota Motor Corporation (TM) hisse senedi verileri (2001-2019).
    • Features: Open, High, Low, Close, Volume.
    • Target: Close (Kapanış fiyatı).
    • Veri Bölümü:
        ◦ Training: 2001-2016 yılları arası
        ◦ Validation: 2017-2018 yılları arası
        ◦ Test: 2019 yılı
    • Data MinMaxScaler ile 0-1 aralığına çekildi.
    • Window yapısı: 60 günlük geçmiş veri ile bir sonraki günün kapanış fiyatı tahmin edildi. (Makalede net bir window aralığı belirtilmemiştir.)

2.2. Modeller ve Yöntemler
    • LightGBM
    • LSTM
    • SVR
    • Ensemble Yöntemleri:
        ◦ Basit Ortalama (Simple Averaging Ensemble): Üç modelin tahminlerinin aritmetik ortalaması.
        ◦ Error-based Ensemble: Validation setindeki hata oranlarına göre ağırlıklandırılmış birleşik model.
        ◦ Linear Regression Ensemble: Üç modelin tahminleriyle lineer regresyon modeli eğitilerek oluşturulan birleşik model.
2.3. Hata Metrikleri
    • RMSE (Root Mean Squared Error).
    • MAE (Mean Absolute Error)



4. Deneysel Sonuçlar

Model	RMSE	MAE
Linear Regression Ensemble	1.1090	0.8683
LightGBM	1.6145	1.2962
LSTM	1.6199	1.3582
Weighted Ensemble	1.6726	1.4041
Simple Ensemble	1.7103	1.4375
SVR	2.2719	1.8815

Çıkarımlar
    • Linear Regression Ensemble (Stacking) yöntemi, tüm diğer yöntemlerden daha düşük RMSE (1.1090) ve MAE (0.8683) ile en iyi sonucu vermiştir.
    • Tek başına kullanılan modeller (LightGBM, LSTM, SVR) arasında LightGBM en düşük hata değerlerine sahiptir.
    • Ağırlıklı ensemble (validation setindeki hatalara göre ağırlıklandırma), basit ortalama ensemble’dan daha iyi sonuç verirken, Linear Regression Ensemble yönteminden daha başarılı değildir.
    • SVR modeli tek başına en yüksek hata değerlerine sahiptir.
    • Ensemble yöntemlerinin tamamı, en kötü temel modelden (SVR) daha iyi performans göstermiştir.
    • Sonuçlar, makalede de vurgulandığı gibi, farklı tahmin modellerinin birleştirilmesinin (özellikle uygun ağırlıklandırma ve linear regression ile) tahmin doğruluğunu önemli ölçüde artırdığını göstermektedir.
Grafik:
plt.figure(figsize=(12,6))
plt.plot(y_test_real, label='Gerçek')
plt.plot(lgbm_pred_real, label='stacking_pred')
plt.legend()
plt.title('Linear Regression Ensemble vs Gerçek Değerler')
plt.show()



Model Performans Metrikleri

rmse_lgbm_test, mae_lgbm_test = calculate_errors(y_test_real, lgbm_test_pred_real)
rmse_svr_test, mae_svr_test = calculate_errors(y_test_real, svr_test_pred_real)
rmse_lstm_test, mae_lstm_test = calculate_errors(y_test_real, lstm_test_pred_real)
rmse_simple_ensemble, mae_simple_ensemble = calculate_errors(y_test_real, simple_ensemble)
rmse_weighted_ensemble, mae_weighted_ensemble = calculate_errors(y_test_real, ensemble_pred)
rmse_stacking_ensemble, mae_stacking_ensemble = calculate_errors(y_test_real, stacking_pred)

comparison_results = {
    'Model': ['LightGBM', 'SVR', 'LSTM'],
    'Validation RMSE': [rmse_lgbm, rmse_svr, rmse_lstm],
    'Test RMSE': [rmse_lgbm_test, rmse_svr_test, rmse_lstm_test]
}

comparison_df = pd.DataFrame(comparison_results)
print("\n--- Validation vs Test RMSE ---")
print(comparison_df)





Model	Validation RMSE	Test RMSE
LightGBM	0.015323	1.614474
SVR	0.020868	2.271901
LSTM	0.016755	1.619854

5. Sonuç
Bu projede, hisse senedi fiyat tahmini için farklı makine öğrenmesi modelleri ve ensemble yöntemleri karşılaştırılmıştır. Sonuçlar, ensemble yöntemlerinin tekil modellere göre daha iyi performans gösterebileceğini, özellikle de lineer regresyon ile yapılan ensemble’ın hata oranlarını önemli ölçüde düşürdüğünü ortaya koymuştur. Bu çalışma, finansal zaman serisi tahminlerinde model çeşitliliği ve birleşik model kullanımının önemini vurgulamaktadır. Bu projede vardığımız nokta gelişmiş ensemble’ın tekil modelden de basit bir ensemble’dan da daha iyi performans gösterdiği olmuştur. Çıktılarımız makaledeki bu tezi kanıtlar nitelikte olmuştur.



6. Kaynakça
    • Kaggle Toyota Stock Data: https://www.kaggle.com/datasets/mhassansaboor/toyota-motors-stock-data-2980-2024
    • J. Zhao, A. Takai and E. Kita, "Weight-Training Ensemble Model for Stock Price Forecast," 2022 IEEE International Conference on Data Mining Workshops (ICDMW), Orlando, FL, USA, 2022, pp. 1-6, doi: 10.1109/ICDMW58026.2022.00024.
