"""
Akıllı Müşteri Hizmetleri Botu - Modüler Test Sistemi
======================================================

Bu dosya, modüler hale getirilmiş bot'un tüm fonksiyonlarını test eder.
Her fonksiyon ayrı ayrı test edilir ve sonuçlar detaylı şekilde raporlanır.

Test Kategorileri:
- Temel Fonksiyon Testleri
- Intent Recognition Testleri
- Sentiment Analysis Testleri
- Yanıt Üretme Testleri
- Hata Yönetimi Testleri
- Performans Testleri
- Modüler Yapı Testleri

Yazar: AI Assistant
Tarih: 2024
"""

import unittest
import json
import time
from customer_service_bot import (
    CustomerServiceBot, 
    metin_temizle, 
    konfigurasyon_yukle, 
    intent_verilerini_yukle,
    sentiment_modeli_baslat,
    intent_vektorleştirici_hazirla
)
from loguru import logger

class TestYardimciFonksiyonlar(unittest.TestCase):
    """
    Yardımcı fonksiyonları test eden sınıf
    """
    
    def test_metin_temizle(self):
        """
        Metin temizleme fonksiyonunu test eder
        """
        # Normal metin
        self.assertEqual(metin_temizle("Merhaba! Nasılsın?"), "merhaba nasılsın")
        
        # Türkçe karakterler
        self.assertEqual(metin_temizle("Çok güzel!"), "çok güzel")
        
        # Fazla boşluklar
        self.assertEqual(metin_temizle("Merhaba    dünya"), "merhaba dünya")
        
        # Boş metin
        self.assertEqual(metin_temizle(""), "")
        
        # Özel karakterler
        self.assertEqual(metin_temizle("Test@#$%^&*()"), "test")
    
    def test_konfigurasyon_yukle(self):
        """
        Konfigürasyon yükleme fonksiyonunu test eder
        """
        # Geçerli dosya
        config = konfigurasyon_yukle("config.yaml")
        self.assertIsInstance(config, dict)
        
        # Geçersiz dosya
        config = konfigurasyon_yukle("olmayan_dosya.yaml")
        self.assertEqual(config, {})
    
    def test_intent_verilerini_yukle(self):
        """
        Intent verilerini yükleme fonksiyonunu test eder
        """
        # Geçerli dosya
        data = intent_verilerini_yukle("data/intent_training_data.json")
        self.assertIsInstance(data, dict)
        self.assertIn('intents', data)
        
        # Geçersiz dosya
        data = intent_verilerini_yukle("olmayan_dosya.json")
        self.assertEqual(data, {"intents": []})


class TestCustomerServiceBot(unittest.TestCase):
    """
    Customer Service Bot ana sınıfı için test sınıfı
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Test sınıfı başlatıldığında çalışır
        """
        try:
            cls.bot = CustomerServiceBot()
            logger.info("Test bot'u başlatıldı")
        except Exception as e:
            logger.error(f"Test bot'u başlatılamadı: {e}")
            cls.bot = None
    
    def test_bot_initialization(self):
        """
        Bot'un doğru şekilde başlatıldığını test eder
        """
        self.assertIsNotNone(self.bot, "Bot başlatılamadı")
        self.assertIsNotNone(self.bot.config, "Konfigürasyon yüklenemedi")
        self.assertIsNotNone(self.bot.intent_data, "Intent verileri yüklenemedi")
    
    def test_karşılama_intent(self):
        """
        Karşılama mesajlarını test eder (yeni fonksiyon isimleri ile)
        """
        if not self.bot:
            self.skipTest("Bot başlatılamadı")
        
        test_messages = [
            "merhaba",
            "selam",
            "iyi günler",
            "günaydın",
            "hello"
        ]
        
        for message in test_messages:
            with self.subTest(message=message):
                # Yeni fonksiyon ismi ile test
                response = self.bot.mesaja_yanit_uret(message)
                self.assertEqual(response['intent'], 'greeting')
                self.assertGreater(response['intent_confidence'], 0.5)
                self.assertIn('merhaba', response['text'].lower())
                
                # Geriye uyumluluk testi
                response_eski = self.bot.get_response(message)
                self.assertEqual(response['intent'], response_eski['intent'])
                self.assertEqual(response['text'], response_eski['text'])
    
    def test_product_info_intent(self):
        """
        Ürün bilgisi intent'ini test eder
        """
        if not self.bot:
            self.skipTest("Bot başlatılamadı")
        
        test_messages = [
            "hangi ürünler var",
            "ürün bilgisi istiyorum",
            "katalog",
            "fiyat listesi"
        ]
        
        for message in test_messages:
            with self.subTest(message=message):
                response = self.bot.get_response(message)
                self.assertEqual(response['intent'], 'product_info')
                self.assertGreater(response['intent_confidence'], 0.5)
    
    def test_order_status_intent(self):
        """
        Sipariş durumu intent'ini test eder
        """
        if not self.bot:
            self.skipTest("Bot başlatılamadı")
        
        test_messages = [
            "siparişim nerede",
            "kargo takibi",
            "sipariş durumu",
            "ne zaman gelecek"
        ]
        
        for message in test_messages:
            with self.subTest(message=message):
                response = self.bot.get_response(message)
                self.assertEqual(response['intent'], 'order_status')
                self.assertGreater(response['intent_confidence'], 0.5)
    
    def test_refund_intent(self):
        """
        İade intent'ini test eder
        """
        if not self.bot:
            self.skipTest("Bot başlatılamadı")
        
        test_messages = [
            "iade etmek istiyorum",
            "para iadesi",
            "geri ödeme",
            "ürünü geri vermek"
        ]
        
        for message in test_messages:
            with self.subTest(message=message):
                response = self.bot.get_response(message)
                self.assertEqual(response['intent'], 'refund')
                self.assertGreater(response['intent_confidence'], 0.5)
    
    def test_technical_support_intent(self):
        """
        Teknik destek intent'ini test eder
        """
        if not self.bot:
            self.skipTest("Bot başlatılamadı")
        
        test_messages = [
            "teknik destek",
            "sorun yaşıyorum",
            "çalışmıyor",
            "hata alıyorum"
        ]
        
        for message in test_messages:
            with self.subTest(message=message):
                response = self.bot.get_response(message)
                self.assertEqual(response['intent'], 'technical_support')
                self.assertGreater(response['intent_confidence'], 0.5)
    
    def test_complaint_intent(self):
        """
        Şikayet intent'ini test eder
        """
        if not self.bot:
            self.skipTest("Bot başlatılamadı")
        
        test_messages = [
            "şikayet",
            "memnun değilim",
            "kötü hizmet",
            "rahatsızım"
        ]
        
        for message in test_messages:
            with self.subTest(message=message):
                response = self.bot.get_response(message)
                self.assertEqual(response['intent'], 'complaint')
                self.assertGreater(response['intent_confidence'], 0.5)
    
    def test_pozitif_sentiment_analizi(self):
        """
        Pozitif sentiment analizini test eder (yeni fonksiyon isimleri ile)
        """
        if not self.bot:
            self.skipTest("Bot başlatılamadı")
        
        test_messages = [
            "çok memnunum",
            "harika hizmet",
            "teşekkürler",
            "mükemmel"
        ]
        
        for message in test_messages:
            with self.subTest(message=message):
                # Yeni fonksiyon ismi ile test
                sentiment = self.bot.duygu_analizi_yap(message)
                self.assertIn(sentiment['label'], ['POSITIVE', 'LABEL_2'])
                self.assertGreater(sentiment['score'], 0.3)
                
                # Geriye uyumluluk testi
                sentiment_eski = self.bot.analyze_sentiment(message)
                self.assertEqual(sentiment['label'], sentiment_eski['label'])
                self.assertEqual(sentiment['score'], sentiment_eski['score'])
    
    def test_sentiment_analysis_negative(self):
        """
        Negatif sentiment analizini test eder
        """
        if not self.bot:
            self.skipTest("Bot başlatılamadı")
        
        test_messages = [
            "çok kötü",
            "berbat hizmet",
            "sinirliyim",
            "memnun değilim"
        ]
        
        for message in test_messages:
            with self.subTest(message=message):
                sentiment = self.bot.analyze_sentiment(message)
                self.assertIn(sentiment['label'], ['NEGATIVE', 'LABEL_0'])
                self.assertGreater(sentiment['score'], 0.3)
    
    def test_unknown_intent(self):
        """
        Bilinmeyen intent'leri test eder
        """
        if not self.bot:
            self.skipTest("Bot başlatılamadı")
        
        test_messages = [
            "asdfghjkl",
            "123456789",
            "xyz abc def",
            "random text"
        ]
        
        for message in test_messages:
            with self.subTest(message=message):
                response = self.bot.get_response(message)
                self.assertEqual(response['intent'], 'unknown')
                self.assertLess(response['intent_confidence'], 0.7)
    
    def test_response_structure(self):
        """
        Yanıt yapısının doğru olduğunu test eder
        """
        if not self.bot:
            self.skipTest("Bot başlatılamadı")
        
        response = self.bot.get_response("merhaba")
        
        # Gerekli alanları kontrol et
        required_fields = ['text', 'intent', 'intent_confidence', 'sentiment', 'timestamp', 'bot_name']
        for field in required_fields:
            self.assertIn(field, response, f"Yanıtta {field} alanı eksik")
        
        # Veri tiplerini kontrol et
        self.assertIsInstance(response['text'], str)
        self.assertIsInstance(response['intent'], str)
        self.assertIsInstance(response['intent_confidence'], float)
        self.assertIsInstance(response['sentiment'], dict)
        self.assertIsInstance(response['timestamp'], str)
        self.assertIsInstance(response['bot_name'], str)
        
        # Sentiment yapısını kontrol et
        sentiment_fields = ['label', 'score', 'confidence']
        for field in sentiment_fields:
            self.assertIn(field, response['sentiment'], f"Sentiment'te {field} alanı eksik")
    
    def test_performance(self):
        """
        Bot'un performansını test eder
        """
        if not self.bot:
            self.skipTest("Bot başlatılamadı")
        
        test_message = "merhaba, ürün bilgisi istiyorum"
        
        # 10 mesaj için ortalama süreyi ölç
        times = []
        for _ in range(10):
            start_time = time.time()
            response = self.bot.get_response(test_message)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        
        # Ortalama yanıt süresi 5 saniyeden az olmalı
        self.assertLess(avg_time, 5.0, f"Ortalama yanıt süresi çok yüksek: {avg_time:.2f}s")
        
        logger.info(f"Ortalama yanıt süresi: {avg_time:.2f}s")
    
    def test_bot_info(self):
        """
        Bot bilgilerinin doğru olduğunu test eder
        """
        if not self.bot:
            self.skipTest("Bot başlatılamadı")
        
        bot_info = self.bot.get_bot_info()
        
        # Gerekli alanları kontrol et
        required_fields = ['name', 'version', 'language', 'supported_intents', 'features']
        for field in required_fields:
            self.assertIn(field, bot_info, f"Bot bilgilerinde {field} alanı eksik")
        
        # Veri tiplerini kontrol et
        self.assertIsInstance(bot_info['name'], str)
        self.assertIsInstance(bot_info['version'], str)
        self.assertIsInstance(bot_info['language'], str)
        self.assertIsInstance(bot_info['supported_intents'], list)
        self.assertIsInstance(bot_info['features'], list)
        
        # Desteklenen intent'lerin sayısını kontrol et
        self.assertGreater(len(bot_info['supported_intents']), 0)
        self.assertGreater(len(bot_info['features']), 0)


def modüler_fonksiyon_testleri():
    """
    Modüler fonksiyonları ayrı ayrı test eder
    """
    print("🔧 Modüler Fonksiyon Testleri")
    print("=" * 50)
    
    # Metin temizleme testleri
    print("📝 Metin Temizleme Testleri:")
    test_metinler = [
        ("Merhaba! Nasılsın?", "merhaba nasılsın"),
        ("Çok güzel!", "çok güzel"),
        ("Test@#$%^&*()", "test"),
        ("", "")
    ]
    
    for ham_metin, beklenen in test_metinler:
        sonuc = metin_temizle(ham_metin)
        if sonuc == beklenen:
            print(f"✅ '{ham_metin}' -> '{sonuc}'")
        else:
            print(f"❌ '{ham_metin}' -> '{sonuc}' (Beklenen: '{beklenen}')")
    
    print()
    
    # Konfigürasyon yükleme testleri
    print("⚙️ Konfigürasyon Yükleme Testleri:")
    config = konfigurasyon_yukle("config.yaml")
    if config:
        print("✅ Konfigürasyon başarıyla yüklendi")
        print(f"   Bot Adı: {config.get('bot', {}).get('name', 'Bilinmiyor')}")
    else:
        print("❌ Konfigürasyon yüklenemedi")
    
    print()
    
    # Intent verileri yükleme testleri
    print("🎯 Intent Verileri Yükleme Testleri:")
    intent_data = intent_verilerini_yukle("data/intent_training_data.json")
    if intent_data and intent_data.get('intents'):
        print(f"✅ {len(intent_data['intents'])} intent yüklendi")
        for intent in intent_data['intents'][:3]:  # İlk 3 intent'i göster
            print(f"   • {intent.get('tag', 'Bilinmiyor')}")
    else:
        print("❌ Intent verileri yüklenemedi")


def kapsamlı_modüler_test():
    """
    Modüler yapı ile kapsamlı test senaryolarını çalıştırır
    """
    print("🧪 Modüler Yapı - Kapsamlı Test")
    print("=" * 60)
    
    try:
        bot = CustomerServiceBot()
        print(f"✅ Bot başarıyla başlatıldı: {bot.bot_bilgilerini_al()['name']}")
        print()
        
        # Test senaryoları
        test_senaryoları = [
            {
                "kategori": "Karşılama",
                "mesajlar": ["merhaba", "selam", "iyi günler"],
                "beklenen_intent": "greeting"
            },
            {
                "kategori": "Ürün Bilgisi", 
                "mesajlar": ["hangi ürünler var", "ürün bilgisi istiyorum"],
                "beklenen_intent": "product_info"
            },
            {
                "kategori": "Sipariş Takibi",
                "mesajlar": ["siparişim nerede", "kargo takibi"],
                "beklenen_intent": "order_status"
            }
        ]
        
        toplam_test = 0
        başarılı_test = 0
        
        for senaryo in test_senaryoları:
            print(f"🎭 {senaryo['kategori']} Testleri")
            print("-" * 40)
            
            for mesaj in senaryo['mesajlar']:
                toplam_test += 1
                
                # Yeni modüler fonksiyonlar ile test
                yanit = bot.mesaja_yanit_uret(mesaj)
                intent = yanit['intent']
                sentiment = yanit['sentiment']
                
                print(f"👤 '{mesaj}'")
                print(f"🤖 {yanit['text']}")
                print(f"   📊 Intent: {intent} (Güven: {yanit['intent_confidence']:.2f})")
                print(f"   😊 Sentiment: {sentiment['label']} (Skor: {sentiment['score']:.2f})")
                
                # Başarı kontrolü
                if intent == senaryo['beklenen_intent'] and yanit['intent_confidence'] > 0.5:
                    print("   ✅ Başarılı")
                    başarılı_test += 1
                else:
                    print(f"   ❌ Başarısız (Beklenen: {senaryo['beklenen_intent']})")
                
                print("-" * 30)
        
        # Test sonuçları
        print("\n📊 Modüler Test Sonuçları")
        print("=" * 40)
        print(f"📈 Toplam Test: {toplam_test}")
        print(f"✅ Başarılı Test: {başarılı_test}")
        print(f"📊 Başarı Oranı: {(başarılı_test/toplam_test)*100:.1f}%")
        
        # Geriye uyumluluk testi
        print("\n🔄 Geriye Uyumluluk Testi")
        print("-" * 30)
        test_mesaj = "merhaba"
        yeni_yanit = bot.mesaja_yanit_uret(test_mesaj)
        eski_yanit = bot.get_response(test_mesaj)
        
        if (yeni_yanit['intent'] == eski_yanit['intent'] and 
            yeni_yanit['text'] == eski_yanit['text']):
            print("✅ Geriye uyumluluk korunuyor")
        else:
            print("❌ Geriye uyumluluk sorunu")
        
    except Exception as e:
        print(f"❌ Test sırasında hata oluştu: {e}")


if __name__ == "__main__":
    print("🎮 Modüler Test Sistemi")
    print("=" * 30)
    print("1. Unit Testleri (unittest)")
    print("2. Modüler Fonksiyon Testleri")
    print("3. Kapsamlı Modüler Test")
    print("4. Tüm Testleri Çalıştır")
    print("5. Çıkış")
    
    try:
        seçim = input("\nSeçiminiz (1-5): ").strip()
        
        if seçim == '1':
            print("\n🧪 Unit Testleri Çalıştırılıyor...")
            unittest.main(argv=[''], exit=False, verbosity=2)
            
        elif seçim == '2':
            modüler_fonksiyon_testleri()
            
        elif seçim == '3':
            kapsamlı_modüler_test()
            
        elif seçim == '4':
            print("\n🧪 Unit Testleri Çalıştırılıyor...")
            unittest.main(argv=[''], exit=False, verbosity=2)
            print("\n" + "="*60 + "\n")
            modüler_fonksiyon_testleri()
            print("\n" + "="*60 + "\n")
            kapsamlı_modüler_test()
            
        elif seçim == '5':
            print("👋 Görüşürüz!")
            
        else:
            print("⚠️ Geçersiz seçim, tüm testler çalıştırılıyor...")
            print("\n🧪 Unit Testleri Çalıştırılıyor...")
            unittest.main(argv=[''], exit=False, verbosity=2)
            print("\n" + "="*60 + "\n")
            modüler_fonksiyon_testleri()
            print("\n" + "="*60 + "\n")
            kapsamlı_modüler_test()
            
    except KeyboardInterrupt:
        print("\n👋 Test kullanıcı tarafından sonlandırıldı.")
    except Exception as e:
        print(f"❌ Hata: {e}")
        print("Tüm testler çalıştırılıyor...")
        unittest.main(argv=[''], exit=False, verbosity=2)
