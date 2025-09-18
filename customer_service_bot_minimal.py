"""
Akıllı Müşteri Hizmetleri Botu - Minimal Versiyon
=================================================

Bu versiyon, büyük ML modelleri olmadan çalışır.
Sadece temel NLP kütüphaneleri kullanır.

Özellikler:
- TextBlob ile sentiment analysis
- TF-IDF ile intent recognition
- Flask web arayüzü
- Türkçe dil desteği

Yazar: AI Assistant
Tarih: 2024
"""

import json
import yaml
import re
import random
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path

# Minimal NLP kütüphaneleri
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Loglama
from loguru import logger

# =============================================================================
# YARDIMCI FONKSİYONLAR - Minimal Versiyon
# =============================================================================

def metin_temizle(ham_metin: str) -> str:
    """
    Kullanıcı mesajını temizler ve analiz için hazırlar.
    
    Args:
        ham_metin (str): Temizlenecek ham metin
        
    Returns:
        str: Temizlenmiş metin
    """
    try:
        # Küçük harfe çevir
        temiz_metin = ham_metin.lower()
        
        # Özel karakterleri temizle (Türkçe karakterleri koru)
        temiz_metin = re.sub(r'[^\w\sçğıöşüÇĞIİÖŞÜ]', '', temiz_metin)
        
        # Fazla boşlukları temizle
        temiz_metin = re.sub(r'\s+', ' ', temiz_metin).strip()
        
        return temiz_metin
        
    except Exception as e:
        logger.error(f"Metin temizleme hatası: {e}")
        return ham_metin.lower() if ham_metin else ""


def konfigurasyon_yukle(dosya_yolu: str) -> Dict[str, Any]:
    """
    Bot konfigürasyon dosyasını yükler.
    
    Args:
        dosya_yolu (str): Konfigürasyon dosyasının yolu
        
    Returns:
        Dict[str, Any]: Konfigürasyon verileri
    """
    try:
        with open(dosya_yolu, 'r', encoding='utf-8') as dosya:
            konfig = yaml.safe_load(dosya)
        
        logger.info(f"Konfigürasyon başarıyla yüklendi: {dosya_yolu}")
        return konfig
        
    except FileNotFoundError:
        logger.error(f"Konfigürasyon dosyası bulunamadı: {dosya_yolu}")
        return {}
    except Exception as e:
        logger.error(f"Konfigürasyon yükleme hatası: {e}")
        return {}


def intent_verilerini_yukle(dosya_yolu: str) -> Dict[str, Any]:
    """
    Intent eğitim verilerini yükler.
    
    Args:
        dosya_yolu (str): Intent veri dosyasının yolu
        
    Returns:
        Dict[str, Any]: Intent verileri
    """
    try:
        with open(dosya_yolu, 'r', encoding='utf-8') as dosya:
            veriler = json.load(dosya)
        
        # Intent verilerini doğrula
        if 'intents' not in veriler:
            logger.warning("Intent verilerinde 'intents' anahtarı bulunamadı")
            return {"intents": []}
        
        logger.info(f"Intent verileri yüklendi: {len(veriler.get('intents', []))} intent")
        return veriler
        
    except FileNotFoundError:
        logger.error(f"Intent veri dosyası bulunamadı: {dosya_yolu}")
        return {"intents": []}
    except Exception as e:
        logger.error(f"Intent verileri yükleme hatası: {e}")
        return {"intents": []}


# =============================================================================
# ANA BOT SINIFI - Minimal Versiyon
# =============================================================================

class CustomerServiceBotMinimal:
    """
    Akıllı Müşteri Hizmetleri Botu - Minimal Versiyon
    
    Bu sınıf, büyük ML modelleri olmadan çalışır.
    TextBlob ve scikit-learn kullanarak temel NLP işlemleri yapar.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Bot'u başlatır.
        
        Args:
            config_path (str): Konfigürasyon dosyasının yolu
        """
        logger.info("Minimal Customer Service Bot başlatılıyor...")
        
        # Konfigürasyon ve veri yükleme
        self.config = konfigurasyon_yukle(config_path)
        self.intent_data = intent_verilerini_yukle(
            self.config.get('nlp', {}).get('intent_data_path', 'data/intent_training_data.json')
        )
        
        # NLP bileşenleri
        self.vectorizer = None
        self.intent_vectors = None
        self.intent_labels = []
        
        # Intent recognition hazırla
        self._intent_tanima_hazirla()
        
        logger.info("Minimal Customer Service Bot başarıyla başlatıldı")
    
    def _intent_tanima_hazirla(self):
        """
        Intent recognition için gerekli bileşenleri hazırlar.
        """
        try:
            intent_verileri = self.intent_data.get('intents', [])
            
            if not intent_verileri:
                logger.warning("Intent verileri boş, intent recognition hazırlanamıyor")
                return
            
            # Tüm pattern'leri topla
            tum_patternler = []
            intent_etiketleri = []
            
            for intent in intent_verileri:
                for pattern in intent.get('patterns', []):
                    temiz_pattern = metin_temizle(pattern)
                    if temiz_pattern:
                        tum_patternler.append(temiz_pattern)
                        intent_etiketleri.append(intent['tag'])
            
            if not tum_patternler:
                logger.warning("Geçerli pattern bulunamadı")
                return
            
            # TF-IDF vektörleştirici oluştur
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words=None,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            
            # Vektörleri oluştur
            self.intent_vectors = self.vectorizer.fit_transform(tum_patternler)
            self.intent_labels = intent_etiketleri
            
            logger.info(f"Intent recognition hazırlandı: {len(tum_patternler)} pattern")
            
        except Exception as e:
            logger.error(f"Intent recognition hazırlama hatası: {e}")
            self.vectorizer = None
            self.intent_vectors = None
            self.intent_labels = []
    
    def duygu_analizi_yap(self, metin: str) -> Dict[str, Any]:
        """
        TextBlob ile basit sentiment analizi yapar.
        
        Args:
            metin (str): Analiz edilecek metin
            
        Returns:
            Dict[str, Any]: Sentiment analiz sonuçları
        """
        try:
            # Metni temizle
            temiz_metin = metin_temizle(metin)
            
            if not temiz_metin:
                return self._varsayilan_sentiment_sonucu()
            
            # TextBlob ile analiz
            blob = TextBlob(temiz_metin)
            polarity = blob.sentiment.polarity
            
            # Sentiment etiketini belirle
            if polarity > 0.1:
                etiket = 'POSITIVE'
            elif polarity < -0.1:
                etiket = 'NEGATIVE'
            else:
                etiket = 'NEUTRAL'
            
            # Güven seviyesini belirle
            skor = abs(polarity)
            if skor > 0.5:
                güven = 'high'
            elif skor > 0.3:
                güven = 'medium'
            else:
                güven = 'low'
            
            return {
                'label': etiket,
                'score': skor,
                'confidence': güven
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis hatası: {e}")
            return self._varsayilan_sentiment_sonucu()
    
    def _varsayilan_sentiment_sonucu(self) -> Dict[str, Any]:
        """
        Hata durumunda varsayılan sentiment sonucu döndürür.
        """
        return {
            'label': 'NEUTRAL',
            'score': 0.5,
            'confidence': 'low'
        }
    
    def intent_tani(self, metin: str) -> Tuple[str, float]:
        """
        Metnin intent'ini belirler.
        
        Args:
            metin (str): Analiz edilecek metin
            
        Returns:
            Tuple[str, float]: (intent_etiketi, güven_skoru)
        """
        try:
            # Vektörleştirici kontrolü
            if not self._intent_sistemi_hazir_mi():
                return 'unknown', 0.0
            
            # Metni temizle
            temiz_metin = metin_temizle(metin)
            
            if not temiz_metin:
                return 'unknown', 0.0
            
            # TF-IDF vektörüne çevir
            metin_vektörü = self.vectorizer.transform([temiz_metin])
            
            # Benzerlik skorlarını hesapla
            benzerlik_skorları = cosine_similarity(metin_vektörü, self.intent_vectors).flatten()
            
            # En yüksek skorlu intent'i bul
            en_iyi_idx = np.argmax(benzerlik_skorları)
            en_iyi_skor = benzerlik_skorları[en_iyi_idx]
            
            intent_etiketi = self.intent_labels[en_iyi_idx]
            
            # Güven eşiğini kontrol et
            eşik = self.config.get('nlp', {}).get('confidence_threshold', 0.7)
            if en_iyi_skor < eşik:
                intent_etiketi = 'unknown'
            
            logger.info(f"Intent tanındı: {intent_etiketi} (güven: {en_iyi_skor:.3f})")
            
            return intent_etiketi, en_iyi_skor
            
        except Exception as e:
            logger.error(f"Intent recognition hatası: {e}")
            return 'unknown', 0.0
    
    def _intent_sistemi_hazir_mi(self) -> bool:
        """
        Intent recognition sisteminin hazır olup olmadığını kontrol eder.
        """
        return (self.vectorizer is not None and 
                self.intent_vectors is not None and 
                len(self.intent_labels) > 0)
    
    def mesaja_yanit_uret(self, kullanici_mesaji: str) -> Dict[str, Any]:
        """
        Kullanıcı mesajına uygun yanıt üretir.
        
        Args:
            kullanici_mesaji (str): Kullanıcının gönderdiği mesaj
            
        Returns:
            Dict[str, Any]: Yanıt ve analiz sonuçları
        """
        try:
            # Giriş kontrolü
            if not kullanici_mesaji or not kullanici_mesaji.strip():
                return self._hata_yaniti_uret("Mesaj boş olamaz")
            
            # Analiz işlemleri
            sentiment_sonucu = self.duygu_analizi_yap(kullanici_mesaji)
            intent_etiketi, intent_güveni = self.intent_tani(kullanici_mesaji)
            
            # Yanıt metni üret
            yanit_metni = self._uygun_yaniti_bul(intent_etiketi, sentiment_sonucu)
            
            # Yanıt objesi oluştur
            yanit = {
                'text': yanit_metni,
                'intent': intent_etiketi,
                'intent_confidence': intent_güveni,
                'sentiment': sentiment_sonucu,
                'timestamp': self._zaman_damgasi_al(),
                'bot_name': self.config.get('bot', {}).get('name', 'Minimal Müşteri Hizmetleri Asistanı')
            }
            
            logger.info(f"Yanıt üretildi - Intent: {intent_etiketi}, Sentiment: {sentiment_sonucu['label']}")
            
            return yanit
            
        except Exception as e:
            logger.error(f"Yanıt üretilirken hata: {e}")
            return self._hata_yaniti_uret("Bir hata oluştu, lütfen tekrar deneyin")
    
    def _hata_yaniti_uret(self, hata_mesaji: str) -> Dict[str, Any]:
        """
        Hata durumunda varsayılan yanıt üretir.
        """
        return {
            'text': hata_mesaji,
            'intent': 'error',
            'intent_confidence': 0.0,
            'sentiment': {'label': 'NEUTRAL', 'score': 0.5, 'confidence': 'low'},
            'timestamp': self._zaman_damgasi_al(),
            'bot_name': self.config.get('bot', {}).get('name', 'Minimal Müşteri Hizmetleri Asistanı')
        }
    
    def _uygun_yaniti_bul(self, intent_etiketi: str, sentiment_sonucu: Dict[str, Any]) -> str:
        """
        Intent ve sentiment'e göre uygun yanıt metni bulur.
        """
        try:
            # Olumsuz sentiment için özel mesaj
            if (sentiment_sonucu.get('label') == 'NEGATIVE' and 
                sentiment_sonucu.get('score', 0) > 0.6):
                return self._olumsuz_sentiment_yaniti_al()
            
            # Intent'e göre yanıt bul
            yanit = self._intent_yaniti_bul(intent_etiketi)
            if yanit:
                return yanit
            
            # Bilinmeyen intent için varsayılan yanıt
            if intent_etiketi == 'unknown':
                return self._bilinmeyen_intent_yaniti_al()
            
            # Fallback yanıt
            return self._varsayilan_yanit_al()
            
        except Exception as e:
            logger.error(f"Yanıt bulma hatası: {e}")
            return self._varsayilan_yanit_al()
    
    def _olumsuz_sentiment_yaniti_al(self) -> str:
        """
        Olumsuz sentiment için özel yanıt döndürür.
        """
        yanitlar = [
            "Anlıyorum, bu durum sizi rahatsız etmiş. Size en iyi şekilde yardımcı olmaya çalışacağım.",
            "Üzgünüm bu deneyimi yaşadığınız için. Sorununuzu çözmek için buradayım."
        ]
        return random.choice(yanitlar)
    
    def _intent_yaniti_bul(self, intent_etiketi: str) -> Optional[str]:
        """
        Intent etiketine göre yanıt bulur.
        """
        try:
            # Konfigürasyondan yanıt ara
            intent_yanitları = self.config.get('intents', {}).get(intent_etiketi, {}).get('responses', [])
            if intent_yanitları:
                return random.choice(intent_yanitları)
            
            # Intent verilerinden yanıt ara
            for intent_verisi in self.intent_data.get('intents', []):
                if intent_verisi.get('tag') == intent_etiketi:
                    yanitlar = intent_verisi.get('responses', [])
                    if yanitlar:
                        return random.choice(yanitlar)
            
            return None
            
        except Exception as e:
            logger.error(f"Intent yanıt bulma hatası: {e}")
            return None
    
    def _bilinmeyen_intent_yaniti_al(self) -> str:
        """
        Bilinmeyen intent için yanıt döndürür.
        """
        yanitlar = [
            "Üzgünüm, sorunuzu tam olarak anlayamadım. Lütfen daha detaylı açıklayabilir misiniz?",
            "Bu konuda size yardımcı olmak için daha fazla bilgiye ihtiyacım var."
        ]
        return random.choice(yanitlar)
    
    def _varsayilan_yanit_al(self) -> str:
        """
        Varsayılan yanıt döndürür.
        """
        return "Size nasıl yardımcı olabilirim?"
    
    def _zaman_damgasi_al(self) -> str:
        """
        Mevcut zamanı string formatında döndürür.
        """
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def bot_bilgilerini_al(self) -> Dict[str, Any]:
        """
        Bot hakkında detaylı bilgileri döndürür.
        """
        try:
            return {
                'name': self.config.get('bot', {}).get('name', 'Minimal Müşteri Hizmetleri Asistanı'),
                'version': self.config.get('bot', {}).get('version', '1.0.0-minimal'),
                'language': self.config.get('bot', {}).get('language', 'tr'),
                'supported_intents': [intent.get('tag', '') for intent in self.intent_data.get('intents', [])],
                'features': ['Intent Recognition', 'Sentiment Analysis (TextBlob)', 'Automatic Response', 'Turkish Language Support']
            }
        except Exception as e:
            logger.error(f"Bot bilgileri alma hatası: {e}")
            return {
                'name': 'Minimal Müşteri Hizmetleri Asistanı',
                'version': '1.0.0-minimal',
                'language': 'tr',
                'supported_intents': [],
                'features': ['Intent Recognition', 'Sentiment Analysis', 'Automatic Response']
            }
    
    # Geriye uyumluluk için eski fonksiyon isimleri
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        return self.duygu_analizi_yap(text)
    
    def recognize_intent(self, text: str) -> Tuple[str, float]:
        return self.intent_tani(text)
    
    def get_response(self, text: str) -> Dict[str, Any]:
        return self.mesaja_yanit_uret(text)
    
    def get_bot_info(self) -> Dict[str, Any]:
        return self.bot_bilgilerini_al()


# =============================================================================
# TEST FONKSİYONLARI - Minimal Versiyon
# =============================================================================

def minimal_bot_test():
    """
    Minimal bot'u test eder.
    """
    print("🤖 Minimal Customer Service Bot - Test")
    print("=" * 50)
    
    try:
        # Bot'u başlat
        print("🔄 Minimal bot başlatılıyor...")
        bot = CustomerServiceBotMinimal()
        print("✅ Minimal bot başarıyla başlatıldı!")
        
        # Bot bilgilerini göster
        bot_bilgileri = bot.bot_bilgilerini_al()
        print(f"📋 Bot Adı: {bot_bilgileri['name']}")
        print(f"🔢 Versiyon: {bot_bilgileri['version']}")
        print(f"🌍 Dil: {bot_bilgileri['language']}")
        print(f"🎯 Desteklenen Intent'ler: {len(bot_bilgileri['supported_intents'])}")
        print()
        
        # Test mesajları
        test_mesajları = [
            "merhaba",
            "ürün bilgisi istiyorum",
            "siparişim nerede",
            "iade etmek istiyorum",
            "çok memnunum",
            "berbat hizmet"
        ]
        
        for mesaj in test_mesajları:
            print(f"👤 Müşteri: {mesaj}")
            yanit = bot.mesaja_yanit_uret(mesaj)
            print(f"🤖 Bot: {yanit['text']}")
            print(f"   📊 Intent: {yanit['intent']} (Güven: {yanit['intent_confidence']:.2f})")
            print(f"   😊 Sentiment: {yanit['sentiment']['label']} (Skor: {yanit['sentiment']['score']:.2f})")
            print("-" * 40)
        
        print("✅ Minimal bot testi tamamlandı!")
        
    except Exception as e:
        print(f"❌ Test sırasında hata oluştu: {e}")


if __name__ == "__main__":
    minimal_bot_test()
