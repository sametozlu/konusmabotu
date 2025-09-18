"""
Akıllı Müşteri Hizmetleri Botu - Modüler Yapı
==============================================

Bu modül, NLP teknolojilerini kullanarak müşteri mesajlarını analiz eder ve
otomatik yanıtlar üretir. Kod, mantıklı fonksiyonlara ayrılmış ve her fonksiyon
belirli bir görevi yerine getirir.

Ana Fonksiyonlar:
- Metin Ön İşleme: Kullanıcı mesajlarını temizler ve hazırlar
- Intent Tanıma: Mesajın amacını belirler
- Duygu Analizi: Müşterinin duygusal durumunu analiz eder
- Yanıt Üretme: Uygun yanıtları oluşturur
- Konfigürasyon Yönetimi: Ayarları yönetir

Özellikler:
- Intent Recognition: Müşteri mesajının amacını belirler
- Sentiment Analysis: Müşterinin duygusal durumunu analiz eder
- Otomatik Yanıt: Uygun yanıtları üretir
- Türkçe dil desteği
- Modüler ve genişletilebilir yapı

Yazar: AI Assistant
Tarih: 2024
"""

import json
import yaml
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import re
from datetime import datetime

# NLP kütüphaneleri
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from textblob import TextBlob
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Loglama ayarları
from loguru import logger

# =============================================================================
# YARDIMCI FONKSİYONLAR - Metin İşleme ve Analiz
# =============================================================================

def metin_temizle(ham_metin: str) -> str:
    """
    Kullanıcı mesajını temizler ve analiz için hazırlar.
    
    Bu fonksiyon:
    - Metni küçük harfe çevirir
    - Özel karakterleri temizler
    - Fazla boşlukları kaldırır
    - Türkçe karakterleri korur
    
    Args:
        ham_metin (str): Temizlenecek ham metin
        
    Returns:
        str: Temizlenmiş metin
        
    Örnek:
        >>> metin_temizle("Merhaba! Nasılsın?")
        "merhaba nasılsın"
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
    Bot konfigürasyon dosyasını yükler ve doğrular.
    
    Args:
        dosya_yolu (str): Konfigürasyon dosyasının yolu
        
    Returns:
        Dict[str, Any]: Konfigürasyon verileri
        
    Hata Durumları:
        - Dosya bulunamazsa: Boş dict döner
        - Geçersiz format: Boş dict döner
    """
    try:
        with open(dosya_yolu, 'r', encoding='utf-8') as dosya:
            konfig = yaml.safe_load(dosya)
        
        # Temel konfigürasyon kontrolü
        if not isinstance(konfig, dict):
            logger.warning("Konfigürasyon dosyası geçersiz format")
            return {}
        
        logger.info(f"Konfigürasyon başarıyla yüklendi: {dosya_yolu}")
        return konfig
        
    except FileNotFoundError:
        logger.error(f"Konfigürasyon dosyası bulunamadı: {dosya_yolu}")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Konfigürasyon dosyası parse hatası: {e}")
        return {}
    except Exception as e:
        logger.error(f"Konfigürasyon yükleme hatası: {e}")
        return {}


def intent_verilerini_yukle(dosya_yolu: str) -> Dict[str, Any]:
    """
    Intent eğitim verilerini yükler ve doğrular.
    
    Args:
        dosya_yolu (str): Intent veri dosyasının yolu
        
    Returns:
        Dict[str, Any]: Intent verileri
        
    Hata Durumları:
        - Dosya bulunamazsa: Boş intent listesi döner
        - Geçersiz JSON: Boş intent listesi döner
    """
    try:
        with open(dosya_yolu, 'r', encoding='utf-8') as dosya:
            veriler = json.load(dosya)
        
        # Intent verilerini doğrula
        if 'intents' not in veriler:
            logger.warning("Intent verilerinde 'intents' anahtarı bulunamadı")
            return {"intents": []}
        
        if not isinstance(veriler['intents'], list):
            logger.warning("Intent verileri liste formatında değil")
            return {"intents": []}
        
        # Her intent için gerekli alanları kontrol et
        gecerli_intentler = []
        for intent in veriler['intents']:
            if all(anahtar in intent for anahtar in ['tag', 'patterns', 'responses']):
                gecerli_intentler.append(intent)
            else:
                logger.warning(f"Geçersiz intent formatı: {intent.get('tag', 'bilinmeyen')}")
        
        logger.info(f"Intent verileri yüklendi: {len(gecerli_intentler)} intent")
        return {"intents": gecerli_intentler}
        
    except FileNotFoundError:
        logger.error(f"Intent veri dosyası bulunamadı: {dosya_yolu}")
        return {"intents": []}
    except json.JSONDecodeError as e:
        logger.error(f"Intent veri dosyası JSON hatası: {e}")
        return {"intents": []}
    except Exception as e:
        logger.error(f"Intent verileri yükleme hatası: {e}")
        return {"intents": []}


def sentiment_modeli_baslat(model_adi: str) -> Optional[Any]:
    """
    Sentiment analysis modelini başlatır ve yükler.
    
    Args:
        model_adi (str): Kullanılacak model adı
        
    Returns:
        Optional[Any]: Yüklenen model veya None (hata durumunda)
        
    Hata Durumları:
        - Model indirilemezse: None döner
        - Bellek yetersizse: None döner
    """
    try:
        logger.info(f"Sentiment modeli yükleniyor: {model_adi}")
        
        model = pipeline(
            "sentiment-analysis",
            model=model_adi,
            tokenizer=model_adi,
            return_all_scores=True
        )
        
        logger.info("Sentiment modeli başarıyla yüklendi")
        return model
        
    except Exception as e:
        logger.error(f"Sentiment modeli yükleme hatası: {e}")
        logger.info("Fallback sentiment analysis kullanılacak")
        return None


def intent_vektorleştirici_hazirla(intent_verileri: List[Dict]) -> Tuple[Optional[Any], Optional[Any], List[str]]:
    """
    Intent recognition için TF-IDF vektörleştirici hazırlar.
    
    Args:
        intent_verileri (List[Dict]): Intent eğitim verileri
        
    Returns:
        Tuple[Optional[Any], Optional[Any], List[str]]: 
        - Vektörleştirici
        - Vektörler
        - Intent etiketleri
        
    Hata Durumları:
        - Veri yoksa: (None, None, []) döner
        - Vektörleştirme hatası: (None, None, []) döner
    """
    try:
        if not intent_verileri:
            logger.warning("Intent verileri boş, vektörleştirici hazırlanamıyor")
            return None, None, []
        
        # Tüm pattern'leri topla
        tum_patternler = []
        intent_etiketleri = []
        
        for intent in intent_verileri:
            for pattern in intent.get('patterns', []):
                temiz_pattern = metin_temizle(pattern)
                if temiz_pattern:  # Boş olmayan pattern'leri ekle
                    tum_patternler.append(temiz_pattern)
                    intent_etiketleri.append(intent['tag'])
        
        if not tum_patternler:
            logger.warning("Geçerli pattern bulunamadı")
            return None, None, []
        
        # TF-IDF vektörleştirici oluştur
        vektörleştirici = TfidfVectorizer(
            max_features=1000,
            stop_words=None,  # Türkçe stop words için özel işlem gerekebilir
            ngram_range=(1, 2),
            min_df=1,  # En az 1 dokümanda geçmeli
            max_df=0.95  # En fazla %95 dokümanda geçmeli
        )
        
        # Vektörleri oluştur
        intent_vektörleri = vektörleştirici.fit_transform(tum_patternler)
        
        logger.info(f"Intent vektörleştirici hazırlandı: {len(tum_patternler)} pattern")
        return vektörleştirici, intent_vektörleri, intent_etiketleri
        
    except Exception as e:
        logger.error(f"Intent vektörleştirici hazırlama hatası: {e}")
        return None, None, []


# =============================================================================
# ANA BOT SINIFI - Modüler Yapı
# =============================================================================

class CustomerServiceBot:
    """
    Akıllı Müşteri Hizmetleri Botu Ana Sınıfı
    
    Bu sınıf, müşteri mesajlarını analiz ederek uygun yanıtlar üretir.
    NLP teknolojilerini kullanarak intent recognition ve sentiment analysis yapar.
    
    Modüler yapı sayesinde her fonksiyon belirli bir görevi yerine getirir:
    - Metin işleme
    - Intent tanıma
    - Duygu analizi
    - Yanıt üretme
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Bot'u başlatır ve gerekli modelleri yükler.
        
        Args:
            config_path (str): Konfigürasyon dosyasının yolu
            
        Başlatma Süreci:
        1. Konfigürasyon yüklenir
        2. Intent verileri yüklenir
        3. NLP modelleri başlatılır
        4. Vektörleştirici hazırlanır
        """
        logger.info("Customer Service Bot başlatılıyor...")
        
        # Konfigürasyon ve veri yükleme
        self.config = konfigurasyon_yukle(config_path)
        self.intent_data = intent_verilerini_yukle(
            self.config.get('nlp', {}).get('intent_data_path', 'data/intent_training_data.json')
        )
        
        # NLP modelleri
        self.sentiment_analyzer = None
        self.vectorizer = None
        self.intent_vectors = None
        self.intent_labels = []
        
        # Modelleri başlat
        self._modelleri_baslat()
        
        # Intent recognition hazırla
        self._intent_tanima_hazirla()
        
        logger.info("Customer Service Bot başarıyla başlatıldı")
    
    def _modelleri_baslat(self):
        """
        NLP modellerini başlatır ve yükler.
        
        Bu fonksiyon:
        1. Sentiment analysis modelini yükler
        2. Hata durumunda fallback mekanizması devreye girer
        3. Model durumunu loglar
        """
        try:
            model_adi = self.config.get('nlp', {}).get('sentiment_model', 'cardiffnlp/twitter-xlm-roberta-base-sentiment')
            self.sentiment_analyzer = sentiment_modeli_baslat(model_adi)
            
            if self.sentiment_analyzer:
                logger.info("Sentiment analysis modeli başarıyla yüklendi")
            else:
                logger.warning("Sentiment modeli yüklenemedi, fallback kullanılacak")
                
        except Exception as e:
            logger.error(f"Model başlatma hatası: {e}")
            self.sentiment_analyzer = None
    
    def _intent_tanima_hazirla(self):
        """
        Intent recognition için gerekli bileşenleri hazırlar.
        
        Bu fonksiyon:
        1. Intent verilerini alır
        2. Vektörleştirici hazırlar
        3. Vektörleri oluşturur
        4. Etiketleri saklar
        """
        try:
            intent_verileri = self.intent_data.get('intents', [])
            
            if not intent_verileri:
                logger.warning("Intent verileri boş, intent recognition hazırlanamıyor")
                return
            
            self.vectorizer, self.intent_vectors, self.intent_labels = intent_vektorleştirici_hazirla(intent_verileri)
            
            if self.vectorizer is not None:
                logger.info("Intent recognition başarıyla hazırlandı")
            else:
                logger.warning("Intent recognition hazırlanamadı")
                
        except Exception as e:
            logger.error(f"Intent recognition hazırlama hatası: {e}")
            self.vectorizer = None
            self.intent_vectors = None
            self.intent_labels = []
    
    def duygu_analizi_yap(self, metin: str) -> Dict[str, Any]:
        """
        Metnin duygusal analizini yapar ve sonuçları döndürür.
        
        Bu fonksiyon:
        1. Metni temizler
        2. Sentiment modeli ile analiz eder
        3. Fallback mekanizması kullanır
        4. Sonuçları standart formatta döndürür
        
        Args:
            metin (str): Analiz edilecek metin
            
        Returns:
            Dict[str, Any]: Sentiment analiz sonuçları
            {
                'label': 'POSITIVE/NEGATIVE/NEUTRAL',
                'score': 0.0-1.0 arası güven skoru,
                'confidence': 'high/medium/low'
            }
            
        Örnek:
            >>> bot.duygu_analizi_yap("Çok memnunum!")
            {'label': 'POSITIVE', 'score': 0.85, 'confidence': 'high'}
        """
        try:
            # Metni temizle
            temiz_metin = metin_temizle(metin)
            
            if not temiz_metin:
                return self._varsayilan_sentiment_sonucu()
            
            # Ana sentiment modeli ile analiz
            if self.sentiment_analyzer:
                return self._gelişmiş_sentiment_analizi(temiz_metin)
            else:
                return self._basit_sentiment_analizi(temiz_metin)
                
        except Exception as e:
            logger.error(f"Sentiment analysis hatası: {e}")
            return self._varsayilan_sentiment_sonucu()
    
    def _gelişmiş_sentiment_analizi(self, metin: str) -> Dict[str, Any]:
        """
        Gelişmiş sentiment modeli ile analiz yapar.
        
        Args:
            metin (str): Temizlenmiş metin
            
        Returns:
            Dict[str, Any]: Sentiment analiz sonuçları
        """
        try:
            # Transformers modeli ile analiz
            sonuclar = self.sentiment_analyzer(metin)
            
            # En yüksek skorlu sentiment'i bul
            en_iyi_sentiment = max(sonuclar[0], key=lambda x: x['score'])
            
            # Güven seviyesini belirle
            skor = en_iyi_sentiment['score']
            if skor > 0.8:
                güven = 'high'
            elif skor > 0.6:
                güven = 'medium'
            else:
                güven = 'low'
            
            return {
                'label': en_iyi_sentiment['label'],
                'score': skor,
                'confidence': güven
            }
            
        except Exception as e:
            logger.error(f"Gelişmiş sentiment analizi hatası: {e}")
            return self._basit_sentiment_analizi(metin)
    
    def _basit_sentiment_analizi(self, metin: str) -> Dict[str, Any]:
        """
        TextBlob ile basit sentiment analizi yapar.
        
        Args:
            metin (str): Temizlenmiş metin
            
        Returns:
            Dict[str, Any]: Sentiment analiz sonuçları
        """
        try:
            blob = TextBlob(metin)
            polarity = blob.sentiment.polarity
            
            # Sentiment etiketini belirle
            if polarity > 0.1:
                etiket = 'POSITIVE'
            elif polarity < -0.1:
                etiket = 'NEGATIVE'
            else:
                etiket = 'NEUTRAL'
            
            return {
                'label': etiket,
                'score': abs(polarity),
                'confidence': 'medium'
            }
            
        except Exception as e:
            logger.error(f"Basit sentiment analizi hatası: {e}")
            return self._varsayilan_sentiment_sonucu()
    
    def _varsayilan_sentiment_sonucu(self) -> Dict[str, Any]:
        """
        Hata durumunda varsayılan sentiment sonucu döndürür.
        
        Returns:
            Dict[str, Any]: Varsayılan sentiment sonuçları
        """
        return {
            'label': 'NEUTRAL',
            'score': 0.5,
            'confidence': 'low'
        }
    
    def intent_tani(self, metin: str) -> Tuple[str, float]:
        """
        Metnin intent'ini (amacını) belirler ve güven skorunu döndürür.
        
        Bu fonksiyon:
        1. Metni temizler ve ön işler
        2. TF-IDF vektörüne çevirir
        3. Benzerlik skorlarını hesaplar
        4. En yüksek skorlu intent'i bulur
        5. Güven eşiğini kontrol eder
        
        Args:
            metin (str): Analiz edilecek metin
            
        Returns:
            Tuple[str, float]: (intent_etiketi, güven_skoru)
            
        Örnek:
            >>> bot.intent_tani("merhaba")
            ('greeting', 0.95)
        """
        try:
            # Vektörleştirici kontrolü
            if not self._intent_sistemi_hazir_mi():
                return 'unknown', 0.0
            
            # Metni temizle ve ön işle
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
            if not self._güven_eşiğini_kontrol_et(en_iyi_skor):
                intent_etiketi = 'unknown'
            
            logger.info(f"Intent tanındı: {intent_etiketi} (güven: {en_iyi_skor:.3f})")
            
            return intent_etiketi, en_iyi_skor
            
        except Exception as e:
            logger.error(f"Intent recognition hatası: {e}")
            return 'unknown', 0.0
    
    def _intent_sistemi_hazir_mi(self) -> bool:
        """
        Intent recognition sisteminin hazır olup olmadığını kontrol eder.
        
        Returns:
            bool: Sistem hazırsa True, değilse False
        """
        return (self.vectorizer is not None and 
                self.intent_vectors is not None and 
                len(self.intent_labels) > 0)
    
    def _güven_eşiğini_kontrol_et(self, skor: float) -> bool:
        """
        Intent güven skorunun eşik değerini geçip geçmediğini kontrol eder.
        
        Args:
            skor (float): Intent güven skoru
            
        Returns:
            bool: Eşiği geçiyorsa True, geçmiyorsa False
        """
        try:
            eşik = self.config.get('nlp', {}).get('confidence_threshold', 0.7)
            return skor >= eşik
        except Exception:
            return skor >= 0.7  # Varsayılan eşik
    
    def mesaja_yanit_uret(self, kullanici_mesaji: str) -> Dict[str, Any]:
        """
        Kullanıcı mesajına uygun yanıt üretir ve analiz sonuçlarını döndürür.
        
        Bu fonksiyon:
        1. Kullanıcı mesajını analiz eder
        2. Intent ve sentiment analizi yapar
        3. Uygun yanıt metni üretir
        4. Tüm sonuçları standart formatta döndürür
        
        Args:
            kullanici_mesaji (str): Kullanıcının gönderdiği mesaj
            
        Returns:
            Dict[str, Any]: Yanıt ve analiz sonuçları
            {
                'text': 'Bot yanıtı',
                'intent': 'intent_etiketi',
                'intent_confidence': 0.0-1.0,
                'sentiment': {'label': 'POSITIVE/NEGATIVE/NEUTRAL', 'score': 0.0-1.0, 'confidence': 'high/medium/low'},
                'timestamp': '2024-01-01 12:00:00',
                'bot_name': 'Bot Adı'
            }
            
        Örnek:
            >>> bot.mesaja_yanit_uret("merhaba")
            {
                'text': 'Merhaba! Size nasıl yardımcı olabilirim?',
                'intent': 'greeting',
                'intent_confidence': 0.95,
                'sentiment': {'label': 'POSITIVE', 'score': 0.8, 'confidence': 'high'},
                'timestamp': '2024-01-01 12:00:00',
                'bot_name': 'Akıllı Müşteri Hizmetleri Asistanı'
            }
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
                'bot_name': self.config.get('bot', {}).get('name', 'Akıllı Müşteri Hizmetleri Asistanı')
            }
            
            logger.info(f"Yanıt üretildi - Intent: {intent_etiketi}, Sentiment: {sentiment_sonucu['label']}")
            
            return yanit
            
        except Exception as e:
            logger.error(f"Yanıt üretilirken hata: {e}")
            return self._hata_yaniti_uret("Bir hata oluştu, lütfen tekrar deneyin")
    
    def _hata_yaniti_uret(self, hata_mesaji: str) -> Dict[str, Any]:
        """
        Hata durumunda varsayılan yanıt üretir.
        
        Args:
            hata_mesaji (str): Gösterilecek hata mesajı
            
        Returns:
            Dict[str, Any]: Hata yanıtı
        """
        return {
            'text': hata_mesaji,
            'intent': 'error',
            'intent_confidence': 0.0,
            'sentiment': {'label': 'NEUTRAL', 'score': 0.5, 'confidence': 'low'},
            'timestamp': self._zaman_damgasi_al(),
            'bot_name': self.config.get('bot', {}).get('name', 'Akıllı Müşteri Hizmetleri Asistanı')
        }
    
    def _uygun_yaniti_bul(self, intent_etiketi: str, sentiment_sonucu: Dict[str, Any]) -> str:
        """
        Intent ve sentiment'e göre uygun yanıt metni bulur ve döndürür.
        
        Bu fonksiyon:
        1. Olumsuz sentiment için özel mesajlar kontrol eder
        2. Intent'e göre yanıt arar
        3. Fallback mekanizmaları kullanır
        4. Rastgele yanıt seçimi yapar
        
        Args:
            intent_etiketi (str): Tanınan intent etiketi
            sentiment_sonucu (Dict[str, Any]): Sentiment analiz sonuçları
            
        Returns:
            str: Uygun yanıt metni
        """
        import random
        
        try:
            # Olumsuz sentiment için özel mesaj kontrolü
            if self._olumsuz_sentiment_kontrolu(sentiment_sonucu):
                return self._olumsuz_sentiment_yaniti_al()
            
            # Intent'e göre yanıt bulma
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
    
    def _olumsuz_sentiment_kontrolu(self, sentiment_sonucu: Dict[str, Any]) -> bool:
        """
        Sentiment sonucunun olumsuz olup olmadığını kontrol eder.
        
        Args:
            sentiment_sonucu (Dict[str, Any]): Sentiment analiz sonuçları
            
        Returns:
            bool: Olumsuz sentiment ise True
        """
        return (sentiment_sonucu.get('label') == 'NEGATIVE' and 
                sentiment_sonucu.get('score', 0) > 0.6)
    
    def _olumsuz_sentiment_yaniti_al(self) -> str:
        """
        Olumsuz sentiment için özel yanıt döndürür.
        
        Returns:
            str: Olumsuz sentiment yanıtı
        """
        import random
        
        try:
            yanitlar = self.config.get('responses', {}).get('negative_sentiment', [
                "Anlıyorum, bu durum sizi rahatsız etmiş. Size en iyi şekilde yardımcı olmaya çalışacağım.",
                "Üzgünüm bu deneyimi yaşadığınız için. Sorununuzu çözmek için buradayım."
            ])
            return random.choice(yanitlar)
        except Exception:
            return "Anlıyorum, bu durum sizi rahatsız etmiş. Size yardımcı olmaya çalışacağım."
    
    def _intent_yaniti_bul(self, intent_etiketi: str) -> Optional[str]:
        """
        Intent etiketine göre yanıt bulur.
        
        Args:
            intent_etiketi (str): Intent etiketi
            
        Returns:
            Optional[str]: Bulunan yanıt veya None
        """
        import random
        
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
        
        Returns:
            str: Bilinmeyen intent yanıtı
        """
        import random
        
        try:
            yanitlar = self.config.get('responses', {}).get('unknown', [
                "Üzgünüm, sorunuzu tam olarak anlayamadım. Lütfen daha detaylı açıklayabilir misiniz?",
                "Bu konuda size yardımcı olmak için daha fazla bilgiye ihtiyacım var."
            ])
            return random.choice(yanitlar)
        except Exception:
            return "Üzgünüm, sorunuzu tam olarak anlayamadım. Lütfen daha detaylı açıklayabilir misiniz?"
    
    def _varsayilan_yanit_al(self) -> str:
        """
        Varsayılan yanıt döndürür.
        
        Returns:
            str: Varsayılan yanıt
        """
        return "Size nasıl yardımcı olabilirim?"
    
    def _zaman_damgasi_al(self) -> str:
        """
        Mevcut zamanı string formatında döndürür.
        
        Returns:
            str: Zaman damgası (YYYY-MM-DD HH:MM:SS)
        """
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def bot_bilgilerini_al(self) -> Dict[str, Any]:
        """
        Bot hakkında detaylı bilgileri döndürür.
        
        Returns:
            Dict[str, Any]: Bot bilgileri
            {
                'name': 'Bot Adı',
                'version': 'Versiyon',
                'language': 'Dil',
                'supported_intents': ['intent1', 'intent2'],
                'features': ['Özellik1', 'Özellik2']
            }
        """
        try:
            return {
                'name': self.config.get('bot', {}).get('name', 'Akıllı Müşteri Hizmetleri Asistanı'),
                'version': self.config.get('bot', {}).get('version', '1.0.0'),
                'language': self.config.get('bot', {}).get('language', 'tr'),
                'supported_intents': [intent.get('tag', '') for intent in self.intent_data.get('intents', [])],
                'features': ['Intent Recognition', 'Sentiment Analysis', 'Automatic Response', 'Turkish Language Support']
            }
        except Exception as e:
            logger.error(f"Bot bilgileri alma hatası: {e}")
            return {
                'name': 'Akıllı Müşteri Hizmetleri Asistanı',
                'version': '1.0.0',
                'language': 'tr',
                'supported_intents': [],
                'features': ['Intent Recognition', 'Sentiment Analysis', 'Automatic Response']
            }


    # =============================================================================
    # GERİYE UYUMLULUK - Eski Fonksiyon İsimleri
    # =============================================================================
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Geriye uyumluluk için eski fonksiyon adı.
        Yeni kullanım: duygu_analizi_yap()
        """
        return self.duygu_analizi_yap(text)
    
    def recognize_intent(self, text: str) -> Tuple[str, float]:
        """
        Geriye uyumluluk için eski fonksiyon adı.
        Yeni kullanım: intent_tani()
        """
        return self.intent_tani(text)
    
    def get_response(self, text: str) -> Dict[str, Any]:
        """
        Geriye uyumluluk için eski fonksiyon adı.
        Yeni kullanım: mesaja_yanit_uret()
        """
        return self.mesaja_yanit_uret(text)
    
    def get_bot_info(self) -> Dict[str, Any]:
        """
        Geriye uyumluluk için eski fonksiyon adı.
        Yeni kullanım: bot_bilgilerini_al()
        """
        return self.bot_bilgilerini_al()


# =============================================================================
# TEST FONKSİYONLARI - Modüler Test Sistemi
# =============================================================================

def bot_test_et():
    """
    Bot'u kapsamlı şekilde test eder ve sonuçları gösterir.
    
    Bu fonksiyon:
    1. Bot'u başlatır
    2. Çeşitli test senaryolarını çalıştırır
    3. Sonuçları detaylı şekilde gösterir
    4. Performans metriklerini hesaplar
    """
    print("🤖 Akıllı Müşteri Hizmetleri Botu - Kapsamlı Test")
    print("=" * 60)
    
    try:
        # Bot'u başlat
        print("🔄 Bot başlatılıyor...")
        bot = CustomerServiceBot()
        print("✅ Bot başarıyla başlatıldı!")
        
        # Bot bilgilerini göster
        bot_bilgileri = bot.bot_bilgilerini_al()
        print(f"📋 Bot Adı: {bot_bilgileri['name']}")
        print(f"🔢 Versiyon: {bot_bilgileri['version']}")
        print(f"🌍 Dil: {bot_bilgileri['language']}")
        print(f"🎯 Desteklenen Intent'ler: {len(bot_bilgileri['supported_intents'])}")
        print()
        
        # Test senaryoları
        test_senaryoları = [
            {
                "kategori": "Karşılama",
                "mesajlar": [
                    "merhaba",
                    "selam",
                    "iyi günler",
                    "günaydın"
                ]
            },
            {
                "kategori": "Ürün Bilgisi",
                "mesajlar": [
                    "hangi ürünler var",
                    "ürün bilgisi istiyorum",
                    "katalog",
                    "fiyat listesi"
                ]
            },
            {
                "kategori": "Sipariş Takibi",
                "mesajlar": [
                    "siparişim nerede",
                    "kargo takibi",
                    "sipariş durumu",
                    "ne zaman gelecek"
                ]
            },
            {
                "kategori": "İade",
                "mesajlar": [
                    "iade etmek istiyorum",
                    "para iadesi",
                    "geri ödeme",
                    "ürünü geri vermek"
                ]
            },
            {
                "kategori": "Teknik Destek",
                "mesajlar": [
                    "teknik destek",
                    "sorun yaşıyorum",
                    "çalışmıyor",
                    "hata alıyorum"
                ]
            },
            {
                "kategori": "Şikayet",
                "mesajlar": [
                    "şikayet",
                    "memnun değilim",
                    "kötü hizmet",
                    "rahatsızım"
                ]
            }
        ]
        
        # Test istatistikleri
        toplam_test = 0
        başarılı_test = 0
        intent_doğruluk = {}
        sentiment_doğruluk = {}
        
        # Her kategori için test çalıştır
        for senaryo in test_senaryoları:
            print(f"🎭 {senaryo['kategori']} Testleri")
            print("-" * 40)
            
            for mesaj in senaryo['mesajlar']:
                toplam_test += 1
                
                print(f"\n👤 Müşteri: {mesaj}")
                
                # Bot'tan yanıt al
                yanit = bot.mesaja_yanit_uret(mesaj)
                
                print(f"🤖 Bot: {yanit['text']}")
                print(f"   📊 Intent: {yanit['intent']} (Güven: {yanit['intent_confidence']:.2f})")
                print(f"   😊 Sentiment: {yanit['sentiment']['label']} (Skor: {yanit['sentiment']['score']:.2f})")
                
                # İstatistikleri güncelle
                intent = yanit['intent']
                sentiment = yanit['sentiment']['label']
                
                if intent in intent_doğruluk:
                    intent_doğruluk[intent] += 1
                else:
                    intent_doğruluk[intent] = 1
                
                if sentiment in sentiment_doğruluk:
                    sentiment_doğruluk[sentiment] += 1
                else:
                    sentiment_doğruluk[sentiment] = 1
                
                # Başarılı test sayısını güncelle (güven skoru > 0.5)
                if yanit['intent_confidence'] > 0.5:
                    başarılı_test += 1
                
                print("-" * 30)
        
        # Test sonuçlarını göster
        print("\n📊 Test Sonuçları")
        print("=" * 40)
        print(f"📈 Toplam Test: {toplam_test}")
        print(f"✅ Başarılı Test: {başarılı_test}")
        print(f"📊 Başarı Oranı: {(başarılı_test/toplam_test)*100:.1f}%")
        
        print(f"\n🎯 Intent Dağılımı:")
        for intent, sayı in sorted(intent_doğruluk.items(), key=lambda x: x[1], reverse=True):
            yüzde = (sayı / toplam_test) * 100
            print(f"   • {intent}: {sayı} ({yüzde:.1f}%)")
        
        print(f"\n😊 Sentiment Dağılımı:")
        for sentiment, sayı in sorted(sentiment_doğruluk.items(), key=lambda x: x[1], reverse=True):
            yüzde = (sayı / toplam_test) * 100
            print(f"   • {sentiment}: {sayı} ({yüzde:.1f}%)")
        
        if başarılı_test == toplam_test:
            print("\n🎉 Tüm testler başarıyla geçti!")
        else:
            print(f"\n⚠️ {toplam_test - başarılı_test} test başarısız oldu.")
        
    except Exception as e:
        print(f"❌ Test sırasında hata oluştu: {e}")


def hızlı_test():
    """
    Bot'un temel işlevlerini hızlıca test eder.
    """
    print("⚡ Hızlı Test - Temel İşlevler")
    print("=" * 40)
    
    try:
        bot = CustomerServiceBot()
        
        test_mesajları = [
            "merhaba",
            "ürün bilgisi istiyorum",
            "siparişim nerede",
            "iade etmek istiyorum"
        ]
        
        for mesaj in test_mesajları:
            print(f"\n👤 {mesaj}")
            yanit = bot.mesaja_yanit_uret(mesaj)
            print(f"🤖 {yanit['text']}")
            print(f"   Intent: {yanit['intent']}, Sentiment: {yanit['sentiment']['label']}")
        
        print("\n✅ Hızlı test tamamlandı!")
        
    except Exception as e:
        print(f"❌ Hızlı test hatası: {e}")


if __name__ == "__main__":
    # Kullanıcı seçimi
    print("🎮 Test Modu Seçin:")
    print("1. Kapsamlı Test (Tüm özellikler)")
    print("2. Hızlı Test (Temel işlevler)")
    print("3. Çıkış")
    
    try:
        seçim = input("\nSeçiminiz (1-3): ").strip()
        
        if seçim == '1':
            bot_test_et()
        elif seçim == '2':
            hızlı_test()
        elif seçim == '3':
            print("👋 Görüşürüz!")
        else:
            print("⚠️ Geçersiz seçim, kapsamlı test çalıştırılıyor...")
            bot_test_et()
            
    except KeyboardInterrupt:
        print("\n👋 Test kullanıcı tarafından sonlandırıldı.")
    except Exception as e:
        print(f"❌ Hata: {e}")
        print("Kapsamlı test çalıştırılıyor...")
        bot_test_et()
