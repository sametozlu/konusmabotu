# 🚀 Kurulum Talimatları - Akıllı Müşteri Hizmetleri Botu

## 📋 Gereksinimler

- **Python**: 3.8+ (Python 3.13 önerilir)
- **İşletim Sistemi**: Windows, macOS, Linux
- **Bellek**: En az 4GB RAM (büyük modeller için 8GB+ önerilir)
- **Disk**: En az 2GB boş alan

## 🎯 Kurulum Seçenekleri

### Seçenek 1: Minimal Kurulum (Önerilen - Hızlı Başlangıç)

Bu seçenek, büyük ML modelleri olmadan çalışır ve hızlı kurulum sağlar.

```bash
# 1. Sanal ortam oluştur
python -m venv venv

# 2. Sanal ortamı aktifleştir
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Minimal bağımlılıkları yükle
pip install -r requirements-minimal.txt

# 4. NLTK verilerini indir
python -c "import nltk; nltk.download('punkt')"

# 5. Minimal bot'u test et
python customer_service_bot_minimal.py
```

### Seçenek 2: Tam Kurulum (Gelişmiş Özellikler)

Bu seçenek, tüm NLP modellerini içerir ancak daha fazla disk alanı ve bellek gerektirir.

```bash
# 1. Sanal ortam oluştur
python -m venv venv

# 2. Sanal ortamı aktifleştir
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Tam bağımlılıkları yükle
pip install -r requirements.txt

# 4. NLTK verilerini indir
python -c "import nltk; nltk.download('punkt')"

# 5. Tam bot'u test et
python customer_service_bot.py
```

### Seçenek 3: Pipenv ile Kurulum

```bash
# 1. Pipenv kurulumu (eğer yoksa)
pip install pipenv

# 2. Sanal ortam oluştur ve aktifleştir
pipenv shell

# 3. Bağımlılıkları yükle
pipenv install -r requirements-minimal.txt

# 4. Bot'u test et
python customer_service_bot_minimal.py
```

## 🔧 Sorun Giderme

### PyTorch Kurulum Sorunu

**Sorun**: `ERROR: Could not find a version that satisfies the requirement torch==2.1.0`

**Çözüm**: 
```bash
# Minimal versiyonu kullan
pip install -r requirements-minimal.txt

# Veya PyTorch'u manuel yükle
pip install torch>=2.6.0
```

### Transformers Kurulum Sorunu

**Sorun**: Transformers modeli indirilemiyor

**Çözüm**:
```bash
# Minimal versiyonu kullan (transformers olmadan)
python customer_service_bot_minimal.py
```

### Bellek Sorunu

**Sorun**: Model yüklenirken bellek hatası

**Çözüm**:
1. Minimal versiyonu kullan
2. Daha küçük model kullan
3. Sistem belleğini artır

### Python 3.13 Uyumluluk Sorunu

**Sorun**: Bazı paketler Python 3.13 ile uyumlu değil

**Çözüm**:
```bash
# Python 3.11 veya 3.12 kullan
# Veya minimal versiyonu kullan
pip install -r requirements-minimal.txt
```

## 🧪 Test Etme

### Minimal Bot Testi
```bash
python customer_service_bot_minimal.py
```

### Tam Bot Testi
```bash
python customer_service_bot.py
```

### Web Arayüzü Testi
```bash
python app.py
# Tarayıcıda http://localhost:5000 adresine git
```

### Kapsamlı Test
```bash
python test_bot.py
```

## 📊 Performans Karşılaştırması

| Özellik | Minimal Versiyon | Tam Versiyon |
|---------|------------------|--------------|
| **Kurulum Süresi** | ~2 dakika | ~10-15 dakika |
| **Disk Kullanımı** | ~500MB | ~3-5GB |
| **Bellek Kullanımı** | ~100MB | ~1-2GB |
| **Sentiment Doğruluğu** | %70-80 | %85-95 |
| **Intent Doğruluğu** | %80-85 | %85-90 |
| **Başlatma Süresi** | ~2 saniye | ~10-30 saniye |

## 🎮 Kullanım Örnekleri

### Temel Kullanım
```python
from customer_service_bot_minimal import CustomerServiceBotMinimal

# Bot'u başlat
bot = CustomerServiceBotMinimal()

# Mesaj gönder
yanit = bot.mesaja_yanit_uret("merhaba")
print(yanit['text'])
```

### Web Arayüzü
```bash
python app.py
# Tarayıcıda http://localhost:5000
```

### API Kullanımı
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "merhaba"}'
```

## 🔄 Güncelleme

### Minimal Versiyondan Tam Versiyona Geçiş
```bash
pip install -r requirements.txt
python customer_service_bot.py
```

### Tam Versiyondan Minimal Versiyona Geçiş
```bash
pip uninstall transformers torch spacy
pip install -r requirements-minimal.txt
python customer_service_bot_minimal.py
```

## 📞 Yardım

Sorun yaşıyorsanız:

1. **Minimal versiyonu deneyin** - En az sorun çıkarır
2. **Python versiyonunu kontrol edin** - 3.8+ gerekli
3. **Sanal ortam kullanın** - Çakışmaları önler
4. **Log dosyalarını kontrol edin** - Hata detayları için

## 🎉 Başarılı Kurulum Kontrolü

Kurulum başarılı ise şu çıktıyı görmelisiniz:

```
🤖 Minimal Customer Service Bot - Test
==================================================
🔄 Minimal bot başlatılıyor...
✅ Minimal bot başarıyla başlatıldı!
📋 Bot Adı: Akıllı Müşteri Hizmetleri Asistanı
🔢 Versiyon: 1.0.0-minimal
🌍 Dil: tr
🎯 Desteklenen Intent'ler: 7
```

Bu çıktıyı gördüyseniz, bot başarıyla kurulmuş ve çalışıyor demektir! 🎉
