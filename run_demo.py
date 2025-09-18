"""
Akıllı Müşteri Hizmetleri Botu - Demo Çalıştırıcı
=================================================

Bu dosya, bot'un tüm özelliklerini gösteren interaktif bir demo sağlar.
Kullanıcılar bu demo ile bot'un nasıl çalıştığını test edebilir.

Özellikler:
- Interaktif sohbet
- Real-time analiz sonuçları
- Performans metrikleri
- Örnek senaryolar

Yazar: AI Assistant
Tarih: 2024
"""

import time
import json
from datetime import datetime
from customer_service_bot import CustomerServiceBot
from loguru import logger

class BotDemo:
    """
    Bot demo sınıfı - Interaktif gösterim
    """
    
    def __init__(self):
        """
        Demo'yu başlatır
        """
        self.bot = None
        self.conversation_history = []
        self.stats = {
            'total_messages': 0,
            'intents': {},
            'sentiments': {},
            'start_time': datetime.now()
        }
        
        print("🤖 Akıllı Müşteri Hizmetleri Botu - Demo")
        print("=" * 50)
        
    def initialize_bot(self):
        """
        Bot'u başlatır
        """
        try:
            print("🔄 Bot başlatılıyor...")
            self.bot = CustomerServiceBot()
            print("✅ Bot başarıyla başlatıldı!")
            print(f"📋 Bot Adı: {self.bot.get_bot_info()['name']}")
            print(f"🔢 Versiyon: {self.bot.get_bot_info()['version']}")
            print(f"🌍 Dil: {self.bot.get_bot_info()['language']}")
            print(f"🎯 Desteklenen Intent'ler: {len(self.bot.get_bot_info()['supported_intents'])}")
            print()
            return True
        except Exception as e:
            print(f"❌ Bot başlatılamadı: {e}")
            return False
    
    def show_welcome(self):
        """
        Hoş geldin mesajını gösterir
        """
        print("🎉 Hoş Geldiniz!")
        print("Bu demo ile Akıllı Müşteri Hizmetleri Botu'nu test edebilirsiniz.")
        print()
        print("📝 Kullanım:")
        print("  • Mesajınızı yazın ve Enter'a basın")
        print("  • 'quit' yazarak çıkış yapın")
        print("  • 'help' yazarak yardım alın")
        print("  • 'stats' yazarak istatistikleri görün")
        print("  • 'examples' yazarak örnek mesajları görün")
        print()
        print("💡 Örnek mesajlar:")
        print("  • 'merhaba' - Karşılama")
        print("  • 'ürün bilgisi istiyorum' - Ürün bilgisi")
        print("  • 'siparişim nerede' - Sipariş takibi")
        print("  • 'iade etmek istiyorum' - İade işlemi")
        print("  • 'sorun yaşıyorum' - Teknik destek")
        print("  • 'memnun değilim' - Şikayet")
        print()
        print("-" * 50)
    
    def show_examples(self):
        """
        Örnek mesajları gösterir
        """
        examples = {
            "Karşılama": [
                "merhaba",
                "selam",
                "iyi günler",
                "günaydın"
            ],
            "Ürün Bilgisi": [
                "hangi ürünler var",
                "ürün bilgisi istiyorum",
                "katalog",
                "fiyat listesi"
            ],
            "Sipariş Takibi": [
                "siparişim nerede",
                "kargo takibi",
                "sipariş durumu",
                "ne zaman gelecek"
            ],
            "İade": [
                "iade etmek istiyorum",
                "para iadesi",
                "geri ödeme",
                "ürünü geri vermek"
            ],
            "Teknik Destek": [
                "teknik destek",
                "sorun yaşıyorum",
                "çalışmıyor",
                "hata alıyorum"
            ],
            "Şikayet": [
                "şikayet",
                "memnun değilim",
                "kötü hizmet",
                "rahatsızım"
            ]
        }
        
        print("📚 Örnek Mesajlar:")
        print("-" * 30)
        
        for category, messages in examples.items():
            print(f"\n🔹 {category}:")
            for message in messages:
                print(f"   • {message}")
    
    def show_stats(self):
        """
        İstatistikleri gösterir
        """
        print("📊 Konuşma İstatistikleri:")
        print("-" * 30)
        print(f"💬 Toplam Mesaj: {self.stats['total_messages']}")
        
        # Konuşma süresi
        duration = datetime.now() - self.stats['start_time']
        minutes = int(duration.total_seconds() / 60)
        seconds = int(duration.total_seconds() % 60)
        print(f"⏱️  Konuşma Süresi: {minutes}dk {seconds}sn")
        
        # Intent istatistikleri
        if self.stats['intents']:
            print(f"\n🎯 Intent Dağılımı:")
            for intent, count in sorted(self.stats['intents'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / self.stats['total_messages']) * 100
                print(f"   • {intent}: {count} ({percentage:.1f}%)")
        
        # Sentiment istatistikleri
        if self.stats['sentiments']:
            print(f"\n😊 Sentiment Dağılımı:")
            for sentiment, count in sorted(self.stats['sentiments'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / self.stats['total_messages']) * 100
                print(f"   • {sentiment}: {count} ({percentage:.1f}%)")
    
    def process_message(self, message):
        """
        Mesajı işler ve yanıt üretir
        """
        if not self.bot:
            return "Bot başlatılamadı!"
        
        # Mesajı kaydet
        self.conversation_history.append({
            'user_message': message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Bot'tan yanıt al
        start_time = time.time()
        response = self.bot.get_response(message)
        end_time = time.time()
        
        # İstatistikleri güncelle
        self.stats['total_messages'] += 1
        
        # Intent istatistikleri
        intent = response['intent']
        if intent in self.stats['intents']:
            self.stats['intents'][intent] += 1
        else:
            self.stats['intents'][intent] = 1
        
        # Sentiment istatistikleri
        sentiment = response['sentiment']['label']
        if sentiment in self.stats['sentiments']:
            self.stats['sentiments'][sentiment] += 1
        else:
            self.stats['sentiments'][sentiment] = 1
        
        # Yanıt süresini hesapla
        response_time = end_time - start_time
        
        # Yanıtı formatla
        formatted_response = f"""
🤖 Bot: {response['text']}

📊 Analiz Sonuçları:
   🎯 Intent: {intent} (Güven: {response['intent_confidence']:.2f})
   😊 Sentiment: {sentiment} (Skor: {response['sentiment']['score']:.2f})
   ⚡ Yanıt Süresi: {response_time:.2f}s
   🕐 Zaman: {response['timestamp']}
"""
        
        return formatted_response
    
    def run_interactive_demo(self):
        """
        Interaktif demo'yu çalıştırır
        """
        if not self.initialize_bot():
            return
        
        self.show_welcome()
        
        while True:
            try:
                # Kullanıcı girişi al
                user_input = input("\n👤 Siz: ").strip()
                
                # Özel komutları kontrol et
                if user_input.lower() == 'quit':
                    print("\n👋 Görüşürüz! Demo sona erdi.")
                    self.show_stats()
                    break
                
                elif user_input.lower() == 'help':
                    self.show_welcome()
                    continue
                
                elif user_input.lower() == 'examples':
                    self.show_examples()
                    continue
                
                elif user_input.lower() == 'stats':
                    self.show_stats()
                    continue
                
                elif user_input.lower() == 'clear':
                    self.conversation_history = []
                    self.stats = {
                        'total_messages': 0,
                        'intents': {},
                        'sentiments': {},
                        'start_time': datetime.now()
                    }
                    print("🧹 Konuşma geçmişi temizlendi.")
                    continue
                
                elif not user_input:
                    print("⚠️  Lütfen bir mesaj yazın.")
                    continue
                
                # Mesajı işle
                response = self.process_message(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Demo kullanıcı tarafından sonlandırıldı.")
                self.show_stats()
                break
            except Exception as e:
                print(f"\n❌ Hata oluştu: {e}")
                continue
    
    def run_automated_demo(self):
        """
        Otomatik demo senaryolarını çalıştırır
        """
        if not self.initialize_bot():
            return
        
        print("🤖 Otomatik Demo Senaryoları")
        print("=" * 40)
        
        demo_scenarios = [
            {
                "name": "Karşılama Senaryosu",
                "messages": [
                    "merhaba",
                    "selam",
                    "iyi günler"
                ]
            },
            {
                "name": "Ürün Bilgisi Senaryosu",
                "messages": [
                    "hangi ürünler var",
                    "ürün bilgisi istiyorum",
                    "katalog"
                ]
            },
            {
                "name": "Sipariş Takibi Senaryosu",
                "messages": [
                    "siparişim nerede",
                    "kargo takibi",
                    "ne zaman gelecek"
                ]
            },
            {
                "name": "İade Senaryosu",
                "messages": [
                    "iade etmek istiyorum",
                    "para iadesi",
                    "geri ödeme"
                ]
            },
            {
                "name": "Teknik Destek Senaryosu",
                "messages": [
                    "teknik destek",
                    "sorun yaşıyorum",
                    "çalışmıyor"
                ]
            },
            {
                "name": "Şikayet Senaryosu",
                "messages": [
                    "şikayet",
                    "memnun değilim",
                    "kötü hizmet"
                ]
            }
        ]
        
        for scenario in demo_scenarios:
            print(f"\n🎭 {scenario['name']}")
            print("-" * 30)
            
            for message in scenario['messages']:
                print(f"\n👤 Müşteri: {message}")
                response = self.process_message(message)
                print(response)
                time.sleep(1)  # Demo için kısa bekleme
        
        print("\n📊 Demo Tamamlandı!")
        self.show_stats()


def main():
    """
    Ana fonksiyon
    """
    demo = BotDemo()
    
    print("🎮 Demo Modu Seçin:")
    print("1. Interaktif Demo (Manuel test)")
    print("2. Otomatik Demo (Senaryo testleri)")
    print("3. Çıkış")
    
    while True:
        try:
            choice = input("\nSeçiminiz (1-3): ").strip()
            
            if choice == '1':
                demo.run_interactive_demo()
                break
            elif choice == '2':
                demo.run_automated_demo()
                break
            elif choice == '3':
                print("👋 Görüşürüz!")
                break
            else:
                print("⚠️  Lütfen 1, 2 veya 3 girin.")
                
        except KeyboardInterrupt:
            print("\n👋 Görüşürüz!")
            break
        except Exception as e:
            print(f"❌ Hata: {e}")


if __name__ == "__main__":
    main()
