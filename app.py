"""
Akıllı Müşteri Hizmetleri Botu - Web Arayüzü
============================================

Flask tabanlı web arayüzü. Müşteriler bu arayüz üzerinden bot ile etkileşim kurabilir.

Özellikler:
- Gerçek zamanlı sohbet arayüzü
- Sentiment ve intent analiz sonuçlarını görüntüleme
- Bot istatistikleri
- Responsive tasarım

Yazar: AI Assistant
Tarih: 2024
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import json
import uuid
from datetime import datetime
from customer_service_bot import CustomerServiceBot
from loguru import logger

# Flask uygulamasını başlat
app = Flask(__name__)
app.secret_key = 'customer_service_bot_secret_key_2024'
CORS(app)

# Bot instance'ını oluştur
try:
    bot = CustomerServiceBot()
    logger.info("Bot web arayüzü için hazırlandı")
except Exception as e:
    logger.error(f"Bot başlatılamadı: {e}")
    bot = None

# Konuşma geçmişi (gerçek uygulamada veritabanı kullanılmalı)
conversations = {}

@app.route('/')
def index():
    """
    Ana sayfa - Sohbet arayüzü
    """
    # Yeni konuşma ID'si oluştur
    if 'conversation_id' not in session:
        session['conversation_id'] = str(uuid.uuid4())
    
    conversation_id = session['conversation_id']
    
    # Konuşma geçmişini başlat
    if conversation_id not in conversations:
        conversations[conversation_id] = {
            'messages': [],
            'start_time': datetime.now().isoformat(),
            'stats': {
                'total_messages': 0,
                'intents': {},
                'sentiments': {}
            }
        }
    
    return render_template('index.html', 
                         bot_info=bot.get_bot_info() if bot else None,
                         conversation_id=conversation_id)

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Chat API endpoint - Müşteri mesajlarını işler
    """
    try:
        if not bot:
            return jsonify({
                'error': 'Bot şu anda kullanılamıyor',
                'success': False
            }), 500
        
        # JSON verisini al
        data = request.get_json()
        message = data.get('message', '').strip()
        conversation_id = session.get('conversation_id')
        
        if not message:
            return jsonify({
                'error': 'Mesaj boş olamaz',
                'success': False
            }), 400
        
        if not conversation_id:
            return jsonify({
                'error': 'Geçersiz konuşma',
                'success': False
            }), 400
        
        # Bot'tan yanıt al
        response = bot.get_response(message)
        
        # Konuşma geçmişine ekle
        if conversation_id in conversations:
            conversations[conversation_id]['messages'].append({
                'user_message': message,
                'bot_response': response,
                'timestamp': datetime.now().isoformat()
            })
            
            # İstatistikleri güncelle
            conversations[conversation_id]['stats']['total_messages'] += 1
            
            # Intent istatistikleri
            intent = response['intent']
            if intent in conversations[conversation_id]['stats']['intents']:
                conversations[conversation_id]['stats']['intents'][intent] += 1
            else:
                conversations[conversation_id]['stats']['intents'][intent] = 1
            
            # Sentiment istatistikleri
            sentiment = response['sentiment']['label']
            if sentiment in conversations[conversation_id]['stats']['sentiments']:
                conversations[conversation_id]['stats']['sentiments'][sentiment] += 1
            else:
                conversations[conversation_id]['stats']['sentiments'][sentiment] = 1
        
        logger.info(f"Chat mesajı işlendi - Intent: {intent}, Sentiment: {sentiment}")
        
        return jsonify({
            'success': True,
            'response': response
        })
        
    except Exception as e:
        logger.error(f"Chat API hatası: {e}")
        return jsonify({
            'error': 'Bir hata oluştu',
            'success': False
        }), 500

@app.route('/api/stats')
def get_stats():
    """
    Bot istatistiklerini döndürür
    """
    try:
        conversation_id = session.get('conversation_id')
        
        if not conversation_id or conversation_id not in conversations:
            return jsonify({
                'total_conversations': 0,
                'total_messages': 0,
                'intents': {},
                'sentiments': {}
            })
        
        stats = conversations[conversation_id]['stats']
        
        return jsonify({
            'total_conversations': 1,
            'total_messages': stats['total_messages'],
            'intents': stats['intents'],
            'sentiments': stats['sentiments'],
            'conversation_start': conversations[conversation_id]['start_time']
        })
        
    except Exception as e:
        logger.error(f"Stats API hatası: {e}")
        return jsonify({
            'error': 'İstatistikler alınamadı'
        }), 500

@app.route('/api/bot-info')
def get_bot_info():
    """
    Bot bilgilerini döndürür
    """
    try:
        if bot:
            return jsonify({
                'success': True,
                'bot_info': bot.get_bot_info()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Bot bilgileri alınamadı'
            }), 500
            
    except Exception as e:
        logger.error(f"Bot info API hatası: {e}")
        return jsonify({
            'success': False,
            'error': 'Bot bilgileri alınamadı'
        }), 500

@app.route('/api/reset-conversation', methods=['POST'])
def reset_conversation():
    """
    Konuşma geçmişini sıfırlar
    """
    try:
        conversation_id = session.get('conversation_id')
        
        if conversation_id and conversation_id in conversations:
            # Yeni konuşma başlat
            conversations[conversation_id] = {
                'messages': [],
                'start_time': datetime.now().isoformat(),
                'stats': {
                    'total_messages': 0,
                    'intents': {},
                    'sentiments': {}
                }
            }
        
        return jsonify({
            'success': True,
            'message': 'Konuşma sıfırlandı'
        })
        
    except Exception as e:
        logger.error(f"Reset conversation hatası: {e}")
        return jsonify({
            'success': False,
            'error': 'Konuşma sıfırlanamadı'
        }), 500

@app.errorhandler(404)
def not_found(error):
    """
    404 hata sayfası
    """
    return jsonify({
        'error': 'Sayfa bulunamadı',
        'success': False
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """
    500 hata sayfası
    """
    return jsonify({
        'error': 'Sunucu hatası',
        'success': False
    }), 500

if __name__ == '__main__':
    # Geliştirme modunda çalıştır
    app.run(debug=True, host='0.0.0.0', port=5000)
