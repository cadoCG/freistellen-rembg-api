from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from rembg import remove, new_session
from PIL import Image
import io
import os
import time
import logging
import gc
import psutil
from functools import lru_cache

@app.route('/test', methods=['GET'])
def test_interface():
    """HTML Test-Interface für die API"""
    return render_template('index.html')

@app.route('/demo', methods=['GET'])  
def demo_page():
    """Alternative Demo-Seite"""
    return render_template('index.html')

# Flask App initialisieren
app = Flask(__name__)
CORS(app)

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RembgAPIService:
    """REMBG API Service - Optimiert für Railway Pro Plan"""
    
    def __init__(self):
        self.sessions = {}
        self.model_descriptions = {
            'u2net': 'Standard-Modell (beste Balance)',
            'silueta': 'Kompakt-Modell (schneller, 43MB)',
            'u2net_human_seg': 'Speziell für Menschen',
            'isnet-general-use': 'Verbessertes Modell (neueste Version)'
        }
        self._initialize_models()
    
    def _detect_railway_plan(self):
        """Automatische Railway Plan-Erkennung basierend auf verfügbarem RAM"""
        try:
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            
            if total_memory_gb > 16:  # Pro Plan hat 32GB+
                return "Railway Pro Plan"
            elif total_memory_gb > 4:   # Hobby Plan hat ~1-4GB  
                return "Railway Hobby Plan"
            else:
                return "Railway Trial Plan"
        except Exception as e:
            logger.warning(f"Plan-Erkennung fehlgeschlagen: {e}")
            return "Railway Pro Plan"  # Fallback für Pro Plan
    
    def get_memory_info(self):
        """Aktuelle Memory-Information für Debugging"""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'used_percent': round(memory.percent, 1),
                'railway_plan': self._detect_railway_plan()
            }
        except:
            return {'error': 'Memory info nicht verfügbar'}
    
    def _initialize_models(self):
        """Modelle beim Start laden - optimiert für Pro Plan"""
        logger.info("🚀 Starte REMBG API für freistellen.online...")
        
        # Memory-Check vor Model-Loading
        memory_info = self.get_memory_info()
        logger.info(f"💾 Verfügbares RAM: {memory_info.get('total_gb', 'N/A')}GB")
        logger.info(f"📊 Railway Plan: {memory_info.get('railway_plan', 'N/A')}")
        
        # Basis-Modelle (garantiert laden)
        essential_models = ['u2net', 'silueta']
        
        for model_name in essential_models:
            try:
                logger.info(f"Lade {model_name}...")
                self.sessions[model_name] = new_session(model_name)
                logger.info(f"✅ {model_name} erfolgreich geladen")
            except Exception as e:
                logger.error(f"❌ Fehler beim Laden von {model_name}: {e}")
        
        # Erweiterte Modelle - Pro Plan kann alle laden
        optional_models = ['u2net_human_seg']
        
        # Mit Pro Plan können wir aggressiver laden
        if memory_info.get('total_gb', 0) > 16:
            logger.info("🚀 Pro Plan erkannt - lade alle verfügbaren Modelle")
            optional_models.extend(['isnet-general-use']) # Weitere Modelle möglich
            
        for model_name in optional_models:
            try:
                logger.info(f"Lade optional: {model_name}...")
                self.sessions[model_name] = new_session(model_name)
                logger.info(f"✅ {model_name} geladen")
            except Exception as e:
                logger.warning(f"⚠️ Optional: {model_name} nicht geladen - {e}")
        
        logger.info(f"🎉 API bereit! Verfügbare Modelle: {list(self.sessions.keys())}")
        
        # Final Memory-Check
        final_memory = self.get_memory_info()
        logger.info(f"💾 Nach Model-Loading: {final_memory.get('used_percent', 'N/A')}% RAM verwendet")
    
    def process_image(self, image_file, model_name='u2net', max_size=2000):  # Größere Bilder für Pro Plan
        """Bild verarbeiten - optimiert für Pro Plan Ressourcen"""
        start_time = time.time()
        
        try:
            # Pre-processing Memory-Check
            memory_before = psutil.virtual_memory().percent
            
            # Bild laden
            image = Image.open(image_file.stream)
            original_size = image.size
            original_pixels = original_size[0] * original_size[1]
            
            logger.info(f"📸 Originalbildgröße: {original_size} ({original_pixels:,} Pixel)")
            
            # Pro Plan kann größere Bilder verarbeiten
            max_pixels = 4000000 if psutil.virtual_memory().total > 16 * (1024**3) else 1500000
            
            # Intelligente Größenanpassung
            if original_pixels > max_pixels:
                # Berechne optimale Größe basierend auf Pixel-Count
                scale_factor = (max_pixels / original_pixels) ** 0.5
                new_width = int(original_size[0] * scale_factor)
                new_height = int(original_size[1] * scale_factor)
                
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"🔄 Intelligente Skalierung: {original_size} → {image.size}")
            elif max(image.size) > max_size:
                # Fallback: Standard-Thumbnail
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                logger.info(f"📏 Standard-Verkleinerung: {original_size} → {image.size}")
            
            # Modell auswählen mit Fallback
            if model_name not in self.sessions:
                logger.warning(f"Modell '{model_name}' nicht verfügbar, verwende 'u2net'")
                model_name = 'u2net'
            
            session = self.sessions[model_name]
            
            # Memory cleanup vor AI-Processing
            gc.collect()
            
            # Background entfernen
            logger.info(f"🤖 Verarbeite mit {model_name}...")
            result = remove(image, session=session)
            
            # Optimierte PNG-Ausgabe mit besserer Komprimierung
            output = io.BytesIO()
            result.save(output, format='PNG', optimize=True, compress_level=9)
            output.seek(0)
            
            processing_time = time.time() - start_time
            memory_after = psutil.virtual_memory().percent
            
            logger.info(f"✅ Verarbeitung abgeschlossen in {processing_time:.2f}s")
            logger.info(f"💾 Memory: {memory_before:.1f}% → {memory_after:.1f}%")
            
            # Memory cleanup
            del image, result
            gc.collect()
            
            return output, processing_time, model_name
            
        except Exception as e:
            logger.error(f"❌ Verarbeitungsfehler: {e}")
            # Emergency cleanup
            gc.collect()
            raise

# Service-Instanz erstellen
rembg_service = RembgAPIService()

@app.route('/', methods=['GET'])
def health_check():
    """Health Check und API-Info mit dynamischer Plan-Erkennung"""
    memory_info = rembg_service.get_memory_info()
    
    return jsonify({
        "status": "online",
        "service": "freistellen.online REMBG API",
        "version": "2.0",
        "powered_by": memory_info.get('railway_plan', 'Railway Pro Plan'),
        "available_models": list(rembg_service.sessions.keys()),
        "system_info": {
            "total_memory_gb": memory_info.get('total_gb'),
            "memory_usage_percent": memory_info.get('used_percent'),
            "available_memory_gb": memory_info.get('available_gb')
        },
        "endpoints": {
            "remove_background": "/remove-bg",
            "batch_process": "/batch",
            "models": "/models",
            "health": "/",
            "system": "/system"
        },
        "usage": {
            "single_image": "POST /remove-bg with 'image' file",
            "model_selection": "Add 'model' parameter (u2net, silueta, human)",
            "size_limit": "Add 'max_size' parameter (default: 2000px for Pro Plan)",
            "supported_formats": "JPG, PNG, WebP, TIFF"
        },
        "limits": {
            "max_image_size": "Pro Plan: 10MB+ | Hobby Plan: 2MB",
            "batch_processing": "Pro Plan: 10 images | Hobby Plan: 3 images",
            "max_resolution": "Pro Plan: 4K+ | Hobby Plan: 1500px"
        }
    })

@app.route('/system', methods=['GET'])
def system_info():
    """Detaillierte System-Information für Debugging"""
    memory_info = rembg_service.get_memory_info()
    
    return jsonify({
        "system": {
            "railway_plan": memory_info.get('railway_plan'),
            "memory": memory_info,
            "loaded_models": list(rembg_service.sessions.keys()),
            "model_count": len(rembg_service.sessions)
        },
        "performance": {
            "can_handle_large_images": memory_info.get('total_gb', 0) > 16,
            "recommended_max_size": 2000 if memory_info.get('total_gb', 0) > 16 else 1500,
            "batch_limit": 10 if memory_info.get('total_gb', 0) > 16 else 3
        }
    })

@app.route('/models', methods=['GET'])
def get_available_models():
    """Verfügbare Modelle und Beschreibungen"""
    available_models = {}
    for model_name in rembg_service.sessions.keys():
        available_models[model_name] = rembg_service.model_descriptions.get(
            model_name, "KI-Modell für Background-Removal"
        )
    
    return jsonify({
        "available_models": available_models,
        "default": "u2net",
        "recommendations": {
            "general": "u2net",
            "fast": "silueta", 
            "people": "u2net_human_seg",
            "high_quality": "isnet-general-use"
        },
        "model_info": {
            "u2net": {"size": "176MB", "speed": "medium", "quality": "high"},
            "silueta": {"size": "43MB", "speed": "fast", "quality": "good"},
            "u2net_human_seg": {"size": "176MB", "speed": "medium", "quality": "excellent for humans"}
        }
    })

@app.route('/remove-bg', methods=['POST'])
def remove_background():
    """Hauptendpoint für Background-Removal - Pro Plan optimiert"""
    try:
        # Input-Validierung
        if 'image' not in request.files:
            return jsonify({
                'error': 'Kein Bild gefunden',
                'hint': 'Sende das Bild als "image" Parameter'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'Leere Datei'}), 400
        
        # Parameter auslesen mit Pro Plan Defaults
        model = request.form.get('model', 'u2net')
        max_size = int(request.form.get('max_size', 2000))  # Pro Plan Default
        
        # File-Size Check basierend auf Railway Plan
        file_size_mb = len(file.read()) / (1024 * 1024)
        file.seek(0)  # Reset file pointer
        
        memory_gb = psutil.virtual_memory().total / (1024**3)
        max_file_size = 20 if memory_gb > 16 else 5  # Pro Plan: 20MB, Hobby: 5MB
        
        if file_size_mb > max_file_size:
            return jsonify({
                'error': f'Datei zu groß: {file_size_mb:.1f}MB',
                'limit': f'Maximum: {max_file_size}MB',
                'plan': 'Pro Plan' if memory_gb > 16 else 'Hobby Plan'
            }), 413
        
        logger.info(f"📁 Neue Anfrage: {file.filename} ({file_size_mb:.1f}MB), Modell: {model}, Max-Größe: {max_size}")
        
        # Bild verarbeiten
        result_image, processing_time, used_model = rembg_service.process_image(
            file, model, max_size
        )
        
        # Response mit erweiterten Metadaten
        response = send_file(
            result_image,
            mimetype='image/png',
            as_attachment=False,
            download_name=f'freigestellt_{file.filename.rsplit(".", 1)[0]}.png'
        )
        
        # Erweiterte Headers
        response.headers['X-Processing-Time'] = f"{processing_time:.2f}s"
        response.headers['X-Model-Used'] = used_model
        response.headers['X-Service'] = 'freistellen.online REMBG API v2.0'
        response.headers['X-Railway-Plan'] = rembg_service._detect_railway_plan()
        response.headers['X-File-Size-MB'] = f"{file_size_mb:.1f}"
        
        return response
        
    except Exception as e:
        logger.error(f"❌ API-Fehler: {e}")
        return jsonify({
            'error': f'Verarbeitungsfehler: {str(e)}',
            'status': 'failed',
            'hint': 'Versuche ein kleineres Bild oder anderen Modell'
        }), 500

@app.route('/batch', methods=['POST'])
def batch_process():
    """Batch-Verarbeitung - Pro Plan optimiert"""
    try:
        files = request.files.getlist('images')
        
        if not files or len(files) == 0:
            return jsonify({'error': 'Keine Bilder gefunden'}), 400
        
        # Dynamisches Limit basierend auf Railway Plan
        memory_gb = psutil.virtual_memory().total / (1024**3)
        max_batch_size = 10 if memory_gb > 16 else 3
        plan_name = "Pro Plan" if memory_gb > 16 else "Hobby Plan"
        
        if len(files) > max_batch_size:
            return jsonify({
                'error': f'Zu viele Bilder: {len(files)}',
                'limit': f'Maximum: {max_batch_size} Bilder',
                'plan': plan_name
            }), 400
        
        model = request.form.get('model', 'u2net')
        results = []
        total_time = 0
        
        logger.info(f"📦 Batch-Verarbeitung: {len(files)} Bilder mit {model} ({plan_name})")
        
        for i, file in enumerate(files):
            try:
                result_image, proc_time, used_model = rembg_service.process_image(file, model)
                total_time += proc_time
                
                # Als Base64 für JSON-Response
                import base64
                img_b64 = base64.b64encode(result_image.getvalue()).decode()
                
                results.append({
                    'index': i,
                    'filename': file.filename,
                    'success': True,
                    'processing_time': f"{proc_time:.2f}s",
                    'model_used': used_model,
                    'image_data': f"data:image/png;base64,{img_b64}"
                })
                
            except Exception as e:
                logger.error(f"Batch-Verarbeitung fehlgeschlagen für {file.filename}: {e}")
                results.append({
                    'index': i,
                    'filename': file.filename,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'batch_results': {
                'total_images': len(files),
                'successful': len([r for r in results if r.get('success')]),
                'failed': len([r for r in results if not r.get('success')]),
                'total_time': f"{total_time:.2f}s",
                'average_time': f"{total_time/len(files):.2f}s",
                'railway_plan': plan_name
            },
            'results': results
        })
        
    except Exception as e:
        logger.error(f"❌ Batch-Fehler: {e}")
        return jsonify({'error': f'Batch-Verarbeitungsfehler: {str(e)}'}), 500

# Railway spezifische Konfiguration
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    # Startup-Informationen
    memory_info = psutil.virtual_memory()
    logger.info(f"🚀 Starte freistellen.online REMBG API v2.0 auf Port {port}")
    logger.info(f"💾 Verfügbares RAM: {memory_info.total / (1024**3):.1f}GB")
    logger.info(f"📊 Railway Plan: {'Pro Plan' if memory_info.total > 16*(1024**3) else 'Hobby Plan'}")
    
    app.run(
        host='0.0.0.0', 
        port=port, 
        debug=debug_mode
    )
