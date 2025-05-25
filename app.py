from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from rembg import remove, new_session
from PIL import Image
import io
import os
import time
import logging
import gc
from functools import lru_cache

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
    """REMBG API Service - Optimiert für Railway Hobby Plan"""
    
    def __init__(self):
        self.sessions = {}
        self.model_descriptions = {
            'u2net': 'Standard-Modell (beste Balance)',
            'silueta': 'Kompakt-Modell (schneller, 43MB)',
            'u2net_human_seg': 'Speziell für Menschen',
            'isnet-general-use': 'Verbessertes Modell (neueste Version)'
        }
        self._initialize_models()
    
    def _initialize_models(self):
        """Modelle beim Start laden - wichtig für Performance"""
        logger.info("🚀 Starte REMBG API für freistellen.online...")
        
        # Basis-Modelle (garantiert laden)
        essential_models = ['u2net', 'silueta']
        
        for model_name in essential_models:
            try:
                logger.info(f"Lade {model_name}...")
                self.sessions[model_name] = new_session(model_name)
                logger.info(f"✅ {model_name} erfolgreich geladen")
            except Exception as e:
                logger.error(f"❌ Fehler beim Laden von {model_name}: {e}")
        
        # Erweiterte Modelle (optional)
        optional_models = ['u2net_human_seg']
        for model_name in optional_models:
            try:
                logger.info(f"Lade optional: {model_name}...")
                self.sessions[model_name] = new_session(model_name)
                logger.info(f"✅ {model_name} geladen")
            except Exception as e:
                logger.warning(f"⚠️ Optional: {model_name} nicht geladen - {e}")
        
        logger.info(f"🎉 API bereit! Verfügbare Modelle: {list(self.sessions.keys())}")
    
    def process_image(self, image_file, model_name='u2net', max_size=1500):
        """Bild verarbeiten mit Optimierungen für Railway"""
        start_time = time.time()
        
        try:
            # Bild laden
            image = Image.open(image_file.stream)
            original_size = image.size
            
            # Größe für Performance begrenzen
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                logger.info(f"Bild verkleinert: {original_size} → {image.size}")
            
            # Modell auswählen mit Fallback
            if model_name not in self.sessions:
                logger.warning(f"Modell '{model_name}' nicht verfügbar, verwende 'u2net'")
                model_name = 'u2net'
            
            session = self.sessions[model_name]
            
            # Background entfernen
            logger.info(f"Verarbeite mit {model_name}...")
            result = remove(image, session=session)
            
            # Als optimierte PNG speichern
            output = io.BytesIO()
            result.save(output, format='PNG', optimize=True, compress_level=6)
            output.seek(0)
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Verarbeitung abgeschlossen in {processing_time:.2f}s")
            
            # Memory cleanup
            gc.collect()
            
            return output, processing_time, model_name
            
        except Exception as e:
            logger.error(f"❌ Verarbeitungsfehler: {e}")
            raise

# Service-Instanz erstellen
rembg_service = RembgAPIService()

@app.route('/', methods=['GET'])
def health_check():
    """Health Check und API-Info"""
    return jsonify({
        "status": "online",
        "service": "freistellen.online REMBG API",
        "version": "1.0",
        "powered_by": "Railway Hobby Plan",
        "available_models": list(rembg_service.sessions.keys()),
        "endpoints": {
            "remove_background": "/remove-bg",
            "batch_process": "/batch",
            "models": "/models",
            "health": "/"
        },
        "usage": {
            "single_image": "POST /remove-bg with 'image' file",
            "model_selection": "Add 'model' parameter (u2net, silueta, human)",
            "size_limit": "Add 'max_size' parameter (default: 1500px)"
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
            "people": "u2net_human_seg"
        }
    })

@app.route('/remove-bg', methods=['POST'])
def remove_background():
    """Hauptendpoint für Background-Removal"""
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
        
        # Parameter auslesen
        model = request.form.get('model', 'u2net')
        max_size = int(request.form.get('max_size', 1500))
        
        logger.info(f"Neue Anfrage: {file.filename}, Modell: {model}, Max-Größe: {max_size}")
        
        # Bild verarbeiten
        result_image, processing_time, used_model = rembg_service.process_image(
            file, model, max_size
        )
        
        # Response mit Metadaten
        response = send_file(
            result_image,
            mimetype='image/png',
            as_attachment=False,
            download_name='freigestellt.png'
        )
        
        # Custom Headers für Debugging
        response.headers['X-Processing-Time'] = f"{processing_time:.2f}s"
        response.headers['X-Model-Used'] = used_model
        response.headers['X-Service'] = 'freistellen.online REMBG API'
        
        return response
        
    except Exception as e:
        logger.error(f"❌ API-Fehler: {e}")
        return jsonify({
            'error': f'Verarbeitungsfehler: {str(e)}',
            'status': 'failed'
        }), 500

@app.route('/batch', methods=['POST'])
def batch_process():
    """Batch-Verarbeitung für mehrere Bilder"""
    try:
        files = request.files.getlist('images')
        
        if not files or len(files) == 0:
            return jsonify({'error': 'Keine Bilder gefunden'}), 400
        
        # Limit für Railway Hobby Plan
        if len(files) > 3:
            return jsonify({
                'error': 'Maximal 3 Bilder pro Batch',
                'hint': 'Railway Hobby Plan Limit'
            }), 400
        
        model = request.form.get('model', 'u2net')
        results = []
        total_time = 0
        
        logger.info(f"Batch-Verarbeitung: {len(files)} Bilder mit {model}")
        
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
                'average_time': f"{total_time/len(files):.2f}s"
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
    
    logger.info(f"🚀 Starte freistellen.online REMBG API auf Port {port}")
    app.run(
        host='0.0.0.0', 
        port=port, 
        debug=debug_mode
    )
