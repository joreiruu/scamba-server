import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer
import os
from typing import Dict, Any, List
from flask import Flask, request, jsonify
from flask_cors import CORS
import time
from concurrent.futures import ThreadPoolExecutor
import gc
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('server.log')
    ]
)
logger = logging.getLogger(__name__)

# Enable memory growth for GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

app = Flask(__name__)
CORS(app)
executor = ThreadPoolExecutor(max_workers=8)  # Adjust based on your CPU cores

class TextClassifier:
    def __init__(self, model_path: str, tokenizer_path: str, max_length: int = 128):
        self.max_length = max_length
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        self.model = tf.saved_model.load(self.model_path)

    def load_tokenizer(self) -> None:
        if not os.path.exists(self.tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {self.tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

    def predict(self, text: str) -> Dict[str, Any]:
        try:
            # Tokenize input
            encoded = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="tf"
            )
            
            # Create dummy token_type_ids if needed
            if 'token_type_ids' not in encoded:
                encoded['token_type_ids'] = tf.zeros_like(encoded['input_ids'])
            
            # Prepare inputs as dictionary
            inputs = {
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'token_type_ids': encoded['token_type_ids']
            }
            
            # Run inference using the signature
            outputs = self.model.signatures['serving_default'](**inputs)
            
            # Get logits from the output
            logits = outputs['logits']
            probabilities = tf.nn.softmax(logits, axis=1)
            pred_class = tf.argmax(probabilities, axis=1).numpy()[0]
            probs = probabilities.numpy()[0]
            
            return {
                'raw_prediction': logits.numpy().tolist(),
                'class_0_probability': float(probs[0]),
                'class_1_probability': float(probs[1]),
                'predicted_class': int(pred_class),
                'confidence': float(probs[pred_class] * 100),
                'timestamp': int(time.time() * 1000)
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return {
                'error': str(e),
                'timestamp': int(time.time() * 1000)
            }

# Initialize classifier globally
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'fin_best__XLM-RoBERTa__lr3e-05_ep65_bs32')
tokenizer_path = os.path.join(current_dir, 'tkn_for_fin_best__XLM-RoBERTa__lr3e-05_ep65_bs32')
classifier = TextClassifier(model_path, tokenizer_path)

@app.route('/classify_batch', methods=['POST'])
def classify_batch():
    try:
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({'error': 'No texts provided'}), 400

        texts = data['texts']
        if not isinstance(texts, list) or not texts:
            return jsonify({'error': 'texts must be a non-empty list'}), 400

        # Parallel processing of predictions
        futures = [executor.submit(classifier.predict, text) for text in texts]
        results = []
        
        for text, future in zip(texts, futures):
            result = future.result()
            results.append({
                'text': text,
                **result
            })

        # Explicit garbage collection after batch processing
        gc.collect()
        return jsonify(results)

    except Exception as e:
        logger.error(f"Error in classify_batch endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'timestamp': int(time.time() * 1000)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint to check server status"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': classifier.model is not None,
        'tokenizer_loaded': classifier.tokenizer is not None,
        'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0
    })

def main():
    try:
        logger.info("Starting server initialization...")
        
        # Load model and tokenizer
        logger.info(f"Loading model from {model_path}")
        classifier.load_model()
        logger.info("Model loaded successfully")
        
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        classifier.load_tokenizer()
        logger.info("Tokenizer loaded successfully")
        
        port = int(os.environ.get('PORT', 5000))
        logger.info(f"Starting server on port {port}")
        
        # Use production WSGI server
        from waitress import serve
        logger.info("Initializing Waitress server...")
        serve(app, host='0.0.0.0', port=port, threads=8)
        
    except FileNotFoundError as e:
        logger.error(f"File not found error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
