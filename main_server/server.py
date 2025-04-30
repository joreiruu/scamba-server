import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer
import os
from typing import Dict, Any, Tuple
from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class TextClassifier:
    def __init__(self, model_path: str, tokenizer_path: str, max_length: int = 128):
        self.max_length = max_length
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model = None
        self.tokenizer = None
        logger.info(f"Initializing classifier with model: {model_path}")

    def load_model(self) -> None:
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            logger.info(f"Loading model from {self.model_path}")
            self.model = tf.saved_model.load(self.model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def load_tokenizer(self) -> None:
        try:
            if not os.path.exists(self.tokenizer_path):
                raise FileNotFoundError(f"Tokenizer not found at {self.tokenizer_path}")
            logger.info(f"Loading tokenizer from {self.tokenizer_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            logger.info("Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {str(e)}")
            raise

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
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_dir, "models")
model_path = os.path.join(models_dir, "fin_best__XLM-RoBERTa__lr3e-05_ep65_bs32")
tokenizer_path = os.path.join(models_dir, "tkn_for_fin_best__XLM-RoBERTa__lr3e-05_ep65_bs32")

logger.info(f"Base directory: {base_dir}")
logger.info(f"Models directory: {models_dir}")
logger.info(f"Loading model from: {model_path}")
logger.info(f"Loading tokenizer from: {tokenizer_path}")

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

        results = []
        for text in texts:
            result = classifier.predict(text)
            results.append({
                'text': text,
                **result  # Include all prediction results including timestamp
            })

        return jsonify(results)

    except Exception as e:
        logger.error(f"Error in classify_batch endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'timestamp': int(time.time() * 1000)
        }), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

def main():
    try:
        classifier.load_model()
        classifier.load_tokenizer()
        port = int(os.environ.get('PORT', 10000))
        app.run(host='0.0.0.0', port=port)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise

if __name__ == "__main__":
    main()
