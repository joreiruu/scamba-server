import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer
import os
from typing import Dict, Any, Tuple
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from enum import Enum

app = Flask(__name__)
CORS(app)

class ModelType(Enum):
    MBERT = "mbert"
    XLMR = "xlmr"
    BERT = "bert"

class ModelConfig:
    def __init__(self, name: str, model_path: str, tokenizer_path: str, max_length: int = 128):
        self.name = name
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length

class TextClassifier:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.infer = None
        self.interpreter = None
        self.input_mapping = {}

    def load_model(self) -> bool:
        try:
            if not os.path.exists(self.config.model_path):
                raise FileNotFoundError(f"Model not found at {self.config.model_path}")
            
            print(f"Loading {self.config.name} model...")
            # Handle both .tflite and SavedModel formats
            if self.config.model_path.endswith('.tflite'):
                self.interpreter = tf.lite.Interpreter(model_path=self.config.model_path)
                self.interpreter.allocate_tensors()
                # Setup input mapping for TFLite
                input_details = self.interpreter.get_input_details()
                for detail in input_details:
                    name = detail['name'].lower()
                    if 'attention' in name:
                        self.input_mapping['attention_mask'] = detail['index']
                    elif 'input_id' in name:
                        self.input_mapping['input_ids'] = detail['index']
                    elif 'token_type' in name:
                        self.input_mapping['token_type_ids'] = detail['index']
            else:
                # Load SavedModel - fix path handling
                saved_model_path = self.config.model_path
                if not os.path.exists(os.path.join(saved_model_path, 'saved_model.pb')):
                    raise FileNotFoundError(f"SavedModel.pb not found in {saved_model_path}")
                
                print(f"Loading SavedModel from {saved_model_path}")
                self.model = tf.saved_model.load(saved_model_path)
                if "serving_default" not in self.model.signatures:
                    raise ValueError("Model does not have 'serving_default' signature")
                self.infer = self.model.signatures["serving_default"]
                print("SavedModel loaded successfully")
            
            print(f"Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading model {self.config.name}: {str(e)}")
            return False

    def load_tokenizer(self) -> bool:
        try:
            if not os.path.exists(self.config.tokenizer_path):
                raise FileNotFoundError(f"Tokenizer not found at {self.config.tokenizer_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
            return True
        except Exception as e:
            print(f"Error loading tokenizer for {self.config.name}: {str(e)}")
            return False

    def predict(self, text: str) -> Dict[str, Any]:
        try:
            # Tokenize input
            encoded_inputs = self.tokenizer(
                text,
                return_tensors="tf",
                padding=True,
                truncation=True,
                max_length=self.config.max_length
            )
            
            # Create dummy token_type_ids if needed
            if 'token_type_ids' not in encoded_inputs:
                encoded_inputs['token_type_ids'] = tf.zeros_like(encoded_inputs['input_ids'])
            
            # SavedModel inference
            inputs = {
                "input_ids": encoded_inputs["input_ids"],
                "attention_mask": encoded_inputs["attention_mask"],
                "token_type_ids": encoded_inputs["token_type_ids"],
            }
            
            # Run inference
            outputs = self.infer(**inputs)
            logits = outputs["logits"]
            
            # Process results
            probabilities = tf.nn.softmax(logits)
            pred_class = tf.argmax(probabilities, axis=1).numpy()[0]
            probs = probabilities.numpy()[0]
            
            return {
                'success': True,
                'raw_prediction': logits.numpy().tolist(),
                'probabilities': {
                    'ham': float(probs[0]),
                    'scam': float(probs[1])
                },
                'predicted_class': int(pred_class),
                'confidence': float(probs[pred_class] * 100)
            }
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

class ModelServer:
    def __init__(self):
        self.models: Dict[str, TextClassifier] = {}
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.base_dir, "models")
        
    def initialize_models(self):
        print("Initializing models...")
        print(f"Base directory: {self.base_dir}")
        print(f"Models directory: {self.models_dir}")
        
        # Configure models - all using SavedModel format
        model_configs = {
            ModelType.MBERT.value: ModelConfig(
                "mBERT",
                os.path.join(self.models_dir, "fin_sbest__MBERT__lr5e-05_ep149_bs16"),
                os.path.join(self.models_dir, "tkn_for_fin_sbest__MBERT__lr5e-05_ep149_bs16")
            ),
            ModelType.XLMR.value: ModelConfig(
                "XLM-R",
                os.path.join(self.models_dir, "fin_best__XLM-RoBERTa__lr3e-05_ep65_bs32"),
                os.path.join(self.models_dir, "tkn_for_fin_best__XLM-RoBERTa__lr3e-05_ep65_bs32")
            ),
            ModelType.BERT.value: ModelConfig(
                "BERT",
                os.path.join(self.models_dir, "fin_tbest__BERTB__lr5e-05_ep94_bs16"),
                os.path.join(self.models_dir, "tkn_for_fin_tbest__BERTB__lr5e-05_ep94_bs16")
            )
        }

        # Add debug logging for all models
        for model_type, config in model_configs.items():
            print(f"\n{model_type} Model Details:")
            print(f"Model path: {config.model_path}")
            print(f"SavedModel exists: {os.path.exists(config.model_path)}")
            print(f"SavedModel.pb exists: {os.path.exists(os.path.join(config.model_path, 'saved_model.pb'))}")
            print(f"Variables dir exists: {os.path.exists(os.path.join(config.model_path, 'variables'))}")
            print(f"Tokenizer path: {config.tokenizer_path}")
            print(f"Tokenizer exists: {os.path.exists(config.tokenizer_path)}")
        
        # Initialize models
        for model_type, config in model_configs.items():
            print(f"\nInitializing {config.name}...")
            classifier = TextClassifier(config)
            if classifier.load_model() and classifier.load_tokenizer():
                self.models[model_type] = classifier
                print(f"Successfully loaded {config.name} model")
            else:
                print(f"Failed to load {config.name} model")

# Initialize global model server
model_server = ModelServer()

@app.route('/available_models', methods=['GET'])
def get_available_models():
    return jsonify({
        'models': list(model_server.models.keys())
    })

@app.route('/classify', methods=['POST'])
def classify_text():
    try:
        data = request.get_json()
        if not data or 'text' not in data or 'model' not in data:
            return jsonify({'error': 'Missing text or model selection'}), 400

        model_type = data['model']
        if model_type not in model_server.models:
            return jsonify({'error': f'Invalid model type. Available models: {list(model_server.models.keys())}'}), 400

        classifier = model_server.models[model_type]
        result = classifier.predict(data['text'])
        
        if not result.get('success'):
            return jsonify({'error': result.get('error')}), 500
            
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

def main():
    model_server.initialize_models()
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()