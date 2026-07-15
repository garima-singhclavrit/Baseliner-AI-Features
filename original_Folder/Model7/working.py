import numpy as np
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2Model
from sklearn.preprocessing import OneHotEncoder

class GPTModel:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = TFGPT2Model.from_pretrained('gpt2')

    def get_vectors(self, inps):
        return np.array([self._get_vector(i) for i in inps if isinstance(i, str)])

    def _get_vector(self, inp):
        encoded_inp = self.tokenizer(inp, return_tensors='tf')
        out = self.model(encoded_inp)
        return np.mean(out[0].numpy()[0], axis=0)

class EstimateProcessor:
    @staticmethod
    def preprocess(optimistic, likely, pessimistic, actual_estimate):
        weighted_average = round((optimistic + 4 * likely + pessimistic) / 6, 2)
        weighted_average_int = int(weighted_average * 100)
        standard_deviation = round((pessimistic - optimistic) / 6, 2)
        
        risk_factor = round((standard_deviation / weighted_average), 2)
        standard_confidence = int(weighted_average)
        high_confidence = round((weighted_average + 0.53 * standard_deviation), 2)
        higher_confidence = round((weighted_average + 1.28 * standard_deviation), 2)
        highest_confidence = round((weighted_average + 2.33 * standard_deviation), 2)
        final_estimate = round(highest_confidence, 1)
        variance = round(((6 * pessimistic - optimistic) ** 2), 2)

        confidence_conditions = [
            (0 <= risk_factor <= 0.05, standard_confidence),
            (0.05 < risk_factor <= 0.1, high_confidence),
            (0.1 < risk_factor <= 0.15, higher_confidence)
        ]
        final_estimate = int(np.select(confidence_conditions, [final_estimate]))

        risk_factor_int = int(risk_factor * 100)

        return [weighted_average_int, risk_factor_int, variance, final_estimate]

class Encoder:
    @staticmethod
    def encode_labels(data):
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        return encoder.fit_transform(data.to_numpy().reshape(-1, 1))

    @staticmethod
    def encode_text(text_data):
        return Encoder.encode_labels(text_data)

def master_func(*args):
    optimistic_estimate = args[0]
    most_likely_estimate = args[1]
    pessimistic_estimate = args[2]
    actual_estimate = args[3]

    task_name = args[4]
    task_description = args[5]
    label = args[6]
    original_estimate = args[7]

    gpt_model = GPTModel()
    estimator = EstimateProcessor()
    encoder = Encoder()

    preprocessed_data = estimator.preprocess(optimistic_estimate, most_likely_estimate, pessimistic_estimate, actual_estimate)
    
    task_name_vectors = gpt_model.get_vectors(task_name)
    task_description_vectors = gpt_model.get_vectors(task_description)

    label_encoded = encoder.encode_labels(label)
    original_estimate_encoded = encoder.encode_labels(original_estimate)
    actual_estimate_encoded = encoder.encode_labels([actual_estimate])

    encoded_data = [
        task_name_vectors, task_description_vectors,
        label_encoded, original_estimate_encoded, actual_estimate_encoded,
        np.array([preprocessed_data[0]]), np.array([preprocessed_data[1]]),
        np.array([preprocessed_data[2]]), np.array([preprocessed_data[3]])
    ]

    model7 = tf.keras.models.load_model('drive/MyDrive/Colab Notebooks/Estimates/Model7/model7.h5')
    model7.summary()

    predictions = model7.predict(encoded_data)
    print(predictions.shape)

    ai_estimate = encoder.encode_labels(predictions)
    print(ai_estimate)




