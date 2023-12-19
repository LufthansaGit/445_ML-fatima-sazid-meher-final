import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

class Prediction:
    def __init__(self):
        self.model_path = "src/model/gradient_boosting_model.sav"
        self.vectorizer_filename = 'src/model/count_dict.sav'
        self.loaded_model = pickle.load(open(self.model_path, 'rb'))
        self.vectoriser = self.load_vectorizer()
    
    def load_vectorizer(self):
        try:
            file_size = os.path.getsize(self.vectorizer_filename)
            print(f"Vectorizer file size: {file_size} bytes")

            with open(self.vectorizer_filename, 'rb') as file:
                vectorizer = pickle.load(file)
            return vectorizer
        except Exception as e:
            print(f"Error loading vectorizer: {str(e)}")
            return None

    def get_prediction(self, user_input):
        try:
            if self.vectoriser is not None:
                vectorized_input = self.vectoriser.transform(user_input)
                result = self.loaded_model.predict(vectorized_input)
                return int(result[0])  # Convert 1 to "Yes" and 0 to "No"
            else:
                print("Vectorizer is not loaded.")
                return None
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None

# For testing the Prediction class
if __name__ == "__main__":
    prediction_obj = Prediction()
    test_input = [""]
    print(prediction_obj.get_prediction(test_input))
