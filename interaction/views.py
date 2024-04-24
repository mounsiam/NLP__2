import re
import string
import nltk
from django.shortcuts import render
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import speech_recognition as sr
from django.http import HttpResponse
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def index(request):
    # Your view logic here
    return render(request, 'index.html')

def process_complaint_text(request):
    if request.method == 'POST':
        complaint_text = request.POST.get('complaint_text', '')
        if complaint_text:
            # Print the original complaint text
            print("Original complaint text:", complaint_text)

            # Process the complaint text here
            # For example, you can preprocess the text, analyze it, etc.
            processed_text = preprocess_text(complaint_text)
            # Print the processed complaint text
            print("Processed complaint text:", processed_text)

            # Optionally, you can perform further processing here before returning the response

            # Return an HttpResponse object with a success message
            return HttpResponse('Complaint text processed successfully!')
        else:
            # If the complaint text is empty, return an error response
            return HttpResponse('Text complaint is empty!')
    else:
        # If the request method is not POST, return an error response
        return HttpResponse('Invalid request method!')


@csrf_exempt
def process_complaint_audio(request):
    if request.method == 'POST':
        # Initialize speech recognition
        recognizer = sr.Recognizer()

        # Use the default microphone as audio source
        with sr.Microphone() as source:
            print("Listening...")

            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source)

            # Capture audio from the microphone
            audio_data = recognizer.listen(source)

        try:
            # Convert audio to text
            transcription = recognizer.recognize_google(audio_data)
            # Print the original audio transcription
            print("Original audio transcription:", transcription)

            # Preprocess the audio transcription
            preprocessed_text = preprocess_text(transcription)
            # Print the preprocessed audio transcription
            print("Preprocessed audio transcription:", preprocessed_text)

            # Optionally, you can perform further processing here before returning the response

            return JsonResponse({'status': 'success', 'message': 'Audio transcription processed successfully'})
        except sr.UnknownValueError:
            return JsonResponse({'status': 'error', 'message': 'Could not understand audio'})
        except sr.RequestError as e:
            return JsonResponse({'status': 'error', 'message': f"Could not request results: {e}"})
    else:
        return JsonResponse({'status': 'error', 'message': 'Method not allowed'})


def preprocess_text(text):
    # Step 1: Lowercasing
    text = text.lower()

    # Step 2: Remove Punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Step 3: Removing Stopwords
    text = re.sub(r'\s+', ' ', text)
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    text = ' '.join([word for word in tokens if word not in stop_words])

    # Step 4: Stemming or Lemmatization
    # Initialize stemmer and lemmatizer
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # Lemmatize each word in the text
    text = ' '.join([lemmatizer.lemmatize(word) for word in tokens])

    # Step 5: Handling Special Characters
    text = re.sub(r'[^\w\s]', '', text)

    # Step 6: Remove extra white spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def extract_keywords(text):
    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Fit and transform the text data
    tfidf_matrix = tfidf_vectorizer.fit_transform([text])

    # Get feature names (keywords) sorted by their TF-IDF scores
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Extract keywords with highest TF-IDF scores
    top_keywords = []
    for col in tfidf_matrix.nonzero()[1]:
        top_keywords.append((feature_names[col], tfidf_matrix[0, col]))

    # Sort keywords by TF-IDF score in descending order
    top_keywords.sort(key=lambda x: x[1], reverse=True)

    # Extract only the keyword strings
    keywords = [keyword[0] for keyword in top_keywords]

    return keywords



# Example usage:
patient_story = "The patient complains of abdominal pain, fever, and dyspnea."
keywords = extract_keywords(patient_story)
print("Keywords extracted:", keywords)


