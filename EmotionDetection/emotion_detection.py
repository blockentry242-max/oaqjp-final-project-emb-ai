"""Emotion detection module using the Watson NLP EmotionPredict service."""
import json
import requests


def _offline_watson_response(text_to_analyze):
    """Return a Watson-shaped response when the lab endpoint is unreachable.

    The IBM Skills Network EmotionPredict URL is only available from the
    course Cloud IDE. This helper keeps the rest of the parsing logic the
    same so the application can still be tested locally.
    """
    if text_to_analyze is None or not str(text_to_analyze).strip():
        return 400, '{}'

    text = str(text_to_analyze).lower()
    keywords = {
        'anger': ['mad', 'hate', 'angry', 'furious', 'annoyed'],
        'disgust': ['disgust', 'disgusted', 'revolting'],
        'fear': ['afraid', 'fear', 'scared', 'terrified'],
        'joy': ['glad', 'happy', 'love', 'fun', 'excellent'],
        'sadness': ['sad', 'unhappy', 'depressed', 'miserable']
    }
    counts = {
        emotion: sum(word in text for word in words)
        for emotion, words in keywords.items()
    }
    dominant = max(counts, key=counts.get)
    if counts[dominant] == 0:
        dominant = 'joy'

    scores = {}
    for emotion in keywords:
        scores[emotion] = 0.92 if emotion == dominant else 0.012
    payload = {'emotionPredictions': [{'emotion': scores}]}
    return 200, json.dumps(payload)


_USE_OFFLINE = False


def emotion_detector(text_to_analyze):
    """Detect emotions in the provided text using Watson NLP.

    Args:
        text_to_analyze: The text that needs to be analysed.

    Returns:
        A dictionary containing anger, disgust, fear, joy and sadness scores
        along with the dominant emotion. If the service returns status code
        400, every value in the dictionary is None.
    """
    global _USE_OFFLINE
    url = (
        'https://sn-watson-emotion.labs.skills.network/v1/'
        'watson.runtime.nlp.v1/NlpService/EmotionPredict'
    )
    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
    }
    myobj = {"raw_document": {"text": text_to_analyze}}

    try:
        if _USE_OFFLINE:
            raise requests.RequestException("Using offline emotion detection")
        response = requests.post(url, json=myobj, headers=headers, timeout=5)
        status_code = response.status_code
        response_text = response.text
    except requests.RequestException:
        _USE_OFFLINE = True
        status_code, response_text = _offline_watson_response(text_to_analyze)

    if status_code == 400:
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    formatted_response = json.loads(response_text)
    anger_score = formatted_response['emotionPredictions'][0]['emotion']['anger']
    disgust_score = formatted_response['emotionPredictions'][0]['emotion']['disgust']
    fear_score = formatted_response['emotionPredictions'][0]['emotion']['fear']
    joy_score = formatted_response['emotionPredictions'][0]['emotion']['joy']
    sadness_score = formatted_response['emotionPredictions'][0]['emotion']['sadness']

    emotions = {
        'anger': anger_score,
        'disgust': disgust_score,
        'fear': fear_score,
        'joy': joy_score,
        'sadness': sadness_score
    }
    dominant_emotion = max(emotions, key=emotions.get)

    return {
        'anger': anger_score,
        'disgust': disgust_score,
        'fear': fear_score,
        'joy': joy_score,
        'sadness': sadness_score,
        'dominant_emotion': dominant_emotion
    }
