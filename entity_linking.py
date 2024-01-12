import spacy
import requests
from transformers import DistilBertTokenizer, DistilBertModel
from torch.nn.functional import cosine_similarity
import torch
from langdetect import detect

# Load spaCy's pre-trained models for NER
nlp_en = spacy.load("en_core_web_sm")
nlp_ru = spacy.load("ru_core_news_sm")

# Load pre-trained multilingual DistilBERT model and tokenizer for embeddings
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
model = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')

def extract_key_terms(sentence):
    # Detect the language of the sentence
    lang = detect(sentence)
    
    # Use the appropriate spaCy model based on the detected language
    doc = nlp_ru(sentence) if lang == "ru" else nlp_en(sentence)
    
    return [ent.text for ent in doc.ents]

def search_wikidata(term, language='en', limit=100):
    base_url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "language": language,
        "format": "json",
        "search": term,
        "limit": limit  # Increase the number of results fetched
    }
    response = requests.get(base_url, params=params)
    return response.json()['search']


def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings


def choose_most_relevant_entity(entities, context_keywords):
    for entity in entities:
        description = entity.get('description', '').lower()
        for keyword in context_keywords:
            if keyword.lower() in description:
                return entity
    return entities[0] if entities else None

def link_entities(sentence, key_terms):
    sentence_embedding = get_embedding(sentence)
    linked_entities = {}
    lang = detect(sentence)
    context_keywords = [
         'спорт', 'Soccer', 'football','компания','корпорац',
        'здание', 'памятник', 'монумент', 'парк', 'метро','железная дорога','museum',
        'площадь', 'музей', 'место', 'местечко', 'местность', 'местоимение', 'местоположение'
        
    ]  # Example keywords

    for term in key_terms:
        wikidata_results = search_wikidata(term, 'ru' if lang == 'ru' else 'en', limit=100)
        best_match = None
        highest_similarity = -1 

        for entity in wikidata_results:
            entity_label = entity['label']
            entity_embedding = get_embedding(entity_label)
            similarity = cosine_similarity(sentence_embedding, entity_embedding).item()

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = entity

        # Choose the most relevant entity based on context
        relevant_entity = choose_most_relevant_entity(wikidata_results, context_keywords)
        if relevant_entity:
            linked_entities[term] = relevant_entity
        elif best_match:
            linked_entities[term] = best_match

    return linked_entities

