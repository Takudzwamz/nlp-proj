from flask import Flask, request, render_template
from entity_linking import extract_key_terms, link_entities

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    linked_entities_info = []
    if request.method == 'POST':
        sentence = request.form['sentence']
        key_terms = extract_key_terms(sentence)
        linked_entities = link_entities(sentence, key_terms)

        for term, entity in linked_entities.items():
            entity_info = {
                'term': term,
                'label': entity.get('label', 'No label'),
                'description': entity.get('description', 'No description'),
                'url': f"https://www.wikidata.org/wiki/{entity['id']}" if 'id' in entity else None
            }
            linked_entities_info.append(entity_info)

        return render_template('home.html', sentence=sentence, linked_entities_info=linked_entities_info)
    return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
