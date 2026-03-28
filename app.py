from __future__ import absolute_import, division, print_function, unicode_literals
from flask import Flask, render_template, request
from inference import DecoderFromNamedEntitySequence
import json
import pickle
import torch
from gluonnlp.data import SentencepieceTokenizer
from model.net import KobertCRFViz
from data_utils.utils import Config
from data_utils.vocab_tokenizer import Tokenizer
from data_utils.pad_sequence import keras_pad_fn
from pathlib import Path

app = Flask(__name__)

print(“Starting the server, loading the model globally... (This may take a few seconds)”)

model_dir = Path('./experiments/base_model_with_crf')
model_config = Config(json_path=model_dir / 'config.json')

tok_path = "ptr_lm_model/tokenizer_78b3253a26.model"
ptr_tokenizer = SentencepieceTokenizer(tok_path)

# load vocab & tokenizer
with open(model_dir / "vocab.pkl", 'rb') as f:
    vocab = pickle.load(f)
tokenizer = Tokenizer(vocab=vocab, split_fn=ptr_tokenizer, pad_fn=keras_pad_fn, maxlen=model_config.maxlen)

# load ner_to_index.json
with open(model_dir / "ner_to_index.json", 'rb') as f:
    ner_to_index = json.load(f)
    index_to_ner = {v: k for k, v in ner_to_index.items()}

# model
model = KobertCRFViz(config=model_config, num_classes=len(ner_to_index), vocab=vocab)

checkpoint = torch.load("./experiments/base_model_with_crf/best-epoch-12-step-1000-acc-0.960.bin",
                        map_location=torch.device('cpu'))

model_dict = model.state_dict()
convert_keys = {}
for k, v in checkpoint['model_state_dict'].items():
    new_key_name = k.replace("module.", '')
    if new_key_name not in model_dict:
        print("{} is not in model_dict".format(new_key_name))
        continue
    convert_keys[new_key_name] = v

model.load_state_dict(convert_keys)
model.eval()

# Force CPU usage (best suited for virtual machine environments; switch to GPU usage if possible)
device = torch.device('cpu')
model.to(device)

decoder_from_res = DecoderFromNamedEntitySequence(tokenizer=tokenizer, index_to_ner=index_to_ner)


@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/test')
def test():
    return render_template('post.html')

@app.route('/post', methods=['POST'])
def post():

    value = request.form['input']
    
    list_of_input_ids = tokenizer.list_of_string_to_list_of_cls_sep_token_ids([value])
    x_input = torch.tensor(list_of_input_ids).long().to(device)
    
    # Plus torch.no_grad()
    with torch.no_grad():
        list_of_pred_ids, _ = model(x_input)
        
    list_of_ner_word, decoding_ner_sentence = decoder_from_res(
        list_of_input_ids=list_of_input_ids,
        list_of_pred_ids=list_of_pred_ids
    )
    
    return {'word': list_of_ner_word, 'decoding': decoding_ner_sentence}

if __name__ == '__main__':
# app.debug = True
    app.run(host='0.0.0.0')