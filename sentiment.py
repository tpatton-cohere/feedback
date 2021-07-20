from flask import Flask
from flask import request
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf


app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

@app.route("/sentiment/<text>")
def sentiment_analysis(text, methods=['GET']):
    encoded_input = tokenizer.encode_plus(text, return_tensors='tf')
    class_logits = model(**encoded_input)[0]
    softmax = tf.nn.softmax(class_logits, axis=1).numpy()[0]
    pos_score = softmax[1]
    # print(pos_score)
    return str(pos_score)


if __name__ == '__main__':
    app.run(port=1739)