import sys
import tensorflow
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.python.client import device_lib
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences


def readDatasets(path):
    f = open(path, 'r', encoding='utf-8')
    texts = f.readlines()
    f.close()
    return texts

def get_ngrams(frase, size=2):
    frase = np.concatenate((np.zeros(size), frase))

    ngrams_list = list()
    for i in range(len(frase)-size):
        ngrams_list.append((tuple(frase[i:i+size]),frase[i+size]))
    return ngrams_list

def co_table(lista_ocurrencias):
    table = dict()
    for i in lista_ocurrencias:
        if i[0] in table:
            if i[1] in table[i[0]]:
                table[i[0]][i[1]] += 1
            else:
                table[i[0]][i[1]] = 1
        else:
            table[i[0]] = {}
            table[i[0]][i[1]] = 1
    return table

def generate_toyLM_ngram_a(table, context='aleatorio', n=15):
    if context == 'aleatorio':
        tmp = list(table.keys())
        context = tmp[np.random.randint(len(tmp))]

    cadena = list(context)
    for _ in range(n-len(context)):
        context = tuple(cadena[-len(context):])
        if context not in table.keys():
            break
        else:
            new = max(table[context], key=table[context].get)
            if new == 2000:
                break
            cadena.append(new)
    return tokenizer_train.sequences_to_texts([cadena])

def perplexity_ngrams(frases):
    tmp = list()
    for frase in frases:
        n_grams = get_ngrams(frase)
        perplexity = 1
        for context, following in n_grams:
            if context in table.keys():
                denominador = sum(table[context].values())
                if following in table[context].keys():
                    numerador = table[context][following]
                else:
                    numerador = 0
            else:
                denominador, numerador = 0, 0

            numerador += 1
            denominador += 2000

            perplexity *= 1/(numerador/denominador)
        tmp.append(perplexity**(1/len(n_grams)))
    return np.mean(tmp)

def generate_toyLM_ngram_b(table, context='aleatorio', n=15):
    if context == 'aleatorio':
        tmp = list(table.keys())
        context = tmp[np.random.randint(len(tmp))]

    cadena = list(context)
    for i in range(n-len(context)):
        context = tuple(cadena[-len(context):])
        if context not in table.keys():
            break
        else:
            lista_tmp = list()
            for i in table[context]:
                for j in range(table[context][i]):
                    lista_tmp.append(i)
            new = np.random.randint(len(lista_tmp))
            if new == 2000:
                break
            cadena.append(lista_tmp[new])
    return tokenizer_train.sequences_to_texts([cadena])

def train_generate(text, max_seq_length=10):
    train_set = dict()
    for frase in text:
        for word_index in range(len(frase)):
            if word_index < max_seq_length:
                train_set[tuple(pad_sequences([frase[:word_index]], maxlen=max_seq_length)[0])] = frase[word_index]
    return train_set
    
def train_generate(text, size=2):
    x, y = list(), list()
    for phrase in text:
        for context, following in get_ngrams(phrase,size):
            x.append(list(context))
            y.append(following)
    return np.array(x), to_categorical(np.array(y))

def generate_toyLM_lstm_a(model, context='aleatorio', n=15):
    if context=='aleatorio':
        context = list(np.random.randint(0, 5616,2))

    cadena = context
    for _ in range(n-len(context)):
        context = cadena[-len(context):]
        new = np.argmax(model.predict(np.array([context])))
        if new == 5615:
            break
        cadena.append(new)
    return tokenizer_train.sequences_to_texts([cadena])


def generate_toyLM_lstm_b(model, context='aleatorio', n=15):
    if context=='aleatorio':
        context = list(np.random.randint(0, 5616,2))

    cadena = context
    for _ in range(n-len(context)):
        context = cadena[-len(context):]
        probs = model.predict(np.array([context]))
        new = np.random.choice(range(5616), p=probs[0])
        if new == 5615:
            break
        cadena.append(new)
    return tokenizer_train.sequences_to_texts([cadena])

def perplexity_lstm(model, frases):
    tmp = list()
    for frase in frases:
        n_grams = get_ngrams(frase)
        perplexity = 1
        for context, following in n_grams:
            prob = model.predict(np.array([context]))

            perplexity *= 1/prob[0][int(following)]
        tmp.append(perplexity**(1/len(n_grams)))
    return np.mean(tmp)


if __name__ == '__main__':

    text_train = readDatasets('HerMajestySpeechesDataset/train.txt')
    text_test = readDatasets('HerMajestySpeechesDataset/test.txt')
    text_val = readDatasets('HerMajestySpeechesDataset/dev.txt')

    tokenizer_train = Tokenizer(oov_token='<unk>', num_words = 2000)
    tokenizer_train.fit_on_texts(text_train) 

    texts2ids_train = tokenizer_train.texts_to_sequences(text_train)
    texts2ids_test = tokenizer_train.texts_to_sequences(text_test)
    texts2ids_val = tokenizer_train.texts_to_sequences(text_val)

    for i, j, k in zip(texts2ids_train, texts2ids_test, texts2ids_val):
        i.append(2000)
        j.append(2000)
        k.append(2000)

    if sys.argv[1] == 'markov':
        all_ngrams = list()
        for i in texts2ids_train:
            all_ngrams += get_ngrams(i)

        table = co_table(all_ngrams)

        print("Generated sentence with random context: ", generate_toyLM_ngram_a(table, context='aleatorio', n=int(sys.argv[2]))[0]) 
        print("Mean perplexity: ", perplexity_ngrams(texts2ids_test))

    elif sys.argv[1] == 'lstm':
        train_set = train_generate(texts2ids_train)
        test_set = train_generate(texts2ids_test)
        val_set = train_generate(texts2ids_val)

        x_train, y_train = train_generate(texts2ids_train, 10)
        x_test, y_test = train_generate(texts2ids_test, 10)
        x_val, y_val = train_generate(texts2ids_val, 10)

        model = Sequential([Embedding(2001, 20), LSTM(64), Dense(2001, activation='softmax')])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(x_train, y_train, batch_size=64, epochs=15, verbose=0, validation_data=(x_val, y_val))

        print("Generated sentence with random context: ", generate_toyLM_lstm_a(model, context='aleatorio', n=int(sys.argv[2]))[0])
        print("Mean perplexity: ", perplexity_lstm(model, texts2ids_test))



