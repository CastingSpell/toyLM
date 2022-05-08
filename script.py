import sys, getopt
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


################################# Read Dataset #################################

def readDataset(path):
    f = open(path, 'r', encoding='utf-8')
    texts = f.readlines()
    f.close()
    return texts




####################### Generate ngrams and training sets #######################

def get_ngrams(phrase, size=2):
    phrase = np.concatenate((np.zeros(size), phrase))

    ngrams_list = list()
    for i in range(len(phrase)-size):
        ngrams_list.append((tuple(phrase[i:i+size]), phrase[i+size]))
    return ngrams_list

def train_generate(text, size=2):
    x, y = list(), list()
    for readDataset in text:
        for context, following in get_ngrams(readDataset, size):
            x.append(list(context))
            y.append(following)
    return np.array(x), to_categorical(np.array(y))




####################### Create and Train Models #######################

def co_table(oc_list):
    table = dict()
    for i in oc_list:
        if i[0] in table:
            if i[1] in table[i[0]]:
                table[i[0]][i[1]] += 1
            else:
                table[i[0]][i[1]] = 1
        else:
            table[i[0]] = {}
            table[i[0]][i[1]] = 1
    return table

def lstm_model(vocab_size, embedding_dims, x_train, y_train, x_val, y_val):
    es = EarlyStopping(
        monitor="val_loss",
        patience=7,
    )
    model = Sequential([Embedding(vocab_size, embedding_dims), LSTM(64), Dense(vocab_size, activation='softmax')])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=64, epochs=15, verbose=0, validation_data=(x_val, y_val), callbacks=[es])
    return model




####################### Evaluate Models #######################

def perplexity_markov(phrases, vocab_size, size=2):
    tmp = list()
    for phrase in phrases:
        n_grams = get_ngrams(phrase, size)
        perplexity = 1
        for context, following in n_grams:
            if context in table.keys():
                denominator = sum(table[context].values())
                if following in table[context].keys():
                    numerator = table[context][following]
                else:
                    numerator = 0
            else:
                denominator, numerator = 0, 0

            perplexity *= 1/((numerator+1)/(denominator+vocab_size))
        tmp.append(perplexity**(1/len(n_grams)))
    return np.mean(tmp)

def perplexity_lstm(model, phrases, size=2):
    tmp = []
    for phrase in phrases:
        x_test, _ = train_generate([phrase], size)
        perplexity = 1
        prob = model.predict(x_test)
        for num, i in enumerate(prob):
            perplexity *= 1/i[phrase[num]]
        tmp.append(perplexity**(1/len(x_test)))
    return np.mean(tmp)




####################### Generate Text #######################

def markov_generate(table, final_word, context='aleatorio', n=15, mode = 2):
    if context == 'aleatorio':
        tmp = list(table.keys())
        context = tmp[np.random.randint(len(tmp))]

    cadena = list(context)
    for _ in range(n-len(context)):
        context = tuple(cadena[-len(context):])
        if context not in table.keys():
            new = 1
        else:
            if len(table[context]) > 1 and 1 in table[context].keys():
                table[context].pop(1)
            if mode == 1:
                new = max(table[context], key=table[context].get)
            elif mode == 2:
                words = list(table[context].keys())
                probs = list(table[context].values())/np.sum(list(table[context].values()))
                new = np.random.choice(words, p=probs)

            if new == final_word:
                break
            cadena.append(new)
    return tokenizer_train.sequences_to_texts([cadena])


def lstm_generate(model, final_word, context='aleatorio', n=15, mode=2, size=2):
    if context == 'aleatorio':
        context = list(np.random.randint(2, final_word, size))

    cadena = context
    for _ in range(n-len(context)):
        context = cadena[-len(context):]
        probs = model.predict(np.array([context]))
        probs = probs[0][2:]

        if mode == 1:
            new = np.argmax(probs)+2
        elif mode == 2:
            new = np.random.choice(range(2,final_word+1), p=probs/np.sum(probs))

        if new == final_word:
            break
        cadena.append(new)
    return tokenizer_train.sequences_to_texts([cadena])





if __name__ == '__main__':

    try:
        opts, args = getopt.getopt(sys.argv[1:], "d:o:m:n:w:", ["data", "option", "model", "n", "w"])
    except getopt.GetoptError:
        print('script.py [data] [option] [model] [ngram_size] [num_words]')
        sys.exit()

    data, option, model, ngram_size, num_words = 'queen', None, None, 2, 2000

    for opt, arg in opts:

        if opt in ("-d", "-data"):

            if arg == 'queen':
                data = 'HerMajestySpeechesDataset'
            elif arg == 'doyle':
                data = 'LaGuardiaBlanca'

            text_train = readDataset(data + '/train.txt')
            text_test = readDataset(data + '/test.txt')
            text_val = readDataset(data + '/dev.txt')

            tokenizer_train = Tokenizer(oov_token='<unk>', num_words=num_words)
            tokenizer_train.fit_on_texts(text_train)

            texts2ids_train = tokenizer_train.texts_to_sequences(text_train)
            texts2ids_test = tokenizer_train.texts_to_sequences(text_test)
            texts2ids_val = tokenizer_train.texts_to_sequences(text_val)

            vocabulary_size = int(np.max(np.concatenate(texts2ids_train)) + 1)

        elif opt in ('-option', '-o'):
            if arg not in ('gen', 'eval'):
                print('Sintaxis incorrecta: script.py [data] [option] [model] [ngram_size] [num_words]')
            option = arg

        elif opt in ('-model', "-m"):
            if arg not in ('markov', 'lstm'):
                print('Sintaxis incorrecta: script.py [data] [option] [model] [ngram_size] [num_words]')
            model = arg
        elif opt in ('-ngram_size', "-n"):
            ngram_size = int(arg)
        elif opt in ('-num_words', "-w"):
            if int(arg) > vocabulary_size:
                num_words = vocabulary_size
            else:
                num_words = int(arg)
        else:
            print('Sintaxis incorrecta: script.py [data] [option] [model] [ngram_size] [num_words]')
            sys.exit()

    for phrase in texts2ids_train:
        phrase.append(vocabulary_size)

    for phrase in texts2ids_test:
        phrase.append(vocabulary_size)

    for phrase in texts2ids_val:
        phrase.append(vocabulary_size)

    print("Arguments:", option, model, ngram_size, num_words)

    if model == 'markov':
        all_ngrams = list()
        for i in texts2ids_train:
            all_ngrams += get_ngrams(i, ngram_size)

        table = co_table(all_ngrams)

        if option == 'gen':
            print("Generated sentence with random context: ", markov_generate(table, vocabulary_size, context='aleatorio', n=15))
        
        elif option == 'eval':
            print("Mean perplexity: ", perplexity_markov(texts2ids_test, vocabulary_size, ngram_size))

    elif model == 'lstm':

        x_train, y_train = train_generate(texts2ids_train, ngram_size)
        x_test, y_test = train_generate(texts2ids_test, ngram_size)
        x_val, y_val = train_generate(texts2ids_val, ngram_size)

        lstm = lstm_model(vocabulary_size+1, 20, x_train, y_train, x_val, y_val)

        if option == 'gen':
            print("Generated sentence with random context: ", lstm_generate(lstm, vocabulary_size, context='aleatorio', n=15, mode=2, size=ngram_size))
        
        elif option == 'eval':
            print("Mean perplexity: ", perplexity_lstm(lstm, texts2ids_test, ngram_size))

