def main():

    # Downloading the data from given url and preparing the train dataset and test dataset
    # For downloading the data from given url and to prepare the train and test dataset
    import sys
    import os
    import urllib.request
    import tarfile
    import zipfile
    import glob

    import tensorflow as tf
    import numpy as np

    # For building the train network and for data preprocessing
    from keras.models import Sequential
    from keras.layers import Dense, GRU, Embedding, LSTM
    from keras.optimizers import Adam
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.layers import Embedding, Dropout, Conv1D, MaxPool1D, GRU, LSTM, Dense, Conv2D, Activation
    from keras.models import load_model

    # For Natural language Processing and data preprocessing
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer

    # For Splitting tarin data in to train and valiation sets
    from sklearn.model_selection import train_test_split

    # Function for showing the download progress
    def printing_downloading_status(c, b_size, tot_size):

        pt_comp = float(c * b_size) / tot_size

        pt_comp = min(1.0, pt_comp)

        message = "\r- Downloading status: {0:.1%}".format(pt_comp)

        # It will print this.
        sys.stdout.write(message)
        sys.stdout.flush()

    # Function for downloading the data from given url and download directory
    def download_res(main_url, filename, download_direct):

        store_path = os.path.join(download_direct, filename)

        if not os.path.exists(store_path):

            if not os.path.exists(download_direct):
                os.makedirs(download_direct)

            print("Downloading", filename, "...")

            url = main_url + filename
            f_path, _ = urllib.request.urlretrieve(url=url,
                                                   filename=store_path,
                                                   reporthook=printing_downloading_status)

            print("Completed!")

    # Function for extracting the zip file in the directory
    def download_and_extract(url, download_direct):

        filename = url.split('/')[-1]
        f_path = os.path.join(download_direct, filename)

        if not os.path.exists(f_path):

            if not os.path.exists(download_direct):
                os.makedirs(download_direct)

            f_path, _ = urllib.request.urlretrieve(url=url,
                                                   filename=f_path,
                                                   reporthook=printing_downloading_status)
            print("Downloading completed successfully. Extracting files.")

            if f_path.endswith(".zip"):
                zipfile.ZipFile(file=f_path, mode="r").extractall(download_direct)
            elif f_path.endswith((".tar.gz", ".tgz")):
                tarfile.open(name=f_path, mode="r:gz").extractall(download_direct)

            print("Completed successfully")
        else:
            print("Data already downloaded and extracted.")

    # Defining the path where to download the zip file and extract
    data_direct = "data/"
    # Defining the url from where the data to be downloaded
    data_link = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    # For reading the text file
    def r_text_file(p):

        with open(p, 'rt', encoding="utf8") as f:
            l = f.readlines()

            text = " ".join(l)

        return text

    # Preparing the data by combining the pos and neg folders in train folder and pos and neg folders in test folder
    def load_data(train=True):

        train_test_path = "train" if train else "test"

        direct_base = os.path.join(data_direct, "aclImdb", train_test_path)

        path_pos = os.path.join(direct_base, "pos", "*.txt")
        path_neg = os.path.join(direct_base, "neg", "*.txt")

        paths_positive = glob.glob(path_pos)
        paths_negative = glob.glob(path_neg)

        d_pos = [r_text_file(p) for p in paths_positive]
        d_neg = [r_text_file(p) for p in paths_negative]

        data = d_pos + d_neg

        label = [1] * len(d_pos) + [0] * len(d_neg)

        return data, label

    # Calling the function for downloading the data and extracting the data in directory
    download_and_extract(url=data_link, download_direct=data_direct)

    # Loading the train data and test data using load_data function
    x_train, y_train = load_data(train=True)  # Loading the train data
    x_test, y_test = load_data(train=False)  # Loading the test data
    # Data Preprocessing for Train data and Test data
    import re
    # Downloading stopwords from nltk
    nltk.download('stopwords')
    # Performing Stemming for train data
    sp = PorterStemmer()
    corpus_train = []
    for j in range(0, len(x_train)):
        result = re.sub('[^a-zA-Z]', ' ', x_train[
            j])  # Other than a-zA-z characters removing all things like punctuations, numbers, special characters, etc and replacing them with 'space'in train data
        result = re.sub(r'<[^<>]+>', " ", result)  # Removing <[^<>]+> and replacing them with 'space'in train data
        result = result.lower()  # lowering the train data
        result = result.split()  # splitting the train data
        result = [sp.stem(word) for word in result if
                  not word in stopwords.words('english')]  # Removing all stopwords and performing Stemming
        result = ' '.join(result)  # Joining all the words
        corpus_train.append(result)

    # Performing Stemming for test data
    sp = PorterStemmer()
    corpus_test = []
    for j in range(0, len(x_test)):
        result = re.sub('[^a-zA-Z]', ' ', x_test[
            j])  # Other than a-zA-z characters removing all things like punctuations, numbers, special characters, etc and replacing them with 'space'in testdata
        result = re.sub(r'<[^<>]+>', " ", result)  # Removing <[^<>]+> and replacing them with 'space'in test data
        result = result.lower()  # lowering the test data
        result = result.split()  # splitting the test data
        result = [sp.stem(word) for word in result if
                  not word in stopwords.words('english')]  # Removing all stopwords and performing Stemming
        result = ' '.join(result)  # Joining all the words
        corpus_test.append(result)

    # Tokenization for train data and test data
    max_feat_words = 7000
    tokeniz = Tokenizer(num_words=max_feat_words)  # Selecting max_features as 7000 means it will take most frequent 7000 words.
    tokeniz.fit_on_texts(corpus_train)  # Fitting the data on text
    list_tokeniz_train = tokeniz.texts_to_sequences(corpus_train) # Tokenization for train data
    list_tokeniz_test = tokeniz.texts_to_sequences(corpus_test)  # Tokenization for test data

    # Performing Padding on train data and test data with maximum length of each sentence as 500
    x_train_padding = pad_sequences(list_tokeniz_train, maxlen=500)
    x_test_padding = pad_sequences(list_tokeniz_test, maxlen=500)  # Performing Padding on test data with maximum length of each sentence as 500

    # Splitting of data into train and validation set
    seed = 66
    np.random.seed(seed)
    X_train, X_val, Y_train, Y_val = train_test_split(x_train_padding, y_train, test_size=0.1, random_state=seed)

    # Converting to numpy array
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_val = np.array(X_val)
    Y_val = np.array(Y_val)

    x_test_padding = np.array(x_test_padding)  # Converting preprocessed data into numpy array
    y_test = np.array(y_test)  # Converting preprocessed data into numpy array

    # Exporting preprocessed test data and test labels into .csv files(this step is performed in train_NLP.py file)

    a = np.asarray(x_test_padding)
    np.savetxt(r'./data/test_data_NLP.csv', a, delimiter=",")

    b = np.asarray(y_test)
    np.savetxt(r'./data/test_labels_NLP.csv', b, delimiter=",")

    # Building a Neural Network Model
    model = Sequential()

    model.add(Embedding(input_dim=max_feat_words,
                        output_dim=20,
                        input_length=500))

    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Dropout(0.35))

    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(Dropout(0.45))

    model.add(LSTM(100, activation='sigmoid'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=64, epochs=3, validation_data=(X_val, Y_val))

    # Saving the trained model in model folder
    model.save(r'./models/20865621_NLP_model.model')

if __name__ == "__main__":
    main()



#if __name__ == "__main__":
	# 1. load your training data

	# 2. Train your network
	# 		Make sure to print your training loss and accuracy within training to show progress
	# 		Make sure you print the final training accuracy

	# 3. Save your model