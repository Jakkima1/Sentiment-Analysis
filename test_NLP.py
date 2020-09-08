# import required packages
# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow
def main():
    # Importing libraries
    import numpy as np
    import pandas as pd
    from keras.models import load_model

    # Loading the test data
    test_data = pd.read_csv(r'./data/test_data_NLP.csv')
    test_target = pd.read_csv(r'./data/test_labels_NLP.csv')

    # Converting test data and test target into numpy array

    test_data = np.array(test_data)
    test_target = np.array(test_target)

    #Calling the saved trained model
    saved_model_NLP = load_model("./models/20865621_NLP_model.model")


    loss, accuracy = saved_model_NLP.evaluate(test_data, test_target)
    print("The loss and accuracy for test data is: ", loss, accuracy)
if __name__ == "__main__":
    main()
	# 1. Load your saved model

	# 2. Load your testing data

	# 3. Run prediction on the test data and print the test accuracy