import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

class TFIDF_PROCESSOR():
    def __init__(self):
        self.vocabulary = None
        self.doc_count = None
        self.doc_frequencies = None

    def configure(self, data_col, threshold=1):
        # Assumes pandas column
        # Compute tf_idf model, to be used in new sentences.
        # word count must be AT LEAST threshold to be accepted

        # Text and split into strings
        sentences = data_col.str.lower().str.findall("\w+")
        self.doc_count = len(sentences)

        # All unique words and their counts
        word_counter = pd.Series(sentences.sum()).value_counts()

        unknown_words = set()

        for word in word_counter.keys():
            if word_counter[word] < threshold:
                unknown_words.add(word)

        if threshold > 1:
            print("Number of words with less than " + str(threshold) + " occurences: " + str(len(unknown_words)))

        # Calculate word value counts
        word_doc_frequency = dict()
        word_doc_frequency["_UNKNOWN"] = 0  # Unknown vector

        for sentence in sentences:
            # Get series indexed by word and whose value is the number of occurences
            sentence_word_counter = pd.Series(sentence).value_counts()
            for word in sentence_word_counter.keys():
                if word in unknown_words:
                    word_doc_frequency["_UNKNOWN"] += sentence_word_counter[word]
                else:
                    if word in word_doc_frequency:
                        word_doc_frequency[word] += sentence_word_counter[word]
                    else:
                        word_doc_frequency[word] = sentence_word_counter[word]

        self.doc_frequencies = word_doc_frequency
        self.vocabulary = dict(zip(sorted(self.doc_frequencies.keys()), range(len(self.doc_frequencies.keys()))))

    def apply(self, orig_sentence):
        # determine replacement for low-frequency words
        sentence = pd.Series(orig_sentence).str.lower().str.findall("\w+").to_list()[0]

        # Replace with unknowns whenever valid.
        for i in range(len(sentence)):
            if sentence[i] not in self.doc_frequencies.keys():
                sentence[i] = "_UNKNOWN"

        # Configure term frequency
        term_frequencies = dict(pd.Series(sentence).value_counts() / len(sentence))

        vectors = np.zeros(len(self.vocabulary.keys()))
        # Implementation of tf_idf
        for word in term_frequencies.keys():
            tf = term_frequencies[word]
            count = self.doc_frequencies[word] + 1
            idf = np.log(self.doc_count / count)
            position = self.vocabulary[word]
            vectors[position] = tf * idf
        return vectors


def get_iris_data():
    from sklearn.datasets import load_iris
    data = load_iris()
    X_train = data.data[:, [0, 2]]
    print(type(X_train))
    y_train = data.target
    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer()
    Y_train = lb.fit_transform(y_train)
    Y_train = Y_train
    print(Y_train)
    return X_train, None, Y_train, None

def get_data():
    train_ds = pd.read_csv("train.csv")
    train_ds = train_ds.dropna(axis=0, how="any")
    test_ds = pd.read_csv("test.csv")
    test_ds = test_ds.dropna(axis=0, how="any")

    X_train = train_ds[["id", "text"]]
    Y_train = pd.get_dummies(train_ds["emotions"], dtype='int')

    X_test = test_ds[["id", "text"]]

    # Create a new column storing the calculated vectors
    return X_train, X_test, Y_train, None

def relu(val):
    return np.maximum(val, 0)

def d_relu(val):
    # For this decision, values == 0 will be set to 0.
    return np.where(val > 0, 1, 0)

def sigmoid(val):
    return 1.0 / (1.0 * np.exp(-val))

def d_sigmoid(val):
    return (1.0 - val) * val

def softmax(val):
    f = np.exp(val - np.max(val))  # shift values
    return f / f.sum(axis=0)
    # return np.exp(val) / np.sum(np.exp(val), axis=0)

def mse(actual, pred):
    assert actual.shape[0] > 1
    return 1 / (2 * (actual.shape[0])) * np.sum(np.square(pred - actual))

def cross_entropy(actual, pred):
    assert actual.shape[0] > 1
    return np.squeeze(-np.sum(np.multiply(np.log(pred), actual)) / actual.shape[0])

def accuracy(actual, pred):
    return np.sum(np.equal(actual, pred)) / len(actual)

def forward(input_data, weights_list, biases_list):
    # Takes input data, weights, and biases.
    # EXPECTED INPUT DATA SHAPE: num_features * batch_size
    # EXPECTED OUTPUT DATA SHAPE: num_outputs * batch_size
    # Checks if dimensions of input_data is CORRECT

    # We store outputs in here. Next layers will use these outputs as well in propogating forward
    outputs = []

    last_idx = len(weights_list) - 1
    for layer_idx in range(last_idx+1):
        weights = weights_list[layer_idx]
        biases = biases_list[layer_idx]

        # If first layer the inputs are going to be the input data
        if layer_idx == 0:
            dot_prod = np.matmul(weights, input_data) + biases
        else:
            dot_prod = np.matmul(weights, outputs[layer_idx - 1]) + biases

        if layer_idx != last_idx:
            # If normal layer just relu like normal
            outputs.append(relu(dot_prod))
        else:
            # Otherwise we apply sigmoid or softmax
            outputs.append(softmax(dot_prod))
    return outputs

def backwards(input_data, output_data, predictions, weights_list, biases_list, learning_rate):
    last_idx = len(weights_list) - 1
    # Goes backwards and updates the gradients

    weight_gradients = []
    bias_gradients = []
    # First deal with the final layer.
    batch_size = output_data.shape[1]
    assert batch_size > 10

    # Softmax derivative
    prior_gradient = predictions[-1] - output_data

    weight_gradient = 1/batch_size * np.dot(prior_gradient, predictions[-2].T) # Need output of layer prior to last
    bias_gradient = 1/batch_size * np.sum(prior_gradient, axis=1, keepdims=True)

    weight_gradients.append(weight_gradient)
    bias_gradients.append(bias_gradient)

    prior_gradient = np.dot(weights_list[-1].T, prior_gradient)

    # We already dealt with the last index (aka the softmax layer), skip it!
    for layer_idx in range(last_idx-1, 0, -1):
        # Error function: E
        #
        # First: Derivative of total error with respect to output (this is prior gradient)
        #   dE / dOutput = prior gradient
        # Next: Derivative of output with respect to the input
        #   dOutput / dInput. Use the derivative (either relu or sigmoid derivatives)
        # Finally: Derivative of input with respect to the weights
        #   dInput / dW_:?. This is just the relevant weight.
        # Multiply it all together:
        # print(predictions[layer_idx-1])
        tmp_derivative = d_relu(predictions[layer_idx])  # Derivative relu using relu and predictions
        tmp_derivative2 = np.multiply(prior_gradient, tmp_derivative)

        # INFO:
        # weights_list[layer_idx+1] is output_size x input_size matrix. Stores weight mapping
        # prior_gradient is output_size x batch_size matrix. Stores gradients for each batch member
        # tmp_derivative is derivative, input_size x batch_size matrix.
        weight_gradient = 1 / batch_size * np.dot(tmp_derivative2, predictions[layer_idx-1].T)
        bias_gradient = 1 / batch_size * np.sum(tmp_derivative2, axis=1, keepdims=True)

        prior_gradient = np.dot(weights_list[layer_idx].T, tmp_derivative2)
        # print(prior_gradient)
        # print(weight_gradient)
        # print(bias_gradient)

        weight_gradients.append(weight_gradient)
        bias_gradients.append(bias_gradient)

    # Now... we deal with the very first layer!
    tmp_derivative = d_relu(predictions[0])
    tmp_derivative2 = np.multiply(prior_gradient, tmp_derivative)

    # Can't use predictions, need to use input!
    weight_gradient = 1 / batch_size * np.dot(tmp_derivative2, input_data.T)  # Need output of layer prior to last
    bias_gradient = 1 / batch_size * np.sum(tmp_derivative2, axis=1, keepdims=True)

    # prior_gradient = np.dot(weights_list[0].T, tmp_derivative2)

    weight_gradients.append(weight_gradient)
    bias_gradients.append(bias_gradient)
    # print()
    # print(weight_gradient)
    # print(bias_gradient)

    # After calculating gradients, THEN we update the weights.
    # Run it backwards (last layer is at the front of the gradient adjustments list)
    for layer_idx in range(0, last_idx+1):
        weights_list[layer_idx] = weights_list[layer_idx] - learning_rate * weight_gradients[last_idx-layer_idx]
        biases_list[layer_idx] = biases_list[layer_idx] - learning_rate * bias_gradients[last_idx-layer_idx]

    return weights_list, biases_list

def LR():
    # Builds a logistic regression
    return None

def NN():
    _X_train, _X_test, _Y_train, Y_test = get_data()

    # Apply tf-idf! (during cross validation this will applied to the train set)
    tf_idf_class = TFIDF_PROCESSOR()
    tf_idf_class.configure(_X_train["text"], threshold=2)

    results = []
    for sentence in _X_train["text"]:
        result = tf_idf_class.apply(sentence)
        results.append(result)

    X_train = np.asarray(results)
    Y_train = np.asarray(_Y_train)

    # Define the relu and softmax functions
    # Builds a neural network

    # Weights and biases, indexed by layer position (0 is input layer to output layer)
    # Each weight is a OUTPUT x INPUT sized matrix. So for a 3x10 matrix:
    #       10 columns for the 10 input weights
    #       3 rows for each of the 3 output neurons.
    LayerWeights = []
    LayerBiases = []

    NUM_ITERS = 1000
    LR = 0.03

    input_size = X_train.shape[1]
    layer_one = 100
    layer_two = 100
    output_layer = Y_train.shape[1]

    print("Neural network architecture: Input size " + str(input_size))
    print("Neural network architecture: Output size " + str(Y_train.shape[1]))

    NetworkArchitecture = [input_size, layer_one, layer_two, output_layer]

    # Initialize weights here using gaussian

    np.random.seed(0)
    for layer_idx in range(len(NetworkArchitecture)-1):
        input_size = NetworkArchitecture[layer_idx]
        output_size = NetworkArchitecture[layer_idx+1]

        LayerWeights.append(np.random.normal(0, np.sqrt(2.0 / input_size), (output_size, input_size)))
        LayerBiases.append(np.random.normal(0, np.sqrt(2.0 / input_size), (output_size, 1)))

    # Execute the forward passes!
    for iter in range(NUM_ITERS):
        # Compute the forward outputs

        # Need to transpose into feature * batch size for matmul
        curr_pred = forward(X_train.T, LayerWeights, LayerBiases)

        # Need to determine the error for the last prediction! AKA MSE
        last_pred = curr_pred[-1] # Get the last prediction! Should be num_output * batch size
        # Our loss function is MSE. To make the derivative easier, we use 1/2 * (
        # curr_loss = mse(Y_train, last_pred)
        curr_loss = cross_entropy(Y_train, last_pred.T)
        curr_acc = accuracy(Y_train, last_pred.T)
        print("ITERATION: " + str(iter) + ", LOSS: " + str(curr_loss) + ", ACC: " + str(curr_acc))

        # Our loss function for multiclass classification is cross-entropy loss.
        LayerWeights, LayerBiases = backwards(X_train.T, Y_train.T, curr_pred, LayerWeights, LayerBiases, LR)

    # Make predictions
    print("Final prediction!")
    last_pred = forward(X_train.T, LayerWeights, LayerBiases)[-1]
    curr_loss = cross_entropy(Y_train, last_pred.T)
    curr_acc = accuracy(Y_train, last_pred.T)
    print("FINAL ACC: " + str(curr_acc))


if __name__ == '__main__':
    print ("..................Beginning of Logistic Regression................")
    LR()
    print ("..................End of Logistic Regression................")

    print("------------------------------------------------")

    print ("..................Beginning of Neural Network................")
    NN()
    print ("..................End of Neural Network................")