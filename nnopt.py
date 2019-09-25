import tensorflow as tf
import matplotlib.pyplot as plt


N_INPUTS = 67


class TrainingBatches:

    def __init__(self, nn, batch_size=16):
        self.nn = nn
        self.batch_size = batch_size
        self.bounds = max([d.shape[0] for d in nn.data])
        self.it = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.it >= self.bounds:
            raise StopIteration

        output = np.concatenate([self.getSlice(d, self.it, self.batch_size) for d in self.nn.data])
        self.it += self.batch_size
        return output[:, :N_INPUTS], output[:, N_INPUTS:]

    # Code must work with Python 2 and 3
    next = __next__

    def getSlice(self, array, start, length):
        start = start % array.shape[0]
        if start + length >= array.shape[0]:
            return np.concatenate((array[start:], array[:(start + length) % array.shape[0]], axis=0))
        else:
            return array[start:start + length]


class NeuralNetwork:

    def __init__(self, df, layer_sizes, output_categories, dropout, trainer):
        self.dropout = dropout
        self.output_categories = output_categories
        
        # Modify output data
        self.modifyDataOutputs(df)
        self.data = self.sortDataByOutput(df)

        self.layer_sizes = (N_INPUTS) + layer_sizes + (len(self.data))

        # Set up variables
        self.X = tf.placeholder("float", [None, self.layer_sizes[0]])
        self.Y = tf.placeholder("float", [None, self.layer_sizes[-1]])
        self.weights = [tf.Variable(tf.random_normal([self.layer_sizes[i], self.layer_sizes[i+1]])) for i in range(len(self.layer_sizes) - 1)]
        self.biases = [tf.Variable(tf.random_normal([sz])) for sz in self.layer_sizes[1:]]

        # Create a neural network skeleton
        self.logits = self.multilayerPerceptron()
        
        # Optimization parameters
        self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(self.Y, self.logits))
        self.train = trainer
        self.init = tf.global_variables_initializer()


    def train(self, epochs=20):

        with tf.Session() as sess:
            sess.run(self.init)

            for epoch in range(epochs):

                total_cost = 0.0
                num_batches = 0
                for batch_in, batch_out in iter(TrainingBatches(self, 16)):
                    _, cost = sess.run([self.train, self.loss], feed_dict={X: batch_in, Y: batch_out})
                    total_cost += cost
                    num_batches += 1

                print("Epoch {}: cost = {}".format(epoch, total_cost / num_batches))

                if epoch % 10 == 0:

                    validation_data = 

        
    def multilayerPerceptron(self):
        
        def outputOfLayer(n, func):
            if n == 0:
                return self.X
            else:
                lastLayer = outputOfLayer(n-1, tf.keras.activations.sigmoid)
                arith = tf.add(tf.matmul(lastLayer, self.weights[n-1]), self.biases[n-1])
                return func(arith)
            
        return outputOfLayer(len(self.data) - 1, tf.nn.softmax)


    def modifyDataOutputs(self, df):

        if self.output_categories == "any":
            df["OUTPUT_ANY"] = df["OUTPUT_<30"] + df["OUTPUT_>30"]
            df.drop(["OUTPUT_<30", "OUTPUT_>30"])

        elif self.output_categories == "rapid":
            df["OUTPUT_NO"] = df["OUTPUT_>30"] + df["OUTPUT_NO"]
            df.drop(["OUTPUT_>30"])


    def sortDataByOutput(self, df):

        if self.output_categories == "three":
            return (
                df[df["OUTPUT_<30"] == 1].to_numpy(),
                df[df["OUTPUT_>30"] == 1].to_numpy(),
                df[df["OUTPUT_NO"] == 1].to_numpy()
            )

        elif self.output_categories == "any":
            return (
                df[df["OUTPUT_ANY"] == 1].to_numpy(),
                df[df["OUTPUT_NO"] == 1].to_numpy()
            )

        else:
            return (
                df[df["OUTPUT_<30"] == 1].to_numpy(),
                df[df["OUTPUT_NO"] == 1].to_numpy()
            )
