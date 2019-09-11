"""

"cam_nn.py"

A neural network to process the credit approval data-- built from scratch (no machine learning libraries used)!

Libraries used: numpy
Author: Ryan Park

"""

# Libraries
import credit_analysis_machine.src.cam_loader as c
import numpy as np

# Classes
class Cost(object):
  # class for the different cost functions

  def __init__(self, name, regularization = None, reg_parameter = None):
    self.name = name
    self.regularization = regularization
    self.reg_parameter = reg_parameter

  def derivative(self, a, y):
    # returns ∂Cx/∂a_L
    if self.name == "mse": return a - y
    elif self.name == "cross-entropy": return (a - y) / (a * (1 - a))
    elif self.name == "log-likelihood": return y / a

  def calculate(self, pairs):
    # accepts a list of tuples (a, y) and returns average cost over that list
    if self.name == "mse": return sum(np.linalg.norm(a - y) ** 2.0 for (a, y) in pairs) / (2.0 * len(pairs))
    elif self.name == "cross-entropy":
      return sum(sum(np.nan_to_num(-y * np.log(a) - (1.0 - y) * np.log(1.0 - a) for (a, y) in pairs))) / (len(pairs))
    elif self.name == "log-likelihood": return sum(np.log(np.argmax(a)) for (a, y) in pairs) / (-1.0 * len(pairs))

  def get_error(self, activation, activations, weighted_inputs, label):
    if self.name == "mse": return self.derivative * activation.calculate(weighted_inputs)
    elif self.name == "cross-entropy" or self.name == "log-likelihood": return activations - label

class Activation(object):

  def __init__(self, name):
    self.name = name

  def calculate(self, z):
    if self.name == "sigmoid": return 1.0 / (1.0 + np.exp(-z)) # np.exp(z) returns e^z
    elif self.name == "softmax": return np.exp(z) / np.sum(np.exp(z))

  def derivative(self, z, j = None, i = None):
    if self.name == "sigmoid": return self.calculate(z) * (1.0 - self.calculate(z))
    elif self.name == "softmax":
      if j == None or i == None: raise TypeError("Arguments 'i' or 'j' not provided.")
      gradient = np.array([-1.0 * self.calculate(zk)[j] * self.calculate(zk)[i] if j != i else self.calculate(z)[j] *
                  (1.0 - self.calculate(z)[j]) for zk in z])
      return gradient

class Early_Stop(object):
  """Note that all the methods in this class need to be used in conjunction with some kind of loop-- they need to be
   supplemented with some other code"""

  @staticmethod # method can be called without creating an Early_Stop object
  def GL(accuracy, stop_parameter):
    """returns "stop" if the generalization loss exceeds a parameter. If a ai_credit accuracy maximum has been found,
    this function returns "ai_credit". Otherwise, the function returns None"""
    local_opt = max(accuracy)
    generalization_loss = local_opt - accuracy[-1]
    to_stop = True if generalization_loss > stop_parameter else False
    if local_opt < accuracy[-1]: return "ai_credit"
    elif to_stop: return "stop"
    else: return None

  @staticmethod
  def average_GL(accuracy, stop_parameter, n):
    """returns "stop" if average generalization loss over the last n epochs exceeds a parameter. If a ai_credit accuracy
    maximum has been found, this function returns "ai_credit". Otherwise, the function returns None"""
    local_opt = max(accuracy)
    accuracy = accuracy[np.argmax(accuracy):]
    # the above line ensures that GL is only evaluated after the locally optimal point
    if len(accuracy) == 0: return "ai_credit"
    elif len(accuracy) >= n:
      average_gl = sum([local_opt - accuracy[-i - 1] for i in range(n)]) / len(accuracy)
      to_stop = True if average_gl > stop_parameter else False
      if to_stop: return "stop"
    if local_opt < accuracy[-1]: return "ai_credit"
    else: return None

  @staticmethod
  def strip_GL(accuracy, stop_parameter, k):
    """returns "stop" if generalization loss over the last k epochs exceeds a parameter. If a ai_credit accuracy maximum has
    been found, this function returns "ai_credit". Otherwise, the function returns None"""
    local_opt = max(accuracy)
    accuracy = accuracy[np.argmax(accuracy):]
    if len(accuracy = 0): return "ai_credit"
    elif len(accuracy >= k):
      strip_GL = [0 if accuracy[-i - 1] > stop_parameter else 1 for i in range(k)]
      to_stop = not(bool(strip_GL))
      if to_stop: return "stop"
    if local_opt < accuracy[-1]: return "ai_credit"
    else: return None

class Network(object): 
  
  def __init__(self, layers, cost_function = Cost("mse"), body_activation = Activation("sigmoid"),
               output_activation = Activation("sigmoid")):
    self.layers = layers # "layers" is a list of neurons (ex: [3, 5, 2])
    self.biases = np.array([np.random.randn(layer, 1) for layer in layers[1:]])
    """generates a random list of biases-- one array per bias (makes the indexing easier), and one array of arrays
      for each layer, except the first layer (which has no biases)."""
    self.weights = np.array([np.random.randn(layers[layer + 1], layers[layer]) for layer in range(len(layers) - 1)])
    """generates a random list of weights-- each layer l has an array of n arrays of length m, where m is the
    size of the layer l - 1 and n is the size of layer l."""
    self.cost = cost_function
    self.activation = body_activation
    self.output_activation = output_activation

  def feed_forward(self, a):
    # feeds an input into the network and returns its output
    for b, w in zip(self.biases, self.weights): a = self.activation.calculate(np.dot(w, a) + b)
    a = self.output_activation.calculate(a)
    return a
  
  def SGD(self, training_data, num_epochs, learning_rate, minibatch_size, validation_data = None, test_data = None,
          monitor = False, early_stopping = None, stop_parameter = None, aGL_parameter = None):
    """implements stochastic gradient descent to minimize the cost function. This function relies on the self.backprop
     function to calculate the gradient of the cost function, which is necessary for the updating of weights and biases"""
    
    if monitor or early_stopping != None:
      evaluation = {"validation accuracy": [], "validation cost": [], "train accuracy": [], "train cost": []}
    if early_stopping != None:
      stored_biases = self.biases
      stored_weights = self.weights
    
    for epoch_num in range(num_epochs):
      epoch = training_data
      np.random.shuffle(epoch) # randomly shuffle epoch
      minibatches = [epoch[i:i + minibatch_size] for i in range(0, len(epoch), minibatch_size)]
      """divide epoch into minibatches (the size of the minibatches is a hyperparameter, or a parameter not chosen by 
      the program)"""
      
      for minibatch in minibatches:
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        """creating the (empty) arrays that will be used to store the gradients of the cost function for individual 
        training examples"""
        
        for image, label in minibatch:
          delta_nabla_b, delta_nabla_w = self.backprop(image, label)
          """most of the work is done by the above line, which calculates the gradient of the cost function with 
          respect to the weights and biases"""
          if self.cost.regularization == "L1":
            delta_nabla_w = delta_nabla_w + (self.cost.reg_parameter / len(epoch) * np.sign(self.weights))
          elif self.cost.regularization == "L2":
            delta_nabla_w = delta_nabla_w + (self.cost.reg_parameter / len(epoch) * self.weights)
          nabla_b = nabla_b + delta_nabla_b
          nabla_w = nabla_w + delta_nabla_w
          
        self.biases -= (learning_rate / minibatch_size) * nabla_b
        self.weights -= (learning_rate / minibatch_size) * nabla_w
      
      if test_data is None: print ("Epoch {0} completed".format(epoch_num + 1))
      else:
        validation_evaluate = self.evaluate_accuracy(validation_data)
        print ("Epoch {0}: {1}%".format(epoch_num + 1, validation_evaluate))
        if early_stopping or monitor:
          evaluation["validation accuracy"].append(validation_evaluate)
          evaluation["train accuracy"].append(self.evaluate_accuracy(
            training_data))
        if monitor:
          evaluation["validation cost"].append(self.evaluate_cost(
            validation_data))
          evaluation["train cost"].append(self.evaluate_cost(
            training_data))

      if early_stopping == "GL":
        to_stop = Early_Stop.GL(evaluation["validation accuracy"], stop_parameter)
        if to_stop == "stop":
          print ("End SGD: stop parameter exceeded")
          self.biases = stored_biases
          self.weights = stored_weights
          break
        elif to_stop == "ai_credit":
          stored_biases = self.biases
          stored_weights = self.weights
      elif early_stopping == "aGL":
        to_stop = Early_Stop.average_GL(evaluation["validation accuracy"], stop_parameter, aGL_parameter)
        if to_stop == "stop":
          self.biases = stored_biases
          self.weights = stored_weights
          raise ValueError("End SGD: stop parameter exceeded")
          break
        elif to_stop == "ai_credit":
          stored_biases = self.biases
          stored_weights = self.weights 

    if test_data is not None: print ("Test accuracy: {0}%".format(self.evaluate_accuracy(test_data)))

    if monitor or early_stopping != None: return evaluation

  def backprop(self, image, label):
    weighted_inputs = []
    a = image # the input (i.e., the "a"s of the first layer) is the image
    activations = [a]

    nabla_b = np.asarray([np.zeros(b.shape) for b in self.biases])
    nabla_w = np.asarray([np.zeros(w.shape) for w in self.weights])
    # creating the arrays to store the gradient of the cost function

    # Step 1: forward-propagating the data
    for b, w in zip(self.biases, self.weights):
      z = np.dot(w, a) + b
      weighted_inputs.append(z)
      a = self.activation.calculate(np.dot(w, a) + b)
      activations.append(a)

    if self.output_activation.name != self.activation.name:
      # If the output layer different from the other layers:
      activations[-1] = self.output_activation.calculate(weighted_inputs[-1])
    
    # Step 2: computing the output error
    error = self.cost.get_error(self.output_activation, activations[-1], weighted_inputs[-1], label)
    nabla_b[-1] = error
    nabla_w[-1] = np.outer(error, activations[-2])

    # Step 3: computing the errors for the rest of the layers
    l = len(self.layers) - 2
    while l > 0:
      error = np.dot(self.weights[l].T, error) * self.activation.derivative(weighted_inputs[l - 1])
      # computing the error for layer "l + 1" (index "l")
      nabla_b[l - 1] = error
      nabla_w[l - 1] = np.outer(error, activations[l - 1])
      l -= 1

    return nabla_b, nabla_w # returns the gradient of the cost function

  def evaluate_accuracy(self, test_data):
    # returns number correct when the network is evaluated using test data
    test_results = [(self.feed_forward(image), label) for (image, label) in test_data]
    return round((sum(int(np.round_(guess - 0.1) == label) for (guess, label) in test_results)
                  / len(test_data) * 100.0), 2)

  def evaluate_cost(self, test_data):
    # returns cost when the network is evaluated using test data
    return self.cost.calculate([(self.feed_forward(image), label) for (image, label) in test_data])

# Main
def main(data, structure, learning_rate, minibatch_size, num_epochs, cost_function = Cost("mse"),
         body_activation = Activation("sigmoid"), output_activation = Activation("sigmoid"), monitor = False):
  network = Network(structure, cost_function = cost_function, body_activation = body_activation,
                    output_activation = output_activation)
    
  print ("Evaluation without training: {0}%".format(network.evaluate_accuracy(data["validation"])))
  
  print ("""Learning rate: {0}\nMinibatch size: {1}\nNumber of epochs: {2}\nStructure: {3}\nCost function: {4}\
        \nBody activation function: {5}\nOutput activation function: {6}""".format(learning_rate, minibatch_size,
                                                                                   num_epochs, network.layers,
                                                                                   network.cost.name,
                                                                                   network.activation.name,
                                                                                   network.output_activation.name))
  print ("Training in process...")
  
  network.SGD(data["train"], num_epochs, learning_rate, minibatch_size, validation_data = data["validation"],
              test_data = data["test"], monitor = monitor)

  with open("/Users/Ryan/PycharmProjects/neural-networks/credit_analysis/credit_analysis_machine/cam_results/\
            cam_params.txt", "w") as filestream:
    filestream.write("weights: " + str(network.weights) + "\nbiases: "+ str(network.biases))

  return network

# Testing area
def main2(data):
  network = main(data, [15, 20, 20, 1], 2.50, 50, 10, cost_function = Cost("cross-entropy", regularization = "L2",
                                                                           reg_parameter = 0.5), monitor = False)
  # data = credit_analysis_loader.load_data()
  print ("Validation accuracy: " + str(network.evaluate_accuracy(data["train"])) + "%")
  # Below are the final weights and biases
  # network.biases = np.array([[[83.41199168]]])
  # network.weights = np.array([[[41.51300759,  -13.28574762,   23.72629901,
  #                               -5.71168951, -10.2075779, 27.31207391,
  #                               -67.58494164 , 137.76930164, -233.67085598,
  #                               4.55101101,  85.08417989,   52.34560646,
  #                               -15.42126096,  -89.86721409,  153.21697032]]])
  # # Interestingly enough, the above line works-- the structure is not immutable.
  # print (network.evaluate_accuracy(data["train"]))
  return network
  

if __name__ == "__main__":
  data = c.load_data()
  # import mlp as mlp
  # structure = [15, 20, 20, 1]
  # learning_rate = 2.0
  # minibatch_size = 10
  # num_epochs = 10
  # cost = mlp.Cost("cross-entropy", regularization = "L2", reg_parameter = 5.0)
  # output_activation = mlp.Activation("sigmoid")
  # large_weight_initialization = False
  # write = None
  # lr_variation = ["average_improvement", 0.1, 10, 2, 0.002]
  # early_stopping = None
  #
  # net, n = mlp.main(data, structure, learning_rate, minibatch_size, num_epochs,
  #          cost_function = cost,output_activation = output_activation,
  #          large_weight_initialization = large_weight_initialization,
  #          early_stopping = early_stopping, lr_variation = lr_variation,
  #          monitor = False, show = True, write = write)
  # print (net.feed_forward(data["test"][0][0]), data["train"][0][1])
  # print (net.feed_forward(data["validation"][0][0]), data["validation"][0][1])
  # print (net.evaluate_accuracy(data["validation"]))
  #
