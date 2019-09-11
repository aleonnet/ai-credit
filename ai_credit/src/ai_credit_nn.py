"""

"ai_credit_nn.py"

A program that uses keras to build and train a neural network to grade the quality of a loan.

Accuracy: 96.14% (25 epochs)

"""

# Libraries
from ai_credit.src.ai_credit_loader \
  import load_data, parse_inputs
import keras
import numpy as np

# Neural network (using keras)
def build():
  # builds model
  model = keras.Sequential([
    keras.layers.Dense(200, input_shape = (9, ), activation = "relu"),
    keras.layers.Dense(200, input_shape = (200, ), activation = "relu"),
    keras.layers.Dense(7, input_shape = (200, ), activation = "softmax")])

  optimizer = keras.optimizers.SGD(lr = 0.01, decay = 1e-8, momentum = 0.9, nesterov = True)
  model.compile(loss = "categorical_crossentropy", optimizer = "SGD", metrics = ["categorical_accuracy"])
  
  return model

def train(data, model):
  # trains model
  tr_X, tr_Y = data["train"]
  
  model.fit(tr_X, tr_Y, epochs = 50, batch_size = 16, verbose = 2, validation_data = data["validation"])
  
  return model

def evaluate(data, model):
  # tests model
  te_X, te_Y = data["test"]

  evaluation = model.evaluate(te_X, te_Y, verbose = 0)
  print ("Accuracy: {0}%".format(round(evaluation[1] * 100, 2)))

  return evaluation

def test(model, cols, inputs_ = None):
  # tests model on user input
  # REQUIREMENTS: interest rate must be in decimal form (i.e., if
  # interest rate is 30%, enter 0.3 for "int_rate") and everything
  # must be lowercase
  
  if inputs_ is None: inputs_ = [input(col + ": ") for col in cols]
  # WHEN USING, SET inputs_ EQUAL TO THE INPUT FROM WEBPAGE!
  print (inputs_)

  parsed_inputs = parse_inputs(inputs_, cols)

  range_ = ["A", "B", "C", "D", "E", "F"]
  
  return range_[np.argmax(model.predict(parsed_inputs))]

# Testing area
if __name__ == "__main__":
  data, cols, big_data = load_data(0.6)
  print ("Independents:", list(cols))

  print ("Starting training")
  model = train(data, build())
  evaluation = evaluate(data, model)

  error = 0
  for ex, label in zip(data["test"][0], data["test"][1]):
    prediction = np.argmax(model.predict(ex.reshape(1, len(ex))))
    true = np.argmax(label)
    error += abs(prediction - true)
  print ("Average error:", round(error / len(data["test"][0]), 5))

  for i in range(10): print (test(model, cols))
