"""

"cam_loader.py"

Loads the credit approval database (called creditapproval.txt).

Libraries used: numpy
Author: Ryan Park

"""

# Libraries
import numpy as np
import random

# Functions
def vectorize(label):
  vectorized = np.zeros((1, ))
  if label == "+": vectorized[0] = 1.0 # approved
  return vectorized
  # no need for devectorize-- use np.argmax(), dummy!

def convert_categorical(categorical, range_):
  # converts a categorical to an integer, range = [0, 1]
  to_int = len(range_) - 1
  fractions = [i / to_int for i in range(to_int + 1)]
  return fractions[range_.index(categorical)]

def feature_scale(value, range_):
  # converts a value between any range to [0, 1]-- for sigmoid function!
  if range_ == (0, 1): return value
  # Step 1: subtract minimum from everything
  value -= range_[0]
  # Step 2: divide by range
  range_ = abs(range_[0]) + abs(range_[1])
  value /= range_
  return value
  
def load_file(file, file_length, num_independent, test_to_train):
  big_data = []
  
  ranges = [["b", "a"], [], [], ["u", "y", "l", "t"], ["g", "p", "gg"],
            ["c", "d", "cc", "i", "j", "k", "m", "r", "q", "w", "x", "e", "aa", "ff"],
            ["v", "h", "bb", "j", "n", "z", "dd", "ff", "o"], [], ["t", "f"], ["t", "f"], [], ["t", "f"],
            ["g", "p", "s"], [], []]
  # above is for finding ranges for feature scaling and categorical converting

  with open(file, "r") as filestream:
    for line in filestream:
      pre = line.split(",") # delimiter is comma in the file
      independents = [pre[i] for i in range(len(pre)) if i < num_independent]
      label = vectorize(pre[num_independent].rstrip("\n"))

      if "?" in independents:
        file_length -= 1
        continue # if data is missing, then throw it out

      big_data.append((independents, label))

      # the below loop is for finding ranges for feature scaling
      for var in independents:
        index = independents.index(var)
        try: ranges[index].append(float(var))
        except ValueError: pass
  
  random.shuffle(big_data)
  num_train = int(test_to_train * file_length)
  num_validation = int((file_length - num_train) / 2)
  
  data = {"train": big_data[0:num_train],
          "validation": big_data[num_train:num_validation + num_train],
          "test": big_data[num_validation + num_train:]}

  ranges = [(min(ranges[i]), max(ranges[i])) if (i == 1 or i == 2) or ((i == 7 or i == 10) or (i == 13 or i == 14))
            else ranges[i] for i in range(len(ranges))]

  return data, ranges

def load_data():
  file = "/Users/Ryan/PycharmProjects/neural-networks/credit_analysis/credit_analysis_machine/cam_dataset/\
         cam_dataset.txt"
  parsed, ranges = load_file(file, file_length = 690, num_independent = 15, test_to_train = 0.8)
  data = {"train": [], "validation": [], "test": []}

  for (independents, label) in parsed["train"]:
    processed = []
    for var in independents:
      try: processed.append(feature_scale(float(var), ranges[independents.index(var)]))
      except ValueError: processed.append(convert_categorical(var, ranges[independents.index(var)]))
    data["train"].append((np.array(processed).reshape(len(ranges), 1), label))
  for (independents, label) in parsed["validation"]:
    processed = []
    for var in independents:
      try: processed.append(feature_scale(float(var), ranges[independents.index(var)]))
      except ValueError: processed.append(convert_categorical(var, ranges[independents.index(var)]))
    data["validation"].append((np.array(processed).reshape(len(ranges), 1), label))
  for (independents, label) in parsed["test"]:
    processed = []
    for var in independents:
      try: processed.append(feature_scale(float(var), ranges[independents.index(var)]))
      except ValueError: processed.append(convert_categorical(var, ranges[independents.index(var)]))
    data["test"].append((np.array(processed).reshape(len(ranges), 1), label))

  return data

# Testing area
if __name__ == "__main__":
  data = load_data()
  with open("/Users/Ryan/PycharmProjects/neural-networks/credit_analysis/credit_analysis_machine/cam_results/\
            cam_processed_data.txt", "w") as filestream:
    for key in data.keys():
      for (independents, label) in data[key]:
        filestream.write(str(independents) + "\n")
          # filestream.write("VARIABLE: " + str(var) + "\n")
        filestream.write(str(label) + "\n")
        # filestream.write("LABEL: " + str(label) + "\n\n")
