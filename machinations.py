from Bio import SeqIO
import numpy as np
import configparser, argparse, sys, os
import ast
from tqdm import tqdm
from random import sample
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.metrics import AUC, TruePositives, FalsePositives, TrueNegatives, FalseNegatives, Precision, Recall
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD, Nadam
from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import KFold
import tensorflow as tf
import tensorflow.keras.backend as K

class CyclicLR(tf.keras.callbacks.Callback):
    #This creates a method to control how the learning rate is adapted over cycles
    def __init__(self,base_lr, max_lr, step_size, base_m, max_m, cyclical_momentum):
      self.base_lr = base_lr
      self.max_lr = max_lr
      self.base_m = base_m
      self.max_m = max_m
      self.cyclical_momentum = cyclical_momentum
      self.step_size = step_size
      self.clr_iterations = 0.
      self.cm_iterations = 0.
      self.trn_iterations = 0.
      self.history = {}

    def clr(self):
      cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
      if cycle == 2:
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        return self.base_lr-(self.base_lr-self.base_lr/100)*np.maximum(0,(1-x))
      else:
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0,(1-x))

    def cm(self):
      cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
      if cycle == 2:
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        return self.max_m
      else:
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        return self.max_m - (self.max_m-self.base_m)*np.maximum(0,(1-x))

    def on_train_begin(self, logs={}):
      logs = logs or {}
      if self.clr_iterations == 0:
        K.set_value(self.model.optimizer.learning_rate, self.base_lr)
      else:
        K.set_value(self.model.optimizer.learning_rate, self.clr())
      if self.cyclical_momentum == True:
        if self.clr_iterations == 0:
          K.set_value(self.model.optimizer.momentum, self.cm())
        else:
          K.set_value(self.model.optimizer.momentum, self.cm())

    def on_batch_begin(self, batch, logs=None):
      logs = logs or {}
      self.trn_iterations += 1
      self.clr_iterations += 1
      self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.learning_rate))
      self.history.setdefault('iterations', []).append(self.trn_iterations)
      if self.cyclical_momentum == True:
        self.history.setdefault('momentum', []).append(K.get_value(self.model.optimizer.momentum))
      for k, v in logs.items():
        self.history.setdefault(k, []).append(v)
      K.set_value(self.model.optimizer.learning_rate, self.clr())
      if self.cyclical_momentum == True:
        K.set_value(self.model.optimizer.momentum, self.cm())

def get_model(input_shape, hp):
    #This creates the neural network method. 
    model = Sequential()
    # Convolutional Layers
    #The 2D convolutional layers are the nodes that learn. The range gives the number of layers, 
    #the filters are the number of nodes per layer, the size and stride are the dimensions of each node
    #dropout rates control overfitting
    for i in range(len(hp["filters"])):
      model.add(Conv2D(filters = hp["filters"][i][0], kernel_size = (hp["size"],4 if i == 0 else hp["stride"]),
                      activation='relu', kernel_initializer='he_normal',
                      bias_initializer='he_normal', kernel_regularizer = l2(l=hp["l2_reg"]),
                      input_shape=input_shape))
      model.add(Dropout(rate = hp["filters"][i][1]))
    # Pool
    model.add(MaxPooling2D(pool_size=(hp["max_pooling_size"],1), strides=hp["max_pooling_stride"]))
    model.add(Flatten())
    # Linear Layers
    for _ in range(1):
      model.add(Dense(units = hp["dense_filters"], activation = 'relu',
                      kernel_initializer='he_normal', bias_initializer='he_normal',
                      kernel_regularizer = l2(l=hp["l2_reg"])
                      #kernel_constraint = tf.keras.constraints.MaxNorm(max_value=4, axis=0)
                      )
      )
      model.add(Dropout(rate = hp["dropout"]))

    # output layer
    model.add(Dense(units = 1, activation = 'sigmoid',
                  kernel_initializer='he_normal', bias_initializer='he_normal',
                  kernel_regularizer = l2(l=hp["l2_reg"])))
    myoptimizer = SGD(learning_rate=hp["base_lr"], momentum=hp["max_m"])
    #myoptimizer = Nadam(lr=hp["base_lr"])
    model.compile(optimizer=myoptimizer,
                  loss="binary_crossentropy",
                  #loss=f1_weighted,
                  metrics=[
                          TruePositives(name='TP'),FalsePositives(name='FP'),
                          TrueNegatives(name='TN'), FalseNegatives(name='FN'),
                          AUC(name='auroc', curve='ROC'), AUC(name='auprc', curve='PR')
                          ]
                  )
    model.summary()
    return model

def train_model_clr(x_train, y_train, x_valid, y_valid, hp):
    #have not fully incorporated the class imbalance
    # dealing w/ class imbalance
    total = y_train.shape[0]
    weight_for_0 = (1 / np.sum(y_train==0))*(total)/2.0
    weight_for_1 = (1 / np.sum(y_train==1))*(total)/2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
    # An epoch is calculated by dividing the number of training images by the batchsize
    iterPerEpoch = y_train.shape[0] / hp["batch"] # unit is iter
    # number of training iterations per half cycle.
    # Authors suggest setting step_size = (2-8) x (training iterations in epoch)
    iterations = list(range(0,round(y_train.shape[0]/hp["batch"]*hp["epoch"])+1))
    step_size = len(iterations)/(hp["n_cycles"])
    #
    # set cyclic learning rate
    
    scheduler =  CyclicLR(base_lr=hp["base_lr"],
                max_lr=hp["max_lr"],
                step_size=step_size,
                max_m=hp["max_m"],
                base_m=hp["base_m"],
                cyclical_momentum=True)

    if hp["restart"]:
        model = load_model(os.path.join(hp["working_folder"], hp["model_name"]))
    else:
        model = get_model(x_train.shape[1:], hp)
        
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=(os.path.join(hp["working_folder"],"model_output",hp["model_name"]+"weights.h5")),
                                                                   save_weights_only=False,
                                                                   monitor='val_loss',
                                                                   mode='min',
                                                                   save_best_only=True)
    hist = model.fit(x_train,
                    y_train,
                    batch_size = hp["batch"],
                    epochs = hp["epoch"],
                    verbose = 1,
                    validation_data=(x_valid, y_valid),
                    callbacks = [scheduler,
                                model_checkpoint_callback])
    return model, scheduler, hist, [], []

def onehot_seq(seq, largest):
    #code sequences into a matrix with 4 states which is 1 for that particular nucleotide
    letter_to_index =  {'A':0, 'a':0,
                        'C':1, 'c':1,
                        'G':2, 'g':2,
                        'T':3, 't':3}
    to_return = np.zeros((largest,4), dtype=object)
    for idx,letter in enumerate(seq):
        if letter in letter_to_index:
            to_return[idx,letter_to_index[letter]] = 1
    return np.expand_dims(to_return, axis=2)


def encode_sequence(fasta, largest, pos="+"):
    #Encode sequence into one-hot format and create a positive or negative designation for the matrix
    x = []
    for seq in tqdm(SeqIO.parse(fasta, "fasta")):
        if len(seq) > largest:
            seq = seq[0:largest]
        x.append(onehot_seq(seq, largest))
        x.append(onehot_seq(seq.reverse_complement(), largest))
    x = np.asarray(x).astype(np.float32)
    if pos == "+":
        y = np.ones(len(x))
    elif pos == "-":
        y = np.zeros(len(x))
    y = np.asarray(y).astype(np.float32)
    #print(x.shape, y.shape)
    return x, y

def load_data(hp):
    #Load fasta files
    if os.path.exists(os.path.join(hp["working_folder"],"model_output", hp["model_name"]+"train_fasta.npy")) and not hp["rewrite"]:
        print("output folder model_output already contains loaded data, loading that (to force use --force)")
        print("loading training data...")
        x_train = np.load(os.path.join(hp["working_folder"],"model_output", hp["model_name"]+"train_fasta.npy"), allow_pickle=True)
        y_train = np.load(os.path.join(hp["working_folder"],"model_output", hp["model_name"]+"train_labels.npy"), allow_pickle=True)
        print(f'There are {np.count_nonzero(y_train == 1)} positives and {np.count_nonzero(y_train == 0)} negatives.')
        print("loading validation data...")
        x_valid = np.load(os.path.join(hp["working_folder"],"model_output", hp["model_name"]+"val_fasta.npy"), allow_pickle=True)
        y_valid = np.load(os.path.join(hp["working_folder"],"model_output", hp["model_name"]+"val_labels.npy"), allow_pickle=True)
        print(f'There are {np.count_nonzero(y_valid == 1)} positives and {np.count_nonzero(y_valid == 0)} negatives.')
    else:
        # encode and save
        print("loading training data...")
        (x1_train, y1_train) = encode_sequence(hp["train_pos"], hp["length"])
        (x2_train, y2_train) = encode_sequence(hp["train_neg"], hp["length"], pos="-")
        x_train = np.concatenate((x1_train, x2_train))
        y_train = np.concatenate((y1_train, y2_train))
        print(f'There are {x1_train.shape[0]} positives and {x2_train.shape[0]} negatives.')
        print("loading validation data...")
        (x1_valid, y1_valid) = encode_sequence(hp["valid_pos"], hp["length"])
        (x2_valid, y2_valid) = encode_sequence(hp["valid_neg"], hp["length"], pos="-")        
        x_valid = np.concatenate((x1_valid, x2_valid))
        y_valid = np.concatenate((y1_valid, y2_valid))
        print(f'There are {x1_valid.shape[0]} positives and {x2_valid.shape[0]} negatives.')
        np.save(os.path.join(hp["working_folder"],"model_output", hp["model_name"]+"train_fasta.npy"), x_train, allow_pickle=True)
        np.save(os.path.join(hp["working_folder"],"model_output", hp["model_name"]+"train_labels.npy"), y_train, allow_pickle=True)
        np.save(os.path.join(hp["working_folder"],"model_output", hp["model_name"]+"val_fasta.npy"), x_valid, allow_pickle=True)
        np.save(os.path.join(hp["working_folder"],"model_output", hp["model_name"]+"val_labels.npy"), y_valid, allow_pickle=True)
    return x_train, y_train, x_valid, y_valid

def split_data(fasta, name, ratio):
    #Split data into positive and negative sets based on a random assignment following a ration of positive:negative data
    total_lines = len([1 for line in open(fasta) if line.startswith(">")])
    nums = range(0, total_lines)
    train = sample(nums, int(total_lines * (1-ratio)))
    x = 0

    fasta_sequences = SeqIO.parse(open(fasta),'fasta')
    with open(os.path.join(hp["working_folder"],"model_output",name+"_train_fasta.fa"), "w") as train_file, \
         open(os.path.join(hp["working_folder"],"model_output",name+"_valid_fasta.fa"), "w") as valid_file:
        for fasta in fasta_sequences:
            if x in train:
                train_file.write('>{}\n'.format(fasta.id))
                for i in range(0, len(fasta.seq), 60):
                    train_file.write('{}\n'.format(str(fasta.seq)[i:i + 60]))
            else:
                valid_file.write('>{}\n'.format(fasta.id))
                for i in range(0, len(fasta.seq), 60):
                    valid_file.write('{}\n'.format(str(fasta.seq)[i:i + 60]))
            x += 1
    train_file.close()
    valid_file.close()

def main(hp):
    K.clear_session()
    if hp["train"]:
        # load data
        x_train, y_train, x_valid, y_valid = load_data(hp)
        model, clr, hist, acc_per_fold, loss_per_fold = train_model_clr(x_train, y_train,
                                                                      x_valid, y_valid,
                                                                      hp)
        model.save(os.path.join(hp["working_folder"], hp["model_name"]))
        
    if hp["eval"]:
        model = load_model(os.path.join(hp["working_folder"], hp["model_name"]))
        for key in hp["eval_fasta"].keys():
            evals = encode_sequence(hp["eval_fasta"][key], model.layers[0].input_shape[1])
            scores = model.predict(evals[0]).ravel()
            fasta_sequences = SeqIO.parse(open(hp["eval_fasta"][key]),'fasta')
            with open(os.path.join(hp["working_folder"],"model_output",hp["model_name"]+"_"+key+"_prob.txt"), "w") as prob_file:
                x = 0
                for fasta in fasta_sequences:
                    prob_file.write('{}\t{}\n'.format(fasta.id,scores[x]))
                    prob_file.write('{}\t{}\n'.format(fasta.id+"_RC",scores[x+1]))
                    x += 2
                prob_file.close()
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Config File")
    parser.add_argument("--train", action='store_true', help="Train Model")
    parser.add_argument("--eval", action='store_true', help="Evaluate Model")
    parser.add_argument("--split", action='store_true', help="Split data for training and validation")
    parser.add_argument("--cont", action='store_true', help="Continue training")
    parser.add_argument("--force", action='store_true', help="Force Rewrite")
    
    options, args = parser.parse_known_args()
    
    config = configparser.ConfigParser()
    config.read(options.config)
    
    hp = {"base_lr" : float(config["scheduler"]["base_lr"]), "max_lr" : float(config["scheduler"]["max_lr"]), 
            "base_m": float(config["scheduler"]["base_m"]), "max_m": float(config["scheduler"]["max_m"]),
            "l2_reg" : float(config["scheduler"]["l2_reg"]), "n_cycles" : float(config["scheduler"]["n_cycles"]),
            "filters" : ast.literal_eval(config["model"]["2D_conv_filters"]), "size" : int(config["model"]["2D_conv_size"]), "stride" : int(config["model"]["2D_conv_stride"]),
            "max_pooling_size" : int(config["model"]["max_pooling_size"]), "max_pooling_stride" : int(config["model"]["max_pooling_stride"]),
            "dense_filters" : int(config["model"]["connected_dense_filters"]), "dropout" : float(config["model"]["connected_dense_dropout"]), 
            "train" : False, "eval" : False, 
            "rewrite" : options.force, "restart" : options.cont,
            "working_folder" : config["options"]["working_folder"]}
    
    if not options.train and not options.eval:
        print("You did not choose --train or --eval, will do both")
        options.train = True
        options.eval = True
        
    if not os.path.exists(os.path.join(hp["working_folder"],"model_output")):
        os.makedirs(os.path.join(hp["working_folder"],"model_output"))

    if options.train:
        hp["train"]=True
        hp["train_pos"] = config["data"]["train_pos"]
        hp["train_neg"] = config["data"]["train_neg"]
        hp["valid_pos"] = config["data"]["valid_pos"]
        hp["valid_neg"] = config["data"]["valid_neg"]
        hp["batch"] = int(config["options"]["batch"])
        hp["epoch"] = int(config["options"]["epoch"])
        hp["length"] = int(config["options"]["max_seq_len"])
        hp["ratio"] = float(config["options"]["valid_ratio"])
        hp["model_name"] = config["options"]["name"]
         
    if options.split:
        split_data(config["data"]["fasta_pos"], "pos", hp["ratio"])
        split_data(config["data"]["fasta_neg"], "neg", hp["ratio"])
        hp["train_pos"] = os.path.join(hp["working_folder"],"model_output","pos_train_fasta.fa")
        hp["train_neg"] = os.path.join(hp["working_folder"],"model_output","pos_valid_fasta.fa")
        hp["valid_pos"] = os.path.join(hp["working_folder"],"model_output","neg_train_fasta.fa")
        hp["valid_neg"] = os.path.join(hp["working_folder"],"model_output","neg_valid_fasta.fa")
    
    if options.eval:
        hp["eval"]=True
        hp["working_folder"] = config["options"]["working_folder"]
        hp["eval_fasta"] = {}
        for x in config["eval"]:
            hp["eval_fasta"][x] = config["eval"][x]
        hp["model_name"] = config["options"]["name"]
       
    main(hp)
