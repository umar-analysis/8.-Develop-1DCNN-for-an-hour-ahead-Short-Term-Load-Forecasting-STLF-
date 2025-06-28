# trainingmonitor.py

import os
import json
import matplotlib.pyplot as plt
from keras.callbacks import Callback

class TrainingMonitor(Callback):
    def __init__(self, figPath, jsonPath=None, startAt=0):
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt

    def on_train_begin(self, logs=None):
        self.H = {}
        if self.jsonPath and os.path.exists(self.jsonPath):
            with open(self.jsonPath, "r") as f:
                self.H = json.load(f)
                if self.startAt > 0:
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(float(v))
            self.H[k] = l

        if self.jsonPath:
            with open(self.jsonPath, "w") as f:
                json.dump(self.H, f)

        if len(self.H.get("loss", [])) > 1:
            N = len(self.H["loss"])
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(range(0, N), self.H["loss"], label="train_loss")
            plt.plot(range(0, N), self.H["val_loss"], label="val_loss")
            plt.title("Training Loss [Epoch {}]".format(N))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(self.figPath)
            plt.close()
