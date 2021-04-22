import numpy as np

class StandardScaler:
    def __init__(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean

def CreateMatrix(mode, eemd, timesteps, future):
    x_train, y_train = [], []
    for i in range(timesteps, eemd[0][mode].shape[0]-future):
        x_train.append(eemd[0][mode, i-timesteps:i])
        y_train.append(eemd[0][mode, i-timesteps+1:i+future+1])
    x_val, y_val = [], []
    for i in range(0, 750-future):
        x_val.append(eemd[i][mode, -timesteps:])
        y_val.append(eemd[i+future+1][mode, -timesteps-future:])
    x_test, y_test = [], []
    for i in range(750, 1500):
        x_test.append(eemd[i][mode, -timesteps:])
        y_test.append(eemd[i+1][mode, -1:])
    return np.array(x_train), np.array(y_train), np.array(x_val), np.array(y_val), np.array(x_test), np.array(y_test)

def Shuffle(x_train, y_train):
    index = np.arange(x_train.shape[0])
    np.random.shuffle(index)
    return x_train[index], y_train[index]

if __name__ == '__main__':
    eemd = [np.array([np.arange(i) for _ in range(11)]) for i in range(6088, 7589)]
    print(eemd[0].shape, eemd[-1].shape)
    x_train, y_train, x_val, y_val, x_test, y_test = CreateMatrix(
            mode=0, eemd=eemd, timesteps=60, future=1)

    print(x_train[0])
    print(y_train[0])
    print('------------------------')
    print( x_train[-1])
    print( y_train[-1])
    print('------------------------')
    print(x_val[0])
    print(y_val[0])
    print('------------------------')
    print( x_val[-1])
    print( y_val[-1])
    print('------------------------')
    print( x_test[0])
    print( y_test[0])
    print('------------------------')
    print(x_test[-1])
    print(y_test[-1])


