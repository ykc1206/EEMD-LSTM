import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from preprocess import StandardScaler, CreateMatrix
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

timesteps = 30
future = int(input('future:'))

eemd = [np.load(f'EEMD/EEMD{i:04d}.npy').reshape(11, -1) for i in range(1501)]
data = pd.read_csv('bdi.csv').fillna(method='ffill')['bdi'].values[-750:]
predict = 0
for mode in range(11):
    x_train, y_train, x_val, y_val, x_test, y_test = CreateMatrix(
            mode, eemd, timesteps, future)
    norm = StandardScaler(y_train)
    x_test = norm.transform(x_test)
    inputs = np.array([x_test[start] for start in range(750-future)])
    inputs = torch.FloatTensor(inputs).to(device)
    model = torch.load(f'models/mode{mode}.pt').to(device)
    model.eval()
    with torch.no_grad():
        pred = model(inputs, future, None, 0)[:, -1-future:]
        pred = norm.inverse_transform(pred.cpu().numpy())
    predict += pred

    plt.figure(figsize=(16, 8))
    plt.plot(y_test)
    for start in range(750-future):
        plt.plot(range(start, start+future+1), pred[start], ':')
    plt.title(f'mode:{mode}-future:{future}')
    plt.savefig(f'test/mode{mode}-future{future}.jpg')
    plt.show()
    plt.close()

errors = 0
plt.figure(figsize=(16, 8))
plt.plot(data)
for start in range(750-future):
    error = metrics.mean_absolute_error(data[start:start+future+1], predict[start])
    errors += error
    plt.plot(range(start, start+future+1), predict[start], ':')
plt.title(f'future:{future}-error:{errors / (750-future)}')
plt.savefig(f'test/total-future{future}.jpg')
plt.show()
plt.close()

