# 總體經驗模態分解結合長短期記憶

流程：  
eemd.py -> preprocessing.py -> model.py -> main.py -> retrain.py <-> Predict.py  
  
|檔案             | 內容                                   | 
|-----------------|----------------------------------------|
|eemd.py          | 總體經驗模態分解                       |  
|preprocessing.py | 資料預處理，標準化、切割、打亂         |  
|model.py         | 模型架構及正向傳播流程                 |  
|main.py          | 進行訓練，並儲存模型                   |  
|retrain.py       | 透過修改各項超參數，對模型再次進行訓練 |  
|Predict.py       | 預測未來數天，儲存圖表及誤差數據       |  
  
結論：  
單天預測  
![Mode00](https://github.com/10873028/EEMD-LSTM/blob/master/train/predict_mode00.jpeg?raw=true)  
![Mode01](https://github.com/10873028/EEMD-LSTM/blob/master/train/predict_mode01.jpeg?raw=true)  
![Mode02](https://github.com/10873028/EEMD-LSTM/blob/master/train/predict_mode02.jpeg?raw=true)  
![Mode03](https://github.com/10873028/EEMD-LSTM/blob/master/train/predict_mode03.jpeg?raw=true)  
![Mode04](https://github.com/10873028/EEMD-LSTM/blob/master/train/predict_mode04.jpeg?raw=true)  
![Mode05](https://github.com/10873028/EEMD-LSTM/blob/master/train/predict_mode05.jpeg?raw=true)  
![Mode06](https://github.com/10873028/EEMD-LSTM/blob/master/train/predict_mode06.jpeg?raw=true)  
![Mode07](https://github.com/10873028/EEMD-LSTM/blob/master/train/predict_mode07.jpeg?raw=true)  
![Mode08](https://github.com/10873028/EEMD-LSTM/blob/master/train/predict_mode08.jpeg?raw=true)  
![Mode09](https://github.com/10873028/EEMD-LSTM/blob/master/train/predict_mode09.jpeg?raw=true)  
![Mode10](https://github.com/10873028/EEMD-LSTM/blob/master/train/predict_mode10.jpeg?raw=true)  
連續多天預測  
![Mode00](https://github.com/10873028/EEMD-LSTM/blob/master/test/mode00-future299.jpeg?raw=true)  
![Mode01](https://github.com/10873028/EEMD-LSTM/blob/master/test/mode01-future299.jpeg?raw=true)  
![Mode02](https://github.com/10873028/EEMD-LSTM/blob/master/test/mode02-future299.jpeg?raw=true)  
![Mode03](https://github.com/10873028/EEMD-LSTM/blob/master/test/mode03-future299.jpeg?raw=true)  
![Mode04](https://github.com/10873028/EEMD-LSTM/blob/master/test/mode04-future299.jpeg?raw=true)  
![Mode05](https://github.com/10873028/EEMD-LSTM/blob/master/test/mode05-future299.jpeg?raw=true)  
![Mode06](https://github.com/10873028/EEMD-LSTM/blob/master/test/mode06-future299.jpeg?raw=true)  
![Mode07](https://github.com/10873028/EEMD-LSTM/blob/master/test/mode07-future299.jpeg?raw=true)  
![Mode08](https://github.com/10873028/EEMD-LSTM/blob/master/test/mode08-future299.jpeg?raw=true)  
![Mode09](https://github.com/10873028/EEMD-LSTM/blob/master/test/mode09-future299.jpeg?raw=true)  
![Mode10](https://github.com/10873028/EEMD-LSTM/blob/master/test/mode10-future299.jpeg?raw=true)  
![Total](https://github.com/10873028/EEMD-LSTM/blob/master/test/total-future299.jpeg?raw=true)  

