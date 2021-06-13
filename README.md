googld slid連結:
https://docs.google.com/presentation/d/1viaFwOaX9YonrFx6eQ3Hl2nSnD1uFAgbX6YtFzk663U/edit?usp=sharing

---------------程式操作說明-----------------

preprocsssing.py為資料前處理程式

如果要執行，請將地13行程式碼path指向sales_train.csv的位置

並且資料處理完後，會將資料存成array的形式，名稱叫data_array

--------------------------------------------

train.py為模型訓練程式

請將第12行程式碼指向data_array.npy的前處理完的資料

並且github已經有data_array的壓縮檔可以載了，請直接使用他，這樣比較快

每次訓練完一個shop，會將模型儲存再同個資料夾下面(總共大概佔48G)

-------------------------------------------
