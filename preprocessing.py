import pandas as pd
from itertools import product

train = pd.read_csv("/home/minahwang2001/data/train.csv")

train_data = []

for q,a in list(product([f"질문_{x}" for x in range(1,3)],[f"답변_{x}" for x in range(1,6)])):
    for i in range(len(train)):
        train_data.append(
            "질문: "+ train.at[i,q] + " 답변 : " + train.at[i,a]
        )

pd.DataFrame(train_data).to_csv("/home/minahwang2001/data/train_data.csv",index=False,encoding='utf-8')

