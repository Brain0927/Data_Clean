import pandas as pd            #匯入pandas 函式庫
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

df1 = pd.read_csv("NASA.csv")  # read_some file formats 的type為dataframe
df1=df1[0:]                    # 從檔案第0筆開始

df1 = pd.DataFrame(df1)
print("未清洗資料")
print("-----方法5  文字 轉 數字------")
df1['hazardous'] = df1['hazardous'].rank(method='dense', ascending=False).astype(int)
df1['hazardous'] = df1['hazardous'].replace(1, 0)
df1['hazardous'] = df1['hazardous'].replace(2, 1)
df1['sentry_object'] = df1['sentry_object'].rank(method='dense', ascending=False).astype(int)
df1['sentry_object'] = df1['sentry_object'].replace(1, 0)
df1['orbiting_body'] = df1['orbiting_body'].rank(method='dense', ascending=False).astype(int)


print("已清洗資料")#  如果是空的 補上0
from pandas import ExcelWriter        # 匯入 excel writer
writer = ExcelWriter('NASA(清洗後).xlsx', engine='xlsxwriter')      # 清洗過的資料轉成 NASA(清洗過).xlsx
df1.to_excel(writer, sheet_name='sheet1')                          # 分頁欄位的名稱為sheet1
writer.save() #儲存

