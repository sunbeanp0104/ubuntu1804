#-*- codeing:utf-8 -*- 
import pymysql
from datetime import datetime 
import pandas as pd 
import re
from konlpy.tag import *
from collections import Counter


komoran = Komoran()
cut=2000
st=0
ed=500000
FS_list=['사회','연예','스포츠','경제','정치']

class dbConfig:
    HOST = "111.222.333.444"
    PORT = 3306
    USER = 'trident'
    PWD = 'tridentDB'
    NAME = 'trident'

db = pymysql.connect(host=dbConfig.HOST, 
                         port=dbConfig.PORT, 
                         user=dbConfig.USER, 
                         passwd=dbConfig.PWD, 
                         db=dbConfig.NAME, 
                         charset='utf8', 
                         autocommit=True) 
cursor = db.cursor()

for FS in FS_list:
    for i in range(st,ed,cut):
        start=datetime.now()
        sql="SELECT TitleKorea, ContentKorea,  FeedClass \
            FROM rss_news \
            WHERE FeedClass = '"+str(FS)+"' \
            ORDER BY UpdateDate DESC \
            LIMIT "+str(i)+" , "+str(cut)+"  "
        cursor.execute(sql) 

        if i == st and FS == FS_list[0]:
            data = cursor.fetchall() 
        else:
            data= data + cursor.fetchall() 
        end=datetime.now()
        print("read data from db",str(i),"  collection time : ",str(end-start))

db.close() 
print('download data')

df=pd.DataFrame(data)
df.columns=['TitleKorea', 'ContentKorea', 'FeedClass']
cut=5000
textall=list()
s0=datetime.now()
for i in range(0,len(df['ContentKorea']),cut):
    s=datetime.now()
    textall.extend(list(map(lambda text:",".join(komoran.morphs(text)),df['ContentKorea'][i:(i+cut)])))
    print(str(i),"st (",str(i*100/len(df['ContentKorea'])),')\t',"end runtime : ",str(datetime.now()-s),'\t',"누적 : ",str(datetime.now()-s0))
df.index=range(0,len(df))
df['komoran_text']=textall
df=df.drop_duplicates().sample(frac=1).reset_index(drop=True)
#df=pd.read_csv('./data/db_komoran_50.csv')
print('end')

text_filter0=list(map(lambda a:list(filter(lambda x:len(x)>1,a.split(","))),list(df['komoran_text'])))
text_filter=list(map(lambda text:",".join(text),text_filter0))
df["filtered"]=text_filter

df['test']=list(map(lambda x: len(x.split(",")),list(df['filtered'])))
df=df[df['test']>3][['TitleKorea', 'ContentKorea', 'FeedClass','filtered']]
length=min(Counter(list(df['FeedClass'])).values())
df.to_csv('./data/db_komoran_50.csv',index=False)


data_train=pd.DataFrame()
data_dev=pd.DataFrame()
data_test=pd.DataFrame()

for FS in FS_list:
    newdf=df[df['FeedClass']==FS].reset_index(drop=True)
    data_train=pd.concat([data_train,newdf[:round(length*0.7)]])
    data_dev=pd.concat([data_dev,newdf[round(length*0.7):round(length*0.9)]])
    data_test=pd.concat([data_test,newdf[round(length*0.9):]])

data_train.drop_duplicates().sample(frac=1).reset_index(drop=True)
data_dev.drop_duplicates().sample(frac=1).reset_index(drop=True)
data_test.drop_duplicates().sample(frac=1).reset_index(drop=True)

data_train.to_csv('./data/data_train_50.csv',index=False)
data_dev.to_csv('./data/data_dev_50.csv',index=False)
data_test.to_csv('./data/data_test_50.csv',index=False)
