# CEVAPLAR




#installations
!pip install pymupdf
!pip install trtokenizer

#libraries
from trtokenizer import SentenceTokenizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

#reading PDF
import fitz
pdf_path = "PDFtest (4).pdf"
doc = fitz.open(pdf_path)
text = ""
for page in doc:
    text += page.get_text()
print(text)

#NORMALIZATION \\\

#Sentence based tokenization
tokenizer = SentenceTokenizer()
tokens = tokenizer.tokenize(text)

#adding " " beginning of all sentences
df = pd.DataFrame()
df['sentences'] = tokens

#transform lower and straigh format
df['sentences'] = df['sentences'].str.lower()
df['sentences'] = df['sentences'].str.replace('\n', '')

df = df.applymap(lambda r: f" {r}" if isinstance(r, str) else r)

#removing all punctuations
punctuations = ['.' , ',' , '?' , '’' , '"' , ':' , '!' , ';' , '(' ,')']

for char in punctuations:
  df['sentences']=df['sentences'].str.replace(char,'')

#removing all stopwords
stopwords =['artık','fakat','lakin','ancak','acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç', 'bir şey', 'biz', 'bu', 'çok', 'çünkü', 'da', 'daha', 'de', 'defa', 'diye', 'eğer', 'en', 'gibi', 'hem', 'hep', 'hepsi', 'her', 'hiç', 'için', 'ile', 'ise', 'kez', 'ki', 'kim', 'mı', 'mu', 'mü', 'nasıl', 'ne', 'neden', 'nerde', 'nerede', 'nereye', 'niçin', 'niye', 'o', 'sanki', 'şey', 'siz', 'şu', 'tüm', 've', 'veya', 'ya', 'yani']

for word in stopwords:
  word = " " + word + " "
  df['sentences']=df['sentences'].str.replace(word,' ')

#removing gaps at the start and end of the lines
df = df.applymap(lambda r: r.lstrip() if isinstance(r, str) else r)
df = df.applymap(lambda r: r.rstrip() if isinstance(r, str) else r)

df

#Editing DataFrame
sentence1 = df['sentences'][20]
parts1 = sentence1.split(' ', 4)
df['sentences'][20]=parts1[0]+' '+parts1[1]+' '+parts1[2] +' '+ parts1[3]

sentence2 = df['sentences'][28]
parts2 = sentence2.split(' ', 3)
df['sentences'][27]=df['sentences'][27]+' '+ parts2[0]+' '+parts2[1]
df['sentences'][28]=parts2[2]+' '+parts2[3]

insert_index = 21
empty_row = pd.DataFrame({'sentences': [None]})
df = pd.concat([df.iloc[:insert_index], empty_row, df.iloc[insert_index:]]).reset_index(drop=True)
df['sentences'][21]=parts1[4]

#Modifications
df['sentences'][2]=df['sentences'][2]+' '+df['sentences'][3]
df=df.drop([3])

df['sentences'][8]=df['sentences'][8]+' '+df['sentences'][9]
df=df.drop([9])

df['sentences'][21]=df['sentences'][21]+' '+df['sentences'][22]
df=df.drop([22])

df['sentences'][10]=df['sentences'][10]+' '+df['sentences'][11]
df=df.drop([11])

#Df regeneration
df = df.reset_index(drop=True)

df

# SORULAR

# Questions DataFrame
dq = pd.DataFrame()

# Soruları bir liste olarak tanımlama
questions = [
    'metinin başlığı başlık ne nasıl orman',#0
    'neredeki hangi ormana ali ne biliyordu',#1
    'herkes ne dedi söylerdi söylerlerdi ormanın ne',#2
    'merakı niçin niye neye ağır basıyordu korkusuna gelen neydi hissettiği duygular',#3
    'ne zaman ali nasıl ormana adım attı giriş yaptı',#4
    'nereye',#5
    'orman nasıldı tepki verdi hava yapraklar nasıldı derinlikleri',#6
    'ağacın dibinde ne gördü parlayan küçük ışığı nerede ne zaman karşılaştı',#7
    'yaklaşınca ne gördü taşın üstünde',#8
    'nerede ne yankılandı',#9
    'taş ilk ne dedi bilmek istedi söyledi',#10
    'ali ne yaptı taşı görünce nereye dokundu nasıl',#11
    'ne birdaha yankılandı yankılanan',#12
    'taş ne dedi bilmek istedi söyledi',#13
    'ali arzusu alinin neydi nasıl ne istedi taştan',#14
    'taş ne yaptı aliye oldu noldu',#15
    'uyandığında kendini nerede buldu',#16
    'herkes köylüler nasıldı',#17
    'ne söylüyorlardı köylüler diyordu neyin',#18
    'ali seli görünce ne yaptı tepkisi oldu',#19
    'ali tam olarak seli görünce ne yaptı tepkisi oldu kurtarmak yaptıkları neler nelerdi',#20
    'sonuç olarak ali ne yaptı kimlere yardımcı oldu',#21
    'kahraman kişisi kim kimdi kurtardı kurtaran ne zaman',#22
    'neyi unutmadı hangi anı',#23
    'ali neyi biliyordu',#24
    'ali ne öğrendi çıkardı cesur olmak demek ana fikir konusu konu'#25
]

# Her soruyu DataFrame'e ekleme
for indeks, question in enumerate(questions):
    yeni_satir = pd.DataFrame({'questions': [question]})
    dq = pd.concat([dq.iloc[:indeks], yeni_satir, dq.iloc[indeks:]]).reset_index(drop=True)

jointdf = pd.concat([dq, df], axis=1)

unknown_question = pd.DataFrame({'questions':' ', 'sentences':'Üzgünüm, sorunun yanıtını metinde bulamadım :c'}, index=[24])
jointdf = pd.concat([jointdf,unknown_question], ignore_index=True)

jointdf

# EĞİTİM

#Testing user Input
mesaj = input('Metin hakkinda bir soru sorunuz: ')
while mesaj == '':
  print('Lütfen bir soru giriniz!')
  mesaj = input('Metin hakkinda bir soru sorunuz: ')
else:
  mesajdf = pd.DataFrame({'questions':mesaj, 'sentences':0}, index=[26])
  jointdf = pd.concat([jointdf,mesajdf], ignore_index=True)

  jointdf['questions'] = jointdf['questions'].str.lower()

  #Train-test-split
  train_data, test_data = train_test_split(jointdf, test_size=0.2, random_state=42)
  vectorizer = TfidfVectorizer()
  x = vectorizer.fit_transform(jointdf['questions']).toarray()
  y = jointdf['sentences']

# tahmin = last line of arrayed df: x    ///in other words, the message for test
  tahmin=x[-1].copy()

# x,y: all data except last line
  x=x[:-1]
  y=y[:-1]

  model = LogisticRegression(max_iter=1000)
  model.fit(x,y)
  score=model.score(x,y)

  result = model.predict([tahmin])
  print('Cevap:', str(result), 'Skor:', score)

#Prediction
question = "tüm bu bütün olaylar ne zaman oldu"
question_vec = vectorizer.transform([question])
prediction = model.predict(question_vec)
print(prediction)
