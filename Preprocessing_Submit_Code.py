# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 13:47:13 2019

@author: Namu PARK
"""

# 필요 패키지 임포트
import numpy as np
import pandas as pd
import codecs

# 데이터 불러오기 
with codecs.open('C:\RecoSys\Data\BX-Users.csv', 'r', encoding='latin-1',errors='ignore') as f:
    user = pd.read_csv(f)
user[:5]

with codecs.open('C:\RecoSys\Data\BX-Books.csv', 'r', encoding='latin-1',errors='ignore') as f:
    book = pd.read_csv(f)
book[:5]

with codecs.open('C:\RecoSys\Data\BX-Book-Ratings.csv', 'r', encoding='latin-1',errors='ignore') as f:
    rating = pd.read_csv(f)
rating[:5]

# User 테이블 전처리
user.describe()

user['country'] = user['Location'].apply(lambda x: x.split(',')[-1].strip())
user[:5]

cnt_loc = dict(user['country'].value_counts())
cnt_loc

# country 정보가 없는 행 존재
user[user['country']=='']

# 미표기 값 재정의
user['country'].iloc[360] = 'usa'
user['country'].iloc[900] = 'usa'
user['country'].iloc[1486] = 'usa'
user['country'].iloc[1649] = 'usa'
user['country'].iloc[1795] = 'canada'
user['country'].iloc[2146] = 'usa'
user['country'].iloc[2789] = 'australia'
user['country'].iloc[3602] = 'france'
user['country'].iloc[3770] = 'usa'


# 유저의 국적 중 상위 국적 추출
cnt_country = dict(user['country'].value_counts())
cnt_country

major_country = []
for x,y in cnt_country.items():
    major_country.append(x)
    if x=='france':
        break
    
major_country

# 상위 국적 여부 체크
def check_country(country):
    if country in major_country:
        return country
    else:
        return 'else'

user['country'] = user['country'].apply(lambda x : check_country(x))

# major_country 에 속하지 않는 유저 수 : 587
len(user[user['country']== 'else'] )

user[:10]

# country 컬럼을 categorize 하여 각각의 컬럼으로 변환 
country_dummy = pd.get_dummies(user['country'])
user_table = pd.concat([user,country_dummy], axis=1)
user_table

# categorize 이후 불필요한 컬럼 삭제
user_table.drop('Location', axis=1, inplace=True)
user_table.drop('country', axis=1, inplace=True)
user_table

# 나이 전처리 : 구간을 나누어 10대, 20대, 30대...의 형태로 변
def age_preprocess(x):
    if x >= 90:
        return 90 // 10
    else:
        return x // 10
    
user_table['Age'] = user_table.Age.apply(lambda x : age_preprocess(x))
user_table[:5]

# 작가 전처리
# 작가: 상위 5개 (작품 수가 많은 사람 기준)
def top_author(x):
    if x in ['Stephen King' , 'Agatha Christie' , 'William Shakespeare', 'Terry Pratchett', 'Jack Canfield'] : 
        return 1
    else:
        return 0
    
book[:1]
book['major_author'] = book['Book-Author'].apply(lambda x : top_author(x))
len(book[book['major_author']==1]) # major author가 쓴 책 개수 : 535개 

# 출판사 전처리
# 출판사: 상위 10개 (평점 정보가 많은 책의 출판사 상위 10)
def top_publisher(x): 
    if x in ['Goldmann' , 'Gallimard' , 'Harlequin', 'Heyne' , 'Mira', 'Pocket', 'L?Â¼bbe' , 
            'LGF', 'Rowohlt Tb.', 'Silhouette'] : 
        return 1
    else:
        return 0
    
book['major_publisher'] = book['Publisher'].apply(lambda x : top_publisher(x))
len(book[book['major_publisher']==1]) # 메이저 출판사 책 5185개

# 출판년도 전처리
# Year of Publication hard coding...
# Dalan-i bihisht 
book['Year-Of-Publication'][0] = 2010
# Is That a Gun in Your Pocket?: Women's Experience of Power in Hollywood
book['Year-Of-Publication'][44340] = 2000
# Help Yourself: Celebrating the Rewards of Resilience and Gratitude
book['Year-Of-Publication'][44353] = 2000
# The Cycling Adventures of Coconut Head: A North American Odyssey
book['Year-Of-Publication'][1] = 1996
# LOOK HOMEWARD ANGEL
book['Year-Of-Publication'][44317] = 1982
# The Royals
book['Year-Of-Publication'][44316] = 1997
# Edgar Allen Poe Collected Poems
book['Year-Of-Publication'][44315] = 2000
# Das große Böse- Mädchen- Lesebuch
book['Year-Of-Publication'][44318] = 2006

book.sort_values(by='Year-Of-Publication')[:5]

# ISBN 전처리
# 정리한 ISBN 파일 불러오기
isbn_group = pd.read_excel('C:\RecoSys\Data\ISBN.xlsx').dropna().reset_index(drop=True)

# ISBN에서 해당 책의 country code 추
def isbn_country(isbn):
    if len(isbn) == 10:
        if isbn[0] in ['0', '1', '2', '3', '4', '5']:
            return isbn_group[isbn_group.Identifier==int(isbn[0])].iloc[0].values[1]
        elif isbn[0]=='7':
            return 'China'
        elif isbn[0]=='8':
            return isbn_group[isbn_group.Identifier==int(isbn[:2])].iloc[0].values[1]
        elif isbn[0]=='9':
            if isbn[1] in ['0', '1', '2', '3', '4']:
                return isbn_group[isbn_group.Identifier==int(isbn[:2])].iloc[0].values[1]
            elif isbn[1] in ['5', '6', '7', '8']:
                return isbn_group[isbn_group.Identifier==int(isbn[:3])].iloc[0].values[1]
        else: 
            return 'unknown'
    else:
        return 'unknown'

book['country'] = book.ISBN.apply(lambda x: isbn_country(x))
pd.DataFrame(book.country.value_counts())


def book_major_country(x):
    major = ['English','German','French','Spain']
    if x in major:
        return x
    else:
        return 'else'

# 컬럼명 변경
book['major_country'] = book['country'].apply(lambda x : book_major_country(x))
book = book[['ISBN', 'Year-Of-Publication', 'major_author', 'major_publisher', 'major_country']]
book.rename(columns={'major_country': 'book_country'}, inplace=True)
book[:5]

# book_country 더미 변수 생성 및 불필요 컬럼 삭제 
country_dummy = pd.get_dummies(book['book_country'])
book_table = pd.concat([book,country_dummy], axis=1)
book_table.drop('book_country', axis=1, inplace=True)
book_table.drop_duplicates(inplace=True)
book_table

user_table.to_csv('C:/RecoSys/Data/user.csv', index=False) # 유저 테이블 저장
book_table.to_csv('C:/RecoSys/Data/book.csv', index=False) # 북 테이블 저장 