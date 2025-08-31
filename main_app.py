# -*- coding: utf-8 -*-
import numpy as np      
import pandas as pd     
import HWJ as util      #앞서 구현한 함수들이 담긴 파일
import seaborn as sns   #penguins.csv dataset을 불러오기 위한 라이브러리

# penguins dataset 불러오기
df = sns.load_dataset('penguins')       #to dataframe  

# 결측치(NaN) 포함 데이터 행 제거
df = df.dropna() 

# label encoding; 문자열로 되어있는 범주형 변수들을 숫자값으로 변환
df['island'] = df['island'].astype('category').cat.codes    #Biscoe:0, Dream:1, Torgersen:2
df['sex'] = df['sex'].astype('category').cat.codes          #FEMALE:0, MALE:1
df['species'] = df['species'].astype('category').cat.codes  #Adelie:0, Chinstrap:1, Gentoo:2

# 데이터 shuffle하고 새로운 인덱스 부여
# train/test split 전 데이터 편향 방지
df =  df.sample(frac=1).reset_index(drop=True) 

# 입력 데이터와 정답 라벨 준비
features = ['island', 'bill_length_mm', 'bill_depth_mm', 
                'flipper_length_mm', 'body_mass_g', 'sex']
data = df[features].to_numpy()      #6개 feature 열만 추출한 입력데이터, numpy 변환
label = df['species'].to_numpy()    #label 열만 추출한 정답라벨, numpy 변환

label_num = max(label) + 1      # species의 종류 수

# feature normalization
normalled_data = util.feature_normalization(data)   #defined in HWJ.py
print("Mean:", np.mean(data, 0)) #원래 평균 출력
print("normalled_Mean:", np.mean(normalled_data, 0))    #정규화된 평균 출력

data = normalled_data  #정규화된 데이터로 교체

# spilt data for testing
# 250개는 training data, 나머지 50개는 test data
split_factor=250    
training_data, test_data, training_label, test_label = util.split_data(data, label, split_factor)

# get train parameter of nomal distribution and prior probability
mu, sigma = util.get_normal_parameter(training_data, training_label, label_num) #likelihood parameter 계산
prior = util.get_prior_probability(training_label, label_num) #클래스 비율로 사전확률 계산

# train end

# test

# get postereior probability of each test data based on likelihood and prior
posterior = util.Gaussian_NB(mu, sigma, prior, test_data) #prior와 likelihood parameter로 posterior계산

# classification using posterior
prediction = util.classifier(posterior) # posterior에서 가장 큰 확률을 가지는 class를 예측값으로 설정

# get accuracy
acc, hit_num = util.accuracy(prediction, test_label) # 예측값과 정답라벨 비교하여 정확도 계산

# print result
print(f'accuracy is {acc}% !')
print(f'the number of correct prediction is {hit_num} of {len(test_label)} !')