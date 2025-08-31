# -*- coding: utf-8 -*-
import numpy as np

"""
feature_normalization(): feature 정규화 함수
입력: data - feature만 포함하는 numpy array
출력: normal_feature - 정규화된 feature, numpy array
설명: data를 feature 별로 정규화하여 반환한다.
    표준편차가 너무 작은 feature의 경우 최소값으로 설정한다.   
추가 설명: 정규화된 feature는 평균이 0, 표준편차가 1이 되며,
    큰 값을 가지는 feature에 의해 다른 feature들이 
    영향을 받지 않도록 한다.
"""
def feature_normalization(data): # 10 points
    # parameter 
    
    feature_num = data.shape[1]     #data의 열(feature)수
    data_point = data.shape[0]      #data의 행(데이터포인트)수
    
    # you should get this parameter correctly
    normal_feature = np.zeros([data_point, feature_num]) # 정규화된 dataset
    mu = np.zeros([feature_num])    # feature마다 평균
    std = np.zeros([feature_num])   # feature마다 표준편차
    
    # your code here
    
    # feature별 평균, 표준편차 계산해 정규화 수행
    for i in range(feature_num):
        mu[i] = np.mean(data[:, i]) # i번째 feature 평균
        # std[i]는 0에 가까운 경우를 대비해 np.clip으로 보정
        std[i] = np.clip(np.std(data[:, i]), 1e-3, None) # i번째 feature 표준편차
        normal_feature[:, i] = (data[:, i] - mu[i]) / std[i] # i번째 feature 정규화
    # end

    return normal_feature # 각 feature가 정규화된 data 반환

"""
split_data(): train/test data split
입력: data - feature가 포함된 데이터, numpy array
    label - 정답 label, numpy array
    split_factor - train/test를 나누는 기준 인덱스
출력: training_data, test_data, training_label, test_label
설명: 정규화된 데이터를 train/test로 나누는 함수. 
    split_factor 인덱스까지의 데이터는 training data로,
    그 이후의 데이터는 test data로 나눈다.
    label도 같은 방식으로 나눈다.
"""
def split_data(data, label, split_factor):
    return  data[:split_factor], data[split_factor:], label[:split_factor], label[split_factor:]

"""
get_normal_parameter(): 라벨 별 Gaussian Naive Bayes 파라미터 추정
입력: data - feature, numpy array
    label - 정답 label, numpy array
    label_num - species의 종류 수
출력: mu - 각 label(= species)마다 feature 평균, numpy array
    sigma - 각 species마다 feature 표준편차, numpy array
설명: 각 species 별로 feature의 평균과 표준편차를 계산한다.
    이 값은 Gaussian Naive Bayes의 likelihood의 estimated parameter이다.
추가 설명: MLE(최대우도추정)은 관측된 데이터셋이 가정한 확률분포 하에서 
    발생할 가능도가 최대가 되게 하는 파라미터를 추정하는 방법이다.
    본 모델에서는 각 데이터포인트(datapoint)들이 
    동일한 클래스(label = k) 하에서 조건부로 독립이며, 
    각 데이터포인트는 동일한 분포를 따른다고 가정한다.
    또한, 각 feature는 클래스가 주어졌을 때 서로 조건부 독립이라고 가정하므로,
    joint likelihood는 각 feature의 likelihood의 곱으로 표현된다.
    본 과제에서는 각 feature가 Gaussian 분포를 따른다고 가정하므로,
    평균(mu)과 표준편차(sigma)를 추정하기 위해
    클래스별로 해당 feature의 train 데이터 샘플의 평균과 표준편차를 구하면
    이는 Gaussian의 MLE 해와 일치한다.
"""
def get_normal_parameter(data, label, label_num): # 20 points
    # parameter
    feature_num = data.shape[1]
    
    # you should get this parameter correctly    
    mu = np.zeros([label_num,feature_num])  # train dataset의 각 label 각 feature 별 평균
    sigma = np.zeros([label_num,feature_num])   #표준편차

    # your code here

    # species가 같은 데이터를 모아 feature 별 평균과 표준편차 계산
    for k in range(label_num):
        indices = np.where(label == k)[0]    # 클래스 k에 속하는 데이터의 인덱스들, numpy
        class_data = data[indices, :]   #2차원 data에서 k번째 species에 해당하는 데이터만 추출
        mu[k] = np.mean(class_data, axis=0) # 각 species 별 feature 평균 계산
        sigma[k] = np.std(class_data, axis=0) # 각 species 별 feature 표준편차 계산
        
    # end
    return mu, sigma

"""
get_prior_probability(): 사전확률 계산
입력: label - 정답 label, numpy array
    label_num - species의 종류 수
출력: prior - 각 species마다 사전확률, numpy array
설명: 각 label(species)마다 사전확률을 반환한다.
    각 species의 데이터 개수를 전체 데이터 개수로 나눈 값으로 계산한다.
추가 설명: 사전확률은 임의의 데이터(X)가 특정 클래스(Y)에 속할 확률로,
    test(observe) 전에 Y가 발생할 선험적인 믿음을 나타낸다.
    학습 데이터에서 클래스 비율로 계산하는 것 외에도,
    이전 실험 및 연구 결과나, 전문가의 의견으로부터 설정할 수 있다.
"""
def get_prior_probability(label, label_num): # 10 points
    # parameter
    data_point = label.shape[0]             # 전체 데이터개수
    
    # you should get this parameter correctly
    prior = np.zeros([label_num])           # 각 species 별 사전확률 저장 배열
    
    # your code here
    
    # 학습 데이터에서 클래스 비율로 사전확률 계산
    for k in range(label_num):
        prior[k] = np.sum(label == k) / data_point  #라벨 == k은 boolean 배열로 처리되므로, sum() 사용 가능
    #end
        
    return prior

"""
Gaussian_PDF(): Gaussian PDF 값 계산
입력: x - test(observe) 하는 feature, numpy array
    mu - (최적화된) feature 평균, numpy array
    sigma - (최적화된) feature 표준편차, numpy array
출력: pdf - Gaussian 확률밀도함수 값, numpy array
설명: 각 feature의 Gaussian 확률밀도함수 값을 계산하여 반환한다.
추가 설명: 이 함수는 test 단계에서 test data feature x가 주어졌을 때,
    앞서 구한 likelihood의 파라미터인 mu, sigma를
    Gaussian 확률밀도함수에 대입하여,
    이 feature값을 가진 sample이
    각 species에 속할 likelihood 값을 계산한다.
"""
def Gaussian_PDF(x, mu, sigma): # 10 points
    # calculate a probability (PDF) using given parameters
    # you should get this parameter correctly
    sigma = np.clip(sigma, 1e-3, None) # 표준편차가 너무 작아지는 경우 방지
    
    pdf=0
    # your code here

    # 정규분포 확률밀도함수 공식 적용
    pdf = 1 / (np.sqrt(2 * np.pi*sigma**2)) * np.exp(-((x - mu)**2) / (2 * sigma**2))

    # end
    return pdf

"""
Gaussian_Log_PDF(): Gaussian PDF의 로그값 계산
입력: x - test(observe) 하는 feature, numpy array
    mu - (최적화된) feature 평균, numpy array
    sigma - (최적화된) feature 표준편차, numpy array
출력: log_pdf - Gaussian 확률밀도함수의 로그값, numpy array
설명: 각 feature의 Gaussian 확률밀도함수(PDF)의 로그값을 계산하여 반환한다.
추가 설명: Gaussian Naive Bayes에서는 각 feature를 조건부 독립으로
    가정하므로 각 feature의 likelihood를 곱하여 joint likelihood를 구하지만,
    underflow가 발생할 수 있다.
    따라서, 로그를 취한 후 더하는 방식으로 계산한다.
"""
def Gaussian_Log_PDF(x, mu, sigma): # 10 points
    # calculate a probability (PDF) using given parameters
    # you should get this parameter correctly
    log_pdf=0
    sigma = np.clip(sigma, 1e-3, None) # sigma correction
    
    # your code here
    # 정규분포 확률밀도함수 공식의 로그값을 계산
    log_pdf = -0.5 * np.log(2 * np.pi*sigma**2) - ((x - mu)**2) / (2 * sigma**2) 
    # end
    return log_pdf

"""
Gaussian_NB(): log of Gaussian Naive Bayes posterior probability 계산
입력: mu - 각 species마다 feature 평균, numpy array
    sigma - 각 species마다 feature 표준편차, numpy array
    prior - 각 species마다 사전확률, numpy array
    data - test data, numpy array
출력: posterior - 각 species마다의 사후확률, numpy array
설명: 각 species마다 사후확률에 로그값을 계산하여 반환한다.
    각 species의 log likelihood를 Gaussian_Log_PDF를 통해 계산하고,
    log posterior는 로그 prior와 로그 likelihood를 더해서 얻는다.
    여기에 exp()를 취해서 정확한 posterior 값을 얻을 수 있다.
추가설명: numpy에서 극단적으로 큰 값이나 작은 값이 아니라면
    로그,지수함수를 사용해도 확률간 대소관계는 바뀌지 않는다.
    과제 ppt의 "Estimation of probability distribution"
    에서 보이듯이 의미적으로 차이가 없기에 
    구현한 likelihood, posterior는 확률의 로그를 사용하였으나,
    엄밀한 의미의 likelihood와 posterior를 주석에 추가하였다.
"""
def Gaussian_NB(mu, sigma, prior, data):# 40 points
    # parameter
    data_point = data.shape[0]  # 데이터 개수
    label_num = mu.shape[0]     # 클래스 개수
    # you should get this parameter correctly
    likelihood = np.zeros([data_point, label_num])  # 각 datapoint의 3개 species에 대한 log likelihood 저장
    posterior = np.zeros([data_point, label_num])   # 각 datapoint의 3개 species에 대한 log posterior 저장
    ## evidence can be ommitted because it is a constant
    
    # your code here
    # 각 species 별로 log likelihood, log posterior 계산
    # i번째 test data에 대해 k번째 species의 likelihood, posterior, 그들의 로그값 계산
    for i in range(data_point): #각 샘플에 대해 반복
        for k in range(label_num):  #각 클래스 (3가지 species에 대해 반복)
            # 각 feature에 대한 Gaussian Log PDF의 합 (총 6개 feature에 대해 합산)
            likelihood[i, k] = np.sum(Gaussian_Log_PDF(data[i], mu[k], sigma[k]))
            # log posterior = log likelihood + log prior
            posterior[i, k] = likelihood[i, k] + np.log(prior[k])
    #end
    return posterior

"""
    # 엄밀한 의미의 likelihood, posterior를 별도로 구하는 code
    # your code 지우고 사용
    
    # 각 species 별로 log likelihood, log posterior 계산
    log_likelihood = np.zeros([data_point, label_num])  # 가능도 로그 저장
    log_posterior = np.zeros([data_point, label_num])   # 사후확률 로그 저장
    # i번째 test data에 대해 k번째 species의 likelihood, posterior, 그들의 로그값 계산
    for i in range(data_point): #각 샘플에 대해 반복
        for k in range(label_num):  #각 클래스 (3가지 species에 대해 반복)
            # 각 feature에 대한 Gaussian Log PDF의 합 (총 6개 feature에 대해 합산)
            log_likelihood[i, k] = np.sum(Gaussian_Log_PDF(data[i], mu[k], sigma[k]))
            # log posterior = log likelihood + log prior
            log_posterior[i, k] = log_likelihood[i, k] + np.log(prior[k])
            # 가능도와 사후확률을 지수함수로 다시 얻음
            likelihood[i, k] = np.exp(log_likelihood[i, k])
            posterior[i, k] = np.exp(log_posterior[i, k])

"""
    

"""
classifier(): classification using posterior
입력: posterior - 각 species마다 사후확률, numpy array
출력: prediction - 예측된 species, numpy array
설명: 각 species마다 사후확률을 비교하여 가장 큰 값을 가지는 species를
    예측값으로 설정한다.
"""
def classifier(posterior):
    data_point = posterior.shape[0]
    prediction = np.zeros([data_point])
    
    # 각 샘플에 대해 가장 큰 확률을 가지는 species를 예측값으로 설정
    prediction = np.argmax(posterior, axis=1) 

    return prediction
        
"""
accuracy(): accuracy 계산
입력: pred - 예측된 species, numpy array
    gnd - ground truth species, numpy array
출력: accuracy - 정확도, numpy array
    hit_num - 맞춘 데이터 개수, numpy array
설명: 예측값과 정답을 비교하여 정확도를 계산한다.
"""
def accuracy(pred, gnd):
    data_point = len(gnd)
    
    #prediction과 ground truth 비교하여 맞춘 데이터 개수 계산
    hit_num = np.sum(pred == gnd)
    
    return (hit_num / data_point) * 100, hit_num