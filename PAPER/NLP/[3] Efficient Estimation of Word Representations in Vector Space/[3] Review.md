# Efficient Estimation of Word Representations in Vector Space
 - 저자 : Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean
 - 날짜 : 2013.01.16.
 - 논문 링크 : [[pdf]](https://arxiv.org/pdf/1301.3781.pdf)

## 1. Introduction
기존 NLP에서는 모든 단어를 독립적인 요소(atomic)로 취급했다. 때문에, 단어들 사이의 유사성을 파악하는 것이 불가능했다. 그럼에도 불구하고, 해당 방식이 단순하고 단단한(robust) 특징 등 장점들 가지고 있는 덕분에 해당 방식을 계속 사용했다. 하지만 이런 단순한 방식은 많은 한계점이 존재한다. 최근의 머신러닝 기법의 발전으로 우리는 조금 더 복잡한 모델을 큰 데이터셋을 이용해 학습할 수 있었고, 복잡한 모델은 단순한 모델의 성능을 뛰어넘을 수 있었다. 가장 성공적으로 사용된 방식은, 단어를 distributed repreesentation으로 표현하는 방식이라고 할 수 있다.
본 논문은 큰 데이터셋을 이용해 high-quality의 단어 vector를 생성하는 방식을 제안한다. 기존에 사용되었던 구조들은 큰 dataset에 대해 학습할 수 없었고, 벡터의 크기도 50~100정도 밖에 사용하지 못했다. 본 논문에서는 높은 vector연산 정확도를 제공할 수 있는 vector 학습 구조에 대해 제안한다. Neural Network Language Model(NNLM)은 기존에 제안된 구조로, 단어 VECTOR를 학습할 수 있다. NNLM은 하나의 은닉층만 이용해 전체 학습을 가능케 한다.

## 2. Model Architectures
다양한 구조의 성능을 비교하기 위해 computational complexity를 전체 모델을 학습하기 위해 필요한 parameter 수로 정의한다. 본 논문에서 수행하고자 하는 것은 정확도를 높이면서 computational complexity를 낮추고자 한다. Computational Complexity(O, 훈련 복잡도)는 E x T x Q로 정의한다. 여기서 E:학습 epoch 수, T:전체 training set의 단어 수, Q: 각 모델에 대해 별도로 정의된다.

### 2.1 Feedforward Neural Net Language Model (NNLM)
NNLM모델은 Input, Projection, Hidden, Output layer로 구성되어있다.  
Input layer에서는 n개의 선행 단어들이 one-hot-encoding으로 주어진다. 전체 vocabularyh의 크기를 V라고하면, V크기의 벡터가 주어진다. N개의 Input은 V x D크기의 Projection 행렬에 의해 각각 D차원으로 출력되고, 해당 출력은 또 N x D의 Projection layer에 출력된다. Projection layer의 값은 N x D  x H 크기의 행렬과의 연산을 통해 N x H의 출력을 생성하고, 해당 출력을 H x V의 출력층을 통해 V차원의 벡터를 출력한다. 해당 결과는 Softmax 함수(활성함수)를 통해 확률 분포로 제공되고, target으로 제공된 단어의 index와 가까워지도록 학습된다. 해당 모델의 Computational complexity는 아래와 같다.  
Q = N x D + N x D x H + H x V  
![image](https://github.com/siryuon/paper-review/blob/main/PAPER/NLP/%5B3%5D%20Efficient%20Estimation%20of%20Word%20Representations%20in%20Vector%20Space/images/NNLM2.png)

### 2.2 Recurrent Neural Net Language Model (RNNLM)
RNNLM 모델은 NNLM의 한계를 극복하기 위해 생겨났다. 여기서는 입력의 크기를 제한하지 않고 입력받아 사용할 수 있다. 이론적으로, RNN은 더 복잡한 표현을 학습할 수 있고 Hidden layer를 자기 자신에게 연결시켜 short-term memory의 기능도 보유 중이다. 해당 모델의 Computational complexity는 아래와 같다.  
Q = H x H + H x V  
![image](https://github.com/siryuon/paper-review/blob/main/PAPER/NLP/%5B3%5D%20Efficient%20Estimation%20of%20Word%20Representations%20in%20Vector%20Space/images/RNNLM.png)

### 2.3 Parallel Training of Neural Networks
큰 규모의 데이터를 다루기 위해  **DistBelief**라는 Large-scale distributed framework를 사용했다. 이 Framework는 동일한 모델의 다양한 복사본을 생성해서 동시에 실행시키고, parameter는 한 개의 중앙 서버에서 처리한다. 병렬처리 과정에서는 mini-batch asynchronus gradient descent를 사용했고, Adagrad라는 방식을 사용했다.  
![image](https://github.com/siryuon/paper-review/blob/main/PAPER/NLP/%5B3%5D%20Efficient%20Estimation%20of%20Word%20Representations%20in%20Vector%20Space/images/PARRELL.png)

## 3. New Log-linear Model
본 논문에서 제안하는 새로운 모델은 총 두 단계로 이루어진다. 연속적인 word vector들은 단순한 모델을 이용해 학습되고, 다음으로 그 위에 N-gram NNLM에 분산 단어 표현이 훈련되는 방식이다.

### 3.1 Continuous Bag-of Words Models(CBOW)
CBOW Model은 Hideen layer가 제거되고 모든 단어가 Projection layer를 공유하고 있는 형태의 feedforward NNLM과 유사하다. 모든 단어는 같은 position으로 projection되는데, 이러한 구조를 Bag of Word(BOW)모델이라고 한다. 이러한 모델은 이전에 Projection된 단어들은 이후에 영향을 미치지 못한다. 이러한 문제를 해결하기 위해 우리는 log-linear classifier를 4개의 과거 단어와 4개의 미래 단어를 input으로 사용해 가운데 있는 현재 단어를 훈련해 가장 좋은 성능을 획득한다. 해당 모델의 Computational complexity는 아래와 같다.  
Q = N x D + D x V  
이러한 구조를 기존 BOW모델과 구분하여 CBOW 모델이라고 한다. 이 모델은 연속적인 분산된 문맥을 표현할 수 있다.   
![image](https://github.com/siryuon/paper-review/blob/main/PAPER/NLP/%5B3%5D%20Efficient%20Estimation%20of%20Word%20Representations%20in%20Vector%20Space/images/CBOW.png)

### 3.2 Continuous Skip-gram Model
두 번째 구조는 CBOW와 유사하다. 하지만, 문맥을 기반으로 현재의 단어를 예측하는 대신에 같은 문장 안에서 다른 단어들을 기반으로 단어의 분류를 최대화한다. 단어 사이의 거리가 멀수록 가까운 단어보다 적게 관련되어있기 때문에, 우리는 작은 weight를 거리에 따라 조절하며 훈련할 수 있다. Skip-gram의 computational complexity는 아래와 같다.  
Q = C x (D + D x V)  
C는 단어간 거리의 최대값이다.  
![image](https://github.com/siryuon/paper-review/blob/main/PAPER/NLP/%5B3%5D%20Efficient%20Estimation%20of%20Word%20Representations%20in%20Vector%20Space/images/SKIP-GRAM.png)  

## 4. Results
단순한 대수적인 연산을 통해 유사성을 찾는다... (코사인 거리)

