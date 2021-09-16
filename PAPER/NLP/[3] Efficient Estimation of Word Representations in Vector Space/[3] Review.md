# Efficient Estimation of Word Representations in Vector Space
 - 저자 : Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean
 - 날짜 : 2013.01.16.
 - 논문 링크 : [[pdf]](https://arxiv.org/pdf/1301.3781.pdf)

## 1. Introduction
기존 NLP에서는 모든 단어를 독립적인 요소(atomic)로 취급했다. 때문에, 단어들 사이의 유사성을 파악하는 것이 불가능했다. 그럼에도 불구하고, 해당 방식이 단순하고 단단한(robust) 특징 등 장점들 가지고 있는 덕분에 해당 방식을 계속 사용했다. 하지만 이런 단순한 방식은 많은 한계점이 존재한다. 최근의 머신러닝 기법의 발전으로 우리는 조금 더 복잡한 모델을 큰 데이터셋을 이용해 학습할 수 있었고, 복잡한 모델은 단순한 모델의 성능을 뛰어넘을 수 있었다. 가장 성공적으로 사용된 방식은, 단어를 distributed repreesentation으로 표현하는 방식이라고 할 수 있다.
본 논문은 큰 데이터셋을 이용해 high-quality의 단어 vector를 생성하는 방식을 제안한다. 기존에 사용되었던 구조들은 큰 dataset에 대해 학습할 수 없었고, 벡터의 크기도 50~100정도 밖에 사용하지 못했다. 본 논문에서는 높은 vector연산 정확도를 제공할 수 있는 vector 학습 구조에 대해 제안한다. Neural Network Language Model(NNLM)은 기존에 제안된 구조로, 단어 VECTOR를 학습할 수 있다. NNLM은 하나의 은닉층만 이용해 전체 학습을 가능케 한다.

## 2. Model Architectures
다양한 구조의 성능을 비교하기 위해 computational complexity를 전체 모델을 학습하기 위해 필요한 parameter 수로 정의한다. 본 논문에서 수행하고자 하는 것은 정확도를 높이면서 computational complexity를 낮추고자 한다. Training Complexity(O)는 E x T x Q로 정의한다. 여기서 E:학습 epoch 수, T:전체 training set의 단어 수, Q: 각 모델에 대해 별도로 정의된다.


