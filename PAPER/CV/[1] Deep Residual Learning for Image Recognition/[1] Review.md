# Deep Residual Learning for Image Recognition
 - 저자 : Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
 - 날짜 : 2015.12.10.
 - 논문 링크 : [[pdf]](https://arxiv.org/pdf/1512.03385.pdf)

## Abstract
 - 딥러닝에서 Neural Network가 깊어질수록 성능은 더 좋아지지만, train이 더 어려워진다는 것은 널리 알려진 사실이다.
 - 따라서 본 논문에서는 잔차(Residual)을 사용한 잔차학습(Residual learning framework)를 이용해 깊은 신경망에서도 train이 쉽게 이루어질 수 있는 것을 보이고 방법론을 제시한다.
 - 함수를 새로 만드는 대신, Residual function을 learning에 사용하는 것으로 layer를 재구성한다.
 - 본 논문은 Empirical evidence showing 방법으로 residual을 이용한 최적화를 더 쉽게하는 법, accuracy를 증가시키고 더 깊게 layer를 쌓는 법에 초점을 둔다.
 - 결론적으로, 152개의 layer를 쌓아 기존의 VGGNet보다 좋은 성능을 내면서 동시에 복잡성을 줄였다.

## 1.Introduction
layer를 깊게 쌓을수록 network의 성능은 더 좋아질까? -> problem of vanishing/exploding gradients 문제 발생  
본 논문에서 중점적으로 다루는 문제는 Degradation problem. -> Network가 깊어질수록 accuracy가 떨어지는 문제. 이 문제는 overfitting(과적합)의 문제가 아님.  
이 논문에서는, Degradation의 문제는 layer를 깊게 쌓을수록 최적화가 복잡해지기 때문에 일어나는 부작용으로 해석.  

우선, 간단하게 Identity mapping layer를 추가해보았지만, 좋은 결과가 나오지 않는다는 것을 확인 -> 이 논문에서는 Deepp residual learning framework라는 개념 도입

기존의 방식인 바로 mapping하는 방식을 H(x)라고 한다면, 본 논문에서는 비선형적인 layer F(x) = H(x) - x를 제안함. 이를 H(x)에 대해 고쳐쓰면 H(x) = F(x) + x의 형태를 띔.  
본 논문에서는, residual mapping이 기존의 mapping 방식보다 최적화하기 더 용이할 것이라는 가정을 하고 진행.  

F(x) + x는 Shortcut Connection(Skip)과 동일한데, 이는 하나 또는 하나 이상의 layer를 건너뛰게 만들어줌. 이러한 Identity shortcut connection은 추가적인 parameter도 필요하지 않고, 복잡한 곱셈 연산도 필요하지 않음.  

Input인 x가 model F(x)를 거치고, 여기에 자신(identity)인 x가 더해져서 output으로 F(x) + x가 나온다.  

이후로는 실험적인 방법을 통해 degradation 문제를 보이고 본 논문이 제안한 방법을 평가한다.

본 논문의 목표는 다음 두가지이다.
  - Plain한 net들과 다르게 Residual Net이 더 쉽게 최적화되는 것을 보이는 것.
  - Residual Net이 더 쉽게 accuracy를 높이는 것을 보이는 것.

## 2. Related Work: Residual Representation, Shortcut Connections
본 논문에서 사용하는 Residual Representation과 Shortcut Connection의 개념에 대한 논문들과 이들과 비교했을 때 본 논문이 제안하는 ResNet이 가지는 장점들에 대한 단락

## 3. Deep Residual Learning

### 3-1. Residual Learning
H(x)를 기본 mapping으로 간주하고, x를 input으로 넣는다고 가정했을 때, 다수의 비선형적 layer가 복잡한 함수를 점근적으로 근사할 수 있다고 가정하면 잔차함수인 H(x) - x를 무의식적으로(?) 근사할 수 있다는 가설과 같다.  
-> 복잡한 함수를 여러개의 비선형 layer가 근사시킬 수 있다면, 잔차함수도 근사시킬 수 있음. (잔차 = 예측값 - 실제값)이기 때문  
H(x) = F(x) + x <- 이 형태의 잔차함수는 형태에 따라서 학습의 용이함이 달라짐. F(x)에 대해 정리하는 것보다 H(x)에 대해 정리하는 것이 학습에 용이함.

### 3-2. Identity Mapping by Shortcuts
