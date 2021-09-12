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

![image](https://github.com/siryuon/paper-review/blob/main/PAPER/CV/%5B1%5D%20Deep%20Residual%20Learning%20for%20Image%20Recognition/images/Degradation.png)  
(56-layer가 20-layer model보다 train error, test error 둘 다 높다.)

우선, 간단하게 Identity mapping layer를 추가해보았지만, 좋은 결과가 나오지 않는다는 것을 확인 -> 이 논문에서는 Deepp residual learning framework라는 개념 도입

기존의 방식인 바로 mapping하는 방식을 H(x)라고 한다면, 본 논문에서는 비선형적인 layer F(x) = H(x) - x를 제안함. 이를 H(x)에 대해 고쳐쓰면 H(x) = F(x) + x의 형태를 띔.  
본 논문에서는, residual mapping이 기존의 mapping 방식보다 최적화하기 더 용이할 것이라는 가정을 하고 진행.  

F(x) + x는 Shortcut Connection(Skip)과 동일한데, 이는 하나 또는 하나 이상의 layer를 건너뛰게 만들어줌. 이러한 Identity shortcut connection은 추가적인 parameter도 필요하지 않고, 복잡한 곱셈 연산도 필요하지 않음.  

![image](https://github.com/siryuon/paper-review/blob/main/PAPER/CV/%5B1%5D%20Deep%20Residual%20Learning%20for%20Image%20Recognition/images/Residual%20block.png)  

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

![image](https://github.com/siryuon/paper-review/blob/main/PAPER/CV/%5B1%5D%20Deep%20Residual%20Learning%20for%20Image%20Recognition/images/math1.png)  
![image](https://github.com/siryuon/paper-review/blob/main/PAPER/CV/%5B1%5D%20Deep%20Residual%20Learning%20for%20Image%20Recognition/images/math2.png)  

위의 식 (1), (2)를 설명. 식 1은 F = W_2 * sigma(W_1 * x)를 간소화한 모양이다. 단순 덧셈으로 인해 복잡한 구조와 연산이 필요없다는 것이 본 논문이 제안하는 방법의 핵심이기 때문에
이 로테이션을 통해 직관적으로 보여준다. 여기서 x는 ReLU함수(sigma)를 한 번 통과했고, 여기서 bias 값은 생략했다. 이 때, x와 F의 차원이 동일하도록 맞춰줘야 한다. 따라서, W_1, W_2 등의 Linear Projection을 통해 같은 차원으로 만들어 준다.

### 3-3. Network Architectures (Plain Network, Residual Network)

![image](https://github.com/siryuon/paper-review/blob/main/PAPER/CV/%5B1%5D%20Deep%20Residual%20Learning%20for%20Image%20Recognition/images/net%20comparison.png)

위의 그림에서는 VGGNet, Plain Net, Residual Net을 비교하고 있다.  

#### Plain Net
이 network는 VGGNet을 참고해서 만든 것이다. 모두 동일한 Feature map size를 가지게 하기 위해 layer들은 모두 동일한 수의 filter를 가지게 된다. 또한, feature map size가 절반이 되면,
filter 수는 2배가 되도록 하였다. Conv layer는 3x3의 filter, stride=2로 downsampling, global average pooling layer를 사용했고, 마지막에는 softmax 활성함수를 통해 1000-way fully-connected layer를 적용했다. 최종 layer 수는 34개이다. 이 Network는 VGGNet에 비해 적은 ㄹilter와 복잡도를 가지고 있고, VGGNet의 18%의 연산 밖에 하지 않는다.

#### Residual Net
위의 Plain Net을 기반으로 하지만, 추가적으로 Shortcut connection 개념을 도입한다.  
Identity Shortcut은 input과 outpu을 같은 차원으로 맞춰줘야 한다. 그 후, 차원이 증가할 때 두 가지 방법을 사용한다.
  - 차원을 늘리기 위해 0을 넣어 padding(zero-padding) -> 같은 차원으로 맞춰주기 위해(추가 파라미터 X)
  - 위의 식(2)에서 쓴 방법 사용(Linear projection)
  - 즉, input, output, x, F 모두 같은 차원으로 만들어 주는 것이다.

### 3-4. Implemention
  - Image Resized 224x224
  - Batch Normalization 사용
  - Initialize Weights
  - SGD, mini batch-256
  - Learning rate = 0.1
  - Iteration 60*10^4
  - weight decay = 0.0001
  - momentum = 0.9
  - No DROPOUT

## 4. Experiments

### 4-1. ImageNet Classification

#### Plain Net
Plain Net으로 쌓은 18-layer와 34-layer 모델을 비교하면, 34-layer에서 train error, test error가 높다. -> 높은 gradation problem 발생, plain net의 경우 34-layer의 성능이 더 낮다.
plain net이 BN(Batch Normalization)으로 학습되었기 때문에, non-zero variance를 가지고 최적화가 어려운 이유가 vanishing gradient 때문이 아닌 것을 알 수 있음.

#### Residual Network
구현할 때, 모든 shortcut에 대해 identity mapping을 사용했고, 차원 증가를 위해 zero-padding을 사용했다.

![image](https://github.com/siryuon/paper-review/blob/main/PAPER/CV/%5B1%5D%20Deep%20Residual%20Learning%20for%20Image%20Recognition/images/layer%20comparison.png)

위의 표를 보면 알 수 있는 세 가지가 있다.
  - 18-layer ResNet보다 34-layer ResNet의 성능이 더 좋다. -> layer의 깊이가 증가할 때도 degradation problem을 잘 조절했다는 증거이다.
  - plain net과 비교했을 때 성공적으로 train error를 줄일 수 있었다.
  - 18-layer들끼리 비교했을 때 accuracy는 서로 비슷했으나, 18-layer ResNet의 수렴 속도가 더 빨랐다. -> ResNet이 SGD를 사용한 최적화 작업이 더 쉽다는 것을 알 수 있다.

#### Identity vs Projection Shortcuts
training을 할 때, identity shortcut을 돕기 위해 parameter-free 방법을 사용하는데, 이를 projection shortcut과 비교해본다.

다음 세 가지를 비교했다.
  - 차원 증가를 위해 zero-padding 사용
  - 차원 증가를 위해 projection shortcuts를 사용
  - 모든 shortcut들이 projection인 경우

본 논문의 결과. 성능은 3 > 2 > 1 순이다.  
1은 residual learning이 이루어지지 않기 때문에 2보다 성능이 낮고, projection shortcut에 사용된 extra parameters들 때문에 3이 2보다 성능이 더 좋다.  
하지만, 이런 작은 차이는 주요 문제가 projection shortcut때문이 아니라는 것을 뜻한다. 따라서 본 논문에서는, 3번이 성능이 제일 좋았으나 model이 더욱 복잡해지기 때문에 사용하지 않았다.

#### Deeper Bottleneck Architectures
training 시간을 생각해 모델의 구조를 bottleneck design을 사용한다. 각각 Residual Function(F)를 사용하는데 3층을 쌓을 때, 1x1 -> 3x3 -> 1x1 순서의 병목모양으로 쌓는다.
-> 이는 기존 구조와 비슷한 복잡성을 가지면서 input과 output의 dimension을 줄여주기 때문에 이렇게 사용한다.

#### 50-layer
기존에 만든 34-layer model에 3-layer bottleneck을 추가해 50-layer ResNet을 만듦.

#### 101-layer and 152-layer ResNets
더 많은 3-layer bottleneck을 추가해 101-layer와 152-layer를 만든다. 이 중 152-layer ResNet은 VGGNet보다 작은 복잡성와 적은 연산을 가져서 유의미한 의미를 가진다. 또한, 이 layer가 많이
쌓인 ResNet모델들은 초기 34-layer ResNet보다 더 성능이 좋다 -> Degradation problem X

#### Comparisons with State-of-the-art Methods
결과적으로 위에 만든 6가지의 모델(ResNet-34(1, 2, 3), -50, -101, -152)을 ensemble 하여 error imf를 3.57%까지 줄이는 데 성공.

###4-2. CIFAR-10 and Analysis
CIFAR-10에도 적용해본 결과를 보여줌. 1000개 이상의 layer에서 성능이 낮아지는데, 이는 overfitting 현상이다. 본 논문에서는 dropout, maxout 등의 regularization 기법이 쓰이지 않기 때문에, 추후 regularization을 통해 이러한 문제를 해결해야 한다.
