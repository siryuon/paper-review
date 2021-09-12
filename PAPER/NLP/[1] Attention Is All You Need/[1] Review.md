# Attention Is All You Need
 - 저자 : Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Adian N. Gomex, Lukasz Kaiser, Illia Polosukhin
 - 날짜 : 2017.06.12. (NIPS 2017)
 - 논문 링크 : [[pdf]](https://arxiv.org/pdf/1706.03762.pdf)

## Abstract
성능 좋은 번역 모델은 encoder와 decoder를 포함한 복잡한 recurrent 또는 convolutional 신경망에 기반을 두고 있다. 최고 성능을 내는 모델 또한 attention 메커니즘을 사용해 encode와
decode를 연결한다.  
본 논문에서는, recurrence와 convolution을 전부 제외하고 오직 attention mechanism에만 기반한 **Transformer**라는 간단한 모델을 제안한다. 두 기계번역 실험에서는, 이 모델이 병렬화와
학습시간 감소와 더불어 최고 수준의 품질을 가진다는 것을 보여준다. 이 논문에서, **Transformer**는 크거나 한정된 학습 데이터를 가지고서도 성공적으로 다른 task들에 일반화될 수 있음을 보여준다.

우선 RNN, LSTM, GRU, GPT 등의 논문을 우선적으로 읽어야할 듯 하다.
