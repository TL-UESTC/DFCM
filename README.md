# Uneven Bi-Classifier Learning for Domain Adaptation

Zhiqi Yu, Jingjing Li, Lei Zhu, Ke Lu, Heng Tao Shen




Abstract: The bi-classifier paradigm is widely adopted as an adversarial method to address domain shift challenge in
unsupervised domain adaptation (UDA) by evenly training two classifiers. In this paper, we report that although the generalization ability of the feature extractor can be strengthened by the two even classifiers, the decision boundaries of the two classifiers would be shrank to the source domain in the adversarial process, which weakens the discriminative ability of the learned model. To tame this dilemma, we disentangle the function of the two classifiers and introduce uneven bi-classifier learning for domain adaptation. Specifically, we leverage the Fnorm (Frobenius Norm) of classifier predictions instead of the classifier disagreement to achieve adversarial learning. By this way, our feature extractor can be adversarially trained with a single classifier and the other classifier is used for preserving
the target-specific decision boundaries. The proposed uneven bi-classifier learning protocol can simultaneously enhance the generalization ability of the feature extractor and expand the decision boundary of the target classifier. Extensive experiments on large-scale datasets prove that our method can significantly
surpass previous domain adaptation methods, even with only a single classifier being involved.

## Environments

```pip install -r requirements.txt```
## Usage
python train.py
## Citations
If you want to use our code or dataset, you should cite the following paper in your submissions.
```tex
@article{yu2022uneven,
  title={Uneven Bi-Classifier Learning for Domain Adaptation},
  author={Yu, Zhiqi and Li, Jingjing and Zhu, Lei and Lu, Ke and Shen, Heng Tao},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2022},
  publisher={IEEE}
}
