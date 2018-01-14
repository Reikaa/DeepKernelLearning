# DeepKernelLearning

I have my faith on a combination between Deep Learning and Gaussian Processses. Deep Learning is great but it can be also improved by integrating model uncertainty (i.e. it will know when it works well and when it does not by doing so). 

A combination of deep learning and Gaussian Processes for *regression* is somewhat straightforward (well it is not that straightforward at all, but at least there are some code for it). But meanwhile it is not easy to find a code that works well for *classification*.

I found this peace of code useful, and plan to use it as a basic block to do my research in future. https://gist.github.com/john-bradshaw/11bbf17dbca013d9fc3886a7bfe46840

Note: It should run with TF 1.4, GPflow git commit f618fe4d9aa096b32a3d24576d68f46a3f260116



Baseline: 512-17-output
Epoch:  46
Valid:
F1-BAD:  0.478405315615 F1-OK:  0.897335295079
F1-score multiplied:  0.429289975055
Test:
F1-BAD:  0.455080105455 F1-OK:  0.889682637435
F1-score multiplied:  0.404876868466

RBF:

Epoch 43: \Dev set LL -1.0475469365930898, Acc 0.8370441198348999, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.481529769665 F1-OK:  0.903330362207
F1-score multiplied:  0.434980461245
Epoch 43: 
Test set LL -1.138587147827296, Acc 0.8229991793632507, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.442820292347 F1-OK:  0.894788114954
F1-score multiplied:  0.396230334653

Matern52:


Epoch 38: \Dev set LL -1.025865894273344, Acc 0.8407321572303772, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.480854853072 F1-OK:  0.905937399161
F1-score multiplied:  0.435624394966
Epoch 38: 
Test set LL -1.1147749525122255, Acc 0.8273695707321167, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.441449403447 F1-OK:  0.897908084969
F1-score multiplied:  0.39638098846
[202294 186901 196985 ...,  16384 150178 216051]


Matern32:
Epoch 29: \Dev set LL -0.9932644037945643, Acc 0.8442835807800293, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.482288828338 F1-OK:  0.908360128617
F1-score multiplied:  0.43809194214
Epoch 29: 
Test set LL -1.085620656306074, Acc 0.8301010727882385, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.443649373882 F1-OK:  0.89974210187
F1-score multiplied:  0.39917002015

Matern12:

Epoch 44: \Dev set LL -1.0138522804283852, Acc 0.838546633720398, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.486086956522 F1-OK:  0.904229460379
F1-score multiplied:  0.439534146393
Epoch 44: 
Test set LL -1.088824646373591, Acc 0.8275061249732971, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.45419187554 F1-OK:  0.897566909976
F1-score multiplied:  0.407667598265





Baseline: 512-256-17-output

Baseline:
Epoch:  21
Valid:
F1-BAD:  0.490836653386 F1-OK:  0.894658753709
F1-score multiplied:  0.439131308594
Test:
F1-BAD:  0.459352801894 F1-OK:  0.886870355078
F1-score multiplied:  0.407386382522

Gaussian Processes

Epoch 49: \Dev set LL -0.8574439718192862, Acc 0.8352683782577515, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.397602397602 F1-OK:  0.904588607595
F1-score multiplied:  0.359666599224
Epoch 49: 
Test set LL -0.9088562727056888, Acc 0.826686680316925, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.373023715415 F1-OK:  0.899445324881
F1-score multiplied:  0.3355144369




Matern12:

Epoch 38: \Dev set LL -1.031681831273476, Acc 0.8375905156135559, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.489918489918 F1-OK:  0.903419705954
F1-score multiplied:  0.442602018104
Epoch 38: 
Test set LL -1.0955293990816295, Acc 0.826686680316925, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.45861774744 F1-OK:  0.896829268293
F1-score multiplied:  0.411301818863


RBF:

Epoch 43: \Dev set LL -1.0397760432170096, Acc 0.8419615030288696, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.466574458276 F1-OK:  0.907239637617
F1-score multiplied:  0.423294842448
Epoch 43: 
Test set LL -1.0891987405798425, Acc 0.8307839632034302, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.441891891892 F1-OK:  0.900273663876
F1-score multiplied:  0.397823632551
[ 86065 158811  87623 ...,  10358 112387 134839]



------


Baseline: 512-256-128-17-output

Epoch:  12
Valid:
F1-BAD:  0.491790148178 F1-OK:  0.895512556608
F1-score multiplied:  0.440404252909
Test:
F1-BAD:  0.464313421257 F1-OK:  0.885546162771
F1-score multiplied:  0.411170968517



------
larger-scale with 144 features

baseline:
Epoch:  15
Valid:
F1-BAD:  0.502758077226 F1-OK:  0.895736946464
F1-score multiplied:  0.450338984905
Test:
F1-BAD:  0.489992301771 F1-OK:  0.890004980907
F1-score multiplied:  0.436095589182

RBF:
Epoch 40: \Dev set LL -0.9904265657295217, Acc 0.8485179543495178, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.501573033708 F1-OK:  0.910686961424
F1-score multiplied:  0.456776022
Epoch 40: 
Test set LL -1.0630065542831817, Acc 0.8377492427825928, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.470588235294 F1-OK:  0.904193548387
F1-score multiplied:  0.4255028463


Matern12: (more detailed) https://drive.google.com/file/d/1e95Au14DdZ2WSTMi_I6gpt-ap7vuNIs6/view?usp=sharing

Epoch 23: \Dev set LL -1.0098853155593643, Acc 0.8431908488273621, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.487042001787 F1-OK:  0.907449209932
F1-score multiplied:  0.441965879726
Epoch 23: 
Test set LL -1.067731174472187, Acc 0.8333788514137268, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.470256187581 F1-OK:  0.90114253302
F1-score multiplied:  0.423767852045


