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

Baseline: - with keras
Epoch:  21
Valid:
F1-BAD:  0.490836653386 F1-OK:  0.894658753709
F1-score multiplied:  0.439131308594
Test:
F1-BAD:  0.459352801894 F1-OK:  0.886870355078
F1-score multiplied:  0.407386382522


Baseline - with my tensorflow:
epoch: 
14
Result from the previous epoch on dev:
F1-BAD:  0.469244288225 F1-OK:  0.902312793142
F1-score multiplied:  0.423405124374
Result from the previous epoch on test:
F1-BAD:  0.450201741346 F1-OK:  0.894666178445
F1-score multiplied:  0.40278027146
[ 95272 217111 176486 ..., 183877 161754 214087]
epoch: 

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

Matern12 with pre-training:

Epoch 7: \Dev set LL -0.9269897989714446, Acc 0.8373172879219055, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.493407060825 F1-OK:  0.903099829143
F1-score multiplied:  0.445595832329
Epoch 7: 
Test set LL -1.0031441810312096, Acc 0.8231357336044312, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.461090303787 F1-OK:  0.894207989543
F1-score multiplied:  0.412310633547
[ 44555 140306 112838 ...,  74919 150399 165017]




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


RBF with pre-training:

Epoch 11: \Dev set LL -1.0214825326985937, Acc 0.8356781601905823, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.486993603412 F1-OK:  0.902171261283
F1-score multiplied:  0.439351633427
Epoch 11: 
Test set LL -1.0950924530623292, Acc 0.8225211501121521, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.458428839342 F1-OK:  0.893870717465
F1-score multiplied:  0.409776115529
[ 78595 203579  77628 ..., 171197 169790 184869]


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





----
now I run the code for tensorflow
512-256-17-1
baseline:
6
[211235 108099  35584 ..., 173533 203627 154044]
Result from the previous epoch on dev:
F1-BAD:  0.499796830557 F1-OK:  0.898940973647
F1-score multiplied:  0.449287849487
Result from the previous epoch on test:
F1-BAD:  0.485190409027 F1-OK:  0.894964028777
F1-score multiplied:  0.434227963187

RBF:

Epoch 39: \Dev set LL -1.0627435747788827, Acc 0.8366343379020691, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.497478991597 F1-OK:  0.902462893492
F1-score multiplied:  0.448956330208
Epoch 39: 
Test set LL -1.1335080609318582, Acc 0.8257306814193726, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.468333333333 F1-OK:  0.895785690951
F1-score multiplied:  0.419526298595


matern12:
Epoch 17: \Dev set LL -0.9767212579140047, Acc 0.8451031446456909, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.495102404274 F1-OK:  0.908518877057
F1-score multiplied:  0.44980988036
Epoch 17: 
Test set LL -1.022837668817143, Acc 0.8361786603927612, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.471702268223 F1-OK:  0.903058956641
F1-score multiplied:  0.425974958187
[151463 159307 100073 ...,  99926  64887  45399]




----
now I run the code for tensorflow
512-256-17-1
baseline:
10
Result from the previous epoch on dev:
F1-BAD:  0.515399610136 F1-OK:  0.897077088681
F1-score multiplied:  0.462353181769
Result from the previous epoch on test:
F1-BAD:  0.494732735076 F1-OK:  0.892806886847
F1-score multiplied:  0.441700793025
[ 79978  62975 104982 ..., 106252 110351  75418]

and I used the results from the baseline as pre-training

Matern12
Epoch 7: \Dev set LL -0.9495129103340999, Acc 0.8344488739967346, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.512077294686 F1-OK:  0.900312551406
F1-score multiplied:  0.461029615696
Epoch 7: 
Test set LL -0.9774145143733359, Acc 0.8273695707321167, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.494804156675 F1-OK:  0.895898534014
F1-score multiplied:  0.443294318589
[ 44555 140306 112838 ...,  74919 150399 165017]

RBF
Epoch 2: \Dev set LL -0.7937541243405875, Acc 0.8328097462654114, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.508828250401 F1-OK:  0.899259259259
F1-score multiplied:  0.457568515546
Epoch 2: 
Test set LL -0.8319089715712372, Acc 0.8263452649116516, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.490686961746 F1-OK:  0.895328256843
F1-score multiplied:  0.439325902116



Matern52:

Epoch 3: \Dev set LL -0.8526977758464622, Acc 0.8343122601509094, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.502256873205 F1-OK:  0.900614502253
F1-score multiplied:  0.452339823865
Epoch 3: 
Test set LL -0.8996582731967095, Acc 0.8266184329986572, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.483207815998 F1-OK:  0.895835897436
F1-score multiplied:  0.432874907493
[ 97998  95754  98742 ..., 203012 213193 125451]


Matern32:

Epoch 3: \Dev set LL -0.8293870780170994, Acc 0.8341756463050842, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.5 F1-OK:  0.900605862125
F1-score multiplied:  0.450302931063
Epoch 3: 
Test set LL -0.8693412485538202, Acc 0.8281890153884888, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.486320947325 F1-OK:  0.89684296843
F1-score multiplied:  0.436153522009


deeper network:
512-256-128-17-output

baseline - epoch 6:

Result from the previous epoch on dev:
F1-BAD:  0.498638661999 F1-OK:  0.893215143733
F1-score multiplied:  0.445391604148
Result from the previous epoch on test:
F1-BAD:  0.484056857472 F1-OK:  0.888464413255
F1-score multiplied:  0.430067291856
[211235 108099  35584 ..., 173533 203627 154044]


Matern12

Epoch 4: \Dev set LL -0.9304697710619161, Acc 0.8232482075691223, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.503834355828 F1-OK:  0.892471331228
F1-score multiplied:  0.449657718265
Epoch 4: 
Test set LL -0.9586520539179797, Acc 0.8188336491584778, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.492831198624 F1-OK:  0.889720247745
F1-score multiplied:  0.438481896136
[ 82707 140148 157986 ...,  51883 106201 103473]



216 features: train with pre-training

network: 512-256-17-output
baseline:
epoch: 
4
Result from the previous epoch on dev:
F1-BAD:  0.501399440224 F1-OK:  0.897290173791
F1-score multiplied:  0.449900790857
Result from the previous epoch on test:
F1-BAD:  0.484513710791 F1-OK:  0.892109500805
F1-score multiplied:  0.432239284667
[ 82707 140148 157986 ...,  51883 106201 103473]

Our - Matern12
Epoch 7: \Dev set LL -0.9919056734849215, Acc 0.8246141076087952, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.512528473804 F1-OK:  0.893071285809
F1-score multiplied:  0.457724463114
Epoch 7: 
Test set LL -1.0410420331300183, Acc 0.8158972859382629, Outputs [1 1 1 ..., 1 0 1]
Result from the previous epoch on test:
F1-BAD:  0.489393939394 F1-OK:  0.887704098634
F1-score multiplied:  0.434437005847

RBF:

Epoch 13: \Dev set LL -0.9810681383297054, Acc 0.8430542349815369, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.504100129478 F1-OK:  0.90677484787
F1-score multiplied:  0.457105318219
Epoch 13: 
Test set LL -1.0474645150024602, Acc 0.8326959609985352, Outputs [1 1 1 ..., 1 0 1]
Result from the previous epoch on test:
F1-BAD:  0.478501489996 F1-OK:  0.90036600244
F1-score multiplied:  0.430826473709


Matern12 without pre-training

Epoch 19: \Dev set LL -1.013438401281861, Acc 0.8397759795188904, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.479360852197 F1-OK:  0.905319234805
F1-score multiplied:  0.433974599907
Epoch 19: 
Test set LL -1.0505727824796036, Acc 0.8345397710800171, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.466181978409 F1-OK:  0.902097054426
F1-score multiplied:  0.42054138955
[   873  43953 181616 ..., 181234   7816  90895]



RBF without pre-training

Epoch 20: \Dev set LL -1.0611289695849617, Acc 0.8343122601509094, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.490977759127 F1-OK:  0.901052288115
F1-score multiplied:  0.442396633275
Epoch 20: 
Test set LL -1.141914249989954, Acc 0.821906566619873, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.460264900662 F1-OK:  0.893359502781
F1-score multiplied:  0.411182022803
[139636 198046 199708 ...,  32597 183833  57999]




=---------------------------------------------------------
Results with English-to-German
Baseline:72-512-256-17-1
maximum 30 iterations

29
Result from the previous epoch on dev:
F1-BAD:  0.530168716042 F1-OK:  0.702732042699
F1-score multiplied:  0.3725665448
Result from the previous epoch on test:
F1-BAD:  0.518600601551 F1-OK:  0.699891443797
F1-score multiplied:  0.362964123774
Result from the previous epoch on test:
F1-BAD:  0.50467159392 F1-OK:  0.680259249257
F1-score multiplied:  0.343307519602
[236416  34551  51158 ..., 242429  27062  62833]
Done!


Matern12 with pre-train:

Epoch 49: \Dev set LL -2.304943293720357, Acc 0.6436170339584351, Outputs [1 0 0 ..., 0 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.530373831776 F1-OK:  0.712857142857
F1-score multiplied:  0.378080774366
Epoch 49: 
Test set LL -2.3342309966711405, Acc 0.636231005191803, Outputs [1 0 0 ..., 0 0 1]
Result from the previous epoch on test:
F1-BAD:  0.517574975814 F1-OK:  0.708040593286
F1-score multiplied:  0.366464092946
Epoch 49: 
Test set LL -2.432163270645982, Acc 0.6236870884895325, Outputs [1 1 0 ..., 1 1 0]
Result from the previous epoch on test:
F1-BAD:  0.503643841547 F1-OK:  0.696973701599
F1-score multiplied:  0.351026512531
[170550 153814  80248 ..., 180904 272190 220843]


RBF with pre-train:


[128949 262150 113820 ...,  94096 261192  78313]
Epoch 33: \Dev set LL -2.3081468257326656, Acc 0.6423980593681335, Outputs [1 0 0 ..., 0 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.51929092805 F1-OK:  0.715306572563
F1-score multiplied:  0.371452213906
Epoch 33: 
Test set LL -2.3571965879883665, Acc 0.634650468826294, Outputs [1 1 0 ..., 0 0 1]
Result from the previous epoch on test:
F1-BAD:  0.506406044678 F1-OK:  0.709998069871
F1-score multiplied:  0.359547314292
Epoch 33: 
Test set LL -2.4491446479420333, Acc 0.6204594969749451, Outputs [1 1 0 ..., 1 1 0]
Result from the previous epoch on test:
F1-BAD:  0.491125128356 F1-OK:  0.697374160342
F1-score multiplied:  0.34249797401
[215935 137192 311492 ..., 304011  32029 253544]


Matern32 with pre-train
Epoch 49: \Dev set LL -2.390197871213981, Acc 0.6337544322013855, Outputs [1 0 0 ..., 0 1 0]
Result from the previous epoch on dev:
F1-BAD:  0.536400617197 F1-OK:  0.697316604085
F1-score multiplied:  0.374041056813
Epoch 49: 
Test set LL -2.4537811196708676, Acc 0.6229179501533508, Outputs [1 0 0 ..., 0 0 1]
Result from the previous epoch on test:
F1-BAD:  0.522736016004 F1-OK:  0.688338441441
F1-score multiplied:  0.359819294541
Epoch 49: 
Test set LL -2.5909655766607576, Acc 0.6041028499603271, Outputs [1 1 0 ..., 1 1 0]
Result from the previous epoch on test:
F1-BAD:  0.50141233207 F1-OK:  0.671716942617
F1-score multiplied:  0.336807158689
[170550 153814  80248 ..., 180904 272190 220843]
Done!

Matern32 without pre-train

Epoch 27: \Dev set LL -2.41257187262274, Acc 0.6289893388748169, Outputs [1 0 0 ..., 0 1 0]
Result from the previous epoch on dev:
F1-BAD:  0.51882724921 F1-OK:  0.698106402164
F1-score multiplied:  0.36219662429
Epoch 27: 
Test set LL -2.4734187258226923, Acc 0.6200000047683716, Outputs [1 0 0 ..., 0 0 1]
Result from the previous epoch on test:
F1-BAD:  0.507601417881 F1-OK:  0.690621133383
F1-score multiplied:  0.350560266524
Epoch 27: 
Test set LL -2.536608869289372, Acc 0.6087527275085449, Outputs [0 1 0 ..., 1 1 0]
Result from the previous epoch on test:
F1-BAD:  0.501185660483 F1-OK:  0.678156781568
F1-score multiplied:  0.339882454481
[125917  94384 282921 ..., 287261 206878 288006]

Matern52 without pre-train
Epoch 11: \Dev set LL -2.414924391611122, Acc 0.6191267967224121, Outputs [1 0 0 ..., 0 1 0]
Result from the previous epoch on dev:
F1-BAD:  0.508929847121 F1-OK:  0.688931124989
F1-score multiplied:  0.350617612117
Epoch 11: 
Test set LL -2.4710455014376866, Acc 0.6121580600738525, Outputs [1 0 0 ..., 0 0 1]
Result from the previous epoch on test:
F1-BAD:  0.497162673392 F1-OK:  0.684345933109
F1-score multiplied:  0.34023125363
Epoch 11: 
Test set LL -2.57481057104963, Acc 0.5980306267738342, Outputs [1 1 0 ..., 1 1 0]
Result from the previous epoch on test:
F1-BAD:  0.493520816101 F1-OK:  0.666787592962
F1-score multiplied:  0.329073557045
[236944  67973 205470 ..., 168422 308493 212732]

RBF without pre-train

Epoch 33: \Dev set LL -2.585161781631039, Acc 0.6068262457847595, Outputs [1 0 0 ..., 0 1 0]
Result from the previous epoch on dev:
F1-BAD:  0.528567632208 F1-OK:  0.662801748717
F1-score multiplied:  0.350335550943
Epoch 33: 
Test set LL -2.664540284145497, Acc 0.5942857265472412, Outputs [1 0 0 ..., 0 0 1]
Result from the previous epoch on test:
F1-BAD:  0.514335613448 F1-OK:  0.651633782232
F1-score multiplied:  0.335158461128
Epoch 33: 
Test set LL -2.7623289783155762, Acc 0.5807986855506897, Outputs [1 1 0 ..., 1 1 0]
Result from the previous epoch on test:
F1-BAD:  0.503530936184 F1-OK:  0.63725443787
F1-score multiplied:  0.320877323688

Matern12 without pre-train
Epoch 33: \Dev set LL -2.51836834805866, Acc 0.610704779624939, Outputs [1 0 0 ..., 0 1 0]
Result from the previous epoch on dev:
F1-BAD:  0.523273171394 F1-OK:  0.671036613915
F1-score multiplied:  0.351135457085
Epoch 33: 
Test set LL -2.571270450345819, Acc 0.603161096572876, Outputs [1 0 0 ..., 0 0 1]
Result from the previous epoch on test:
F1-BAD:  0.518938835667 F1-OK:  0.662286601138
F1-score multiplied:  0.343686237672
Epoch 33: 
Test set LL -2.695880803336656, Acc 0.5867067575454712, Outputs [0 1 0 ..., 1 1 0]
Result from the previous epoch on test:
F1-BAD:  0.502010414607 F1-OK:  0.646781055683
F1-score multiplied:  0.324690825923
[215935 137192 311492 ..., 304011  32029 253544]



Here is the result with a better resampling method


Baseline (as the above):

epoch: 27
Result from the previous epoch on dev:
F1-BAD:  0.4988336623 F1-OK:  0.776112224449
F1-score multiplied:  0.387150903278
Result from the previous epoch on test:
F1-BAD:  0.493337281611 F1-OK:  0.774561904344
F1-score multiplied:  0.382120264328
Result from the previous epoch on test:
F1-BAD:  0.475737734326 F1-OK:  0.769981504073
F1-score multiplied:  0.366309256221
[ 43379 199156 245551 ..., 193210 207728 258779]

Matern12 with pretrain

Epoch 44: \Dev set LL -2.0058208174533307, Acc 0.6848404407501221, Outputs [1 0 0 ..., 1 1 0]
Result from the previous epoch on dev:
F1-BAD:  0.509655172414 F1-OK:  0.767798824298
F1-score multiplied:  0.391312642177
Epoch 44: 
Test set LL -2.0204724763739503, Acc 0.6822492480278015, Outputs [1 1 0 ..., 0 0 1]
Result from the previous epoch on test:
F1-BAD:  0.501858381778 F1-OK:  0.76672468425
F1-score multiplied:  0.384787209307
Epoch 44: 
Test set LL -2.0975594509681383, Acc 0.6716630458831787, Outputs [1 1 0 ..., 1 1 0]
Result from the previous epoch on test:
F1-BAD:  0.486920841169 F1-OK:  0.758587402462
F1-score multiplied:  0.369372016107
[  8239  75951  18537 ...,  75920 112273 199137]


RBF with pretrain

Epoch 46: \Dev set LL -2.125521476360115, Acc 0.6727614998817444, Outputs [1 0 0 ..., 1 1 0]
Result from the previous epoch on dev:
F1-BAD:  0.523786486051 F1-OK:  0.750738583608
F1-score multiplied:  0.39322672465
Epoch 46: 
Test set LL -2.1786543521156587, Acc 0.6633434891700745, Outputs [1 1 0 ..., 0 0 1]
Result from the previous epoch on test:
F1-BAD:  0.50851970181 F1-OK:  0.743990384615
F1-score multiplied:  0.378333768534
Epoch 46: 
Test set LL -2.225349368863511, Acc 0.6572757363319397, Outputs [1 1 0 ..., 1 1 0]
Result from the previous epoch on test:
F1-BAD:  0.495287198904 F1-OK:  0.740547480018
F1-score multiplied:  0.366783687034

RBF without pre-train
Epoch 44: \Dev set LL -2.10585268231703, Acc 0.675642728805542, Outputs [1 0 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.500767525158 F1-OK:  0.759786622897
F1-score multiplied:  0.380476466796
Epoch 44: 
Test set LL -2.138964521563216, Acc 0.6699696183204651, Outputs [1 1 0 ..., 0 0 1]
Result from the previous epoch on test:
F1-BAD:  0.48681349844 F1-OK:  0.756776130102
F1-score multiplied:  0.368408835431
Epoch 44: 
Test set LL -2.178761008595202, Acc 0.663238525390625, Outputs [1 1 0 ..., 1 1 0]
Result from the previous epoch on test:
F1-BAD:  0.482776004033 F1-OK:  0.750344715711
F1-score multiplied:  0.362248423498
[  8239  75951  18537 ...,  75920 112273 199137]

Matern12 without pre-train
Epoch 41: \Dev set LL -2.0237285916763716, Acc 0.6868351101875305, Outputs [1 0 0 ..., 1 1 0]
Result from the previous epoch on dev:
F1-BAD:  0.500706713781 F1-OK:  0.771876009041
F1-score multiplied:  0.386483499933
Epoch 41: 
Test set LL -2.0362838900515965, Acc 0.6817021369934082, Outputs [1 1 1 ..., 0 0 1]
Result from the previous epoch on test:
F1-BAD:  0.487671232877 F1-OK:  0.769135802469
F1-score multiplied:  0.37508540504
Epoch 41: 
Test set LL -2.0797453362915417, Acc 0.677625834941864, Outputs [1 1 0 ..., 1 1 0]
Result from the previous epoch on test:
F1-BAD:  0.483206173814 F1-OK:  0.765751083198
F1-score multiplied:  0.370015651006
[ 25448 105319 255368 ...,  51909 188489  55076]




Matern32 without pre-training

Epoch 47: \Dev set LL -2.0401124842159915, Acc 0.6876108050346375, Outputs [1 0 0 ..., 1 1 0]
Result from the previous epoch on dev:
F1-BAD:  0.490327246429 F1-OK:  0.774786290645
F1-score multiplied:  0.379898828463
Epoch 47: 
Test set LL -2.0587980699834336, Acc 0.6844376921653748, Outputs [1 1 1 ..., 0 0 1]
Result from the previous epoch on test:
F1-BAD:  0.48095190481 F1-OK:  0.773308878117
F1-score multiplied:  0.371924377936
Epoch 47: 
Test set LL -2.0945572769050247, Acc 0.6785557866096497, Outputs [1 1 0 ..., 1 1 0]
Result from the previous epoch on test:
F1-BAD:  0.475169703466 F1-OK:  0.768333070494
F1-score multiplied:  0.365088597269
[ 59597  49440  42322 ..., 153440 186116   7463]








Matern52 without pre-training
Epoch 42: \Dev set LL -2.070641107828316, Acc 0.6858377456665039, Outputs [1 0 0 ..., 1 1 0]
Result from the previous epoch on dev:
F1-BAD:  0.497072911123 F1-OK:  0.771573604061
F1-score multiplied:  0.383528337516
Epoch 42: 
Test set LL -2.1084234953543355, Acc 0.6784802675247192, Outputs [1 1 1 ..., 0 0 1]
Result from the previous epoch on test:
F1-BAD:  0.47917282127 F1-OK:  0.767465377006
F1-score multiplied:  0.367748549927
Epoch 42: 
Test set LL -2.145120792928431, Acc 0.6743982434272766, Outputs [1 1 0 ..., 1 1 0]
Result from the previous epoch on test:
F1-BAD:  0.480446927374 F1-OK:  0.762906309751
F1-score multiplied:  0.366535992395
[221375 139101 249196 ..., 213715 137850   7294]





------------------------
Sent - task 1 - prediction

baseline: 17-512-256-17-3
10
Result from the previous epoch on dev:
0.517142857143
Result from the previous epoch on test normal:
0.544444444444
Result from the previous epoch on test adaptation:
0.431524547804
epoch: 

full batch is better than mini-batch size

0.54 0.524547803618 0.528571428571 [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1] get at epoch: 20


Matern12 - with pre-training

0.537777777778 0.421188630491 0.528571428571 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1] 1

RBF - with pre-training
0.548888888889 0.335917312661 0.538095238095 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 0 0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 0 0 1 0 1 0 1 0
 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1
 1 1 1 0 0 1 1 1 1 1 1 1 0 1 1 1 1 0 1 1 1 1 0 1 1 1 1 1 0 1 1 0 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 0 0 1 0 1 0
 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 0
 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1
 1 1 1 1 1 0 0 1 1 1 1 1 1 1 0 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 1 0 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 0 0 1 0
 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1
 1 0 1 1 1 1] 3



RBF - without pre-training

0.544444444444 0.348837209302 0.519047619048 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1] 2
 
 Matern12 - without pre-training
 
 0.542222222222 0.354005167959 0.533333333333 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1]
 
 Matern52 - without pre-training
 
 0.671111111111 0.333333333333 0.542857142857 [1 1 0 1 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 1 1 0 0 0 0
 1 0 1 1 1 0 0 1 1 1 0 0 1 0 1 1 0 1 1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0
 1 0 0 0 1 0 1 0 1 0 0 0 1 0 1 0 0 1 1 0 1 0 0 0 1 0 0 1 0 0 0 0 1 1 0 0 0
 0 1 0 1 1 0 1 1 0 1 0 0 0 0 1 0 0 0 0 0 1 0 0 0 1 0 1 1 0 0 1 0 0 0 0 0 1
 0 1 1 1 0 1 0 0 1 0 0 0 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 1 0 0 0 1 1 0 0
 0 0 1 0 1 1 1 0 0 1 1 1 0 0 0 0 1 1 0 1 1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 1
 0 0 1 1 0 0 1 0 1 0 1 0 0 0 0 1 1 0 1 1 1 0 1 0 0 0 1 1 0 1 0 0 0 0 1 1 0
 0 0 0 1 0 1 1 0 1 1 0 1 0 0 0 0 1 0 0 0 0 0 1 0 0 0 1 0 1 1 0 0 1 0 0 0 0
 0 1 0 1 1 1 0 1 0 0 1 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 1 1
 0 0 0 0 1 0 1 1 1 0 0 1 1 1 0 0 0 0 1 1 0 1 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0
 0 1 0 0 1 1 0 0 1 0 1 0 1 0 0 0 0 1 1 0 1 1 1 0 1 0 0 0 1 1 0 1 0 0 0 0 1
 1 0 0 0 0 1 0 1 1 0 1 1 0 1 0 0 0 0 1 0 0 0 0 0 1 0 0 0 1 0 1 1 0 0 1 0 0
 0 0 0 1 0 1] 6
 
 Matern32 - without pretraining
 
 0.544444444444 0.33850129199 0.538095238095 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1] 0
 
 
 Matern52 - with pretraining
 
 0.54 0.369509043928 0.533333333333 [1 1 1 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
 1 1 1 1 1 1 1 1 1 1 0 1 1 0 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1
 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 0 0 0 1 1 0 0 0
 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1
 1 1 1 1 1 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 0 0 0 1 1 0
 0 0 1 0 0 1 1 1 1 1 1 1 0 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1
 1 1 1 1 1 1 0 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 1 0 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 0 0 1 1
 1 0 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1
 1 1 1 1 1 1] 5
 
 
 Matern32 - with pre-training

0.528888888889 0.410852713178 0.52380952381 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1
 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1
 1 1 1 1 1 1] 4
 

 

----
small scale
Gaussian Processes:

Matern52
0.708888888889 0.64857881137 0.680952380952 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 0 1 1 1 2 1 2 2 1 0 1 1 1 1 1 0 1 1 0
 1 1 0 2 2 1 2 1 1 2 2 1 1 1 1 2 1 2 1 2 1 1 1 1 1 2 1 2 1 1 1 1 0 1 1 2 0
 1 2 2 1 0 1 1 2 1 0 2 1 1 1 1 2 1 1 0 2 0 1 1 0 2 1 2 1 1 1 1 2 1 1 1 2 1
 1 0 2 0 1 1 1 0 1 1 0 1 0 1 0 1 1 1 1 1 1 0 0 0 1 1 1 2 2 1 2 2 0 2 1 1 1
 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 2 1 1 0 1 1 1 2 1 2 2 1 0 1 1 1 1 1 0 1
 1 0 1 1 0 2 2 1 2 2 1 2 1 1 1 1 1 2 1 2 1 2 1 1 1 1 1 2 1 2 1 1 1 1 0 1 1
 2 0 1 2 2 1 0 1 1 2 1 0 2 1 1 1 2 2 1 1 2 2 0 1 1 0 2 1 2 1 1 1 1 2 1 1 1
 2 1 2 0 2 1 1 1 1 0 1 1 0 1 2 1 2 1 1 1 1 1 1 0 2 0 1 1 1 1 0 1 2 1 0 1 1
 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 2 1 1 2 1 1 1 2 1 2 2 1 2 1 1 1 1 1
 0 2 2 0 1 1 0 2 2 1 2 1 1 2 1 1 1 1 1 2 1 1 1 2 1 1 1 1 1 2 1 2 1 1 1 1 2
 1 1 2 2 1 2 2 1 0 1 1 2 1 0 2 1 1 1 1 2 1 1 1 2 0 1 1 0 2 1 2 1 1 1 1 2 1
 1 1 2 1 2 0 2 1 1 1 1 2 1 1 0 1 2 1 2 1 1 1 2 1 1 0 2 2 1 1 1 2 2 1 2 2 0
 1 1 1 1 1 1] 0


Matern12

0.622222222222 0.537467700258 0.719047619048 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 0 1 1 1 0 1 0 2 1 2 1 1 1 1 0 1 1 1 1
 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 2 1 1 1 1 0 1 1 0 2
 1 1 0 1 0 1 1 1 1 0 2 1 1 1 1 2 1 1 1 1 2 1 1 0 2 1 1 1 1 1 1 2 1 1 1 0 1
 1 0 0 1 1 1 1 0 1 1 0 1 2 1 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 2 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 2 1 1 1 0 1 0 0 1 2 1 1 1 1 0 1 1
 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 2 1 1 1 1 2 1 1
 1 2 1 1 2 1 1 1 1 1 1 2 2 1 1 1 1 2 1 1 1 1 2 1 1 0 2 1 1 1 1 1 1 2 1 1 1
 0 1 1 0 0 1 1 1 1 2 1 1 0 1 1 1 1 1 1 1 1 1 1 0 2 0 1 1 1 1 1 1 2 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 2 1 1 1 0 1 2 1 1 2 1 1 1 1 2
 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 2
 1 1 1 2 1 1 2 1 1 1 1 1 1 2 2 1 1 1 1 2 1 1 1 1 2 1 1 0 2 1 1 1 1 1 1 2 1
 1 1 2 1 1 0 2 1 1 1 1 2 1 1 0 1 1 1 1 1 1 1 1 1 1 1 2 2 1 1 1 1 1 1 2 1 1
 1 1 1 1 1 1] 4
 
 Matern32
 
 0.984444444444 0.617571059432 0.695238095238 [2 1 0 2 1 2 0 2 1 1 1 1 1 1 0 2 2 2 0 2 1 2 2 2 2 2 2 0 2 0 2 0 2 2 2 2 0
 1 2 2 2 2 2 2 1 1 2 2 2 2 2 1 2 1 2 1 0 2 1 1 0 2 2 2 2 1 2 2 2 0 2 0 0 2
 2 2 2 1 2 2 0 2 1 0 2 2 2 2 2 2 2 1 2 2 2 0 1 0 2 2 0 2 2 2 0 2 1 1 0 2 1
 2 0 0 1 1 1 2 0 1 1 0 2 2 0 2 2 0 0 2 2 1 0 2 2 1 0 2 2 2 1 2 2 0 2 0 2 1
 2 1 2 1 0 2 1 1 0 2 1 1 1 2 1 1 0 2 0 2 2 2 1 0 2 2 0 2 2 0 2 1 2 0 2 2 2
 2 0 1 2 2 2 0 2 2 1 1 2 2 0 1 2 1 2 1 2 1 0 2 1 1 0 2 2 2 2 1 2 2 2 2 2 0
 0 2 2 2 2 1 2 2 0 2 1 2 2 0 1 2 2 2 2 2 2 2 2 0 1 0 2 2 0 2 2 2 0 2 1 1 0
 0 1 2 0 0 1 1 1 2 2 1 1 0 2 2 0 2 2 0 2 2 2 1 0 2 2 2 2 2 2 0 1 2 2 2 0 2
 2 1 2 1 2 1 2 2 1 2 0 2 1 2 1 2 1 1 2 2 0 2 2 2 1 2 2 2 2 2 2 2 2 1 2 0 2
 2 2 2 0 1 2 2 2 0 2 2 1 1 2 2 2 1 2 1 2 1 2 1 2 2 2 1 0 2 2 2 2 1 2 2 2 2
 2 2 0 2 2 2 2 1 2 2 0 2 1 2 2 2 1 2 2 2 2 2 2 2 2 0 1 2 2 2 2 2 1 2 0 2 1
 1 1 2 1 2 0 2 1 1 1 2 2 1 1 0 2 2 0 2 2 0 0 2 2 1 2 2 2 2 2 2 2 2 1 2 2 2
 2 2 2 1 2 1] 7
 
 RBF 
 0.808888888889 0.66149870801 0.614285714286 [0 0 0 1 1 0 1 0 1 1 0 1 0 0 0 2 0 0 2 0 2 2 2 0 2 2 0 2 2 0 0 0 0 0 0 0 2
 0 0 0 2 0 0 2 1 0 0 2 2 0 0 1 0 1 2 1 0 1 0 1 0 2 0 2 2 0 2 2 0 2 0 0 0 0
 0 0 2 1 0 2 0 2 1 2 2 2 0 2 2 2 1 1 0 2 0 2 2 0 2 2 0 0 0 1 1 2 1 1 2 2 1
 0 0 0 1 0 1 2 0 0 1 2 0 2 0 0 2 0 0 0 0 1 0 0 2 0 0 1 0 0 1 0 0 0 2 0 2 1
 0 1 0 2 0 1 1 0 1 2 1 1 1 2 0 0 0 2 0 0 2 0 2 0 2 0 2 2 2 0 0 1 0 0 0 0 0
 0 2 0 0 0 2 0 0 2 1 0 0 2 0 0 2 1 0 1 1 1 2 1 0 1 0 0 2 0 2 1 2 2 0 2 0 0
 0 0 0 0 0 1 0 2 0 2 1 2 2 2 0 2 2 0 0 0 0 0 2 2 2 0 0 1 0 0 0 1 1 2 1 1 2
 2 1 0 0 0 1 0 1 2 0 0 1 2 0 0 0 0 0 0 0 0 0 1 0 2 2 0 0 1 0 0 1 0 0 0 0 0
 2 1 0 1 2 2 0 1 1 0 1 0 1 2 1 2 0 0 0 2 0 2 2 0 2 0 2 0 2 2 2 0 0 1 0 0 2
 0 0 2 2 0 2 0 0 0 0 2 1 0 0 1 0 0 0 1 2 1 2 1 0 2 0 1 0 0 2 0 2 1 2 2 2 2
 0 0 0 0 0 0 0 1 0 2 0 0 1 2 2 0 0 2 2 2 2 2 0 0 0 2 2 0 0 2 2 0 0 1 1 2 1
 1 2 2 0 2 0 0 1 2 1 2 2 0 1 2 0 0 0 0 0 0 0 2 2 1 0 2 0 0 0 1 2 0 2 2 2 0
 0 0 2 1 1 1] 0




---89 features - pretty big


baseline - 89-512-256-17-3


0.542222222222 0.542222222222 0.485714285714 [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1] 12


Matern12 - with pretraining
0.528888888889 0.528888888889 0.471428571429 [1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
 1 1 1 1 1 1 1 0 1 1 1 1 0 1 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 0 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 1 1 0 1 1 0 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 1 0 1 1 1 1 1 0 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 0 1 1 1 1 1 0 1 1 1 1 0
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1
 1 0 1 1 1 1] 6

RBF - with pretraining

0.542222222222 0.542222222222 0.490476190476 [1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0
 1 0 1 1 1 1 1 0 1 1 1 1 0 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 0 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0
 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0 1 1 1 1 1
 1 0 1 1 1 1] 0


Matern52 - with pretraining


0.54 0.54 0.480952380952 [1 1 1 1 1 1 1 0 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
 1 0 1 1 1 1 1 0 0 1 1 1 0 1 1 1 1 0 1 1 1 0 1 0 1 1 1 1 1 1 1 1 0 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1
 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 1 0 0 0 1 1 1 0 1 1 0 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 0 1 0 1 1 1 1 1 0 0 1 1 1 0 1 1 1 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1
 1 1 1 0 1 1 1 1 0 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1
 1 1 1 1 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 0 1 1 1 1 1 1 0
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 0 1 0 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 0 1 1 0 1 0
 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 0 0 1 1 1 1
 1 0 1 1 1 1] 0
 
 Matern32 - with pretraining
 
 0.524444444444 0.524444444444 0.485714285714 [1 1 1 1 1 1 1 0 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
 1 0 1 1 1 1 1 0 0 0 1 1 0 1 1 1 1 0 0 1 1 0 1 0 1 1 1 1 1 1 1 1 0 1 1 1 1
 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1
 1 1 1 0 1 1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 1 0 1 1 0 1 1
 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 0 1 0 1 1 1 1 1 0 0 1 1 1 0 1 1 1 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1
 1 1 1 0 1 1 1 1 0 1 1 1 0 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1
 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 0 1 1 1 1 1 1 0
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 0 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 0 1 1 0 1 0
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 0 1 1 1 1 1
 1 0 1 1 1 1] 2
 
 Matern32 - without pretraining
 
 0.531111111111 0.531111111111 0.5 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1
 1 0 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 1
 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1] 2
 
 
 Matern52 - without pretraining
 
 0.595555555556 0.595555555556 0.52380952381 [0 0 1 0 1 0 0 1 1 1 1 1 1 0 1 0 1 0 1 1 1 0 1 1 1 0 1 1 1 0 0 0 1 0 0 0 0
 0 0 0 0 0 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 1 0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 1
 0 0 0 1 0 0 0 0 0 1 0 1 1 1 1 1 1 1 0 1 0 1 1 0 0 1 0 1 1 1 1 0 1 1 1 0 1
 0 0 0 0 1 1 0 0 1 0 1 1 1 0 1 0 1 1 1 1 0 0 1 0 1 0 1 0 0 0 0 0 1 1 0 1 0
 1 0 0 0 0 0 1 0 0 1 1 1 1 1 1 0 1 0 1 1 1 1 0 0 1 1 0 0 1 1 1 1 1 0 1 0 0
 0 0 1 1 0 1 0 1 0 0 0 0 0 1 0 0 1 1 1 0 0 1 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0
 0 1 0 0 0 1 0 1 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 1 0 1 0 0 0 1 1 1 1 0 1 1 1
 0 1 0 0 1 1 1 1 1 0 1 0 1 1 1 0 1 1 1 1 1 1 0 0 1 0 1 1 1 0 1 0 0 1 1 1 0
 0 1 0 1 0 0 1 0 1 0 0 0 1 1 1 1 1 1 1 0 1 1 1 0 1 0 1 1 1 0 1 1 1 1 0 0 1
 0 0 1 0 1 1 0 0 1 1 0 1 0 0 0 0 1 0 1 1 1 1 0 0 1 0 0 0 0 0 1 0 1 1 1 0 0
 0 0 0 1 1 0 1 1 1 0 0 1 0 1 0 0 1 1 1 0 0 1 1 0 0 1 0 1 0 1 0 1 1 1 0 0 1
 1 0 0 1 0 0 0 0 1 1 1 0 1 0 1 1 1 1 1 0 1 0 1 1 0 1 1 1 1 0 1 0 0 0 0 1 1
 1 0 1 0 1 0] 10
 
 
 RBF - without pretraining
 
 0.542222222222 0.542222222222 0.5 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1] 10
 
 Matern12 - without pretraining
 
 
0.535555555556 0.535555555556 0.480952380952 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1] 2
 
 Gaussian Processes - Matern12
 
 0.662222222222 0.662222222222 0.642857142857 [0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1
 0 0 1 0 0 0 0 1 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0
 0 0 0 1 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
 0 0 0 0 0 0 1 1 0 0 0 1 0 1 1 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 0 0 0 0 0 0
 0 1 0 0 1 0 0 0 0 1 1 1 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 0 0 0 1 0 0 0 1 0 0 0 0 1 0 1 1 0
 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0
 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 0 0 0 0
 0 0 0 1 1 1 1 1 0 0 0 1 1 1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 1 0 1 0 1 1 0 0
 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0
 0 0 0 1 0 0] 0


RBF

0.662222222222 0.662222222222 0.633333333333 [0 0 0 0 1 1 0 0 1 0 0 0 0 0 1 0 1 1 0 1 1 0 0 0 0 0 0 0 1 0 0 0 0 1 0 1 1
 1 1 1 1 1 1 1 1 0 0 2 0 0 0 1 1 1 1 0 1 0 0 1 0 1 0 1 0 1 0 1 1 0 0 0 0 1
 0 0 0 0 1 0 0 0 0 0 0 1 0 1 1 0 0 0 0 0 0 1 1 0 0 1 0 0 0 1 1 0 0 1 1 1 0
 0 2 0 1 1 0 0 0 1 1 1 0 0 2 0 1 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0
 0 1 0 0 0 0 1 1 0 0 1 2 0 0 0 1 1 0 1 1 0 1 1 0 0 1 0 0 1 1 0 1 0 0 0 1 1
 1 1 0 1 1 1 1 1 1 1 0 1 2 2 0 0 1 1 0 1 0 1 0 2 1 0 1 0 1 0 1 2 1 1 0 1 0
 0 0 0 0 0 0 1 1 0 0 1 0 0 0 0 0 1 1 2 1 1 0 1 1 1 0 0 1 0 1 0 2 1 0 0 1 1
 1 0 0 0 0 0 1 0 0 0 1 1 2 0 0 2 0 1 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0
 1 0 0 1 0 0 0 0 1 1 0 0 1 2 0 0 2 0 2 0 1 1 0 1 1 0 0 1 0 0 1 1 0 0 0 0 0
 0 0 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 0 0 1 1 1 0 0 0 1 1 0 1 0 0 1 1 1 2 1 0
 0 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 1 1 0 0 1 0 1 1 1 0 1 1 0 1 0 2 1 0 0
 1 1 1 0 0 2 0 0 1 1 1 0 0 1 0 1 0 1 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 1
 0 0 0 0 0 1] 0


Matern52

0.66 0.66 0.647619047619 [0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1
 0 1 1 0 0 1 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 1 1 1 0
 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
 0 1 0 0 0 0 1 1 0 0 0 2 0 0 0 1 1 0 0 1 0 1 1 0 0 1 0 0 1 0 0 0 0 0 0 0 0
 1 1 0 1 1 1 0 1 1 1 0 1 0 0 0 0 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0
 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 1 0 0 1 0 0 0 1 1 0 0 1 0 1 0 0 1 0 0 1 1
 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0
 0 0 0 0 0 0 0 0 0 1 0 0 1 2 0 0 2 0 1 0 0 1 0 1 1 0 0 1 0 0 1 0 0 0 0 0 0
 0 0 0 1 1 1 1 1 1 1 1 1 0 1 0 0 0 0 0 1 1 1 0 0 0 0 0 0 1 0 0 1 1 2 2 0 0
 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 0 0 0 1 0 1 0 1 1 0 0
 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0
 0 0 0 0 0 0] 4
 
 Matern32
 
 0.653333333333 0.653333333333 0.642857142857 [0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1
 0 1 1 0 0 1 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0
 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
 0 0 0 0 0 0 1 1 0 0 0 2 0 0 0 0 1 0 0 1 0 1 1 0 0 1 0 0 1 0 0 0 0 0 0 0 0
 1 1 0 1 1 0 0 1 0 1 0 1 0 0 0 0 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 1 0
 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 1 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 0 0 0 0
 0 0 0 1 1 1 1 1 0 1 0 1 0 1 0 0 0 0 0 1 1 1 0 0 0 0 0 0 1 0 0 0 0 2 2 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 1 0 1 0 1 1 0 0
 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0
 0 0 0 1 0 0] 11
 
 
 
 



-----
German - English
baseline: 17-512-256-17-3

0.713333333333 0.713333333333 0.72380952381 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 2, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1] 22

RBF - with pre-training

0.651111111111 0.651111111111 0.585714285714 [0 1 1 1 2 2 2 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1
 1 1 1 1 1 1 1 1 1 1 1 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1
 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 0 1 1 1 2 2 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 0 1 1 1 2 2 2 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 2 1 1 1 1 1 1 1 1 2 1 1 1 1 2 2 1 1 1 2 1 2 1 1 1 1 1 1 1 1 1 1 1 1 2 1
 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 2 1 1 1 1 1] 35

Matern12 - with pre-training

0.653333333333 0.653333333333 0.590476190476 [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1] 8
 
 
Matern12 - without pretraining

0.653333333333 0.653333333333 0.585714285714 [0 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 0 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 0 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1] 12
 
 





-----------------

phrase level - 144 features - matern 52 from scratch


Epoch 1: \Dev set LL -0.7493147904988721, Acc 0.839093029499054, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.420845624385 F1-OK:  0.906567258883
F1-score multiplied:  0.381524864112
Epoch 1: 
Test set LL -0.7783776118377148, Acc 0.8318082690238953, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.402909090909 F1-OK:  0.902118189405
F1-score multiplied:  0.363471619586
[ 10604 112260   2170 ...,  24253  38081  64020]
Epoch 2: \Dev set LL -0.879300563405222, Acc 0.8281655311584473, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.451612903226 F1-OK:  0.898121153223
F1-score multiplied:  0.405603101456
Epoch 2: 
Test set LL -0.9003143742573952, Acc 0.8228626251220703, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.443108630313 F1-OK:  0.894681282988
F1-score multiplied:  0.396440997872
[ 16587 157088  51334 ...,  99848  92538  64243]
Epoch 3: \Dev set LL -0.9124609975698444, Acc 0.8470154404640198, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.440559440559 F1-OK:  0.911392405063
F1-score multiplied:  0.401522528105
Epoch 3: 
Test set LL -0.9459802497826799, Acc 0.8389784097671509, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.422341989221 F1-OK:  0.906450845037
F1-score multiplied:  0.382832253024
[ 97998  95754  98742 ..., 203012 213193 125451]
Epoch 4: \Dev set LL -0.9672487047990705, Acc 0.8380002975463867, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.456461961503 F1-OK:  0.90481540931
F1-score multiplied:  0.413013816532
Epoch 4: 
Test set LL -1.0041998135212165, Acc 0.8349494934082031, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.45055694476 F1-OK:  0.902888826389
F1-score multiplied:  0.406802831076
[ 82707 140148 157986 ...,  51883 106201 103473]
Epoch 5: \Dev set LL -1.0081210912293341, Acc 0.8359513878822327, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.477144101001 F1-OK:  0.902713649251
F1-score multiplied:  0.430724492633
Epoch 5: 
Test set LL -1.0455956600449658, Acc 0.8268915414810181, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.457289659602 F1-OK:  0.897022382906
F1-score multiplied:  0.410199060134
[ 96593 186600 181132 ...,  40532  34553 129546]
Epoch 6: \Dev set LL -1.020641456214472, Acc 0.8333560824394226, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.470026064292 F1-OK:  0.90113452188
F1-score multiplied:  0.423556712717
Epoch 6: 
Test set LL -1.0544540613194733, Acc 0.8288035988807678, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.463284093342 F1-OK:  0.898159808263
F1-score multiplied:  0.416103152447
[211235 108099  35584 ..., 173533 203627 154044]
Epoch 7: \Dev set LL -0.9839530767813401, Acc 0.8433274030685425, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.472643678161 F1-OK:  0.907997112377
F1-score multiplied:  0.429159094953
Epoch 7: 
Test set LL -1.0346109731798498, Acc 0.8359054923057556, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.456457814974 F1-OK:  0.903365906623
F1-score multiplied:  0.412348427859
[ 44555 140306 112838 ...,  74919 150399 165017]
Epoch 8: \Dev set LL -0.9860655202902947, Acc 0.8445567488670349, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.473148148148 F1-OK:  0.908828713347
F1-score multiplied:  0.430010622704
Epoch 8: 
Test set LL -1.0413509122011808, Acc 0.8382272720336914, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.45276045276 F1-OK:  0.905084338315
F1-score multiplied:  0.409786394802
[169730  20673 186182 ...,  67812 116880 108578]
Epoch 9: \Dev set LL -1.0573214688448014, Acc 0.8339024782180786, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.474503025065 F1-OK:  0.90136275146
F1-score multiplied:  0.427699352249
Epoch 9: 
Test set LL -1.0871795191209193, Acc 0.8292133212089539, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.466396415618 F1-OK:  0.898337465957
F1-score multiplied:  0.418981374137
[ 41613 123879  70949 ...,  98250 105271 211104]
Epoch 10: \Dev set LL -1.0459707227828303, Acc 0.8367709517478943, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.471004869411 F1-OK:  0.903496729387
F1-score multiplied:  0.425551359038
Epoch 10: 
Test set LL -1.0822425755217928, Acc 0.8320813775062561, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.460627330555 F1-OK:  0.900562093089
F1-score multiplied:  0.414823512939
[ 79978  62975 104982 ..., 106252 110351  75418]
Epoch 11: \Dev set LL -1.0330078106800569, Acc 0.8395028114318848, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.471434997751 F1-OK:  0.905386907158
F1-score multiplied:  0.42683107454
Epoch 11: 
Test set LL -1.0526317259132754, Acc 0.8357006311416626, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.461744966443 F1-OK:  0.90305423483
F1-score multiplied:  0.416980747358
[ 78595 203579  77628 ..., 171197 169790 184869]
Epoch 12: \Dev set LL -1.0615618464829928, Acc 0.8363611698150635, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.472246696035 F1-OK:  0.903168444876
F1-score multiplied:  0.426518314056
Epoch 12: 
Test set LL -1.0769240227441526, Acc 0.8334471583366394, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.46943658908 F1-OK:  0.901219067676
F1-score multiplied:  0.423065205144
[161145   8624 103629 ..., 131598  77971 145710]
Epoch 13: \Dev set LL -1.0611722838248603, Acc 0.8344488739967346, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.480274442539 F1-OK:  0.901543460601
F1-score multiplied:  0.432988282965
Epoch 13: 
Test set LL -1.1031822336067936, Acc 0.8262087106704712, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.469902103728 F1-OK:  0.896067301017
F1-score multiplied:  0.42106390983
[ 88180  37890  33658 ..., 138465 165899 187250]
Epoch 14: \Dev set LL -1.0686174185183, Acc 0.8359513878822327, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.473937801139 F1-OK:  0.902823853063
F1-score multiplied:  0.427882351736
Epoch 14: 
Test set LL -1.093445991745239, Acc 0.8316716551780701, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.468864468864 F1-OK:  0.899987828133
F1-score multiplied:  0.421972315022
[ 95272 217111 176486 ..., 183877 161754 214087]
Epoch 15: \Dev set LL -1.1775949966916288, Acc 0.8166916966438293, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.46914556962 F1-OK:  0.889219085356
F1-score multiplied:  0.417173194316
Epoch 15: 
Test set LL -1.1930409318947213, Acc 0.8135072588920593, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.467745078932 F1-OK:  0.886947882601
F1-score multiplied:  0.414865507356
[ 43428  23203 165494 ...,  87260 127806  73850]
Epoch 16: \Dev set LL -1.0227206770733492, Acc 0.8440103530883789, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.471785383904 F1-OK:  0.908493589744
F1-score multiplied:  0.428613997011
Epoch 16: 
Test set LL -1.0824350163999676, Acc 0.8335837125778198, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.446765039728 F1-OK:  0.902061648515
F1-score multiplied:  0.403009608236
[204644  31141 103650 ...,  30090  81702 136580]
Epoch 17: \Dev set LL -1.045133706304, Acc 0.8399125933647156, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.46581586144 F1-OK:  0.905848329049
F1-score multiplied:  0.42195851973
Epoch 17: 
Test set LL -1.0817940661582846, Acc 0.8341982960700989, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.459242761693 F1-OK:  0.902088878135
F1-score multiplied:  0.414277787687
[151463 159307 100073 ...,  99926  64887  45399]
Epoch 18: \Dev set LL -1.0485657011768017, Acc 0.8384100794792175, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.472581364244 F1-OK:  0.904589079764
F1-score multiplied:  0.427491941396
Epoch 18: 
Test set LL -1.1046500052693027, Acc 0.8303059339523315, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.460134694764 F1-OK:  0.899331577881
F1-score multiplied:  0.41381366108
[ 77558  65635  39900 ...,  52102 175656  30661]
Epoch 19: \Dev set LL -1.042973610559692, Acc 0.8396393656730652, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.473070017953 F1-OK:  0.905429353955
F1-score multiplied:  0.428331480731
Epoch 19: 
Test set LL -1.095536271489737, Acc 0.8334471583366394, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.464779460171 F1-OK:  0.901378836278
F1-score multiplied:  0.418942368935
[   873  43953 181616 ..., 181234   7816  90895]
Epoch 20: \Dev set LL -1.0847062754606531, Acc 0.8341756463050842, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.48252344416 F1-OK:  0.90126870527
F1-score multiplied:  0.434883279781
Epoch 20: 
Test set LL -1.1256111127763424, Acc 0.8275744318962097, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.471204188482 F1-OK:  0.896993432056
F1-score multiplied:  0.422667062226
[139636 198046 199708 ...,  32597 183833  57999]
Epoch 21: \Dev set LL -1.0366936740511363, Acc 0.8429176211357117, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.468576709797 F1-OK:  0.907837794518
F1-score multiplied:  0.425391646784
Epoch 21: 
Test set LL -1.068370803859053, Acc 0.8366566300392151, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.454628362973 F1-OK:  0.903943458357
F1-score multiplied:  0.410958334693
[ 48250  85392  96772 ..., 159417  47120  80258]
Epoch 22: \Dev set LL -1.0520967694257068, Acc 0.8396393656730652, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.472596585804 F1-OK:  0.905444587629
F1-score multiplied:  0.427910020748
Epoch 22: 
Test set LL -1.0903415570110888, Acc 0.8337885737419128, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.464819700967 F1-OK:  0.901616814875
F1-score multiplied:  0.419089258277
[ 94976 169993 129145 ..., 212870  49486 180803]
Epoch 23: \Dev set LL -1.063189014642339, Acc 0.8381368517875671, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.475431606906 F1-OK:  0.904304288137
F1-score multiplied:  0.429934840841
Epoch 23: 
Test set LL -1.1023510022976344, Acc 0.8320813775062561, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.465783184879 F1-OK:  0.900384849099
F1-score multiplied:  0.41938412263
[188706 216106  14316 ...,  23716  33182   2019]
Epoch 24: \Dev set LL -1.063432629971611, Acc 0.8375905156135559, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.475055187638 F1-OK:  0.903934717621
F1-score multiplied:  0.429418876892
Epoch 24: 
Test set LL -1.0865836438885264, Acc 0.8342666029930115, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.470203012443 F1-OK:  0.90176872951
F1-score multiplied:  0.424014373142
[ 49555 204099 199462 ..., 142630  30997  94732]
Epoch 25: \Dev set LL -1.1511239037147998, Acc 0.8222920298576355, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.478974769724 F1-OK:  0.89287772746
F1-score multiplied:  0.427665903901
Epoch 25: 
Test set LL -1.1760984913062835, Acc 0.8192433714866638, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.471551207826 F1-OK:  0.890975740352
F1-score multiplied:  0.420140686506
[ 10586  40093 110450 ..., 135123  16134  39003]
Epoch 26: \Dev set LL -1.0546522873129494, Acc 0.8397759795188904, Outputs [1 1 1 ..., 1 1 0]
Result from the previous epoch on dev:
F1-BAD:  0.478434859938 F1-OK:  0.905349794239
F1-score multiplied:  0.433150902001
Epoch 26: 
Test set LL -1.0988168981208266, Acc 0.8329008221626282, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.469542597008 F1-OK:  0.900830800405
F1-score multiplied:  0.422978433487
[124016  93299 142951 ..., 121315 130112 138434]
Epoch 27: \Dev set LL -1.0417070743216614, Acc 0.8416882753372192, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.476276547673 F1-OK:  0.906750341942
F1-score multiplied:  0.431863922461
Epoch 27: 
Test set LL -1.0876959978852727, Acc 0.8339934349060059, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.462762430939 F1-OK:  0.901829342164
F1-score multiplied:  0.417332738672
[  9469 193383  39949 ..., 202484  63420  57130]
Epoch 28: \Dev set LL -1.0982520275345955, Acc 0.8339024782180786, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.482112436116 F1-OK:  0.901089962583
F1-score multiplied:  0.434426677021
Epoch 28: 
Test set LL -1.1236956763394434, Acc 0.8291450142860413, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.477006688963 F1-OK:  0.897894221352
F1-score multiplied:  0.428301549566
[ 70289  17736 199270 ...,  16586  34471 159323]
Epoch 29: \Dev set LL -1.0671044961788871, Acc 0.8362245559692383, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.47389205792 F1-OK:  0.903017067055
F1-score multiplied:  0.427932616244
Epoch 29: 
Test set LL -1.1008019184249518, Acc 0.8322179913520813, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.4680666811 F1-OK:  0.900401313389
F1-score multiplied:  0.421447854416
[ 44559 117975   7356 ..., 164492 159093  15768]
Done!


phrase level - 144 features - matern 32 from scratch


Epoch 1: \Dev set LL -0.6687387965291192, Acc 0.840049147605896, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.422298963986 F1-OK:  0.907173999207
F1-score multiplied:  0.38309864002
Epoch 1: 
Test set LL -0.7042616977965287, Acc 0.834130048751831, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.414275379793 F1-OK:  0.903384909113
F1-score multiplied:  0.374250126322
[ 10604 112260   2170 ...,  24253  38081  64020]
Epoch 2: \Dev set LL -0.8595031504621647, Acc 0.8341756463050842, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.467543859649 F1-OK:  0.901795825918
F1-score multiplied:  0.421629101065
Epoch 2: 
Test set LL -0.9062876240169994, Acc 0.8236137628555298, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.449605795866 F1-OK:  0.894978654198
F1-score multiplied:  0.402387590104
[ 16587 157088  51334 ...,  99848  92538  64243]
Epoch 3: \Dev set LL -0.9018523466079391, Acc 0.8461958765983582, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.46021093001 F1-OK:  0.910321758522
F1-score multiplied:  0.418940023097
Epoch 3: 
Test set LL -0.940677096186665, Acc 0.836246907711029, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.438933083762 F1-OK:  0.904133685136
F1-score multiplied:  0.39685418655
[ 97998  95754  98742 ..., 203012 213193 125451]
Epoch 4: \Dev set LL -0.9323303084485602, Acc 0.840049147605896, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.474180511899 F1-OK:  0.905678614579
F1-score multiplied:  0.429455149077
Epoch 4: 
Test set LL -0.9903253499445506, Acc 0.831261932849884, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.453681185054 F1-OK:  0.900222087624
F1-score multiplied:  0.408413823525
[ 82707 140148 157986 ...,  51883 106201 103473]
Epoch 5: \Dev set LL -0.9449707417404618, Acc 0.8438737988471985, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.469112865769 F1-OK:  0.908479461926
F1-score multiplied:  0.426179403876
Epoch 5: 
Test set LL -1.0111407672328852, Acc 0.8346763253211975, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.451642129105 F1-OK:  0.902665540948
F1-score multiplied:  0.407681786784
[ 96593 186600 181132 ...,  40532  34553 129546]
Epoch 6: \Dev set LL -0.9592126915237605, Acc 0.8442835807800293, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.475620975161 F1-OK:  0.908565928778
F1-score multiplied:  0.432133013043
Epoch 6: 
Test set LL -1.0204373062455978, Acc 0.8357006311416626, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.459568733154 F1-OK:  0.903124496698
F1-score multiplied:  0.415047780828
[211235 108099  35584 ..., 173533 203627 154044]
Epoch 7: \Dev set LL -0.9930843909015468, Acc 0.8444201350212097, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.470970738504 F1-OK:  0.908799743775
F1-score multiplied:  0.428018086478
Epoch 7: 
Test set LL -1.0303194106359206, Acc 0.8382272720336914, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.45752232654 F1-OK:  0.904939609165
F1-score multiplied:  0.414030075363
[ 44555 140306 112838 ...,  74919 150399 165017]
Epoch 8: \Dev set LL -0.9829333956026811, Acc 0.8446933627128601, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.481532147743 F1-OK:  0.90866736284
F1-score multiplied:  0.437552546812
Epoch 8: 
Test set LL -1.0640342389489683, Acc 0.8340617418289185, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.457347029924 F1-OK:  0.902055622733
F1-score multiplied:  0.412552459883
[169730  20673 186182 ...,  67812 116880 108578]
Epoch 9: \Dev set LL -1.0819580581856216, Acc 0.827619194984436, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.486156351792 F1-OK:  0.896438536025
F1-score multiplied:  0.435809288279
Epoch 9: 
Test set LL -1.140674294238243, Acc 0.8178776502609253, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.467345715998 F1-OK:  0.890161031259
F1-score multiplied:  0.416012944507
[ 41613 123879  70949 ...,  98250 105271 211104]
Epoch 10: \Dev set LL -1.0319669261951478, Acc 0.838546633720398, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.478835978836 F1-OK:  0.904477129465
F1-score multiplied:  0.433096191622
Epoch 10: 
Test set LL -1.0847070418996865, Acc 0.8297596573829651, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.462367910287 F1-OK:  0.898868200073
F1-score multiplied:  0.415607811291
[ 79978  62975 104982 ..., 106252 110351  75418]
Epoch 11: \Dev set LL -1.0665535601480076, Acc 0.8322633504867554, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.475661827498 F1-OK:  0.900162601626
F1-score multiplied:  0.428172988135
Epoch 11: 
Test set LL -1.12289845395134, Acc 0.8236820697784424, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.462754889721 F1-OK:  0.894534760232
F1-score multiplied:  0.413950334323
[ 78595 203579  77628 ..., 171197 169790 184869]
Epoch 12: \Dev set LL -1.0437649730813339, Acc 0.8384100794792175, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.48227571116 F1-OK:  0.904264789188
F1-score multiplied:  0.436104944283
Epoch 12: 
Test set LL -1.0950396953536154, Acc 0.8321496844291687, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.469114470842 F1-OK:  0.900316327358
F1-score multiplied:  0.422351417499
[161145   8624 103629 ..., 131598  77971 145710]
Epoch 13: \Dev set LL -1.011480249446454, Acc 0.8438737988471985, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.472542685741 F1-OK:  0.908376753507
F1-score multiplied:  0.429246790767
Epoch 13: 
Test set LL -1.051771319145348, Acc 0.8366566300392151, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.456610631531 F1-OK:  0.903881700555
F1-score multiplied:  0.41272199412
[ 88180  37890  33658 ..., 138465 165899 187250]
Epoch 14: \Dev set LL -1.0838946112094447, Acc 0.8295314908027649, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.480865224626 F1-OK:  0.898022552705
F1-score multiplied:  0.431827816525
Epoch 14: 
Test set LL -1.126889606654188, Acc 0.8239551782608032, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.474306688418 F1-OK:  0.894274934383
F1-score multiplied:  0.424160582662
[ 95272 217111 176486 ..., 183877 161754 214087]
Epoch 15: \Dev set LL -1.0896069660884016, Acc 0.8313071727752686, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.491141326741 F1-OK:  0.898894801474
F1-score multiplied:  0.441484385396
Epoch 15: 
Test set LL -1.148065845497997, Acc 0.8219748735427856, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.470444850701 F1-OK:  0.893002257336
F1-score multiplied:  0.420108313628
[ 43428  23203 165494 ...,  87260 127806  73850]
Epoch 16: \Dev set LL -1.0357786703444862, Acc 0.8404589295387268, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.473873873874 F1-OK:  0.905973273225
F1-score multiplied:  0.429317064609
Epoch 16: 
Test set LL -1.0918020290868835, Acc 0.8323545455932617, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.461268378319 F1-OK:  0.900731874975
F1-score multiplied:  0.41547913127
[204644  31141 103650 ...,  30090  81702 136580]
Epoch 17: \Dev set LL -1.0381645670538604, Acc 0.8403223752975464, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.481596452328 F1-OK:  0.905626866877
F1-score multiplied:  0.436146686221
Epoch 17: 
Test set LL -1.1139147712573898, Acc 0.8288035988807678, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.457711442786 F1-OK:  0.898357997162
F1-score multiplied:  0.411188735019
[151463 159307 100073 ...,  99926  64887  45399]
Epoch 18: \Dev set LL -1.0596041256064515, Acc 0.835131824016571, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.493495593789 F1-OK:  0.901541724447
F1-score multiplied:  0.444906868632
Epoch 18: 
Test set LL -1.1316669742172445, Acc 0.8253209590911865, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.468633153303 F1-OK:  0.895480918526
F1-score multiplied:  0.419652046571
[ 77558  65635  39900 ...,  52102 175656  30661]
Epoch 19: \Dev set LL -1.1101627180605902, Acc 0.8273459672927856, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.485342019544 F1-OK:  0.896274413261
F1-score multiplied:  0.434999633798
Epoch 19: 
Test set LL -1.186153755618713, Acc 0.816511869430542, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.466547548144 F1-OK:  0.889200445342
F1-score multiplied:  0.414854287583
[   873  43953 181616 ..., 181234   7816  90895]
Epoch 20: \Dev set LL -0.9979634118461422, Acc 0.8476983904838562, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.494789306751 F1-OK:  0.910333735424
F1-score multiplied:  0.450423397863
Epoch 20: 
Test set LL -1.0692094044867804, Acc 0.8361103534698486, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.46571682992 F1-OK:  0.903210195193
F1-score multiplied:  0.420640188856
[139636 198046 199708 ...,  32597 183833  57999]
Epoch 21: \Dev set LL -1.028674347247688, Acc 0.840595543384552, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.473612990528 F1-OK:  0.906076458753
F1-score multiplied:  0.429129581277
Epoch 21: 
Test set LL -1.0654030929682563, Acc 0.8357689380645752, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.467566969227 F1-OK:  0.902910661661
F1-score multiplied:  0.422171201556
[ 48250  85392  96772 ..., 159417  47120  80258]
Epoch 22: \Dev set LL -1.0431592616373733, Acc 0.8410053253173828, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.483584738243 F1-OK:  0.906038101388
F1-score multiplied:  0.438146198098
Epoch 22: 
Test set LL -1.1043228768716191, Acc 0.8321496844291687, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.464488017429 F1-OK:  0.900477771479
F1-score multiplied:  0.418261134814
[ 94976 169993 129145 ..., 212870  49486 180803]
Epoch 23: \Dev set LL -1.0506880404816121, Acc 0.8388198614120483, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.493127147766 F1-OK:  0.904174110768
F1-score multiplied:  0.445872800327
Epoch 23: 
Test set LL -1.1100787603704128, Acc 0.8300327658653259, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.478743455497 F1-OK:  0.89846204055
F1-score multiplied:  0.430132821926
[188706 216106  14316 ...,  23716  33182   2019]
Epoch 24: \Dev set LL -1.0535326131481924, Acc 0.8382734656333923, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.482969432314 F1-OK:  0.90414507772
F1-score multiplied:  0.436674434916
Epoch 24: 
Test set LL -1.1057168562283024, Acc 0.8313302397727966, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.469957081545 F1-OK:  0.899707649829
F1-score multiplied:  0.422823981358
[ 49555 204099 199462 ..., 142630  30997  94732]
Epoch 25: \Dev set LL -1.0362864263475464, Acc 0.840595543384552, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.499356499356 F1-OK:  0.905206725692
F1-score multiplied:  0.452020861736
Epoch 25: 
Test set LL -1.126633814476552, Acc 0.8268915414810181, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.469554300063 F1-OK:  0.89656860745
F1-score multiplied:  0.42098764493
[ 10586  40093 110450 ..., 135123  16134  39003]
Epoch 26: \Dev set LL -1.0694228784890885, Acc 0.8347220420837402, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.493723849372 F1-OK:  0.901240613777
F1-score multiplied:  0.444963985045
Epoch 26: 
Test set LL -1.1509498638865527, Acc 0.8234089016914368, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.464817880795 F1-OK:  0.894259077527
F1-score multiplied:  0.415667609298
[124016  93299 142951 ..., 121315 130112 138434]
Epoch 27: \Dev set LL -1.038598634712906, Acc 0.8408687114715576, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.477812640072 F1-OK:  0.906131657401
F1-score multiplied:  0.432961159475
Epoch 27: 
Test set LL -1.0727203830580962, Acc 0.8355640769004822, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.458389563653 F1-OK:  0.903067385879
F1-score multiplied:  0.413956664962
[  9469 193383  39949 ..., 202484  63420  57130]
Epoch 28: \Dev set LL -1.0539005633158913, Acc 0.840049147605896, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.486178148311 F1-OK:  0.905281889509
F1-score multiplied:  0.440128272741
Epoch 28: 
Test set LL -1.0978509204440017, Acc 0.8322862386703491, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.468168038112 F1-OK:  0.900445885691
F1-score multiplied:  0.42155998373
[ 70289  17736 199270 ...,  16586  34471 159323]
Epoch 29: \Dev set LL -1.0603212776244912, Acc 0.8367709517478943, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.487783969138 F1-OK:  0.902916565115
F1-score multiplied:  0.440428225933
Epoch 29: 
Test set LL -1.1039123992242594, Acc 0.8322179913520813, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.476677316294 F1-OK:  0.900093522547
F1-score multiplied:  0.429054164741
[ 44559 117975   7356 ..., 164492 159093  15768]
Done!


phrase level - shallow Neural network


epoch: 
0
[157230  58090 106355 ...,  43269 152588 159419]
epoch: 
1
Result from the previous epoch on dev:
F1-BAD:  0.393292682927 F1-OK:  0.905791383936
F1-score multiplied:  0.35624112356
Result from the previous epoch on test:
F1-BAD:  0.397518610422 F1-OK:  0.903872040542
F1-score multiplied:  0.359305957555
[ 10604 112260   2170 ...,  24253  38081  64020]
epoch: 
2
Result from the previous epoch on dev:
F1-BAD:  0.41037962518 F1-OK:  0.902316694531
F1-score multiplied:  0.370292386895
Result from the previous epoch on test:
F1-BAD:  0.414931801866 F1-OK:  0.902624556932
F1-score multiplied:  0.374527633817
[ 16587 157088  51334 ...,  99848  92538  64243]
epoch: 
3
Result from the previous epoch on dev:
F1-BAD:  0.439622641509 F1-OK:  0.905126976521
F1-score multiplied:  0.39791431232
Result from the previous epoch on test:
F1-BAD:  0.429496070273 F1-OK:  0.90112971717
F1-score multiplied:  0.387031672331
[ 97998  95754  98742 ..., 203012 213193 125451]
epoch: 
4
Result from the previous epoch on dev:
F1-BAD:  0.456293706294 F1-OK:  0.899303869192
F1-score multiplied:  0.410346695558
Result from the previous epoch on test:
F1-BAD:  0.449582530507 F1-OK:  0.895559978876
F1-score multiplied:  0.402628121524
[ 82707 140148 157986 ...,  51883 106201 103473]
epoch: 
5
Result from the previous epoch on dev:
F1-BAD:  0.455650607834 F1-OK:  0.9026648418
F1-score multiplied:  0.411299783837
Result from the previous epoch on test:
F1-BAD:  0.446088321004 F1-OK:  0.900471261127
F1-score multiplied:  0.401689712989
[ 96593 186600 181132 ...,  40532  34553 129546]
epoch: 
6
Result from the previous epoch on dev:
F1-BAD:  0.467023172906 F1-OK:  0.903532827875
F1-score multiplied:  0.421970768099
Result from the previous epoch on test:
F1-BAD:  0.450254819411 F1-OK:  0.899858728557
F1-score multiplied:  0.405165729321
[211235 108099  35584 ..., 173533 203627 154044]
epoch: 
7
Result from the previous epoch on dev:
F1-BAD:  0.45911674683 F1-OK:  0.899878591663
F1-score multiplied:  0.413149331546
Result from the previous epoch on test:
F1-BAD:  0.459465333623 F1-OK:  0.899258719164
F1-score multiplied:  0.413178207414
[ 44555 140306 112838 ...,  74919 150399 165017]
epoch: 
8
Result from the previous epoch on dev:
F1-BAD:  0.472647702407 F1-OK:  0.902484421785
F1-score multiplied:  0.426557188415
Result from the previous epoch on test:
F1-BAD:  0.460083571586 F1-OK:  0.900771997898
F1-score multiplied:  0.414430397977
[169730  20673 186182 ...,  67812 116880 108578]
epoch: 
9
Result from the previous epoch on dev:
F1-BAD:  0.481858948227 F1-OK:  0.895725654278
F1-score multiplied:  0.43161342167
Result from the previous epoch on test:
F1-BAD:  0.474164133739 F1-OK:  0.893442286371
F1-score multiplied:  0.423638287763
[ 41613 123879  70949 ...,  98250 105271 211104]
epoch: 
10
Result from the previous epoch on dev:
F1-BAD:  0.481814291827 F1-OK:  0.901584721658
F1-score multiplied:  0.434396404188
Result from the previous epoch on test:
F1-BAD:  0.471513605442 F1-OK:  0.898877318581
F1-score multiplied:  0.423832885334
[ 79978  62975 104982 ..., 106252 110351  75418]
epoch: 
11
Result from the previous epoch on dev:
F1-BAD:  0.478204574881 F1-OK:  0.901906693712
F1-score multiplied:  0.431295907049
Result from the previous epoch on test:
F1-BAD:  0.466292134831 F1-OK:  0.899837793998
F1-score multiplied:  0.419587285966
[ 78595 203579  77628 ..., 171197 169790 184869]
epoch: 
12
Result from the previous epoch on dev:
F1-BAD:  0.485875706215 F1-OK:  0.904140669314
F1-score multiplied:  0.43929998622
Result from the previous epoch on test:
F1-BAD:  0.471934369603 F1-OK:  0.900794938352
F1-score multiplied:  0.425116091372
[161145   8624 103629 ..., 131598  77971 145710]
epoch: 
13
Result from the previous epoch on dev:
F1-BAD:  0.486554621849 F1-OK:  0.900342521611
F1-score multiplied:  0.438065815137
Result from the previous epoch on test:
F1-BAD:  0.476569037657 F1-OK:  0.897910886241
F1-score multiplied:  0.427916526958
[ 88180  37890  33658 ..., 138465 165899 187250]
epoch: 
14
Result from the previous epoch on dev:
F1-BAD:  0.489522212909 F1-OK:  0.900620104439
F1-score multiplied:  0.440873546515
Result from the previous epoch on test:
F1-BAD:  0.477736163129 F1-OK:  0.89747569643
F1-score multiplied:  0.428756595714
[ 95272 217111 176486 ..., 183877 161754 214087]
epoch: 
15
Result from the previous epoch on dev:
F1-BAD:  0.482337549062 F1-OK:  0.903878856588
F1-score multiplied:  0.435974712336
Result from the previous epoch on test:
F1-BAD:  0.467044958533 F1-OK:  0.901157613535
F1-score multiplied:  0.420881120246
[ 43428  23203 165494 ...,  87260 127806  73850]
epoch: 
16
Result from the previous epoch on dev:
F1-BAD:  0.480349344978 F1-OK:  0.903659326425
F1-score multiplied:  0.434072165532
Result from the previous epoch on test:
F1-BAD:  0.478714161599 F1-OK:  0.902771025766
F1-score multiplied:  0.432169274715
[204644  31141 103650 ...,  30090  81702 136580]
epoch: 
17
Result from the previous epoch on dev:
F1-BAD:  0.485494880546 F1-OK:  0.901935274028
F1-score multiplied:  0.437884958125
Result from the previous epoch on test:
F1-BAD:  0.474410702909 F1-OK:  0.899304284145
F1-score multiplied:  0.426639577571
[151463 159307 100073 ...,  99926  64887  45399]
epoch: 
18
Result from the previous epoch on dev:
F1-BAD:  0.484120171674 F1-OK:  0.902371669916
F1-score multiplied:  0.436856327753
Result from the previous epoch on test:
F1-BAD:  0.47825155346 F1-OK:  0.901100686406
F1-score multiplied:  0.430952803098
[ 77558  65635  39900 ...,  52102 175656  30661]
epoch: 
19
Result from the previous epoch on dev:
F1-BAD:  0.489983305509 F1-OK:  0.900212314225
F1-score multiplied:  0.441089005384
Result from the previous epoch on test:
F1-BAD:  0.481612300021 F1-OK:  0.898059244127
F1-score multiplied:  0.432516378119
[   873  43953 181616 ..., 181234   7816  90895]
epoch: 
20
Result from the previous epoch on dev:
F1-BAD:  0.481529769665 F1-OK:  0.903330362207
F1-score multiplied:  0.434980461245
Result from the previous epoch on test:
F1-BAD:  0.476025917927 F1-OK:  0.901614080623
F1-score multiplied:  0.429191670344
[139636 198046 199708 ...,  32597 183833  57999]
epoch: 
21
Result from the previous epoch on dev:
F1-BAD:  0.489051094891 F1-OK:  0.896517739816
F1-score multiplied:  0.438442982246
Result from the previous epoch on test:
F1-BAD:  0.481376518219 F1-OK:  0.894775751602
F1-score multiplied:  0.430724035893
[ 48250  85392  96772 ..., 159417  47120  80258]
epoch: 
22
Result from the previous epoch on dev:
F1-BAD:  0.492716366752 F1-OK:  0.90380240494
F1-score multiplied:  0.445318237224
Result from the previous epoch on test:
F1-BAD:  0.474178403756 F1-OK:  0.899845541013
F1-score multiplied:  0.426687322264
[ 94976 169993 129145 ..., 212870  49486 180803]
epoch: 
23
Result from the previous epoch on dev:
F1-BAD:  0.492487479132 F1-OK:  0.900702270129
F1-score multiplied:  0.443584590464
Result from the previous epoch on test:
F1-BAD:  0.480133139172 F1-OK:  0.897920836567
F1-score multiplied:  0.431121549989
[188706 216106  14316 ...,  23716  33182   2019]
epoch: 
24
Result from the previous epoch on dev:
F1-BAD:  0.495405179616 F1-OK:  0.901371652515
F1-score multiplied:  0.446544185415
Result from the previous epoch on test:
F1-BAD:  0.482012892493 F1-OK:  0.898239307161
F1-score multiplied:  0.432962926596
[ 49555 204099 199462 ..., 142630  30997  94732]
epoch: 
25
Result from the previous epoch on dev:
F1-BAD:  0.497367355205 F1-OK:  0.898053068266
F1-score multiplied:  0.446662279397
Result from the previous epoch on test:
F1-BAD:  0.48486080065 F1-OK:  0.89596585546
F1-score multiplied:  0.434418722034
[ 10586  40093 110450 ..., 135123  16134  39003]
epoch: 
26
Result from the previous epoch on dev:
F1-BAD:  0.48709408826 F1-OK:  0.899346405229
F1-score multiplied:  0.438066317285
Result from the previous epoch on test:
F1-BAD:  0.485150571132 F1-OK:  0.89870469497
F1-score multiplied:  0.436007096044
[124016  93299 142951 ..., 121315 130112 138434]
epoch: 
27
Result from the previous epoch on dev:
F1-BAD:  0.495967741935 F1-OK:  0.897220851834
F1-score multiplied:  0.444992599901
Result from the previous epoch on test:
F1-BAD:  0.484146341463 F1-OK:  0.895847012475
F1-score multiplied:  0.433721053601
[  9469 193383  39949 ..., 202484  63420  57130]
epoch: 
28
Result from the previous epoch on dev:
F1-BAD:  0.488706365503 F1-OK:  0.898009338904
F1-score multiplied:  0.438862880204
Result from the previous epoch on test:
F1-BAD:  0.481848184818 F1-OK:  0.897217675941
F1-score multiplied:  0.432322708539
[ 70289  17736 199270 ...,  16586  34471 159323]
epoch: 
29
Result from the previous epoch on dev:
F1-BAD:  0.499189627229 F1-OK:  0.89847215377
F1-score multiplied:  0.448507979516
Result from the previous epoch on test:
F1-BAD:  0.484799020608 F1-OK:  0.896461229344
F1-score multiplied:  0.434603525999
[ 44559 117975   7356 ..., 164492 159093  15768]
Done!



deep network - phrase - 144 features


epoch: 
0
[157230  58090 106355 ...,  43269 152588 159419]
epoch: 
1
Result from the previous epoch on dev:
F1-BAD:  0.420588235294 F1-OK:  0.906205364228
F1-score multiplied:  0.381139314955
Result from the previous epoch on test:
F1-BAD:  0.419971333015 F1-OK:  0.903274639471
F1-score multiplied:  0.379349454417
[ 10604 112260   2170 ...,  24253  38081  64020]
epoch: 
2
Result from the previous epoch on dev:
F1-BAD:  0.449499545041 F1-OK:  0.902764384442
F1-score multiplied:  0.405792180086
Result from the previous epoch on test:
F1-BAD:  0.445739910314 F1-OK:  0.90043499275
F1-score multiplied:  0.401359812912
[ 16587 157088  51334 ...,  99848  92538  64243]
epoch: 
3
Result from the previous epoch on dev:
F1-BAD:  0.467590523022 F1-OK:  0.903990326481
F1-score multiplied:  0.422697309566
Result from the previous epoch on test:
F1-BAD:  0.455142231947 F1-OK:  0.899263694474
F1-score multiplied:  0.409292885012
[ 97998  95754  98742 ..., 203012 213193 125451]
epoch: 
4
Result from the previous epoch on dev:
F1-BAD:  0.498397435897 F1-OK:  0.89692079697
F1-score multiplied:  0.447023025413
Result from the previous epoch on test:
F1-BAD:  0.47587289133 F1-OK:  0.889541132699
F1-score multiplied:  0.423308510775
[ 82707 140148 157986 ...,  51883 106201 103473]
epoch: 
5
Result from the previous epoch on dev:
F1-BAD:  0.484047402005 F1-OK:  0.909061696658
F1-score multiplied:  0.44002895253
Result from the previous epoch on test:
F1-BAD:  0.459765018843 F1-OK:  0.90164265246
F1-score multiplied:  0.414543751098
[ 96593 186600 181132 ...,  40532  34553 129546]
epoch: 
6
Result from the previous epoch on dev:
F1-BAD:  0.497388509442 F1-OK:  0.897062453715
F1-score multiplied:  0.446188556729
Result from the previous epoch on test:
F1-BAD:  0.48197492163 F1-OK:  0.890671518359
F1-score multiplied:  0.429281335259
[211235 108099  35584 ..., 173533 203627 154044]
epoch: 
7
Result from the previous epoch on dev:
F1-BAD:  0.49774497745 F1-OK:  0.899614848808
F1-score multiplied:  0.447778772633
Result from the previous epoch on test:
F1-BAD:  0.486 F1-OK:  0.894186429513
F1-score multiplied:  0.434574604743
[ 44555 140306 112838 ...,  74919 150399 165017]
epoch: 
8
Result from the previous epoch on dev:
F1-BAD:  0.501818181818 F1-OK:  0.88479650185
F1-score multiplied:  0.444006971837
Result from the previous epoch on test:
F1-BAD:  0.494815874151 F1-OK:  0.880729298557
F1-score multiplied:  0.435798837756
[169730  20673 186182 ...,  67812 116880 108578]
epoch: 
9
Result from the previous epoch on dev:
F1-BAD:  0.491638795987 F1-OK:  0.869550291795
F1-score multiplied:  0.427504658508
Result from the previous epoch on test:
F1-BAD:  0.48928332217 F1-OK:  0.869188540058
F1-score multiplied:  0.425279456472
[ 41613 123879  70949 ...,  98250 105271 211104]
epoch: 
10
Result from the previous epoch on dev:
F1-BAD:  0.506939371804 F1-OK:  0.886592741935
F1-score multiplied:  0.449448767643
Result from the previous epoch on test:
F1-BAD:  0.490139835066 F1-OK:  0.880050611556
F1-score multiplied:  0.431347861598
[ 79978  62975 104982 ..., 106252 110351  75418]
epoch: 
11
Result from the previous epoch on dev:
F1-BAD:  0.491938507687 F1-OK:  0.886847599165
F1-score multiplied:  0.436274484479
Result from the previous epoch on test:
F1-BAD:  0.483194057567 F1-OK:  0.883571099862
F1-score multiplied:  0.426936304892
[ 78595 203579  77628 ..., 171197 169790 184869]
epoch: 
12
Result from the previous epoch on dev:
F1-BAD:  0.48685558211 F1-OK:  0.871681038163
F1-score multiplied:  0.424382779249
Result from the previous epoch on test:
F1-BAD:  0.485699779997 F1-OK:  0.870011548826
F1-score multiplied:  0.422564417859
[161145   8624 103629 ..., 131598  77971 145710]
epoch: 
13
Result from the previous epoch on dev:
F1-BAD:  0.479713603819 F1-OK:  0.892150395778
F1-score multiplied:  0.427976681507
Result from the previous epoch on test:
F1-BAD:  0.480109739369 F1-OK:  0.890303907381
F1-score multiplied:  0.427443576932
[ 88180  37890  33658 ..., 138465 165899 187250]
epoch: 
14
Result from the previous epoch on dev:
F1-BAD:  0.486552567237 F1-OK:  0.896619625862
F1-score multiplied:  0.436252580798
Result from the previous epoch on test:
F1-BAD:  0.482378420949 F1-OK:  0.891404023297
F1-score multiplied:  0.429994065186
[ 95272 217111 176486 ..., 183877 161754 214087]
epoch: 
15
Result from the previous epoch on dev:
F1-BAD:  0.470690403821 F1-OK:  0.901207553286
F1-score multiplied:  0.424189747183
Result from the previous epoch on test:
F1-BAD:  0.479783243018 F1-OK:  0.898080849326
F1-score multiplied:  0.430884142382
[ 43428  23203 165494 ...,  87260 127806  73850]
epoch: 
16
Result from the previous epoch on dev:
F1-BAD:  0.480190174326 F1-OK:  0.891731308797
F1-score multiplied:  0.428200612624
Result from the previous epoch on test:
F1-BAD:  0.467238689548 F1-OK:  0.886920529801
F1-score multiplied:  0.414403586077
[204644  31141 103650 ...,  30090  81702 136580]
epoch: 
17
Result from the previous epoch on dev:
F1-BAD:  0.473073736537 F1-OK:  0.895976447498
F1-score multiplied:  0.423862925867
Result from the previous epoch on test:
F1-BAD:  0.47144006436 F1-OK:  0.891923013654
F1-score multiplied:  0.420488242961
[151463 159307 100073 ...,  99926  64887  45399]
epoch: 
18
Result from the previous epoch on dev:
F1-BAD:  0.466755912539 F1-OK:  0.903636803484
F1-score multiplied:  0.421777820814
Result from the previous epoch on test:
F1-BAD:  0.461974286337 F1-OK:  0.900036438722
F1-score multiplied:  0.415793691456
[ 77558  65635  39900 ...,  52102 175656  30661]
epoch: 
19
Result from the previous epoch on dev:
F1-BAD:  0.476853463279 F1-OK:  0.872291613561
F1-score multiplied:  0.415955276916
Result from the previous epoch on test:
F1-BAD:  0.484098332474 F1-OK:  0.872140087768
F1-score multiplied:  0.422201562172
[   873  43953 181616 ..., 181234   7816  90895]

epoch: 
20
Result from the previous epoch on dev:
F1-BAD:  0.467674223342 F1-OK:  0.896574225122
F1-score multiplied:  0.419304654402
Result from the previous epoch on test:
F1-BAD:  0.468198604842 F1-OK:  0.893831408208
F1-score multiplied:  0.418490618287
[139636 198046 199708 ...,  32597 183833  57999]
epoch: 
21
Result from the previous epoch on dev:
F1-BAD:  0.468713563933 F1-OK:  0.88673460933
F1-score multiplied:  0.415624539002
Result from the previous epoch on test:
F1-BAD:  0.475572519084 F1-OK:  0.885728542914
F1-score multiplied:  0.421228154378
[ 48250  85392  96772 ..., 159417  47120  80258]
epoch: 
22
Result from the previous epoch on dev:
F1-BAD:  0.4621979735 F1-OK:  0.885723749586
F1-score multiplied:  0.409379722139
Result from the previous epoch on test:
F1-BAD:  0.469737342977 F1-OK:  0.884080885412
F1-score multiplied:  0.41528580609
[ 94976 169993 129145 ..., 212870  49486 180803]
epoch: 
23
Result from the previous epoch on dev:
F1-BAD:  0.464447184546 F1-OK:  0.893275452535
F1-score multiplied:  0.414879268954
Result from the previous epoch on test:
F1-BAD:  0.46935483871 F1-OK:  0.891811903979
F1-score multiplied:  0.418576232351
[188706 216106  14316 ...,  23716  33182   2019]
epoch: 
24
Result from the previous epoch on dev:
F1-BAD:  0.471750296326 F1-OK:  0.889604491784
F1-score multiplied:  0.419671182612
Result from the previous epoch on test:
F1-BAD:  0.477203647416 F1-OK:  0.885447885448
F1-score multiplied:  0.422538960533
[ 49555 204099 199462 ..., 142630  30997  94732]
epoch: 
25
Result from the previous epoch on dev:
F1-BAD:  0.478701825558 F1-OK:  0.894473187156
F1-score multiplied:  0.428185947604
Result from the previous epoch on test:
F1-BAD:  0.466482649842 F1-OK:  0.888255698712
F1-score multiplied:  0.414355872072
[ 10586  40093 110450 ..., 135123  16134  39003]
epoch: 
26
Result from the previous epoch on dev:
F1-BAD:  0.444711538462 F1-OK:  0.885888358307
F1-score multiplied:  0.393964774728
Result from the previous epoch on test:
F1-BAD:  0.459352801894 F1-OK:  0.886870355078
F1-score multiplied:  0.407386382522
[124016  93299 142951 ..., 121315 130112 138434]
epoch: 
27
Result from the previous epoch on dev:
F1-BAD:  0.458349940215 F1-OK:  0.887991428336
F1-score multiplied:  0.407010818089
Result from the previous epoch on test:
F1-BAD:  0.466627748589 F1-OK:  0.886496335252
F1-score multiplied:  0.413663789051
[  9469 193383  39949 ..., 202484  63420  57130]
epoch: 
28
Result from the previous epoch on dev:
F1-BAD:  0.475827940617 F1-OK:  0.885393258427
F1-score multiplied:  0.421294850793
Result from the previous epoch on test:
F1-BAD:  0.46844840386 F1-OK:  0.880167364017
F1-score multiplied:  0.412312996804
[ 70289  17736 199270 ...,  16586  34471 159323]
epoch: 
29
Result from the previous epoch on dev:
F1-BAD:  0.461415433826 F1-OK:  0.889053619965
F1-score multiplied:  0.410223061751
Result from the previous epoch on test:
F1-BAD:  0.465279128574 F1-OK:  0.886155630099
F1-score multiplied:  0.412309719354
[ 44559 117975   7356 ..., 164492 159093  15768]
Done!




----

phrase level - 72 features - shallow network



0
[157230  58090 106355 ...,  43269 152588 159419]
epoch: 
1
Result from the previous epoch on dev:
F1-BAD:  0.386895475819 F1-OK:  0.907304033336
F1-score multiplied:  0.35103182569
Result from the previous epoch on test:
F1-BAD:  0.374714394516 F1-OK:  0.902836403803
F1-score multiplied:  0.338305796398
[ 10604 112260   2170 ...,  24253  38081  64020]
epoch: 
2
Result from the previous epoch on dev:
F1-BAD:  0.397212543554 F1-OK:  0.904139950922
F1-score multiplied:  0.359135729635
Result from the previous epoch on test:
F1-BAD:  0.382301764852 F1-OK:  0.901642588561
F1-score multiplied:  0.344699552873
[ 16587 157088  51334 ...,  99848  92538  64243]
epoch: 
3
Result from the previous epoch on dev:
F1-BAD:  0.414888337469 F1-OK:  0.906628652887
F1-score multiplied:  0.376149654498
Result from the previous epoch on test:
F1-BAD:  0.391697191697 F1-OK:  0.901123327909
F1-score multiplied:  0.352967476915
[ 97998  95754  98742 ..., 203012 213193 125451]
epoch: 
4
Result from the previous epoch on dev:
F1-BAD:  0.431464908149 F1-OK:  0.903586548446
F1-score multiplied:  0.38986588713
Result from the previous epoch on test:
F1-BAD:  0.409101571663 F1-OK:  0.899340659341
F1-score multiplied:  0.367921677197
[ 82707 140148 157986 ...,  51883 106201 103473]
epoch: 
5
Result from the previous epoch on dev:
F1-BAD:  0.42471042471 F1-OK:  0.905171042164
F1-score multiplied:  0.384435577753
Result from the previous epoch on test:
F1-BAD:  0.403634624582 F1-OK:  0.900661196527
F1-score multiplied:  0.363538043935
[ 96593 186600 181132 ...,  40532  34553 129546]
epoch: 
6
Result from the previous epoch on dev:
F1-BAD:  0.434535104364 F1-OK:  0.904898675602
F1-score multiplied:  0.393210240442
Result from the previous epoch on test:
F1-BAD:  0.408918406072 F1-OK:  0.900606253989
F1-score multiplied:  0.36827447388
[211235 108099  35584 ..., 173533 203627 154044]
epoch: 
7
Result from the previous epoch on dev:
F1-BAD:  0.431316042267 F1-OK:  0.905732484076
F1-score multiplied:  0.390656950385
Result from the previous epoch on test:
F1-BAD:  0.410684474124 F1-OK:  0.901534170153
F1-score multiplied:  0.370246086574
[ 44555 140306 112838 ...,  74919 150399 165017]
epoch: 
8
Result from the previous epoch on dev:
F1-BAD:  0.434657730972 F1-OK:  0.905918903848
F1-score multiplied:  0.393764655191
Result from the previous epoch on test:
F1-BAD:  0.408766079085 F1-OK:  0.901076125947
F1-score multiplied:  0.368329354961
[169730  20673 186182 ...,  67812 116880 108578]
epoch: 
9
Result from the previous epoch on dev:
F1-BAD:  0.446892904363 F1-OK:  0.898569465772
F1-score multiplied:  0.401564318331
Result from the previous epoch on test:
F1-BAD:  0.428159005108 F1-OK:  0.896106516038
F1-score multiplied:  0.383676074377
[ 41613 123879  70949 ...,  98250 105271 211104]
epoch: 
10
Result from the previous epoch on dev:
F1-BAD:  0.441176470588 F1-OK:  0.902454676721
F1-score multiplied:  0.398141769141
Result from the previous epoch on test:
F1-BAD:  0.421197522368 F1-OK:  0.898792570901
F1-score multiplied:  0.378569203986
[ 79978  62975 104982 ..., 106252 110351  75418]
epoch: 
11
Result from the previous epoch on dev:
F1-BAD:  0.44896073903 F1-OK:  0.904384066683
F1-score multiplied:  0.406032938945
Result from the previous epoch on test:
F1-BAD:  0.420253164557 F1-OK:  0.899009742212
F1-score multiplied:  0.377811689132
[ 78595 203579  77628 ..., 171197 169790 184869]
epoch: 
12
Result from the previous epoch on dev:
F1-BAD:  0.445185891325 F1-OK:  0.907206632653
F1-score multiplied:  0.403875593374
Result from the previous epoch on test:
F1-BAD:  0.413858566682 F1-OK:  0.901491584909
F1-score multiplied:  0.373090015207
[161145   8624 103629 ..., 131598  77971 145710]
epoch: 
13
Result from the previous epoch on dev:
F1-BAD:  0.450442477876 F1-OK:  0.899693102891
F1-score multiplied:  0.405259990594
Result from the previous epoch on test:
F1-BAD:  0.428635749662 F1-OK:  0.897850760686
F1-score multiplied:  0.384850933891
[ 88180  37890  33658 ..., 138465 165899 187250]
epoch: 
14
Result from the previous epoch on dev:
F1-BAD:  0.447284345048 F1-OK:  0.902738735845
F1-score multiplied:  0.403780904212
Result from the previous epoch on test:
F1-BAD:  0.426259402781 F1-OK:  0.8989197221
F1-score multiplied:  0.38317298389
[ 95272 217111 176486 ..., 183877 161754 214087]
epoch: 
15
Result from the previous epoch on dev:
F1-BAD:  0.444231689804 F1-OK:  0.90751214849
F1-score multiplied:  0.403145655241
Result from the previous epoch on test:
F1-BAD:  0.417938931298 F1-OK:  0.902773350335
F1-score multiplied:  0.377304129243
[ 43428  23203 165494 ...,  87260 127806  73850]
epoch: 
16
Result from the previous epoch on dev:
F1-BAD:  0.442669172932 F1-OK:  0.905226146716
F1-score multiplied:  0.400715709683
Result from the previous epoch on test:
F1-BAD:  0.424341338307 F1-OK:  0.901236049442
F1-score multiplied:  0.382431711351
[204644  31141 103650 ...,  30090  81702 136580]
epoch: 
17
Result from the previous epoch on dev:
F1-BAD:  0.459299681673 F1-OK:  0.904444265852
F1-score multiplied:  0.415410963397
Result from the previous epoch on test:
F1-BAD:  0.426551094891 F1-OK:  0.899052361066
F1-score multiplied:  0.383491768977
[151463 159307 100073 ...,  99926  64887  45399]
epoch: 
18
Result from the previous epoch on dev:
F1-BAD:  0.448275862069 F1-OK:  0.905249679898
F1-score multiplied:  0.405801580644
Result from the previous epoch on test:
F1-BAD:  0.423960993731 F1-OK:  0.900684520235
F1-score multiplied:  0.381855104237
[ 77558  65635  39900 ...,  52102 175656  30661]
epoch: 
19
Result from the previous epoch on dev:
F1-BAD:  0.458240285842 F1-OK:  0.902201080384
F1-score multiplied:  0.413424880962
Result from the previous epoch on test:
F1-BAD:  0.43245673185 F1-OK:  0.898345344015
F1-score multiplied:  0.388495491545
[   873  43953 181616 ..., 181234   7816  90895]
epoch: 
20
Result from the previous epoch on dev:
F1-BAD:  0.459962756052 F1-OK:  0.907155434609
F1-score multiplied:  0.41725771387
Result from the previous epoch on test:
F1-BAD:  0.422934076137 F1-OK:  0.900480384307
F1-score multiplied:  0.380843839417
[139636 198046 199708 ...,  32597 183833  57999]
epoch: 
21
Result from the previous epoch on dev:
F1-BAD:  0.46160635481 F1-OK:  0.901422107304
F1-score multiplied:  0.416102173098
Result from the previous epoch on test:
F1-BAD:  0.433962264151 F1-OK:  0.89567327133
F1-score multiplied:  0.388688400766
[ 48250  85392  96772 ..., 159417  47120  80258]
epoch: 
22
Result from the previous epoch on dev:
F1-BAD:  0.454377880184 F1-OK:  0.905067350866
F1-score multiplied:  0.411242584311
Result from the previous epoch on test:
F1-BAD:  0.423556581986 F1-OK:  0.899991986537
F1-score multiplied:  0.381197529633
[ 94976 169993 129145 ..., 212870  49486 180803]
epoch: 
23
Result from the previous epoch on dev:
F1-BAD:  0.462169553327 F1-OK:  0.905205655527
F1-score multiplied:  0.418358493484
Result from the previous epoch on test:
F1-BAD:  0.426460559218 F1-OK:  0.898629916831
F1-score multiplied:  0.383230216862
[188706 216106  14316 ...,  23716  33182   2019]
epoch: 
24
Result from the previous epoch on dev:
F1-BAD:  0.461047835991 F1-OK:  0.904957017755
F1-score multiplied:  0.417228474701
Result from the previous epoch on test:
F1-BAD:  0.428440786465 F1-OK:  0.899654812555
F1-score multiplied:  0.385448815439
[ 49555 204099 199462 ..., 142630  30997  94732]
epoch: 
25
Result from the previous epoch on dev:
F1-BAD:  0.458553791887 F1-OK:  0.900759657346
F1-score multiplied:  0.413046756455
Result from the previous epoch on test:
F1-BAD:  0.434724983433 F1-OK:  0.896651993054
F1-score multiplied:  0.389797022825
[ 10586  40093 110450 ..., 135123  16134  39003]
epoch: 
26
Result from the previous epoch on dev:
F1-BAD:  0.46091278807 F1-OK:  0.904014804087
F1-score multiplied:  0.416671983809
Result from the previous epoch on test:
F1-BAD:  0.432224485178 F1-OK:  0.89911134344
F1-score multiplied:  0.388617937536
[124016  93299 142951 ..., 121315 130112 138434]
epoch: 
27
Result from the previous epoch on dev:
F1-BAD:  0.458813108946 F1-OK:  0.901324289406
F1-score multiplied:  0.413539399391
Result from the previous epoch on test:
F1-BAD:  0.437569060773 F1-OK:  0.897225699633
F1-score multiplied:  0.39259820669
[  9469 193383  39949 ..., 202484  63420  57130]
epoch: 
28
Result from the previous epoch on dev:
F1-BAD:  0.460296096904 F1-OK:  0.903085474905
F1-score multiplied:  0.41568671927
Result from the previous epoch on test:
F1-BAD:  0.436704621567 F1-OK:  0.898303035189
F1-score multiplied:  0.392293087035
[ 70289  17736 199270 ...,  16586  34471 159323]
epoch: 
29
Result from the previous epoch on dev:
F1-BAD:  0.461059190031 F1-OK:  0.90229931424
F1-score multiplied:  0.416013390989
Result from the previous epoch on test:
F1-BAD:  0.435612788632 F1-OK:  0.897433828276
F1-score multiplied:  0.390933652548
[ 44559 117975   7356 ..., 164492 159093  15768]
Done!





-------

phrase - 72 features - deeper network 




Using TensorFlow backend.
217872
14644
7321
2018-03-13 17:39:34.717750: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
epoch: 
0
[157230  58090 106355 ...,  43269 152588 159419]
epoch: 
1
Result from the previous epoch on dev:
F1-BAD:  0.423357664234 F1-OK:  0.905855247478
F1-score multiplied:  0.383500761706
Result from the previous epoch on test:
F1-BAD:  0.398635477583 F1-OK:  0.902001270648
F1-score multiplied:  0.359569707305
[ 10604 112260   2170 ...,  24253  38081  64020]
epoch: 
2
Result from the previous epoch on dev:
F1-BAD:  0.437209302326 F1-OK:  0.903138008325
F1-score multiplied:  0.394860338524
Result from the previous epoch on test:
F1-BAD:  0.414279004227 F1-OK:  0.900359568518
F1-score multiplied:  0.373000065492
[ 16587 157088  51334 ...,  99848  92538  64243]
epoch: 
3
Result from the previous epoch on dev:
F1-BAD:  0.432203389831 F1-OK:  0.903658731427
F1-score multiplied:  0.390564366973
Result from the previous epoch on test:
F1-BAD:  0.414870944826 F1-OK:  0.901416317574
F1-score multiplied:  0.373971439354
[ 97998  95754  98742 ..., 203012 213193 125451]
epoch: 
4
Result from the previous epoch on dev:
F1-BAD:  0.457142857143 F1-OK:  0.896478815971
F1-score multiplied:  0.409818887301
Result from the previous epoch on test:
F1-BAD:  0.437977909941 F1-OK:  0.892351505289
F1-score multiplied:  0.390830247219
[ 82707 140148 157986 ...,  51883 106201 103473]
epoch: 
5
Result from the previous epoch on dev:
F1-BAD:  0.459955257271 F1-OK:  0.902716208592
F1-score multiplied:  0.415209065965
Result from the previous epoch on test:
F1-BAD:  0.432685867381 F1-OK:  0.897577492039
F1-score multiplied:  0.388369095685
[ 96593 186600 181132 ...,  40532  34553 129546]
epoch: 
6
Result from the previous epoch on dev:
F1-BAD:  0.45871559633 F1-OK:  0.899700477617
F1-score multiplied:  0.412706641109
Result from the previous epoch on test:
F1-BAD:  0.442894507411 F1-OK:  0.896518218623
F1-score multiplied:  0.397062994822
[211235 108099  35584 ..., 173533 203627 154044]
epoch: 
7
Result from the previous epoch on dev:
F1-BAD:  0.466895958727 F1-OK:  0.899317960377
F1-score multiplied:  0.419887921311
Result from the previous epoch on test:
F1-BAD:  0.447357112162 F1-OK:  0.895591458959
F1-score multiplied:  0.400649208757
[ 44555 140306 112838 ...,  74919 150399 165017]
epoch: 
8
Result from the previous epoch on dev:
F1-BAD:  0.472817133443 F1-OK:  0.895202226953
F1-score multiplied:  0.4232669508
Result from the previous epoch on test:
F1-BAD:  0.456673511294 F1-OK:  0.891637316734
F1-score multiplied:  0.407187144233
[169730  20673 186182 ...,  67812 116880 108578]
epoch: 
9
Result from the previous epoch on dev:
F1-BAD:  0.473604826546 F1-OK:  0.883569641368
F1-score multiplied:  0.418462846741
Result from the previous epoch on test:
F1-BAD:  0.459065986009 F1-OK:  0.880786699446
F1-score multiplied:  0.404339214644
[ 41613 123879  70949 ...,  98250 105271 211104]
epoch: 
10
Result from the previous epoch on dev:
F1-BAD:  0.477216238608 F1-OK:  0.896794242722
F1-score multiplied:  0.427964775317
Result from the previous epoch on test:
F1-BAD:  0.456632134877 F1-OK:  0.890211368767
F1-score multiplied:  0.406499117812
[ 79978  62975 104982 ..., 106252 110351  75418]
epoch: 
11
Result from the previous epoch on dev:
F1-BAD:  0.46980424823 F1-OK:  0.896005228331
F1-score multiplied:  0.420947062706
Result from the previous epoch on test:
F1-BAD:  0.455461839128 F1-OK:  0.89163630409
F1-score multiplied:  0.406106310894
[ 78595 203579  77628 ..., 171197 169790 184869]
epoch: 
12
Result from the previous epoch on dev:
F1-BAD:  0.473773265651 F1-OK:  0.898680566868
F1-score multiplied:  0.425770826942
Result from the previous epoch on test:
F1-BAD:  0.445275181724 F1-OK:  0.890859314346
F1-score multiplied:  0.396677543086
[161145   8624 103629 ..., 131598  77971 145710]
epoch: 
13
Result from the previous epoch on dev:
F1-BAD:  0.471590909091 F1-OK:  0.893085892593
F1-score multiplied:  0.421171187984
Result from the previous epoch on test:
F1-BAD:  0.452669902913 F1-OK:  0.888843246796
F1-score multiplied:  0.402352586232
[ 88180  37890  33658 ..., 138465 165899 187250]
epoch: 
14
Result from the previous epoch on dev:
F1-BAD:  0.469342743714 F1-OK:  0.902787878788
F1-score multiplied:  0.423716940022
Result from the previous epoch on test:
F1-BAD:  0.445736434109 F1-OK:  0.895552670021
F1-score multiplied:  0.399180453692
[ 95272 217111 176486 ..., 183877 161754 214087]
epoch: 
15
Result from the previous epoch on dev:
F1-BAD:  0.475998386446 F1-OK:  0.893200690619
F1-score multiplied:  0.425162087507
Result from the previous epoch on test:
F1-BAD:  0.457667731629 F1-OK:  0.888138385502
F1-score multiplied:  0.406472280266
[ 43428  23203 165494 ...,  87260 127806  73850]
epoch: 
16
Result from the previous epoch on dev:
F1-BAD:  0.47741364039 F1-OK:  0.904715762274
F1-score multiplied:  0.431923645585
Result from the previous epoch on test:
F1-BAD:  0.437975233543 F1-OK:  0.895199513875
F1-score multiplied:  0.392075216157
[204644  31141 103650 ...,  30090  81702 136580]
epoch: 
17
Result from the previous epoch on dev:
F1-BAD:  0.465201465201 F1-OK:  0.906244983143
F1-score multiplied:  0.42158649399
Result from the previous epoch on test:
F1-BAD:  0.43466607103 F1-OK:  0.897988795292
F1-score multiplied:  0.390325261478
[151463 159307 100073 ...,  99926  64887  45399]
epoch: 
18
Result from the previous epoch on dev:
F1-BAD:  0.470739000427 F1-OK:  0.899276481587
F1-score multiplied:  0.42332451205
Result from the previous epoch on test:
F1-BAD:  0.450821376586 F1-OK:  0.892111605866
F1-score multiplied:  0.402182982225
[ 77558  65635  39900 ...,  52102 175656  30661]
epoch: 
19
Result from the previous epoch on dev:
F1-BAD:  0.469037155414 F1-OK:  0.890518164593
F1-score multiplied:  0.417686106765
Result from the previous epoch on test:
F1-BAD:  0.459522878373 F1-OK:  0.88566228179
F1-score multiplied:  0.406982080995
[   873  43953 181616 ..., 181234   7816  90895]
epoch: 
20
Result from the previous epoch on dev:
F1-BAD:  0.459902525476 F1-OK:  0.901574485264
F1-score multiplied:  0.414636382678
Result from the previous epoch on test:
F1-BAD:  0.446748757831 F1-OK:  0.89614339592
F1-score multiplied:  0.400350948966
[139636 198046 199708 ...,  32597 183833  57999]
epoch: 
21
Result from the previous epoch on dev:
F1-BAD:  0.473445862495 F1-OK:  0.895275526079
F1-score multiplied:  0.423864493615
Result from the previous epoch on test:
F1-BAD:  0.454968629832 F1-OK:  0.889390890048
F1-score multiplied:  0.40464495463
[ 48250  85392  96772 ..., 159417  47120  80258]
epoch: 
22
Result from the previous epoch on dev:
F1-BAD:  0.464985994398 F1-OK:  0.889895412995
F1-score multiplied:  0.413788903522
Result from the previous epoch on test:
F1-BAD:  0.460726846424 F1-OK:  0.885808853951
F1-score multiplied:  0.408115919816
[ 94976 169993 129145 ..., 212870  49486 180803]
epoch: 
23
Result from the previous epoch on dev:
F1-BAD:  0.468534680753 F1-OK:  0.907180104292
F1-score multiplied:  0.42504534055
Result from the previous epoch on test:
F1-BAD:  0.43527036123 F1-OK:  0.898634771052
F1-score multiplied:  0.391149081409
[188706 216106  14316 ...,  23716  33182   2019]
epoch: 
24
Result from the previous epoch on dev:
F1-BAD:  0.460669274229 F1-OK:  0.899440888097
F1-score multiplied:  0.414344781131
Result from the previous epoch on test:
F1-BAD:  0.44917756221 F1-OK:  0.893587549906
F1-score multiplied:  0.401379477288
[ 49555 204099 199462 ..., 142630  30997  94732]
epoch: 
25
Result from the previous epoch on dev:
F1-BAD:  0.473817567568 F1-OK:  0.898484601597
F1-score multiplied:  0.425717788426
Result from the previous epoch on test:
F1-BAD:  0.452837977296 F1-OK:  0.891543591212
F1-score multiplied:  0.403724796516
[ 10586  40093 110450 ..., 135123  16134  39003]
epoch: 
26
Result from the previous epoch on dev:
F1-BAD:  0.449079754601 F1-OK:  0.889563007297
F1-score multiplied:  0.399484737019
Result from the previous epoch on test:
F1-BAD:  0.452755122339 F1-OK:  0.886608136515
F1-score multiplied:  0.401416375315
[124016  93299 142951 ..., 121315 130112 138434]
epoch: 
27
Result from the previous epoch on dev:
F1-BAD:  0.458172458172 F1-OK:  0.897408821379
F1-score multiplied:  0.411168005677
Result from the previous epoch on test:
F1-BAD:  0.446228607648 F1-OK:  0.893260028507
F1-score multiplied:  0.398598178789
[  9469 193383  39949 ..., 202484  63420  57130]
epoch: 
28
Result from the previous epoch on dev:
F1-BAD:  0.476735316552 F1-OK:  0.885856905158
F1-score multiplied:  0.422319272101
Result from the previous epoch on test:
F1-BAD:  0.461538461538 F1-OK:  0.880564014851
F1-score multiplied:  0.406414160701
[ 70289  17736 199270 ...,  16586  34471 159323]
epoch: 
29
Result from the previous epoch on dev:
F1-BAD:  0.462940943321 F1-OK:  0.888192095057
F1-score multiplied:  0.411180486336
Result from the previous epoch on test:
F1-BAD:  0.464513633727 F1-OK:  0.885184724468
F1-score multiplied:  0.411180372882
[ 44559 117975   7356 ..., 164492 159093  15768]
Done!





---

phrases 72 features - another attempt to train matern 32 and 52 - deep model 


0
[157230  58090 106355 ...,  43269 152588 159419]
epoch: 
1
Result from the previous epoch on dev:
F1-BAD:  0.421563865954 F1-OK:  0.905348486053
F1-score multiplied:  0.381662207816
Result from the previous epoch on test:
F1-BAD:  0.399420429848 F1-OK:  0.901101523044
F1-score multiplied:  0.359918357671
[ 10604 112260   2170 ...,  24253  38081  64020]
epoch: 
2
Result from the previous epoch on dev:
F1-BAD:  0.449633699634 F1-OK:  0.903515813132
F1-score multiplied:  0.406251157736
Result from the previous epoch on test:
F1-BAD:  0.424172794118 F1-OK:  0.899502726981
F1-score multiplied:  0.38154458502
[ 16587 157088  51334 ...,  99848  92538  64243]
epoch: 
3
Result from the previous epoch on dev:
F1-BAD:  0.443510737628 F1-OK:  0.90464
F1-score multiplied:  0.401217553688
Result from the previous epoch on test:
F1-BAD:  0.416608022519 F1-OK:  0.900619380619
F1-score multiplied:  0.375205259202
[ 97998  95754  98742 ..., 203012 213193 125451]
epoch: 
4
Result from the previous epoch on dev:
F1-BAD:  0.469413233458 F1-OK:  0.895824822289
F1-score multiplied:  0.420512026443
Result from the previous epoch on test:
F1-BAD:  0.441328108888 F1-OK:  0.889152583985
F1-score multiplied:  0.392408028403
[ 82707 140148 157986 ...,  51883 106201 103473]
epoch: 
5
Result from the previous epoch on dev:
F1-BAD:  0.46393588602 F1-OK:  0.902871894159
F1-score multiplied:  0.418874672179
Result from the previous epoch on test:
F1-BAD:  0.431188561215 F1-OK:  0.897388360471
F1-score multiplied:  0.386943596003
[ 96593 186600 181132 ...,  40532  34553 129546]
epoch: 
6
Result from the previous epoch on dev:
F1-BAD:  0.461329715061 F1-OK:  0.904191135066
F1-score multiplied:  0.417130238701
Result from the previous epoch on test:
F1-BAD:  0.427963253417 F1-OK:  0.897160120846
F1-score multiplied:  0.383951564153
[211235 108099  35584 ..., 173533 203627 154044]
epoch: 
7
Result from the previous epoch on dev:
F1-BAD:  0.459854014599 F1-OK:  0.904899598394
F1-score multiplied:  0.41612171313
Result from the previous epoch on test:
F1-BAD:  0.433821871477 F1-OK:  0.898965919607
F1-score multiplied:  0.389991077638
[ 44555 140306 112838 ...,  74919 150399 165017]
epoch: 
8
Result from the previous epoch on dev:
F1-BAD:  0.466638334042 F1-OK:  0.897876149402
F1-score multiplied:  0.418983430533
Result from the previous epoch on test:
F1-BAD:  0.448421052632 F1-OK:  0.893226831853
F1-score multiplied:  0.400541716178
[169730  20673 186182 ...,  67812 116880 108578]
epoch: 
9
Result from the previous epoch on dev:
F1-BAD:  0.477704536642 F1-OK:  0.888336234767
F1-score multiplied:  0.424362249412
Result from the previous epoch on test:
F1-BAD:  0.451014492754 F1-OK:  0.882179737071
F1-score multiplied:  0.397875846633
[ 41613 123879  70949 ...,  98250 105271 211104]
epoch: 
10
Result from the previous epoch on dev:
F1-BAD:  0.472336911643 F1-OK:  0.895417348609
F1-score multiplied:  0.422938665074
Result from the previous epoch on test:
F1-BAD:  0.446443172527 F1-OK:  0.888998196426
F1-score multiplied:  0.396887175183
[ 79978  62975 104982 ..., 106252 110351  75418]
epoch: 
11
Result from the previous epoch on dev:
F1-BAD:  0.468708388815 F1-OK:  0.903382032448
F1-score multiplied:  0.423422736913
Result from the previous epoch on test:
F1-BAD:  0.442835198596 F1-OK:  0.897335328131
F1-score multiplied:  0.39737166824
[ 78595 203579  77628 ..., 171197 169790 184869]
epoch: 
12
Result from the previous epoch on dev:
F1-BAD:  0.474325500435 F1-OK:  0.902138690862
F1-score multiplied:  0.427907386005
Result from the previous epoch on test:
F1-BAD:  0.440450411434 F1-OK:  0.895257397649
F1-score multiplied:  0.394316489133
[161145   8624 103629 ..., 131598  77971 145710]
epoch: 
13
Result from the previous epoch on dev:
F1-BAD:  0.476389469285 F1-OK:  0.897705935178
F1-score multiplied:  0.427657654034
Result from the previous epoch on test:
F1-BAD:  0.452877397832 F1-OK:  0.892862975666
F1-score multiplied:  0.40435746104
[ 88180  37890  33658 ..., 138465 165899 187250]
epoch: 
14
Result from the previous epoch on dev:
F1-BAD:  0.480653482373 F1-OK:  0.901916206561
F1-score multiplied:  0.433509165492
Result from the previous epoch on test:
F1-BAD:  0.449553001277 F1-OK:  0.894835298902
F1-score multiplied:  0.40227589427
[ 95272 217111 176486 ..., 183877 161754 214087]
epoch: 
15
Result from the previous epoch on dev:
F1-BAD:  0.477947072975 F1-OK:  0.892821863681
F1-score multiplied:  0.426721596435
Result from the previous epoch on test:
F1-BAD:  0.459126984127 F1-OK:  0.887578356978
F1-score multiplied:  0.407511174216
[ 43428  23203 165494 ...,  87260 127806  73850]
epoch: 
16
Result from the previous epoch on dev:
F1-BAD:  0.466609589041 F1-OK:  0.898748577929
F1-score multiplied:  0.419364704599
Result from the previous epoch on test:
F1-BAD:  0.45172195225 F1-OK:  0.894318875993
F1-score multiplied:  0.403983468598
[204644  31141 103650 ...,  30090  81702 136580]
epoch: 
17
Result from the previous epoch on dev:
F1-BAD:  0.471359860079 F1-OK:  0.902144880615
F1-score multiplied:  0.425234884697
Result from the previous epoch on test:
F1-BAD:  0.446305841924 F1-OK:  0.895339395908
F1-score multiplied:  0.399595202899
[151463 159307 100073 ...,  99926  64887  45399]
epoch: 
18
Result from the previous epoch on dev:
F1-BAD:  0.466780238501 F1-OK:  0.898161704897
F1-score multiplied:  0.419244134824
Result from the previous epoch on test:
F1-BAD:  0.45443196005 F1-OK:  0.892900906789
F1-score multiplied:  0.405762709202
[ 77558  65635  39900 ...,  52102 175656  30661]
epoch: 
19
Result from the previous epoch on dev:
F1-BAD:  0.48 F1-OK:  0.892933618844
F1-score multiplied:  0.428608137045
Result from the previous epoch on test:
F1-BAD:  0.457967622391 F1-OK:  0.884979926328
F1-score multiplied:  0.405292152724
[   873  43953 181616 ..., 181234   7816  90895]
epoch: 
20
Result from the previous epoch on dev:
F1-BAD:  0.468911917098 F1-OK:  0.900210936232
F1-score multiplied:  0.422119635902
Result from the previous epoch on test:
F1-BAD:  0.453664700927 F1-OK:  0.894295028525
F1-score multiplied:  0.405710086656
[139636 198046 199708 ...,  32597 183833  57999]
epoch: 
21
Result from the previous epoch on dev:
F1-BAD:  0.469729497638 F1-OK:  0.899699504589
F1-score multiplied:  0.422615396316
Result from the previous epoch on test:
F1-BAD:  0.455291394908 F1-OK:  0.894477277359
F1-score multiplied:  0.407247807323
[ 48250  85392  96772 ..., 159417  47120  80258]
epoch: 
22
Result from the previous epoch on dev:
F1-BAD:  0.466409927257 F1-OK:  0.898659081674
F1-score multiplied:  0.419143516913
Result from the previous epoch on test:
F1-BAD:  0.452587991718 F1-OK:  0.891896312045
F1-score multiplied:  0.40366156069
[ 94976 169993 129145 ..., 212870  49486 180803]
epoch: 
23
Result from the previous epoch on dev:
F1-BAD:  0.455719557196 F1-OK:  0.905403238737
F1-score multiplied:  0.41260996304
Result from the previous epoch on test:
F1-BAD:  0.435553534894 F1-OK:  0.900237052513
F1-score multiplied:  0.392101430465
[188706 216106  14316 ...,  23716  33182   2019]
epoch: 
24
Result from the previous epoch on dev:
F1-BAD:  0.454503033131 F1-OK:  0.906472517801
F1-score multiplied:  0.411994508791
Result from the previous epoch on test:
F1-BAD:  0.43558000917 F1-OK:  0.901227633796
F1-score multiplied:  0.392556740993
[ 49555 204099 199462 ..., 142630  30997  94732]
epoch: 
25
Result from the previous epoch on dev:
F1-BAD:  0.47717348154 F1-OK:  0.89136352388
F1-score multiplied:  0.425335036008
Result from the previous epoch on test:
F1-BAD:  0.459653898503 F1-OK:  0.884903706772
F1-score multiplied:  0.406749438617
[ 10586  40093 110450 ..., 135123  16134  39003]
epoch: 
26
Result from the previous epoch on dev:
F1-BAD:  0.46909537454 F1-OK:  0.89367980982
F1-score multiplied:  0.419221065106
Result from the previous epoch on test:
F1-BAD:  0.458649620455 F1-OK:  0.888394695659
F1-score multiplied:  0.407461889979
[124016  93299 142951 ..., 121315 130112 138434]
epoch: 
27
Result from the previous epoch on dev:
F1-BAD:  0.470541401274 F1-OK:  0.890354492993
F1-score multiplied:  0.418948650763
Result from the previous epoch on test:
F1-BAD:  0.458253174298 F1-OK:  0.883105022831
F1-score multiplied:  0.404685679951
[  9469 193383  39949 ..., 202484  63420  57130]
epoch: 
28
Result from the previous epoch on dev:
F1-BAD:  0.473597359736 F1-OK:  0.895563922082
F1-score multiplied:  0.424136708973
Result from the previous epoch on test:
F1-BAD:  0.457739409757 F1-OK:  0.888879746575
F1-score multiplied:  0.406875290542
[ 70289  17736 199270 ...,  16586  34471 159323]
epoch: 
29
Result from the previous epoch on dev:
F1-BAD:  0.473264166002 F1-OK:  0.891232696111
F1-score multiplied:  0.421788498638
Result from the previous epoch on test:
F1-BAD:  0.461449498843 F1-OK:  0.884066390041
F1-score multiplied:  0.407951992629
[ 44559 117975   7356 ..., 164492 159093  15768]
Done!


Matern 32 - phrase - 72 features - with pretraining


Epoch 1: \Dev set LL -0.5317967134389773, Acc 0.8441469669342041, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.445847498786 F1-OK:  0.909322101248
F1-score multiplied:  0.405418984432
Epoch 1: 
Test set LL -0.5643179174809212, Acc 0.8339251279830933, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.417624521073 F1-OK:  0.903153870659
F1-score multiplied:  0.377179202689
[ 10604 112260   2170 ...,  24253  38081  64020]
Epoch 2: \Dev set LL -0.7414332546495933, Acc 0.8326731324195862, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.475823705605 F1-OK:  0.900446972775
F1-score multiplied:  0.428454015287
Epoch 2: 
Test set LL -0.7922857995033465, Acc 0.8214285969734192, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.447262735151 F1-OK:  0.893513051268
F1-score multiplied:  0.399635091204
[ 16587 157088  51334 ...,  99848  92538  64243]
Epoch 3: \Dev set LL -0.822730959338397, Acc 0.842644453048706, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.483408071749 F1-OK:  0.907186593619
F1-score multiplied:  0.438541321938
Epoch 3: 
Test set LL -0.8806207300139655, Acc 0.8286670446395874, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.446014572753 F1-OK:  0.898663112404
F1-score multiplied:  0.400816844128
[ 97998  95754  98742 ..., 203012 213193 125451]
Epoch 4: \Dev set LL -0.9046720874178723, Acc 0.8295314908027649, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.495553759095 F1-OK:  0.897435897436
F1-score multiplied:  0.444727732521
Epoch 4: 
Test set LL -0.973457398827871, Acc 0.8141900897026062, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.4578601315 F1-OK:  0.887881659731
F1-score multiplied:  0.406525613481
[ 82707 140148 157986 ...,  51883 106201 103473]
Epoch 5: \Dev set LL -0.9021023607794195, Acc 0.8401857614517212, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.479537366548 F1-OK:  0.905599483621
F1-score multiplied:  0.434268791523
Epoch 5: 
Test set LL -0.9764071977918957, Acc 0.8279158473014832, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.444689290436 F1-OK:  0.898181818182
F1-score multiplied:  0.39941183541
[ 96593 186600 181132 ...,  40532  34553 129546]
Epoch 6: \Dev set LL -0.9432990488379764, Acc 0.8354049921035767, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.49134655973 F1-OK:  0.901816996659
F1-score multiplied:  0.443104678814
Epoch 6: 
Test set LL -1.0160416595664115, Acc 0.821906566619873, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.451178451178 F1-OK:  0.893707205739
F1-score multiplied:  0.403221432892
[211235 108099  35584 ..., 173533 203627 154044]
Epoch 7: \Dev set LL -0.9431846219489657, Acc 0.8427810668945312, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.479891549932 F1-OK:  0.907393997908
F1-score multiplied:  0.435450712055
Epoch 7: 
Test set LL -1.0101934478071657, Acc 0.8294181823730469, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.444888888889 F1-OK:  0.89922543166
F1-score multiplied:  0.400055403152
[ 44555 140306 112838 ...,  74919 150399 165017]
Epoch 8: \Dev set LL -1.0172457582508028, Acc 0.8281655311584473, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.482730263158 F1-OK:  0.89696969697
F1-score multiplied:  0.432994417863
Epoch 8: 
Test set LL -1.0901821304376451, Acc 0.8162387609481812, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.458442342524 F1-OK:  0.889345779021
F1-score multiplied:  0.407713762248
[169730  20673 186182 ...,  67812 116880 108578]
Epoch 9: \Dev set LL -1.0096187677743234, Acc 0.8323999643325806, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.475865014951 F1-OK:  0.900252012032
F1-score multiplied:  0.428398437165
Epoch 9: 
Test set LL -1.0729610610574134, Acc 0.8224529027938843, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.450781580059 F1-OK:  0.894110939155
F1-score multiplied:  0.4030487419
[ 41613 123879  70949 ...,  98250 105271 211104]
Epoch 10: \Dev set LL -0.9806860989110072, Acc 0.8395028114318848, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.482606781154 F1-OK:  0.905019804381
F1-score multiplied:  0.436768694673
Epoch 10: 
Test set LL -1.0672185837822787, Acc 0.8277792930603027, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.451023073574 F1-OK:  0.897869927918
F1-score multiplied:  0.404960054559
[ 79978  62975 104982 ..., 106252 110351  75418]
Epoch 11: \Dev set LL -1.0342771848170167, Acc 0.8303510546684265, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.476832350463 F1-OK:  0.898761004239
F1-score multiplied:  0.428558322156
Epoch 11: 
Test set LL -1.1117654572047784, Acc 0.8182190656661987, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.457399103139 F1-OK:  0.890821097531
F1-score multiplied:  0.407460771068
[ 78595 203579  77628 ..., 171197 169790 184869]
Epoch 12: \Dev set LL -1.0294109248256922, Acc 0.8343122601509094, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.485799067401 F1-OK:  0.901245624033
F1-score multiplied:  0.437824283655
Epoch 12: 
Test set LL -1.1032648979477784, Acc 0.8220431804656982, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.456403838131 F1-OK:  0.893606597534
F1-score multiplied:  0.407845480894
[161145   8624 103629 ..., 131598  77971 145710]
Epoch 13: \Dev set LL -1.0292192596211092, Acc 0.8318535685539246, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.485582950272 F1-OK:  0.899502000163
F1-score multiplied:  0.436782835015
Epoch 13: 
Test set LL -1.1173749529942627, Acc 0.8178776502609253, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.454935622318 F1-OK:  0.890674318508
F1-score multiplied:  0.405199475373
[ 88180  37890  33658 ..., 138465 165899 187250]
Epoch 14: \Dev set LL -1.0051843695959282, Acc 0.8412784934043884, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.47277676951 F1-OK:  0.906576620035
F1-score multiplied:  0.428608365734
Epoch 14: 
Test set LL -1.0749299691493688, Acc 0.8292816281318665, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.447147279965 F1-OK:  0.899055156263
F1-score multiplied:  0.402010067661
[ 95272 217111 176486 ..., 183877 161754 214087]
Epoch 15: \Dev set LL -0.9827034614185041, Acc 0.8452396988868713, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.481938728852 F1-OK:  0.909032517061
F1-score multiplied:  0.438097975758
Epoch 15: 
Test set LL -1.0720398935261657, Acc 0.8315351009368896, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.44399368943 F1-OK:  0.900728340912
F1-score multiplied:  0.399917699255
[ 43428  23203 165494 ...,  87260 127806  73850]
Epoch 16: \Dev set LL -1.0336514923620046, Acc 0.8334926962852478, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.476149548775 F1-OK:  0.90101502233
F1-score multiplied:  0.429017896322
Epoch 16: 
Test set LL -1.1146243445697819, Acc 0.8216334581375122, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.457415870378 F1-OK:  0.893274495383
F1-score multiplied:  0.408597930792
[204644  31141 103650 ...,  30090  81702 136580]
Epoch 17: \Dev set LL -1.0335432323332616, Acc 0.8373172879219055, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.476022877255 F1-OK:  0.903710890129
F1-score multiplied:  0.430187058125
Epoch 17: 
Test set LL -1.0960867820434046, Acc 0.8261404037475586, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.449156209433 F1-OK:  0.896780994081
F1-score multiplied:  0.402794751993
[151463 159307 100073 ...,  99926  64887  45399]
Epoch 18: \Dev set LL -1.0802811821178793, Acc 0.8284387588500977, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.484823625923 F1-OK:  0.897082923632
F1-score multiplied:  0.434926995789
Epoch 18: 
Test set LL -1.1513586119913428, Acc 0.816511869430542, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.458156886469 F1-OK:  0.889555674298
F1-score multiplied:  0.407556058077
[ 77558  65635  39900 ...,  52102 175656  30661]
Epoch 19: \Dev set LL -1.0281657284548515, Acc 0.8393661975860596, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.488250652742 F1-OK:  0.904731043422
F1-score multiplied:  0.441735522506
Epoch 19: 
Test set LL -1.1127050801873026, Acc 0.8255941271781921, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.460726351351 F1-OK:  0.895975887911
F1-score multiplied:  0.412799701736
[   873  43953 181616 ..., 181234   7816  90895]
Epoch 20: \Dev set LL -1.0510074391165014, Acc 0.833629310131073, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.482142857143 F1-OK:  0.900895036615
F1-score multiplied:  0.434360106939
Epoch 20: 
Test set LL -1.1379515722434475, Acc 0.8206091523170471, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.455544041451 F1-OK:  0.892613334423
F1-score multiplied:  0.406624685816
[139636 198046 199708 ...,  32597 183833  57999]
Epoch 21: \Dev set LL -1.0147181641131544, Acc 0.8408687114715576, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.492816717458 F1-OK:  0.90562980964
F1-score multiplied:  0.446309510018
Epoch 21: 
Test set LL -1.1018311788934476, Acc 0.8279158473014832, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.461768475011 F1-OK:  0.897585954645
F1-score multiplied:  0.414476897468
[ 48250  85392  96772 ..., 159417  47120  80258]
Epoch 22: \Dev set LL -1.0201790815308003, Acc 0.8389564156532288, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.479470198675 F1-OK:  0.904742667852
F1-score multiplied:  0.433797146705
Epoch 22: 
Test set LL -1.1266974187233887, Acc 0.8242966532707214, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.451034777043 F1-OK:  0.89541075566
F1-score multiplied:  0.403861390541
[ 94976 169993 129145 ..., 212870  49486 180803]
Epoch 23: \Dev set LL -1.0064083248866522, Acc 0.84209805727005, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.481149012567 F1-OK:  0.906879329789
F1-score multiplied:  0.436344094046
Epoch 23: 
Test set LL -1.1146642344517816, Acc 0.8261404037475586, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.441666666667 F1-OK:  0.897039792947
F1-score multiplied:  0.396192575218
[188706 216106  14316 ...,  23716  33182   2019]
Epoch 24: \Dev set LL -1.0099869041758414, Acc 0.842644453048706, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.476839237057 F1-OK:  0.907395498392
F1-score multiplied:  0.432681777163
Epoch 24: 
Test set LL -1.0930057723607995, Acc 0.830442488193512, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.446870126977 F1-OK:  0.899874994959
F1-score multiplied:  0.402127253261
[ 49555 204099 199462 ..., 142630  30997  94732]
Epoch 25: \Dev set LL -1.0230482273660328, Acc 0.8414151072502136, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.481464939705 F1-OK:  0.906393614448
F1-score multiplied:  0.436396746929
Epoch 25: 
Test set LL -1.104279161752242, Acc 0.8292133212089539, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.455001089562 F1-OK:  0.89874083971
F1-score multiplied:  0.408928061302
[ 10586  40093 110450 ..., 135123  16134  39003]
Epoch 26: \Dev set LL -1.0208072650859756, Acc 0.841141939163208, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.484714222419 F1-OK:  0.906096083973
F1-score multiplied:  0.43919765878
Epoch 26: 
Test set LL -1.1172976454888421, Acc 0.826072096824646, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.453785116878 F1-OK:  0.896568527919
F1-score multiplied:  0.40684945423
[124016  93299 142951 ..., 121315 130112 138434]
Epoch 27: \Dev set LL -1.01891367264576, Acc 0.8419615030288696, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.485091232755 F1-OK:  0.906655909641
F1-score multiplied:  0.439810832892
Epoch 27: 
Test set LL -1.11134793738784, Acc 0.8270964026451111, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.45595186936 F1-OK:  0.897215230982
F1-score multiplied:  0.409086961784
[  9469 193383  39949 ..., 202484  63420  57130]
Epoch 28: \Dev set LL -1.0167251025134512, Acc 0.8431908488273621, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.478181818182 F1-OK:  0.907731875904
F1-score multiplied:  0.434060878841
Epoch 28: 
Test set LL -1.104578262022107, Acc 0.8299645185470581, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.453227931489 F1-OK:  0.89932885906
F1-score multiplied:  0.40760095852
[ 70289  17736 199270 ...,  16586  34471 159323]
Epoch 29: \Dev set LL -1.0650517178654968, Acc 0.833629310131073, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.49292256453 F1-OK:  0.900490196078
F1-score multiplied:  0.443871936785
Epoch 29: 
Test set LL -1.168633653858893, Acc 0.8166484832763672, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.457027300303 F1-OK:  0.889701351518
F1-score multiplied:  0.40661780676
[ 44559 117975   7356 ..., 164492 159093  15768]
Done!



phrase with Matern 52 - pretraining - 72 features


Epoch 1: \Dev set LL -0.5886904706076116, Acc 0.847152054309845, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.419906687403 F1-OK:  0.911979863132
F1-score multiplied:  0.382946443306
Epoch 1: 
Test set LL -0.6218492284270157, Acc 0.836042046546936, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.388591800357 F1-OK:  0.905327077008
F1-score multiplied:  0.351802678766
[ 10604 112260   2170 ...,  24253  38081  64020]
Epoch 2: \Dev set LL -0.7242870865098034, Acc 0.8414151072502136, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.477732793522 F1-OK:  0.906514212094
F1-score multiplied:  0.433071566911
Epoch 2: 
Test set LL -0.7758411213946459, Acc 0.8255258202552795, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.434108527132 F1-OK:  0.896863520769
F1-score multiplied:  0.389336102039
[ 16587 157088  51334 ...,  99848  92538  64243]
Epoch 3: \Dev set LL -0.8333930505171782, Acc 0.8419615030288696, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.474329850068 F1-OK:  0.907001044932
F1-score multiplied:  0.430217669654
Epoch 3: 
Test set LL -0.884630591843767, Acc 0.8296913504600525, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.439550561798 F1-OK:  0.899589338916
F1-score multiplied:  0.395414999308
[ 97998  95754  98742 ..., 203012 213193 125451]
Epoch 4: \Dev set LL -0.9196170201539091, Acc 0.8319901823997498, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.483193277311 F1-OK:  0.899690099494
F1-score multiplied:  0.434724207739
Epoch 4: 
Test set LL -0.9795385583352432, Acc 0.8180825114250183, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.452977412731 F1-OK:  0.890900155623
F1-score multiplied:  0.403557647496
[ 82707 140148 157986 ...,  51883 106201 103473]
Epoch 5: \Dev set LL -0.9309618816213984, Acc 0.833082914352417, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.495874587459 F1-OK:  0.899983630709
F1-score multiplied:  0.446279011597
Epoch 5: 
Test set LL -1.0070181076861324, Acc 0.8177410364151001, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.460699131138 F1-OK:  0.890340605612
F1-score multiplied:  0.410179143422
[ 96593 186600 181132 ...,  40532  34553 129546]
Epoch 6: \Dev set LL -0.9441419800394448, Acc 0.8407321572303772, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.486343612335 F1-OK:  0.905754930488
F1-score multiplied:  0.440508124784
Epoch 6: 
Test set LL -0.9927683936983358, Acc 0.8277792930603027, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.451739130435 F1-OK:  0.897845106935
F1-score multiplied:  0.405591767872
[211235 108099  35584 ..., 173533 203627 154044]
Epoch 7: \Dev set LL -0.9378020719555039, Acc 0.8431908488273621, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.473877176902 F1-OK:  0.907865168539
F1-score multiplied:  0.430216583075
Epoch 7: 
Test set LL -1.0077464988533336, Acc 0.830442488193512, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.444891571652 F1-OK:  0.89993955269
F1-score multiplied:  0.400375521988
[ 44555 140306 112838 ...,  74919 150399 165017]

Epoch 8: \Dev set LL -0.9768918776571536, Acc 0.8374539017677307, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.478527607362 F1-OK:  0.903721682848
F1-score multiplied:  0.432455774614
Epoch 8: 
Test set LL -1.0372929614513189, Acc 0.8285304307937622, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.460116104064 F1-OK:  0.898080123392
F1-score multiplied:  0.413221127512
[169730  20673 186182 ...,  67812 116880 108578]
Epoch 9: \Dev set LL -1.0226507796747784, Acc 0.8302144408226013, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.482729920932 F1-OK:  0.898439414985
F1-score multiplied:  0.433703587758
Epoch 9: 
Test set LL -1.079272649095397, Acc 0.820882260799408, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.459732234809 F1-OK:  0.892645192977
F1-score multiplied:  0.410377769459
[ 41613 123879  70949 ...,  98250 105271 211104]
Epoch 10: \Dev set LL -0.9833795362607404, Acc 0.8404589295387268, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.472448057814 F1-OK:  0.906018667525
F1-score multiplied:  0.428046759815
Epoch 10: 
Test set LL -1.045458423237323, Acc 0.8286670446395874, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.445279681627 F1-OK:  0.898687664042
F1-score multiplied:  0.400167356927
[ 79978  62975 104982 ..., 106252 110351  75418]
Epoch 11: \Dev set LL -1.0179005733793314, Acc 0.8348585963249207, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.469038208169 F1-OK:  0.90222401941
F1-score multiplied:  0.423177537431
Epoch 11: 
Test set LL -1.063672038101324, Acc 0.8281207084655762, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.458360232408 F1-OK:  0.897853171543
F1-score multiplied:  0.411540188377
[ 78595 203579  77628 ..., 171197 169790 184869]
Epoch 12: \Dev set LL -1.0687241642840206, Acc 0.8280289769172668, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.480396203054 F1-OK:  0.896963744987
F1-score multiplied:  0.430897977369
Epoch 12: 
Test set LL -1.1302291183884479, Acc 0.8165801763534546, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.458686013704 F1-OK:  0.889583162049
F1-score multiplied:  0.408039354459
[161145   8624 103629 ..., 131598  77971 145710]
Epoch 13: \Dev set LL -1.0102787236289477, Acc 0.8381368517875671, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.475431606906 F1-OK:  0.904304288137
F1-score multiplied:  0.429934840841
Epoch 13: 
Test set LL -1.0732465130443738, Acc 0.8290084600448608, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.4568329718 F1-OK:  0.898533106411
F1-score multiplied:  0.410479549263
[ 88180  37890  33658 ..., 138465 165899 187250]
Epoch 14: \Dev set LL -1.0102172951885864, Acc 0.8399125933647156, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.485061511424 F1-OK:  0.905224001294
F1-score multiplied:  0.439089322244
Epoch 14: 
Test set LL -1.092741027894261, Acc 0.825867235660553, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.459287531807 F1-OK:  0.896223343643
F1-score multiplied:  0.411624207449
[ 95272 217111 176486 ..., 183877 161754 214087]
Epoch 15: \Dev set LL -1.017819369350204, Acc 0.8389564156532288, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.479470198675 F1-OK:  0.904742667852
F1-score multiplied:  0.433797146705
Epoch 15: 
Test set LL -1.0782003463422094, Acc 0.8266184329986572, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.449360225548 F1-OK:  0.897110669855
F1-score multiplied:  0.403125852947
[ 43428  23203 165494 ...,  87260 127806  73850]
Epoch 16: \Dev set LL -1.002304406720373, Acc 0.8423712849617004, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.461753731343 F1-OK:  0.907665226436
F1-score multiplied:  0.419117805117
Epoch 16: 
Test set LL -1.055083966857053, Acc 0.8336520195007324, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.442817932296 F1-OK:  0.902231497833
F1-score multiplied:  0.399524286323
[204644  31141 103650 ...,  30090  81702 136580]
Epoch 17: \Dev set LL -0.9913352223501327, Acc 0.8437371850013733, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.466417910448 F1-OK:  0.908465354457
F1-score multiplied:  0.42372451234
Epoch 17: 
Test set LL -1.052630045922301, Acc 0.8344031572341919, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.445207046442 F1-OK:  0.902676887266
F1-score multiplied:  0.401878110871
[151463 159307 100073 ...,  99926  64887  45399]
Epoch 18: \Dev set LL -1.0675908312926887, Acc 0.8308973908424377, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.481574539363 F1-OK:  0.898971764322
F1-score multiplied:  0.432921913304
Epoch 18: 
Test set LL -1.1200798368334022, Acc 0.8206091523170471, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.462011058775 F1-OK:  0.892358123335
F1-score multiplied:  0.412279321369
[ 77558  65635  39900 ...,  52102 175656  30661]
Epoch 19: \Dev set LL -1.0456840858590288, Acc 0.8334926962852478, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.479726845924 F1-OK:  0.900886250915
F1-score multiplied:  0.432179319688
Epoch 19: 
Test set LL -1.1185648564903359, Acc 0.8248429298400879, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.459203036053 F1-OK:  0.895498064779
F1-score multiplied:  0.411215430126
[   873  43953 181616 ..., 181234   7816  90895]
Epoch 20: \Dev set LL -1.0032031673293027, Acc 0.84209805727005, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.477868112014 F1-OK:  0.90698422916
F1-score multiplied:  0.433418841216
Epoch 20: 
Test set LL -1.0791478114854969, Acc 0.8320131301879883, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.451137884873 F1-OK:  0.900830444247
F1-score multiplied:  0.406398741247
[139636 198046 199708 ...,  32597 183833  57999]
Epoch 21: \Dev set LL -1.0635306245103435, Acc 0.833629310131073, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.48433530906 F1-OK:  0.900814332248
F1-score multiplied:  0.436296188015
Epoch 21: 
Test set LL -1.1171551718428436, Acc 0.8241600394248962, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.464322862492 F1-OK:  0.894816388219
F1-score multiplied:  0.415483706783
[ 48250  85392  96772 ..., 159417  47120  80258]
Epoch 22: \Dev set LL -1.0267701772978315, Acc 0.8380002975463867, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.477533039648 F1-OK:  0.90413837698
F1-score multiplied:  0.431755947421
Epoch 22: 
Test set LL -1.0907690872504385, Acc 0.8287352919578552, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.456907752274 F1-OK:  0.898338062424
F1-score multiplied:  0.410457624884
[ 94976 169993 129145 ..., 212870  49486 180803]
Epoch 23: \Dev set LL -1.0300583703643325, Acc 0.8382734656333923, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.461328480437 F1-OK:  0.904853744777
F1-score multiplied:  0.417434803095
Epoch 23: 
Test set LL -1.0943273109255305, Acc 0.8305107951164246, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.440990990991 F1-OK:  0.900112685126
F1-score multiplied:  0.396941585017
[188706 216106  14316 ...,  23716  33182   2019]
Epoch 24: \Dev set LL -1.0320721784201043, Acc 0.8374539017677307, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.478070175439 F1-OK:  0.903737259343
F1-score multiplied:  0.432049830125
Epoch 24: 
Test set LL -1.0998241086417961, Acc 0.8289401531219482, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.459546925566 F1-OK:  0.898389648319
F1-score multiplied:  0.412852200845
[ 49555 204099 199462 ..., 142630  30997  94732]
Epoch 25: \Dev set LL -1.0736388090539843, Acc 0.8319901823997498, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.46288209607 F1-OK:  0.900420984456
F1-score multiplied:  0.41678875263
Epoch 25: 
Test set LL -1.1065006295080897, Acc 0.8273012638092041, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.4567132116 F1-OK:  0.897332846182
F1-score multiplied:  0.409823766054
[ 10586  40093 110450 ..., 135123  16134  39003]
Epoch 26: \Dev set LL -1.053831890048353, Acc 0.835131824016571, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.478166882836 F1-OK:  0.902100738097
F1-score multiplied:  0.43135469794
Epoch 26: 
Test set LL -1.111943750508736, Acc 0.8260038495063782, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.458793542906 F1-OK:  0.896338486574
F1-score multiplied:  0.411234309898
[124016  93299 142951 ..., 121315 130112 138434]
Epoch 27: \Dev set LL -1.0234836347724807, Acc 0.8407321572303772, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.480854853072 F1-OK:  0.905937399161
F1-score multiplied:  0.435624394966
Epoch 27: 
Test set LL -1.0836094275733468, Acc 0.8309205174446106, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.461036134088 F1-OK:  0.899732728598
F1-score multiplied:  0.414809298905
[  9469 193383  39949 ..., 202484  63420  57130]
Epoch 28: \Dev set LL -1.0116532457223018, Acc 0.8419615030288696, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.468534680753 F1-OK:  0.907180104292
F1-score multiplied:  0.42504534055
Epoch 28: 
Test set LL -1.0724263136716359, Acc 0.8333788514137268, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.454626732231 F1-OK:  0.901668412993
F1-score multiplied:  0.409922564154
[ 70289  17736 199270 ...,  16586  34471 159323]
Epoch 29: \Dev set LL -1.0534986115604152, Acc 0.8360879421234131, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.477806788512 F1-OK:  0.902786779002
F1-score multiplied:  0.431357651586
Epoch 29: 
Test set LL -1.1307988660213832, Acc 0.8246380686759949, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.454545454545 F1-OK:  0.895524816924
F1-score multiplied:  0.407056734966
[ 44559 117975   7356 ..., 164492 159093  15768]
Done!




Matern52 from scratch - phrase 72 features

Epoch 1: \Dev set LL -0.6964280706933401, Acc 0.8313071727752686, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.400194269063 F1-OK:  0.901851704681
F1-score multiplied:  0.360915883758
Epoch 1: 
Test set LL -0.7269325071410395, Acc 0.8267549872398376, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.389704113543 F1-OK:  0.899048983327
F1-score multiplied:  0.35036308708
[ 10604 112260   2170 ...,  24253  38081  64020]
Epoch 2: \Dev set LL -0.8398386819371361, Acc 0.8382734656333923, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.423563777994 F1-OK:  0.905942167143
F1-score multiplied:  0.383724286959
Epoch 2: 
Test set LL -0.8893474096376841, Acc 0.8300327658653259, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.393666260658 F1-OK:  0.901163483302
F1-score multiplied:  0.354757658713
[ 16587 157088  51334 ...,  99848  92538  64243]
Epoch 3: \Dev set LL -0.8870523748508732, Acc 0.8498839139938354, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.405624661979 F1-OK:  0.914093644962
F1-score multiplied:  0.370778925755
Epoch 3: 
Test set LL -0.9555119836436368, Acc 0.8390467166900635, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.366568126848 F1-OK:  0.907810849924
F1-score multiplied:  0.332774522788
[ 97998  95754  98742 ..., 203012 213193 125451]
Epoch 4: \Dev set LL -0.9554108302391628, Acc 0.8347220420837402, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.462222222222 F1-OK:  0.902356358941
F1-score multiplied:  0.417089161466
Epoch 4: 
Test set LL -1.0268548144029865, Acc 0.8225894570350647, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.426743159753 F1-OK:  0.895055744062
F1-score multiplied:  0.381958916376
[ 82707 140148 157986 ...,  51883 106201 103473]
Epoch 5: \Dev set LL -0.9777564988170486, Acc 0.8349952101707458, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.461675579323 F1-OK:  0.902564929827
F1-score multiplied:  0.416692186854
Epoch 5: 
Test set LL -1.0493239048071066, Acc 0.8241600394248962, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.426631039857 F1-OK:  0.89615679316
F1-score multiplied:  0.382328304541
[ 96593 186600 181132 ...,  40532  34553 129546]
Epoch 6: \Dev set LL -0.991246149469982, Acc 0.8373172879219055, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.439001413095 F1-OK:  0.904864605799
F1-score multiplied:  0.397236840605
Epoch 6: 
Test set LL -1.0338458799527435, Acc 0.8291450142860413, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.411016949153 F1-OK:  0.900079872204
F1-score multiplied:  0.369948083067
[211235 108099  35584 ..., 173533 203627 154044]
Epoch 7: \Dev set LL -0.978107286374424, Acc 0.8403223752975464, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.455011655012 F1-OK:  0.906457549812
F1-score multiplied:  0.412448749938
Epoch 7: 
Test set LL -1.0485034224996772, Acc 0.8288719058036804, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.421781264421 F1-OK:  0.899575218402
F1-score multiplied:  0.379423973059
[ 44555 140306 112838 ...,  74919 150399 165017]
Epoch 8: \Dev set LL -1.0253391857330956, Acc 0.8313071727752686, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.462342185459 F1-OK:  0.899959497772
F1-score multiplied:  0.416089241025
Epoch 8: 
Test set LL -1.0894925160107165, Acc 0.8225211501121521, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.440955044096 F1-OK:  0.894516822923
F1-score multiplied:  0.394441705096
[169730  20673 186182 ...,  67812 116880 108578]
Epoch 9: \Dev set LL -1.0184721073022156, Acc 0.834585428237915, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.472331154684 F1-OK:  0.901919494614
F1-score multiplied:  0.426004676323
Epoch 9: 
Test set LL -1.0967052816775087, Acc 0.8234089016914368, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.441950798446 F1-OK:  0.895108298856
F1-score multiplied:  0.395593827375
[ 41613 123879  70949 ...,  98250 105271 211104]
Epoch 10: \Dev set LL -0.9930524127954933, Acc 0.842644453048706, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.456090651558 F1-OK:  0.908016608112
F1-score multiplied:  0.41413788642
Epoch 10: 
Test set LL -1.0612854628375026, Acc 0.8326277136802673, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.419881656805 F1-OK:  0.902206439772
F1-score multiplied:  0.378819934711
[ 79978  62975 104982 ..., 106252 110351  75418]
Epoch 11: \Dev set LL -0.9953476371206811, Acc 0.8423712849617004, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.457706766917 F1-OK:  0.907783282723
F1-score multiplied:  0.415498551397
Epoch 11: 
Test set LL -1.0778914939977369, Acc 0.8316716551780701, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.423391812865 F1-OK:  0.901451245352
F1-score multiplied:  0.38166707698
[ 78595 203579  77628 ..., 171197 169790 184869]
Epoch 12: \Dev set LL -1.0200978035209314, Acc 0.840049147605896, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.475593372145 F1-OK:  0.9056330083
F1-score multiplied:  0.430713056344
Epoch 12: 
Test set LL -1.1125243366866848, Acc 0.826276957988739, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.43163538874 F1-OK:  0.897468966629
F1-score multiplied:  0.387379366293
[161145   8624 103629 ..., 131598  77971 145710]
Epoch 13: \Dev set LL -1.0346106626092597, Acc 0.8371807336807251, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.471162377995 F1-OK:  0.903777849532
F1-score multiplied:  0.425826120764
Epoch 13: 
Test set LL -1.0981204387061927, Acc 0.825867235660553, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.438325991189 F1-OK:  0.896961370616
F1-score multiplied:  0.393161481834
[ 88180  37890  33658 ..., 138465 165899 187250]
Epoch 14: \Dev set LL -0.993435770460614, Acc 0.8429176211357117, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.451335877863 F1-OK:  0.908337318667
F1-score multiplied:  0.409965221116
Epoch 14: 
Test set LL -1.0569575186981066, Acc 0.8320131301879883, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.423347398031 F1-OK:  0.901686515866
F1-score multiplied:  0.381726640331
[ 95272 217111 176486 ..., 183877 161754 214087]
Epoch 15: \Dev set LL -1.0385346888882288, Acc 0.8360879421234131, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.470432480141 F1-OK:  0.903038138332
F1-score multiplied:  0.424818471078
Epoch 15: 
Test set LL -1.1021082375516742, Acc 0.8270964026451111, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.446194225722 F1-OK:  0.897556238874
F1-score multiplied:  0.400484411046
[ 43428  23203 165494 ...,  87260 127806  73850]
Epoch 16: \Dev set LL -1.0595034438506457, Acc 0.8358147740364075, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.465302491103 F1-OK:  0.903017589156
F1-score multiplied:  0.420176333744
Epoch 16: 
Test set LL -1.1287622433191273, Acc 0.825867235660553, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.438325991189 F1-OK:  0.896961370616
F1-score multiplied:  0.393161481834
[204644  31141 103650 ...,  30090  81702 136580]
Epoch 17: \Dev set LL -1.0610870234125138, Acc 0.8328097462654114, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.465502183406 F1-OK:  0.900906735751
F1-score multiplied:  0.419374052538
Epoch 17: 
Test set LL -1.1356006347772674, Acc 0.8241600394248962, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.443964586482 F1-OK:  0.895567181733
F1-score multiplied:  0.397600113505
[151463 159307 100073 ...,  99926  64887  45399]
Epoch 18: \Dev set LL -1.0504506524315025, Acc 0.8356781601905823, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.459325842697 F1-OK:  0.903116694854
F1-score multiplied:  0.414824836917
Epoch 18: 
Test set LL -1.1021616330897601, Acc 0.8282573223114014, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.440489432703 F1-OK:  0.898560077441
F1-score multiplied:  0.395806218762
[ 77558  65635  39900 ...,  52102 175656  30661]
Epoch 19: \Dev set LL -1.0473255515813522, Acc 0.8373172879219055, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.46230248307 F1-OK:  0.904160296129
F1-score multiplied:  0.417995549994
Epoch 19: 
Test set LL -1.1029759300845625, Acc 0.8281890153884888, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.441385435169 F1-OK:  0.898482892189
F1-score multiplied:  0.39657726236
[   873  43953 181616 ..., 181234   7816  90895]
Epoch 20: \Dev set LL -1.0364493591720105, Acc 0.8395028114318848, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.478935698448 F1-OK:  0.905142488092
F1-score multiplied:  0.433505049729
Epoch 20: 
Test set LL -1.125530914404405, Acc 0.8243649005889893, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.430217102348 F1-OK:  0.896181480584
F1-score multiplied:  0.385552599755
[139636 198046 199708 ...,  32597 183833  57999]
Epoch 21: \Dev set LL -1.0977898774793768, Acc 0.8292583227157593, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.47168216399 F1-OK:  0.898175301401
F1-score multiplied:  0.423653269807
Epoch 21: 
Test set LL -1.1435918655425994, Acc 0.8217700123786926, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.452141057935 F1-OK:  0.893573642146
F1-score multiplied:  0.404021331902
[ 48250  85392  96772 ..., 159417  47120  80258]
Epoch 22: \Dev set LL -1.0331798002745463, Acc 0.8415517210960388, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.457436856876 F1-OK:  0.9072296865
F1-score multiplied:  0.415000296257
Epoch 22: 
Test set LL -1.0922679410987661, Acc 0.8311253786087036, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.43084004603 F1-OK:  0.900853946999
F1-score multiplied:  0.388123955991
[ 94976 169993 129145 ..., 212870  49486 180803]
Epoch 23: \Dev set LL -1.036878709384679, Acc 0.8388198614120483, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.475088967972 F1-OK:  0.904792641601
F1-score multiplied:  0.429857002326
Epoch 23: 
Test set LL -1.1178248708627492, Acc 0.8268915414810181, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.442489553552 F1-OK:  0.897538498848
F1-score multiplied:  0.397151409651
[188706 216106  14316 ...,  23716  33182   2019]
Epoch 24: \Dev set LL -1.0257761215309138, Acc 0.8422346711158752, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.467496542185 F1-OK:  0.907399983965
F1-score multiplied:  0.424206354883
Epoch 24: 
Test set LL -1.0981794339568416, Acc 0.830647349357605, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.434306569343 F1-OK:  0.900417603598
F1-score multiplied:  0.391057280395
[ 49555 204099 199462 ..., 142630  30997  94732]
Epoch 25: \Dev set LL -1.0706503780950614, Acc 0.833629310131073, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.462014134276 F1-OK:  0.901599612215
F1-score multiplied:  0.416551764301
Epoch 25: 
Test set LL -1.1164638219592415, Acc 0.8266184329986572, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.448642779587 F1-OK:  0.897135680428
F1-score multiplied:  0.402493445334
[ 10586  40093 110450 ..., 135123  16134  39003]
Epoch 26: \Dev set LL -1.0603861197095759, Acc 0.8369075059890747, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.456284153005 F1-OK:  0.904065563233
F1-score multiplied:  0.412510789781
Epoch 26: 
Test set LL -1.1229357640649666, Acc 0.8259355425834656, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.435187236871 F1-OK:  0.897114026236
F1-score multiplied:  0.390412574236
[124016  93299 142951 ..., 121315 130112 138434]
Epoch 27: \Dev set LL -1.0129277604177247, Acc 0.8441469669342041, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.466073935423 F1-OK:  0.908756497401
F1-score multiplied:  0.423547717085
Epoch 27: 
Test set LL -1.0806745965944973, Acc 0.8335154056549072, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.435909301249 F1-OK:  0.902347192181
F1-score multiplied:  0.393341534028
[  9469 193383  39949 ..., 202484  63420  57130]
Epoch 28: \Dev set LL -1.0242422687754063, Acc 0.8429176211357117, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.471507352941 F1-OK:  0.907749077491
F1-score multiplied:  0.428010364662
Epoch 28: 
Test set LL -1.0949228994020304, Acc 0.8301693797111511, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.434901158828 F1-OK:  0.900068308756
F1-score multiplied:  0.391440750502
[ 70289  17736 199270 ...,  16586  34471 159323]
Epoch 29: \Dev set LL -1.0925775583562005, Acc 0.8321267366409302, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.471397849462 F1-OK:  0.900219209223
F1-score multiplied:  0.424361399272
Epoch 29: 
Test set LL -1.1540837504247767, Acc 0.8217700123786926, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.446095076401 F1-OK:  0.893798828125
F1-score multiplied:  0.398719256519
[ 44559 117975   7356 ..., 164492 159093  15768]
Done!



Matern32 from scratch - phrase 72 features



Epoch 1: \Dev set LL -0.6971392896546063, Acc 0.8349952101707458, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.416988416988 F1-OK:  0.903898170247
F1-score multiplied:  0.37691506713
Epoch 1: 
Test set LL -0.7280565044135398, Acc 0.8275744318962097, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.3934662503 F1-OK:  0.899502487562
F1-score multiplied:  0.353923870917
[ 10604 112260   2170 ...,  24253  38081  64020]
Epoch 2: \Dev set LL -0.8491756595364848, Acc 0.8363611698150635, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.434905660377 F1-OK:  0.904328382048
F1-score multiplied:  0.393297532192
Epoch 2: 
Test set LL -0.8900345501822534, Acc 0.8293498754501343, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.409917355372 F1-OK:  0.90025146689
F1-score multiplied:  0.369028700477
[ 16587 157088  51334 ...,  99848  92538  64243]
Epoch 3: \Dev set LL -0.8714825765968035, Acc 0.8423712849617004, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.424725822532 F1-OK:  0.908673630896
F1-score multiplied:  0.385937155296
Epoch 3: 
Test set LL -0.9215029865412192, Acc 0.8372029662132263, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.407554671968 F1-OK:  0.905636478784
F1-score multiplied:  0.369096378033
[ 97998  95754  98742 ..., 203012 213193 125451]
Epoch 4: \Dev set LL -0.9423308429410878, Acc 0.8308973908424377, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.449288256228 F1-OK:  0.900112957883
F1-score multiplied:  0.404410181255
Epoch 4: 
Test set LL -0.9972757572437662, Acc 0.8257989883422852, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.437486218302 F1-OK:  0.896941784834
F1-score multiplied:  0.392399669484
[ 82707 140148 157986 ...,  51883 106201 103473]
Epoch 5: \Dev set LL -0.9484183994624966, Acc 0.8355416059494019, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.45960502693 F1-OK:  0.903012727566
F1-score multiplied:  0.415029188971
Epoch 5: 
Test set LL -1.0065971344199953, Acc 0.8262087106704712, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.430011198208 F1-OK:  0.897474116747
F1-score multiplied:  0.385923920303
[ 96593 186600 181132 ...,  40532  34553 129546]
Epoch 6: \Dev set LL -0.9343963229868061, Acc 0.8475618362426758, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.41935483871 F1-OK:  0.912264150943
F1-score multiplied:  0.382562385879
Epoch 6: 
Test set LL -0.9940387486144447, Acc 0.8393198847770691, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.396821327865 F1-OK:  0.907314767401
F1-score multiplied:  0.360041850791
[211235 108099  35584 ..., 173533 203627 154044]
Epoch 7: \Dev set LL -0.9656145098831234, Acc 0.8415517210960388, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.455909943715 F1-OK:  0.907274180655
F1-score multiplied:  0.413635320637
Epoch 7: 
Test set LL -1.0296460847621205, Acc 0.8322179913520813, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.419560595322 F1-OK:  0.901935741369
F1-score multiplied:  0.378416696591
[ 44555 140306 112838 ...,  74919 150399 165017]
Epoch 8: \Dev set LL -0.9866747916790197, Acc 0.841141939163208, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.465778594396 F1-OK:  0.906698756518
F1-score multiplied:  0.422320872352
Epoch 8: 
Test set LL -1.044718351363289, Acc 0.8320131301879883, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.444945848375 F1-OK:  0.901029932411
F1-score multiplied:  0.400909527688
[169730  20673 186182 ...,  67812 116880 108578]
Epoch 9: \Dev set LL -1.0043640411041788, Acc 0.8393661975860596, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.455555555556 F1-OK:  0.905784329434
F1-score multiplied:  0.412635083409
Epoch 9: 
Test set LL -1.0543190240959963, Acc 0.830647349357605, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.424326833798 F1-OK:  0.900720576461
F1-score multiplied:  0.382199910346
[ 41613 123879  70949 ...,  98250 105271 211104]
Epoch 10: \Dev set LL -0.9876199383527123, Acc 0.8429176211357117, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.439024390244 F1-OK:  0.908672172808
F1-score multiplied:  0.398929246599
Epoch 10: 
Test set LL -1.034563667307349, Acc 0.8389101624488831, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.425614803993 F1-OK:  0.906318255828
F1-score multiplied:  0.38574246681
[ 79978  62975 104982 ..., 106252 110351  75418]
Epoch 11: \Dev set LL -0.9949894249767502, Acc 0.8437371850013733, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.433663366337 F1-OK:  0.909364601489
F1-score multiplied:  0.394358114309
Epoch 11: 
Test set LL -1.035641068094921, Acc 0.83870530128479, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.417365564874 F1-OK:  0.906396132203
F1-score multiplied:  0.378298533717
[ 78595 203579  77628 ..., 171197 169790 184869]
Epoch 12: \Dev set LL -1.0877981371009926, Acc 0.8218822479248047, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.46948738812 F1-OK:  0.892974392646
F1-score multiplied:  0.419240215262
Epoch 12: 
Test set LL -1.1315090917353352, Acc 0.8168533444404602, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.452876376989 F1-OK:  0.890018863282
F1-score multiplied:  0.403068518255
[161145   8624 103629 ..., 131598  77971 145710]



Epoch 13: \Dev set LL -1.0840726149680155, Acc 0.826526403427124, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.466834592779 F1-OK:  0.896411092985
F1-score multiplied:  0.418475707557
Epoch 13: 
Test set LL -1.1185429092391168, Acc 0.8211554288864136, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.455169544414 F1-OK:  0.893019076018
F1-score multiplied:  0.406475085985
[ 88180  37890  33658 ..., 138465 165899 187250]
Epoch 14: \Dev set LL -0.9953766665439399, Acc 0.84209805727005, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.479747974797 F1-OK:  0.90692431562
F1-score multiplied:  0.435095103713
Epoch 14: 
Test set LL -1.068547538935032, Acc 0.8282573223114014, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.433941030835 F1-OK:  0.898772388811
F1-score multiplied:  0.390014216887
[ 95272 217111 176486 ..., 183877 161754 214087]
Epoch 15: \Dev set LL -1.0051519617939926, Acc 0.8431908488273621, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.450191570881 F1-OK:  0.908555042218
F1-score multiplied:  0.409023821688
Epoch 15: 
Test set LL -1.0648082895576916, Acc 0.8354957699775696, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.42629197428 F1-OK:  0.903981824704
F1-score multiplied:  0.385360196766
[ 43428  23203 165494 ...,  87260 127806  73850]
Epoch 16: \Dev set LL -1.0316436553361934, Acc 0.8371807336807251, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.46975088968 F1-OK:  0.903824431176
F1-score multiplied:  0.424572330659
Epoch 16: 
Test set LL -1.0904838304523798, Acc 0.8270281553268433, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.438483706495 F1-OK:  0.897768091375
F1-score multiplied:  0.393656680279
[204644  31141 103650 ...,  30090  81702 136580]
Epoch 17: \Dev set LL -0.9925282436166362, Acc 0.8437371850013733, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.45471877979 F1-OK:  0.908801020408
F1-score multiplied:  0.413248891072
Epoch 17: 
Test set LL -1.056504749292476, Acc 0.8339251279830933, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.423696682464 F1-OK:  0.902983883836
F1-score multiplied:  0.3825912759
[151463 159307 100073 ...,  99926  64887  45399]
Epoch 18: \Dev set LL -1.033278882514026, Acc 0.8384100794792175, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.461538461538 F1-OK:  0.904941743672
F1-score multiplied:  0.417665420156
Epoch 18: 
Test set LL -1.078350567177299, Acc 0.831057071685791, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.441282746161 F1-OK:  0.900482703138
F1-score multiplied:  0.397367480111
[ 77558  65635  39900 ...,  52102 175656  30661]
Epoch 19: \Dev set LL -0.9847758248609813, Acc 0.8463324904441833, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.461979913917 F1-OK:  0.910365707912
F1-score multiplied:  0.420570671374
Epoch 19: 
Test set LL -1.0565889604779244, Acc 0.8365200757980347, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.424242424242 F1-OK:  0.904735376045
F1-score multiplied:  0.383827129231
[   873  43953 181616 ..., 181234   7816  90895]
Epoch 20: \Dev set LL -1.0072214928400185, Acc 0.8451031446456909, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.462049335863 F1-OK:  0.909526089038
F1-score multiplied:  0.42024592539
Epoch 20: 
Test set LL -1.0742100087975899, Acc 0.8339934349060059, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.430011723329 F1-OK:  0.902849378572
F1-score multiplied:  0.388235817187
[139636 198046 199708 ...,  32597 183833  57999]
Epoch 21: \Dev set LL -1.0781545862339494, Acc 0.8319901823997498, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.476595744681 F1-OK:  0.899934917019
F1-score multiplied:  0.428905151941
Epoch 21: 
Test set LL -1.1412960490292534, Acc 0.8211554288864136, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.450828265884 F1-OK:  0.893184877034
F1-score multiplied:  0.402672989227
[ 48250  85392  96772 ..., 159417  47120  80258]
Epoch 22: \Dev set LL -1.0401651329768726, Acc 0.8371807336807251, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.470692717584 F1-OK:  0.903793381759
F1-score multiplied:  0.425408962995
Epoch 22: 
Test set LL -1.105317153658297, Acc 0.8275744318962097, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.443464844611 F1-OK:  0.897983919842
F1-score multiplied:  0.398224299476
[ 94976 169993 129145 ..., 212870  49486 180803]
Epoch 23: \Dev set LL -1.048111037087856, Acc 0.8356781601905823, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.475816993464 F1-OK:  0.902567425285
F1-score multiplied:  0.429456918698
Epoch 23: 
Test set LL -1.113756714683909, Acc 0.8245015144348145, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.444444444444 F1-OK:  0.895791095613
F1-score multiplied:  0.398129375828
[188706 216106  14316 ...,  23716  33182   2019]
Epoch 24: \Dev set LL -0.9963267001316932, Acc 0.8453763127326965, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.465028355388 F1-OK:  0.909627973814
F1-score multiplied:  0.423002800677
Epoch 24: 
Test set LL -1.051323025891098, Acc 0.8372029662132263, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.435606060606 F1-OK:  0.904883498244
F1-score multiplied:  0.394172735978
[ 49555 204099 199462 ..., 142630  30997  94732]
Epoch 25: \Dev set LL -1.0181991420521253, Acc 0.841141939163208, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.46479521399 F1-OK:  0.906728687144
F1-score multiplied:  0.421443154172
Epoch 25: 
Test set LL -1.081146845404494, Acc 0.8321496844291687, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.43623853211 F1-OK:  0.901396020539
F1-score multiplied:  0.39322367685
[ 10586  40093 110450 ..., 135123  16134  39003]
Epoch 26: \Dev set LL -1.0208554280467255, Acc 0.8408687114715576, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.464367816092 F1-OK:  0.906553300714
F1-score multiplied:  0.420974176423
Epoch 26: 
Test set LL -1.0754095510974468, Acc 0.8334471583366394, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.442004118051 F1-OK:  0.902115021873
F1-score multiplied:  0.398738554623
[124016  93299 142951 ..., 121315 130112 138434]
Epoch 27: \Dev set LL -1.0362410451387445, Acc 0.8382734656333923, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.475177304965 F1-OK:  0.90440820281
F1-score multiplied:  0.429754252399
Epoch 27: 
Test set LL -1.1165744226423941, Acc 0.8273012638092041, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.445273086203 F1-OK:  0.897731408468
F1-score multiplied:  0.39973563483
[  9469 193383  39949 ..., 202484  63420  57130]
Epoch 28: \Dev set LL -0.9812788582405763, Acc 0.8497473001480103, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.45652173913 F1-OK:  0.912822951339
F1-score multiplied:  0.416723521264
Epoch 28: 
Test set LL -1.0529424998935204, Acc 0.8390467166900635, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.422445479049 F1-OK:  0.906494227794
F1-score multiplied:  0.382944388316
[ 70289  17736 199270 ...,  16586  34471 159323]
Epoch 29: \Dev set LL -1.0266075586376688, Acc 0.8393661975860596, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.463503649635 F1-OK:  0.905542168675
F1-score multiplied:  0.419722100079
Epoch 29: 
Test set LL -1.0946397923136968, Acc 0.8296913504600525, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.436766034327 F1-OK:  0.899678197908
F1-score multiplied:  0.392948878671
[ 44559 117975   7356 ..., 164492 159093  15768]
Done!




Matern12 pretraining - phrase 72 features



Epoch 1: \Dev set LL -0.40533634995401635, Acc 0.8472886085510254, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.434782608696 F1-OK:  0.911718256475
F1-score multiplied:  0.396399241946
Epoch 1: 
Test set LL -0.4177118722128757, Acc 0.8385003805160522, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.412422360248 F1-OK:  0.906384831572
F1-score multiplied:  0.37381337153
[ 10604 112260   2170 ...,  24253  38081  64020]
Epoch 2: \Dev set LL -0.5202454507999688, Acc 0.8356781601905823, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.479446127218 F1-OK:  0.902441002352
F1-score multiplied:  0.43267184362
Epoch 2: 
Test set LL -0.5507564978752855, Acc 0.8250477910041809, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.448082722964 F1-OK:  0.89604804025
F1-score multiplied:  0.401503645782
[ 16587 157088  51334 ...,  99848  92538  64243]

Epoch 3: \Dev set LL -0.727793010623235, Acc 0.8433274030685425, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.474094452086 F1-OK:  0.907952812776
F1-score multiplied:  0.430455391293
Epoch 3: 
Test set LL -0.7717817091550372, Acc 0.8329691290855408, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.437701149425 F1-OK:  0.901916753549
F1-score multiplied:  0.394769999714
[ 97998  95754  98742 ..., 203012 213193 125451]
Epoch 4: \Dev set LL -0.8264766339718538, Acc 0.8319901823997498, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.49173553719 F1-OK:  0.899361806578
F1-score multiplied:  0.442248161086
Epoch 4: 
Test set LL -0.877999221193535, Acc 0.8172630667686462, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.460701330109 F1-OK:  0.889994244841
F1-score multiplied:  0.410021532387
[ 82707 140148 157986 ...,  51883 106201 103473]
Epoch 5: \Dev set LL -0.8645340755111807, Acc 0.833629310131073, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.494186046512 F1-OK:  0.90044139284
F1-score multiplied:  0.444985572043
Epoch 5: 
Test set LL -0.9334155050132873, Acc 0.8197214007377625, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.462978030919 F1-OK:  0.891678975874
F1-score multiplied:  0.412827776462
[ 96593 186600 181132 ...,  40532  34553 129546]
Epoch 6: \Dev set LL -0.8874903223548433, Acc 0.8429176211357117, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.474885844749 F1-OK:  0.907645358175
F1-score multiplied:  0.43102793265
Epoch 6: 
Test set LL -0.9469550275434354, Acc 0.8318765163421631, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.445745159838 F1-OK:  0.900909603155
F1-score multiplied:  0.401576095058
[211235 108099  35584 ..., 173533 203627 154044]
Epoch 7: \Dev set LL -0.9020265379052844, Acc 0.8414151072502136, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.472991375397 F1-OK:  0.906664522872
F1-score multiplied:  0.428844499697
Epoch 7: 
Test set LL -0.9615233821130824, Acc 0.8296230435371399, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.443949186539 F1-OK:  0.899399217773
F1-score multiplied:  0.399287551104
[ 44555 140306 112838 ...,  74919 150399 165017]
Epoch 8: \Dev set LL -0.9385123170817392, Acc 0.8395028114318848, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.482150727193 F1-OK:  0.905035157197
F1-score multiplied:  0.436363359177
Epoch 8: 
Test set LL -0.9965940907289806, Acc 0.8281207084655762, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.456019018803 F1-OK:  0.897936012327
F1-score multiplied:  0.409475899289
[169730  20673 186182 ...,  67812 116880 108578]
Epoch 9: \Dev set LL -0.9701829316149979, Acc 0.8322633504867554, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.481418918919 F1-OK:  0.899951116181
F1-score multiplied:  0.433253493432
Epoch 9: 
Test set LL -1.0322336260514664, Acc 0.8212237358093262, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.463084495488 F1-OK:  0.892757660167
F1-score multiplied:  0.413422230652
[ 41613 123879  70949 ...,  98250 105271 211104]
Epoch 10: \Dev set LL -0.942130053077485, Acc 0.8412784934043884, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.476104598738 F1-OK:  0.906471345782
F1-score multiplied:  0.431575176351
Epoch 10: 
Test set LL -1.0178326662709372, Acc 0.8295547962188721, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.448275862069 F1-OK:  0.899208528509
F1-score multiplied:  0.403093478297
[ 79978  62975 104982 ..., 106252 110351  75418]
Epoch 11: \Dev set LL -1.042552310212879, Acc 0.8205163478851318, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.485512920908 F1-OK:  0.891297154203
F1-score multiplied:  0.432736284734
Epoch 11: 
Test set LL -1.095577969146159, Acc 0.8129609227180481, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.468258590565 F1-OK:  0.886522765878
F1-score multiplied:  0.415121900854
[ 78595 203579  77628 ..., 171197 169790 184869]
Epoch 12: \Dev set LL -1.0121908067804357, Acc 0.8318535685539246, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.494455852156 F1-OK:  0.89915622184
F1-score multiplied:  0.444593055891
Epoch 12: 
Test set LL -1.0914710701817953, Acc 0.8173313140869141, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.464678807284 F1-OK:  0.889876909143
F1-score multiplied:  0.413506940771
[161145   8624 103629 ..., 131598  77971 145710]
Epoch 13: \Dev set LL -0.9707638359194521, Acc 0.8418248891830444, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.480251346499 F1-OK:  0.906718221363
F1-score multiplied:  0.435452646705
Epoch 13: 
Test set LL -1.0539847147804808, Acc 0.8270281553268433, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.45137535196 F1-OK:  0.897328847635
F1-score multiplied:  0.405032124425
[ 88180  37890  33658 ..., 138465 165899 187250]
Epoch 14: \Dev set LL -1.0074457551421763, Acc 0.8337658643722534, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.480580452411 F1-OK:  0.901048865761
F1-score multiplied:  0.433026471552
Epoch 14: 
Test set LL -1.0835154445684814, Acc 0.8208140134811401, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.452877397832 F1-OK:  0.892862975666
F1-score multiplied:  0.40435746104
[ 95272 217111 176486 ..., 183877 161754 214087]
Epoch 15: \Dev set LL -0.9825868445091737, Acc 0.8418248891830444, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.470749542962 F1-OK:  0.907017825598
F1-score multiplied:  0.426978226858
Epoch 15: 
Test set LL -1.0434995669033866, Acc 0.8288035988807678, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.444985609918 F1-OK:  0.898792943361
F1-score multiplied:  0.399949926092
[ 43428  23203 165494 ...,  87260 127806  73850]
Epoch 16: \Dev set LL -0.9733909176289236, Acc 0.8437371850013733, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.48 F1-OK:  0.908053367626
F1-score multiplied:  0.43586561646
Epoch 16: 
Test set LL -1.0637117445370385, Acc 0.8294181823730469, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.44832155477 F1-OK:  0.899111470113
F1-score multiplied:  0.403091052193
[204644  31141 103650 ...,  30090  81702 136580]
Epoch 17: \Dev set LL -0.9789777974545197, Acc 0.842644453048706, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.476839237057 F1-OK:  0.907395498392
F1-score multiplied:  0.432681777163
Epoch 17: 
Test set LL -1.056811199649368, Acc 0.8288719058036804, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.441373160945 F1-OK:  0.89895976131
F1-score multiplied:  0.396776711412
[151463 159307 100073 ...,  99926  64887  45399]
Epoch 18: \Dev set LL -1.0358839007661567, Acc 0.832536518573761, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.481387478849 F1-OK:  0.900146603681
F1-score multiplied:  0.433319304141
Epoch 18: 
Test set LL -1.1013129842061173, Acc 0.821906566619873, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.467755102041 F1-OK:  0.893062161719
F1-score multiplied:  0.417734382584
[ 77558  65635  39900 ...,  52102 175656  30661]
Epoch 19: \Dev set LL -1.0033435352671654, Acc 0.8380002975463867, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.471479500891 F1-OK:  0.904339409582
F1-score multiplied:  0.426377493466
Epoch 19: 
Test set LL -1.0846894417688535, Acc 0.8251844048500061, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.446845289542 F1-OK:  0.896188158962
F1-score multiplied:  0.400457457375
[   873  43953 181616 ..., 181234   7816  90895]
Epoch 20: \Dev set LL -1.0490796633174513, Acc 0.8310340046882629, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.479157894737 F1-OK:  0.899160348904
F1-score multiplied:  0.430839779811
Epoch 20: 
Test set LL -1.1148491257487907, Acc 0.8207457065582275, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.461759278245 F1-OK:  0.892466510999
F1-score multiplied:  0.412104691977
[139636 198046 199708 ...,  32597 183833  57999]
Epoch 21: \Dev set LL -1.003488728535686, Acc 0.8396393656730652, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.48823016565 F1-OK:  0.904923874312
F1-score multiplied:  0.441811133055
Epoch 21: 
Test set LL -1.0957281763388131, Acc 0.8246380686759949, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.45731191885 F1-OK:  0.895422707281
F1-score multiplied:  0.409487476449
[ 48250  85392  96772 ..., 159417  47120  80258]
Epoch 22: \Dev set LL -0.9979234317614198, Acc 0.841141939163208, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.481497993758 F1-OK:  0.906202113074
F1-score multiplied:  0.436334499385
Epoch 22: 
Test set LL -1.0873870348420578, Acc 0.8266184329986572, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.455267110062 F1-OK:  0.896901774475
F1-score multiplied:  0.408329878875
[ 94976 169993 129145 ..., 212870  49486 180803]
Epoch 23: \Dev set LL -0.9970972347082299, Acc 0.8429176211357117, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.481514878269 F1-OK:  0.907437218287
F1-score multiplied:  0.4369445217
Epoch 23: 
Test set LL -1.0766067376202952, Acc 0.8271647095680237, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.447017697182 F1-OK:  0.897575978309
F1-score multiplied:  0.401232346869
[188706 216106  14316 ...,  23716  33182   2019]
Epoch 24: \Dev set LL -0.9935265197914077, Acc 0.8442835807800293, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.485559566787 F1-OK:  0.908256880734
F1-score multiplied:  0.44101281754
Epoch 24: 
Test set LL -1.0893359764812387, Acc 0.8281207084655762, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.451753430625 F1-OK:  0.898084787626
F1-score multiplied:  0.405712883802
[ 49555 204099 199462 ..., 142630  30997  94732]
Epoch 25: \Dev set LL -1.0271493303060388, Acc 0.8366343379020691, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.468444444444 F1-OK:  0.903486120077
F1-score multiplied:  0.423233053583
Epoch 25: 
Test set LL -1.1008991179159515, Acc 0.8264135718345642, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.450259515571 F1-OK:  0.896934803763
F1-score multiplied:  0.403853430241
[ 10586  40093 110450 ..., 135123  16134  39003]
Epoch 26: \Dev set LL -1.0161001416689897, Acc 0.839093029499054, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.476444444444 F1-OK:  0.90493867011
F1-score multiplied:  0.431153001937
Epoch 26: 
Test set LL -1.0916656447254482, Acc 0.8268232941627502, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.460195828012 F1-OK:  0.896868645791
F1-score multiplied:  0.412735209068
[124016  93299 142951 ..., 121315 130112 138434]
Epoch 27: \Dev set LL -1.0247745761795237, Acc 0.8389564156532288, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.481302243731 F1-OK:  0.904681057482
F1-score multiplied:  0.435425022827
Epoch 27: 
Test set LL -1.1074355925537434, Acc 0.826481819152832, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.458093410109 F1-OK:  0.896703118013
F1-score multiplied:  0.410773789186
[  9469 193383  39949 ..., 202484  63420  57130]
Epoch 28: \Dev set LL -0.999496334588626, Acc 0.8430542349815369, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.471237919926 F1-OK:  0.90785147165
F1-score multiplied:  0.427814039102
Epoch 28: 
Test set LL -1.0700796860590978, Acc 0.8311253786087036, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.451541361721 F1-OK:  0.900197748093
F1-score multiplied:  0.406476516992
[ 70289  17736 199270 ...,  16586  34471 159323]
Epoch 29: \Dev set LL -1.0041363989092245, Acc 0.8414151072502136, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.485600354453 F1-OK:  0.906257569641
F1-score multiplied:  0.440078997043
Epoch 29: 
Test set LL -1.091562356752454, Acc 0.8282573223114014, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.456451264318 F1-OK:  0.898017112039
F1-score multiplied:  0.40990104617
[ 44559 117975   7356 ..., 164492 159093  15768]
Done!












Matern12 - phrase - 72 features - instead of using 17 hidden -> 68 hidden

Epoch 1: \Dev set LL -0.4370331511204081, Acc 0.8373172879219055, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.418172936004 F1-OK:  0.905438666137
F1-score multiplied:  0.37862994539
Epoch 1: 
Test set LL -0.45307886222610594, Acc 0.8300327658653259, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.393666260658 F1-OK:  0.901163483302
F1-score multiplied:  0.354757658713
[ 10604 112260   2170 ...,  24253  38081  64020]
Epoch 2: \Dev set LL -0.6712170421771798, Acc 0.8319901823997498, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.450402144772 F1-OK:  0.900838439213
F1-score multiplied:  0.405739565115
Epoch 2: 
Test set LL -0.7030008328650122, Acc 0.8242283463478088, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.423387096774 F1-OK:  0.896310022559
F1-score multiplied:  0.379486098261
[ 16587 157088  51334 ...,  99848  92538  64243]
Epoch 3: \Dev set LL -0.7467167779369507, Acc 0.8438737988471985, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.437223042836 F1-OK:  0.909364840219
F1-score multiplied:  0.397595262489
Epoch 3: 
Test set LL -0.7964001420101124, Acc 0.835837185382843, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.408173313639 F1-OK:  0.904701498454
F1-score multiplied:  0.369275008478
[ 97998  95754  98742 ..., 203012 213193 125451]
Epoch 4: \Dev set LL -0.8558249773267019, Acc 0.8303510546684265, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.45953002611 F1-OK:  0.899384316267
F1-score multiplied:  0.413294098337
Epoch 4: 
Test set LL -0.8933407109750527, Acc 0.8225211501121521, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.43463128127 F1-OK:  0.894738973715
F1-score multiplied:  0.388881546548
[ 82707 140148 157986 ...,  51883 106201 103473]
Epoch 5: \Dev set LL -0.8968023757522076, Acc 0.8341756463050842, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.460444444444 F1-OK:  0.902033570045
F1-score multiplied:  0.41533634603
Epoch 5: 
Test set LL -0.9450166543691131, Acc 0.8246380686759949, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.434610303831 F1-OK:  0.896225652631
F1-score multiplied:  0.389508903191
[ 96593 186600 181132 ...,  40532  34553 129546]
Epoch 6: \Dev set LL -0.9154513825405859, Acc 0.8414151072502136, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.462251042149 F1-OK:  0.906993511175
F1-score multiplied:  0.419258695763
Epoch 6: 
Test set LL -0.9779505426467995, Acc 0.8307156562805176, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.428143021915 F1-OK:  0.900653228069
F1-score multiplied:  0.385608394763
[211235 108099  35584 ..., 173533 203627 154044]
Epoch 7: \Dev set LL -0.9463019698504198, Acc 0.840595543384552, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.463448275862 F1-OK:  0.906392877196
F1-score multiplied:  0.42006621619
Epoch 7: 
Test set LL -1.0064683442988325, Acc 0.8294864892959595, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.431077694236 F1-OK:  0.899714847986
F1-score multiplied:  0.387847002139
[ 44555 140306 112838 ...,  74919 150399 165017]
Epoch 8: \Dev set LL -0.962137676111767, Acc 0.8393661975860596, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.463503649635 F1-OK:  0.905542168675
F1-score multiplied:  0.419722100079
Epoch 8: 
Test set LL -1.0316380653316581, Acc 0.8277109861373901, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.428797826579 F1-OK:  0.898556551807
F1-score multiplied:  0.385299096473
[169730  20673 186182 ...,  67812 116880 108578]
Epoch 9: \Dev set LL -1.001522632117633, Acc 0.8303510546684265, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.468321917808 F1-OK:  0.899073622623
F1-score multiplied:  0.421055883198
Epoch 9: 
Test set LL -1.0603662103030609, Acc 0.8229308724403381, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.449819647783 F1-OK:  0.894486266531
F1-score multiplied:  0.402357497357
[ 41613 123879  70949 ...,  98250 105271 211104]
Epoch 10: \Dev set LL -0.980943535417408, Acc 0.8410053253173828, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.468493150685 F1-OK:  0.906521040797
F1-score multiplied:  0.424698898565
Epoch 10: 
Test set LL -1.0532128749011147, Acc 0.8301693797111511, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.433872069201 F1-OK:  0.900100421771
F1-score multiplied:  0.390528432483
[ 79978  62975 104982 ..., 106252 110351  75418]
Epoch 11: \Dev set LL -0.9875794267151063, Acc 0.8408687114715576, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.451764705882 F1-OK:  0.906926579851
F1-score multiplied:  0.409717419603
Epoch 11: 
Test set LL -1.0447531311244798, Acc 0.8329691290855408, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.43458159963 F1-OK:  0.902011056806
F1-score multiplied:  0.391997407951
[ 78595 203579  77628 ..., 171197 169790 184869]
Epoch 12: \Dev set LL -0.9914249120554052, Acc 0.840049147605896, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.460119870908 F1-OK:  0.90611721318
F1-score multiplied:  0.416922535156
Epoch 12: 
Test set LL -1.0550858492639557, Acc 0.831261932849884, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.437258027784 F1-OK:  0.900751094509
F1-score multiplied:  0.39386064711
[161145   8624 103629 ..., 131598  77971 145710]
Epoch 13: \Dev set LL -1.041782833232654, Acc 0.8319901823997498, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.469370146678 F1-OK:  0.900194741967
F1-score multiplied:  0.422524538076
Epoch 13: 
Test set LL -1.0934389661347907, Acc 0.8232723474502563, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.447008547009 F1-OK:  0.894830949285
F1-score multiplied:  0.399997082458
[ 88180  37890  33658 ..., 138465 165899 187250]
Epoch 14: \Dev set LL -1.0263667229623221, Acc 0.8369075059890747, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.471213463242 F1-OK:  0.903585271318
F1-score multiplied:  0.425781545032
Epoch 14: 
Test set LL -1.0872272813109929, Acc 0.8268915414810181, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.44614376229 F1-OK:  0.897414107078
F1-score multiplied:  0.400375706064
[ 95272 217111 176486 ..., 183877 161754 214087]
Epoch 15: \Dev set LL -1.0175960143707665, Acc 0.8399125933647156, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.465328467153 F1-OK:  0.905863453815
F1-score multiplied:  0.421524052414
Epoch 15: 
Test set LL -1.0784659406516268, Acc 0.8305791020393372, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.440081245768 F1-OK:  0.900189081546
F1-score multiplied:  0.396156332434
[ 43428  23203 165494 ...,  87260 127806  73850]
Epoch 16: \Dev set LL -1.015878972389153, Acc 0.8396393656730652, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.459484346225 F1-OK:  0.905854049719
F1-score multiplied:  0.41622575581
Epoch 16: 
Test set LL -1.0753483065066416, Acc 0.8324228525161743, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.444041685546 F1-OK:  0.901342767548
F1-score multiplied:  0.400233761757
[204644  31141 103650 ...,  30090  81702 136580]
Epoch 17: \Dev set LL -1.0231028363180448, Acc 0.8404589295387268, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.470534904805 F1-OK:  0.906079125121
F1-score multiplied:  0.426341854884
Epoch 17: 
Test set LL -1.087358966033398, Acc 0.8291450142860413, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.444986690328 F1-OK:  0.899031476998
F1-score multiplied:  0.40005704145
[151463 159307 100073 ...,  99926  64887  45399]
Epoch 18: \Dev set LL -1.036116560419699, Acc 0.8384100794792175, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.471639124609 F1-OK:  0.904619850036
F1-score multiplied:  0.426654114175
Epoch 18: 
Test set LL -1.104277782692106, Acc 0.8275744318962097, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.441495244415 F1-OK:  0.898049824363
F1-score multiplied:  0.396484726704
[ 77558  65635  39900 ...,  52102 175656  30661]
Epoch 19: \Dev set LL -1.049049938742939, Acc 0.8344488739967346, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.473501303215 F1-OK:  0.901782820097
F1-score multiplied:  0.426995340533
Epoch 19: 
Test set LL -1.1112103286359902, Acc 0.8254575133323669, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.452442159383 F1-OK:  0.896181965881
F1-score multiplied:  0.405470503844
[   873  43953 181616 ..., 181234   7816  90895]
Epoch 20: \Dev set LL -1.0198900161752313, Acc 0.8410053253173828, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.474729241877 F1-OK:  0.906325446644
F1-score multiplied:  0.430259192179
Epoch 20: 
Test set LL -1.0944133373517457, Acc 0.8294181823730469, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.443404634581 F1-OK:  0.899274193548
F1-score multiplied:  0.398742345179
[139636 198046 199708 ...,  32597 183833  57999]
Epoch 21: \Dev set LL -1.0470138715895634, Acc 0.8364977240562439, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.467289719626 F1-OK:  0.903428801936
F1-score multiplied:  0.422162991559
Epoch 21: 
Test set LL -1.1106728135411117, Acc 0.8267549872398376, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.446915195117 F1-OK:  0.897291607627
F1-score multiplied:  0.401013253899
[ 48250  85392  96772 ..., 159417  47120  80258]
Epoch 22: \Dev set LL -1.0165361240250905, Acc 0.8434640169143677, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.466480446927 F1-OK:  0.908275972467
F1-score multiplied:  0.42369298157
Epoch 22: 
Test set LL -1.0824086716368664, Acc 0.8321496844291687, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.432594644506 F1-OK:  0.901506651707
F1-score multiplied:  0.389986949515
[ 94976 169993 129145 ..., 212870  49486 180803]
Epoch 23: \Dev set LL -1.0442119433874864, Acc 0.8375905156135559, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.461747397012 F1-OK:  0.904367409314
F1-score multiplied:  0.417589297193
Epoch 23: 
Test set LL -1.1048635068345583, Acc 0.8282573223114014, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.440240373915 F1-OK:  0.89856825973
F1-score multiplied:  0.395586026652
[188706 216106  14316 ...,  23716  33182   2019]
Epoch 24: \Dev set LL -1.0599179190387464, Acc 0.835131824016571, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.464745011086 F1-OK:  0.902559134577
F1-score multiplied:  0.419459855005
Epoch 24: 
Test set LL -1.115530978818112, Acc 0.8255941271781921, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.441382327209 F1-OK:  0.896666127205
F1-score multiplied:  0.395772581955
[ 49555 204099 199462 ..., 142630  30997  94732]
Epoch 25: \Dev set LL -1.042884253176557, Acc 0.8371807336807251, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.470222222222 F1-OK:  0.903808908974
F1-score multiplied:  0.424991033642
Epoch 25: 
Test set LL -1.1088463570923688, Acc 0.8268915414810181, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.444200833151 F1-OK:  0.897480486917
F1-score multiplied:  0.398661580025
[ 10586  40093 110450 ..., 135123  16134  39003]
Epoch 26: \Dev set LL -1.056708872074179, Acc 0.834585428237915, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.465342163355 F1-OK:  0.902157227115
F1-score multiplied:  0.419811795752
Epoch 26: 
Test set LL -1.110978737110131, Acc 0.826686680316925, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.445851528384 F1-OK:  0.897280233123
F1-score multiplied:  0.400053763327
[124016  93299 142951 ..., 121315 130112 138434]
Epoch 27: \Dev set LL -1.0388368474341025, Acc 0.8378636837005615, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.470325747434 F1-OK:  0.904281912749
F1-score multiplied:  0.425307066505
Epoch 27: 
Test set LL -1.0949767005419604, Acc 0.8301010727882385, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.451740855002 F1-OK:  0.899474747475
F1-score multiplied:  0.406329491477
[  9469 193383  39949 ..., 202484  63420  57130]
Epoch 28: \Dev set LL -1.0484888286049452, Acc 0.8367709517478943, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.477481416703 F1-OK:  0.903278025091
F1-score multiplied:  0.431298471097
Epoch 28: 
Test set LL -1.1202636175253262, Acc 0.8245697617530823, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.452121987631 F1-OK:  0.89556486036
F1-score multiplied:  0.404904564718
[ 70289  17736 199270 ...,  16586  34471 159323]
Epoch 29: \Dev set LL -1.0345574602564507, Acc 0.8404589295387268, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.475292003594 F1-OK:  0.905927835052
F1-score multiplied:  0.430580255833
Epoch 29: 
Test set LL -1.109372935649945, Acc 0.8283255696296692, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.444051304732 F1-OK:  0.898489865138
F1-score multiplied:  0.398975596903
[ 44559 117975   7356 ..., 164492 159093  15768]
Done!


Matern12 - phrase - 72 features - instead of using 17 hidden -> 34 hidden

Epoch 1: \Dev set LL -0.4393242920309501, Acc 0.8397759795188904, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.417287630402 F1-OK:  0.907118536701
F1-score multiplied:  0.378529344674
Epoch 1: 
Test set LL -0.4579980038352507, Acc 0.8318082690238953, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.388682055101 F1-OK:  0.902490201512
F1-score multiplied:  0.350781746232
[ 10604 112260   2170 ...,  24253  38081  64020]
Epoch 2: \Dev set LL -0.6654017493928057, Acc 0.8355416059494019, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.438432835821 F1-OK:  0.903664586334
F1-score multiplied:  0.396196227217
Epoch 2: 
Test set LL -0.7038209392449273, Acc 0.8292133212089539, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.415244330138 F1-OK:  0.900003998241
F1-score multiplied:  0.373721557371
[ 16587 157088  51334 ...,  99848  92538  64243]
Epoch 3: \Dev set LL -0.7832289969792856, Acc 0.8448299169540405, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.455938697318 F1-OK:  0.909510912856
F1-score multiplied:  0.414681220804
Epoch 3: 
Test set LL -0.8366870734643029, Acc 0.8319448232650757, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.413628782464 F1-OK:  0.90191702204
F1-score multiplied:  0.37305883971
[ 97998  95754  98742 ..., 203012 213193 125451]
Epoch 4: \Dev set LL -0.8332254794999973, Acc 0.8416882753372192, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.469565217391 F1-OK:  0.906959942201
F1-score multiplied:  0.425876842425
Epoch 4: 
Test set LL -0.8991766614425359, Acc 0.8255258202552795, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.430830920027 F1-OK:  0.896971652083
F1-score multiplied:  0.386443122105
[ 82707 140148 157986 ...,  51883 106201 103473]

Epoch 5: \Dev set LL -0.9151585686026714, Acc 0.8328097462654114, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.465034965035 F1-OK:  0.900922778048
F1-score multiplied:  0.418960592589
Epoch 5: 
Test set LL -0.9748376343684593, Acc 0.8213602900505066, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.437902879244 F1-OK:  0.893805309735
F1-score multiplied:  0.391399918616
[ 96593 186600 181132 ...,  40532  34553 129546]
Epoch 6: \Dev set LL -0.8947169564360042, Acc 0.8476983904838562, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.466251795117 F1-OK:  0.911176611169
F1-score multiplied:  0.424837730626
Epoch 6: 
Test set LL -0.9685912906554305, Acc 0.8335837125778198, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.420727359163 F1-OK:  0.902834815199
F1-score multiplied:  0.379847307559
[211235 108099  35584 ..., 173533 203627 154044]
Epoch 7: \Dev set LL -0.9235077119243292, Acc 0.8418248891830444, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.469294225481 F1-OK:  0.907062600321
F1-score multiplied:  0.425679240481
Epoch 7: 
Test set LL -0.9972296856712511, Acc 0.8301010727882385, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.436083408885 F1-OK:  0.899983920244
F1-score multiplied:  0.392468055882
[ 44555 140306 112838 ...,  74919 150399 165017]
Epoch 8: \Dev set LL -0.9511528038786713, Acc 0.8399125933647156, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.462878093492 F1-OK:  0.905939004815
F1-score multiplied:  0.419339319369
Epoch 8: 
Test set LL -1.0153238080008715, Acc 0.8297596573829651, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.436610169492 F1-OK:  0.899730523268
F1-score multiplied:  0.392831496261
[169730  20673 186182 ...,  67812 116880 108578]
Epoch 9: \Dev set LL -0.9947346736802084, Acc 0.8328097462654114, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.463157894737 F1-OK:  0.900986895324
F1-score multiplied:  0.417299193624
Epoch 9: 
Test set LL -1.0531133496908403, Acc 0.8245015144348145, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.444684528954 F1-OK:  0.895782643958
F1-score multiplied:  0.398340683074
[ 41613 123879  70949 ...,  98250 105271 211104]
Epoch 10: \Dev set LL -0.9886182331866004, Acc 0.8370441198348999, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.469069870939 F1-OK:  0.903751512707
F1-score multiplied:  0.423922605426
Epoch 10: 
Test set LL -1.0396009617086812, Acc 0.8289401531219482, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.445428381669 F1-OK:  0.898873682936
F1-score multiplied:  0.400383849915
[ 79978  62975 104982 ..., 106252 110351  75418]
Epoch 11: \Dev set LL -0.9805687956467195, Acc 0.8425078392028809, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.471827759963 F1-OK:  0.90745645718
F1-score multiplied:  0.428163147455
Epoch 11: 
Test set LL -1.047012352869056, Acc 0.8311253786087036, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.442390078918 F1-OK:  0.900494910071
F1-score multiplied:  0.398370014331
[ 78595 203579  77628 ..., 171197 169790 184869]
Epoch 12: \Dev set LL -0.9955299948108033, Acc 0.8380002975463867, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.472419928826 F1-OK:  0.904308536389
F1-score multiplied:  0.427213374397
Epoch 12: 
Test set LL -1.059147593057356, Acc 0.8277792930603027, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.440301819796 F1-OK:  0.898232588169
F1-score multiplied:  0.395493443171
[161145   8624 103629 ..., 131598  77971 145710]
Epoch 13: \Dev set LL -1.0310698813752817, Acc 0.8337658643722534, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.470639408438 F1-OK:  0.901401604148
F1-score multiplied:  0.424235117742
Epoch 13: 
Test set LL -1.07778278793983, Acc 0.8265501260757446, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.452586206897 F1-OK:  0.896949042519
F1-score multiplied:  0.405946764933
[ 88180  37890  33658 ..., 138465 165899 187250]
Epoch 14: \Dev set LL -0.9907175364012892, Acc 0.8414151072502136, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.471552116523 F1-OK:  0.906709521896
F1-score multiplied:  0.427560794121
Epoch 14: 
Test set LL -1.064597477433907, Acc 0.8287352919578552, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.437415881561 F1-OK:  0.898993153443
F1-score multiplied:  0.393233882731
[ 95272 217111 176486 ..., 183877 161754 214087]
Epoch 15: \Dev set LL -0.990240136486085, Acc 0.8433274030685425, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.466263378315 F1-OK:  0.908188585608
F1-score multiplied:  0.423455078073
Epoch 15: 
Test set LL -1.064192912767892, Acc 0.8327642679214478, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.440228571429 F1-OK:  0.901697908722
F1-score multiplied:  0.396953182217
[ 43428  23203 165494 ...,  87260 127806  73850]
Epoch 16: \Dev set LL -0.9840960319160043, Acc 0.8434640169143677, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.471889400922 F1-OK:  0.908114175754
F1-score multiplied:  0.428529454365
Epoch 16: 
Test set LL -1.0598725637975304, Acc 0.8307839632034302, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.436818181818 F1-OK:  0.900433944069
F1-score multiplied:  0.393325918296
[204644  31141 103650 ...,  30090  81702 136580]
Epoch 17: \Dev set LL -1.022821116401845, Acc 0.8371807336807251, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.46927871772 F1-OK:  0.90383994837
F1-score multiplied:  0.424152851996
Epoch 17: 
Test set LL -1.0834987493536234, Acc 0.8284621834754944, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.444739168877 F1-OK:  0.898562429333
F1-score multiplied:  0.399625908006
[151463 159307 100073 ...,  99926  64887  45399]
Epoch 18: \Dev set LL -1.019432709327731, Acc 0.8371807336807251, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.471631205674 F1-OK:  0.903762312288
F1-score multiplied:  0.426242508987
Epoch 18: 
Test set LL -1.079742231176521, Acc 0.8268915414810181, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.445416757821 F1-OK:  0.897439009589
F1-score multiplied:  0.399734373993
[ 77558  65635  39900 ...,  52102 175656  30661]
Epoch 19: \Dev set LL -1.074110000739413, Acc 0.8289850950241089, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.469940728196 F1-OK:  0.898045602606
F1-score multiplied:  0.422028204442
Epoch 19: 
Test set LL -1.1195106552941303, Acc 0.8208140134811401, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.451734224822 F1-OK:  0.892906701494
F1-score multiplied:  0.403356516638
[   873  43953 181616 ..., 181234   7816  90895]
Epoch 20: \Dev set LL -0.9864409728239754, Acc 0.8441469669342041, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.448525857902 F1-OK:  0.909249980116
F1-score multiplied:  0.407822127379
Epoch 20: 
Test set LL -1.0560522983711886, Acc 0.8344714641571045, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.423954372624 F1-OK:  0.903349282297
F1-score multiplied:  0.382978878236
[139636 198046 199708 ...,  32597 183833  57999]
Epoch 21: \Dev set LL -1.0392370919914053, Acc 0.8366343379020691, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.478184991274 F1-OK:  0.903157894737
F1-score multiplied:  0.431876550014
Epoch 21: 
Test set LL -1.1140059987094237, Acc 0.8243649005889893, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.4499572284 F1-OK:  0.895498130993
F1-score multiplied:  0.402935857059
[ 48250  85392  96772 ..., 159417  47120  80258]
Epoch 22: \Dev set LL -1.028656827423876, Acc 0.8367709517478943, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.473336271485 F1-OK:  0.903418734341
F1-score multiplied:  0.427620855303
Epoch 22: 
Test set LL -1.1086235870566379, Acc 0.8240234851837158, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.444492347489 F1-OK:  0.89545214816
F1-score multiplied:  0.398021627399
[ 94976 169993 129145 ..., 212870  49486 180803]
Epoch 23: \Dev set LL -1.0259442930337719, Acc 0.8410053253173828, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.475202885482 F1-OK:  0.906310367032
F1-score multiplied:  0.430681301556
Epoch 23: 
Test set LL -1.0925876992847494, Acc 0.8285304307937622, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.448979591837 F1-OK:  0.898467510412
F1-score multiplied:  0.403393576103
[188706 216106  14316 ...,  23716  33182   2019]
Epoch 24: \Dev set LL -1.012437073520265, Acc 0.840595543384552, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.461964038728 F1-OK:  0.906437905877
F1-score multiplied:  0.418741715855
Epoch 24: 
Test set LL -1.0788405965263157, Acc 0.830852210521698, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.436917481246 F1-OK:  0.900478122866
F1-score multiplied:  0.393434633359
[ 49555 204099 199462 ..., 142630  30997  94732]
Epoch 25: \Dev set LL -1.084800439305873, Acc 0.8242043256759644, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.466196598922 F1-OK:  0.894775570272
F1-score multiplied:  0.417141327659
Epoch 25: 
Test set LL -1.1401793852237825, Acc 0.8181507587432861, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.455307833913 F1-OK:  0.89085618263
F1-score multiplied:  0.405613798841
[ 10586  40093 110450 ..., 135123  16134  39003]
Epoch 26: \Dev set LL -1.0550215491059087, Acc 0.833082914352417, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.465441819773 F1-OK:  0.901100679832
F1-score multiplied:  0.419409940219
Epoch 26: 
Test set LL -1.1142373427065364, Acc 0.8252526521682739, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.451916898694 F1-OK:  0.896055891791
F1-score multiplied:  0.404942799674
[124016  93299 142951 ..., 121315 130112 138434]
Epoch 27: \Dev set LL -1.0083568256740023, Acc 0.8438737988471985, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.473514509443 F1-OK:  0.908347365889
F1-score multiplied:  0.430115657362
Epoch 27: 
Test set LL -1.0953286953406018, Acc 0.8291450142860413, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.435723951286 F1-OK:  0.899332099461
F1-score multiplied:  0.391860535895
[  9469 193383  39949 ..., 202484  63420  57130]
Epoch 28: \Dev set LL -1.015595003485752, Acc 0.841141939163208, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.459321245932 F1-OK:  0.906892962933
F1-score multiplied:  0.416555205662
Epoch 28: 
Test set LL -1.0848848744608095, Acc 0.830647349357605, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.432494279176 F1-OK:  0.900473553255
F1-score multiplied:  0.389449660332
[ 70289  17736 199270 ...,  16586  34471 159323]
Epoch 29: \Dev set LL -1.0618763990549094, Acc 0.8333560824394226, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.471403812825 F1-OK:  0.901086427761
F1-score multiplied:  0.424775577731
Epoch 29: 
Test set LL -1.129714656730309, Acc 0.8211554288864136, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.443594646272 F1-OK:  0.893454293967
F1-score multiplied:  0.396331541492
[ 44559 117975   7356 ..., 164492 159093  15768]
Done!




Matern12 - phrase - 72 features - instead of using 17 hidden -> 100 hidden

Epoch 1: \Dev set LL -0.44776423640250096, Acc 0.838546633720398, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.414271555996 F1-OK:  0.906368821293
F1-score multiplied:  0.375482821903
Epoch 1: 
Test set LL -0.4647355119731794, Acc 0.8318082690238953, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.388074534161 F1-OK:  0.90250564066
F1-score multiplied:  0.350239456077
[ 10604 112260   2170 ...,  24253  38081  64020]
Epoch 2: \Dev set LL -0.6691451806445993, Acc 0.8363611698150635, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.449448529412 F1-OK:  0.903898604203
F1-score multiplied:  0.406255898397
Epoch 2: 
Test set LL -0.7075913160882169, Acc 0.8298962116241455, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.421907635182 F1-OK:  0.900276232035
F1-score multiplied:  0.379833416069
[ 16587 157088  51334 ...,  99848  92538  64243]
Epoch 3: \Dev set LL -0.7754166139163946, Acc 0.8431908488273621, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.437254901961 F1-OK:  0.908903348675
F1-score multiplied:  0.397422444617
Epoch 3: 
Test set LL -0.8245307631372202, Acc 0.835837185382843, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.414800389484 F1-OK:  0.904527402701
F1-score multiplied:  0.375198318939
[ 97998  95754  98742 ..., 203012 213193 125451]
Epoch 4: \Dev set LL -0.833131028342462, Acc 0.8396393656730652, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.469738030714 F1-OK:  0.905535886707
F1-score multiplied:  0.425364644162
Epoch 4: 
Test set LL -0.8995078258913888, Acc 0.8261404037475586, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.434975588105 F1-OK:  0.89726414333
F1-score multiplied:  0.39028799843
[ 82707 140148 157986 ...,  51883 106201 103473]
Epoch 5: \Dev set LL -0.8792139403834678, Acc 0.8362245559692383, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.471573380344 F1-OK:  0.90309544977
F1-score multiplied:  0.425875774021
Epoch 5: 
Test set LL -0.9378459405890391, Acc 0.8229308724403381, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.436916395223 F1-OK:  0.894947939878
F1-score multiplied:  0.391017427803
[ 96593 186600 181132 ...,  40532  34553 129546]
Epoch 6: \Dev set LL -0.9120177672583853, Acc 0.8408687114715576, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.456876456876 F1-OK:  0.90677762663
F1-score multiplied:  0.41428534923
Epoch 6: 
Test set LL -0.9675335600023591, Acc 0.831261932849884, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.428670520231 F1-OK:  0.90101349998
F1-score multiplied:  0.386237925772
[211235 108099  35584 ..., 173533 203627 154044]
Epoch 7: \Dev set LL -0.9162561416397664, Acc 0.8479716181755066, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.460494425594 F1-OK:  0.911519198664
F1-score multiplied:  0.419749509807
Epoch 7: 
Test set LL -0.9789218372406274, Acc 0.8349494934082031, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.41940907999 F1-OK:  0.903800995025
F1-score multiplied:  0.379062343818
[ 44555 140306 112838 ...,  74919 150399 165017]
Epoch 8: \Dev set LL -0.9435850420653584, Acc 0.8437371850013733, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.459867799811 F1-OK:  0.908655381667
F1-score multiplied:  0.417861351154
Epoch 8: 
Test set LL -1.0055441497745576, Acc 0.8347446322441101, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.430051813472 F1-OK:  0.90336235125
F1-score multiplied:  0.388492617377
[169730  20673 186182 ...,  67812 116880 108578]
Epoch 9: \Dev set LL -0.9750543326738952, Acc 0.8388198614120483, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.463148316652 F1-OK:  0.905175184828
F1-score multiplied:  0.419230363128
Epoch 9: 
Test set LL -1.0285809144094513, Acc 0.8298279047012329, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.43721770551 F1-OK:  0.899758648431
F1-score multiplied:  0.39339041178
[ 41613 123879  70949 ...,  98250 105271 211104]
Epoch 10: \Dev set LL -1.0051153314991728, Acc 0.8356781601905823, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.469810489202 F1-OK:  0.902772165198
F1-score multiplied:  0.42413183257
Epoch 10: 
Test set LL -1.0624925437029356, Acc 0.8251160979270935, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.444829828745 F1-OK:  0.896210739615
F1-score multiplied:  0.398661269822
[ 79978  62975 104982 ..., 106252 110351  75418]
Epoch 11: \Dev set LL -0.988200206581095, Acc 0.8431908488273621, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.464552238806 F1-OK:  0.908145303249
F1-score multiplied:  0.421880933785
Epoch 11: 
Test set LL -1.052249693825962, Acc 0.8309888243675232, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.437883261413 F1-OK:  0.900542495479
F1-score multiplied:  0.394332484961
[ 78595 203579  77628 ..., 171197 169790 184869]
Epoch 12: \Dev set LL -0.9876502162023854, Acc 0.8423712849617004, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.462756052142 F1-OK:  0.907635665119
F1-score multiplied:  0.420013897173
Epoch 12: 
Test set LL -1.0489722996158177, Acc 0.8329691290855408, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.436405529954 F1-OK:  0.901956068623
F1-score multiplied:  0.393618616122
[161145   8624 103629 ..., 131598  77971 145710]
Epoch 13: \Dev set LL -1.0061379293517567, Acc 0.8408687114715576, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.466819221968 F1-OK:  0.906478285301
F1-score multiplied:  0.423161487875
Epoch 13: 
Test set LL -1.0764199504698182, Acc 0.8285304307937622, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.437121721587 F1-OK:  0.898860111975
F1-score multiplied:  0.392911279612
[ 88180  37890  33658 ..., 138465 165899 187250]
Epoch 14: \Dev set LL -1.0010724430650118, Acc 0.841141939163208, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.470641784251 F1-OK:  0.906548814785
F1-score multiplied:  0.426659751701
Epoch 14: 
Test set LL -1.0694452566163848, Acc 0.8311936855316162, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.44623655914 F1-OK:  0.900418949404
F1-score multiplied:  0.401799853766
[ 95272 217111 176486 ..., 183877 161754 214087]
Epoch 15: \Dev set LL -0.9882306402402267, Acc 0.8452396988868713, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.459188544153 F1-OK:  0.909699529768
F1-score multiplied:  0.417723602691
Epoch 15: 
Test set LL -1.0466908575928495, Acc 0.8364517688751221, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.432060706664 F1-OK:  0.904471301504
F1-score multiplied:  0.390786509685
[ 43428  23203 165494 ...,  87260 127806  73850]
Epoch 16: \Dev set LL -1.0435095050779577, Acc 0.8355416059494019, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.466312056738 F1-OK:  0.902793476506
F1-score multiplied:  0.420983482839
Epoch 16: 
Test set LL -1.0943209990479494, Acc 0.8273012638092041, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.448899542384 F1-OK:  0.897607190575
F1-score multiplied:  0.402935457089
[204644  31141 103650 ...,  30090  81702 136580]
Epoch 17: \Dev set LL -1.0170590663510983, Acc 0.840595543384552, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.465414567109 F1-OK:  0.90633277149
F1-score multiplied:  0.4218204745
Epoch 17: 
Test set LL -1.0793258627333207, Acc 0.8309205174446106, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.443094916779 F1-OK:  0.900330086144
F1-score multiplied:  0.398931684594
[151463 159307 100073 ...,  99926  64887  45399]
Epoch 18: \Dev set LL -1.0137538818779985, Acc 0.8425078392028809, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.476622787108 F1-OK:  0.907307661388
F1-score multiplied:  0.432443506335
Epoch 18: 
Test set LL -1.0847251246023504, Acc 0.8297596573829651, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.442655935614 F1-OK:  0.899536570623
F1-score multiplied:  0.398185202288
[ 77558  65635  39900 ...,  52102 175656  30661]
Epoch 19: \Dev set LL -1.0172075599977113, Acc 0.8412784934043884, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.462534690102 F1-OK:  0.906891025641
F1-score multiplied:  0.419468559501
Epoch 19: 
Test set LL -1.0738683180612814, Acc 0.8325594067573547, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.435543278085 F1-OK:  0.901699807569
F1-score multiplied:  0.392729290037
[   873  43953 181616 ..., 181234   7816  90895]
Epoch 20: \Dev set LL -1.0197375793894354, Acc 0.840595543384552, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.474561008555 F1-OK:  0.90604621206
F1-score multiplied:  0.429974204192
Epoch 20: 
Test set LL -1.0915724453397668, Acc 0.8305107951164246, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.451370468612 F1-OK:  0.899773865288
F1-score multiplied:  0.40613135122
[139636 198046 199708 ...,  32597 183833  57999]
Epoch 21: \Dev set LL -1.0341317385667785, Acc 0.8386832475662231, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.468257541648 F1-OK:  0.90491908864
F1-score multiplied:  0.423735187837
Epoch 21: 
Test set LL -1.099027884621566, Acc 0.8293498754501343, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.44773480663 F1-OK:  0.899083309777
F1-score multiplied:  0.402550891847
[ 48250  85392  96772 ..., 159417  47120  80258]
Epoch 22: \Dev set LL -1.0179367944625475, Acc 0.841141939163208, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.468221307727 F1-OK:  0.906623845845
F1-score multiplied:  0.424500602718
Epoch 22: 
Test set LL -1.0844912644366251, Acc 0.8316034078598022, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.446091644205 F1-OK:  0.900708648736
F1-score multiplied:  0.401798602064
[ 94976 169993 129145 ..., 212870  49486 180803]
Epoch 23: \Dev set LL -1.0224608563043358, Acc 0.8416882753372192, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.460679385761 F1-OK:  0.907228047707
F1-score multiplied:  0.417941259763
Epoch 23: 
Test set LL -1.0889829305820629, Acc 0.8316034078598022, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.4375 F1-OK:  0.900979762287
F1-score multiplied:  0.394178646001
[188706 216106  14316 ...,  23716  33182   2019]
Epoch 24: \Dev set LL -1.0092273778249015, Acc 0.8446933627128601, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.460370194589 F1-OK:  0.909293976865
F1-score multiplied:  0.418611845068
Epoch 24: 
Test set LL -1.0626666422618354, Acc 0.8366566300392151, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.433980123048 F1-OK:  0.904556699386
F1-score multiplied:  0.392559627703
[ 49555 204099 199462 ..., 142630  30997  94732]
Epoch 25: \Dev set LL -1.0422784591795924, Acc 0.8389564156532288, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.466757123474 F1-OK:  0.90515646368
F1-score multiplied:  0.422488227281
Epoch 25: 
Test set LL -1.1077568306024206, Acc 0.8288035988807678, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.448404840484 F1-OK:  0.898678414097
F1-score multiplied:  0.40297175092
[ 10586  40093 110450 ..., 135123  16134  39003]
Epoch 26: \Dev set LL -1.055283201621927, Acc 0.833082914352417, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.46356453029 F1-OK:  0.901164671627
F1-score multiplied:  0.417747977717
Epoch 26: 
Test set LL -1.1107009587991328, Acc 0.8262087106704712, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.448298287449 F1-OK:  0.8968591692
F1-score multiplied:  0.402060429635
[124016  93299 142951 ..., 121315 130112 138434]
Epoch 27: \Dev set LL -1.0231107087285385, Acc 0.8395028114318848, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.458774758176 F1-OK:  0.905781412878
F1-score multiplied:  0.415549648653
Epoch 27: 
Test set LL -1.0813735034953684, Acc 0.8322862386703491, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.443336355394 F1-OK:  0.901270300691
F1-score multiplied:  0.399565890334
[  9469 193383  39949 ..., 202484  63420  57130]
Epoch 28: \Dev set LL -1.0199393858127412, Acc 0.8418248891830444, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.474591651543 F1-OK:  0.906898215147
F1-score multiplied:  0.430406321708
Epoch 28: 
Test set LL -1.0891891030029268, Acc 0.8307156562805176, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.449233503666 F1-OK:  0.899987896881
F1-score multiplied:  0.404304716173
[ 70289  17736 199270 ...,  16586  34471 159323]
Epoch 29: \Dev set LL -1.0251688557651633, Acc 0.840049147605896, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.466514806378 F1-OK:  0.905921105487
F1-score multiplied:  0.42262560912
Epoch 29: 
Test set LL -1.0978298740826162, Acc 0.8282573223114014, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.443460942686 F1-OK:  0.898461786911
F1-score multiplied:  0.398432710991
[ 44559 117975   7356 ..., 164492 159093  15768]
Done!


Gaussia processes - only - 72 features - matern52 - phrase

Epoch 1: \Dev set LL -0.5460106441549424, Acc 0.7987979650497437, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.330758746025 F1-OK:  0.881601157463
F1-score multiplied:  0.291597293336
Epoch 1: 
Test set LL -0.5487441200641229, Acc 0.7947282195091248, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.331108144192 F1-OK:  0.878760990562
F1-score multiplied:  0.290964920774
[ 10604 112260   2170 ...,  24253  38081  64020]
Epoch 2: \Dev set LL -0.4387871884787938, Acc 0.8370441198348999, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.324844368987 F1-OK:  0.907339805825
F1-score multiplied:  0.29474422668
Epoch 2: 
Test set LL -0.45018598618107714, Acc 0.8292816281318665, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.310535024821 F1-OK:  0.902579689814
F1-score multiplied:  0.280282606379
[ 16587 157088  51334 ...,  99848  92538  64243]
Epoch 3: \Dev set LL -0.5105204372392332, Acc 0.8456494808197021, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.315151515152 F1-OK:  0.913023399015
F1-score multiplied:  0.287740707568
Epoch 3: 
Test set LL -0.5316783109048019, Acc 0.8389784097671509, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.297377830751 F1-OK:  0.909069875058
F1-score multiplied:  0.270337227446
[ 97998  95754  98742 ..., 203012 213193 125451]
Epoch 4: \Dev set LL -0.59296528763708, Acc 0.8456494808197021, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.323353293413 F1-OK:  0.912889300031
F1-score multiplied:  0.295185761687
Epoch 4: 
Test set LL -0.6222670752965531, Acc 0.840412437915802, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.310008857396 F1-OK:  0.909771823482
F1-score multiplied:  0.282037323489
[ 82707 140148 157986 ...,  51883 106201 103473]
Epoch 5: \Dev set LL -0.6483049671285707, Acc 0.846059262752533, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.332741267022 F1-OK:  0.912993129005
F1-score multiplied:  0.303790490527
Epoch 5: 
Test set LL -0.6823718142622606, Acc 0.840617299079895, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.31993006993 F1-OK:  0.909730816832
F1-score multiplied:  0.291050243846
[ 96593 186600 181132 ...,  40532  34553 129546]
Epoch 6: \Dev set LL -0.6837085175958232, Acc 0.8419615030288696, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.339988590987 F1-OK:  0.91023353247
F1-score multiplied:  0.309469016173
Epoch 6: 
Test set LL -0.7215108528173237, Acc 0.8377492427825928, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.330704225352 F1-OK:  0.90768513482
F1-score multiplied:  0.300175309374
[211235 108099  35584 ..., 173533 203627 154044]
Epoch 7: \Dev set LL -0.7159680639754394, Acc 0.8410053253173828, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.348264277716 F1-OK:  0.909458618544
F1-score multiplied:  0.316731948899
Epoch 7: 
Test set LL -0.7564291068328312, Acc 0.836042046546936, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.331756192597 F1-OK:  0.90655769605
F1-score multiplied:  0.300756129611
[ 44555 140306 112838 ...,  74919 150399 165017]
Epoch 8: \Dev set LL -0.7317261285790353, Acc 0.838546633720398, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.357608695652 F1-OK:  0.907670676457
F1-score multiplied:  0.324590926689
Epoch 8: 
Test set LL -0.7741302014193568, Acc 0.8333105444908142, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.339019767127 F1-OK:  0.90462981051
F1-score multiplied:  0.306687387695
[169730  20673 186182 ...,  67812 116880 108578]
Epoch 9: \Dev set LL -0.748061452858849, Acc 0.8381368517875671, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.372019077901 F1-OK:  0.907095256762
F1-score multiplied:  0.337456740989
Epoch 9: 
Test set LL -0.7920156883278913, Acc 0.8311253786087036, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.349381741647 F1-OK:  0.902970141641
F1-score multiplied:  0.315481280742
[ 41613 123879  70949 ...,  98250 105271 211104]
Epoch 10: \Dev set LL -0.7636789410540683, Acc 0.8374539017677307, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.380853277836 F1-OK:  0.906446540881
F1-score multiplied:  0.345223136277
Epoch 10: 
Test set LL -0.8095271253579883, Acc 0.830237627029419, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.358947911294 F1-OK:  0.902164502165
F1-score multiplied:  0.323830063696
[ 79978  62975 104982 ..., 106252 110351  75418]
Epoch 11: \Dev set LL -0.7800583687909963, Acc 0.8370441198348999, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.385368366821 F1-OK:  0.906070388158
F1-score multiplied:  0.34917086571
Epoch 11: 
Test set LL -0.8274086188097932, Acc 0.8294181823730469, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.363728986246 F1-OK:  0.901506190364
F1-score multiplied:  0.327903932715
[ 78595 203579  77628 ..., 171197 169790 184869]
Epoch 12: \Dev set LL -0.7968654236492794, Acc 0.8358147740364075, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.387359836901 F1-OK:  0.905205047319
F1-score multiplied:  0.350640079491
Epoch 12: 
Test set LL -0.8471855170787189, Acc 0.8289401531219482, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.364696931271 F1-OK:  0.90116393766
F1-score multiplied:  0.328651722636
[161145   8624 103629 ..., 131598  77971 145710]
Epoch 13: \Dev set LL -0.8091234679776717, Acc 0.8352683782577515, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.392749244713 F1-OK:  0.904709228824
F1-score multiplied:  0.355323866306
Epoch 13: 
Test set LL -0.859462175623407, Acc 0.8285987377166748, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.370927318296 F1-OK:  0.900782670567
F1-score multiplied:  0.334124900361
[ 88180  37890  33658 ..., 138465 165899 187250]
Epoch 14: \Dev set LL -0.8206563908536897, Acc 0.8348585963249207, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.395197598799 F1-OK:  0.904373961876
F1-score multiplied:  0.35740641815
Epoch 14: 
Test set LL -0.8728538627462618, Acc 0.8280524611473083, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.373007968127 F1-OK:  0.900364039253
F1-score multiplied:  0.335842960857
[ 95272 217111 176486 ..., 183877 161754 214087]
Epoch 15: \Dev set LL -0.8395616402847395, Acc 0.8344488739967346, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.390954773869 F1-OK:  0.904204868795
F1-score multiplied:  0.353503210011
Epoch 15: 
Test set LL -0.8945791534055972, Acc 0.8274378776550293, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.367775831874 F1-OK:  0.90008303349
F1-score multiplied:  0.331028786397
[ 43428  23203 165494 ...,  87260 127806  73850]
Epoch 16: \Dev set LL -0.8481081971004485, Acc 0.8349952101707458, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.393574297189 F1-OK:  0.904505928854
F1-score multiplied:  0.355990285252
Epoch 16: 
Test set LL -0.9035665107131705, Acc 0.8279158473014832, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.369053580371 F1-OK:  0.900371629635
F1-score multiplied:  0.332285373581
[204644  31141 103650 ...,  30090  81702 136580]
Epoch 17: \Dev set LL -0.8519223449769541, Acc 0.8334926962852478, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.396833250866 F1-OK:  0.903414943348
F1-score multiplied:  0.35850508885
Epoch 17: 
Test set LL -0.907494114963491, Acc 0.8267549872398376, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.374352651048 F1-OK:  0.899457060199
F1-score multiplied:  0.336714134989
[151463 159307 100073 ...,  99926  64887  45399]
Epoch 18: \Dev set LL -0.860645823449573, Acc 0.8340390920639038, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.392196098049 F1-OK:  0.903899390967
F1-score multiplied:  0.354505814166
Epoch 18: 
Test set LL -0.920449388308449, Acc 0.8270964026451111, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.370775347913 F1-OK:  0.899778340722
F1-score multiplied:  0.333615627325
[ 77558  65635  39900 ...,  52102 175656  30661]
Epoch 19: \Dev set LL -0.8627530118647371, Acc 0.833082914352417, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.397435897436 F1-OK:  0.903123513556
F1-score multiplied:  0.358933704106
Epoch 19: 
Test set LL -0.9223407699664837, Acc 0.8262087106704712, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.372070071552 F1-OK:  0.899148008718
F1-score multiplied:  0.33454606394
[   873  43953 181616 ..., 181234   7816  90895]
Epoch 20: \Dev set LL -0.8722410809101663, Acc 0.833629310131073, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.398221343874 F1-OK:  0.903471231574
F1-score multiplied:  0.359781527988
Epoch 20: 
Test set LL -0.9313910067636264, Acc 0.8259355425834656, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.373556156304 F1-OK:  0.898925413379
F1-score multiplied:  0.335799122226
[139636 198046 199708 ...,  32597 183833  57999]
Epoch 21: \Dev set LL -0.8779724231434555, Acc 0.833629310131073, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.395833333333 F1-OK:  0.903532393474
F1-score multiplied:  0.357648239083
Epoch 21: 
Test set LL -0.938220159577997, Acc 0.8269598484039307, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.372461614661 F1-OK:  0.899643564356
F1-score multiplied:  0.335082694599
[ 48250  85392  96772 ..., 159417  47120  80258]


Epoch 1: \Dev set LL -0.5671985422869743, Acc 0.8229749798774719, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.317894736842 F1-OK:  0.898289122587
F1-score multiplied:  0.285561384233
Epoch 1: 
Test set LL -0.5695403622563391, Acc 0.8168533444404602, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.312307692308 F1-OK:  0.89435953994
F1-score multiplied:  0.279315364012
[ 10604 112260   2170 ...,  24253  38081  64020]
Epoch 2: \Dev set LL -0.43372405967888605, Acc 0.8425078392028809, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.292203806016 F1-OK:  0.911396296012
F1-score multiplied:  0.266313466483
Epoch 2: 
Test set LL -0.4411925081098339, Acc 0.8365883827209473, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.277257626095 F1-OK:  0.907880047735
F1-score multiplied:  0.251716666814
[ 16587 157088  51334 ...,  99848  92538  64243]
Epoch 3: \Dev set LL -0.4896923653190095, Acc 0.846605658531189, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.284257488846 F1-OK:  0.914097758739
F1-score multiplied:  0.259839133459
Epoch 3: 
Test set LL -0.505055670562966, Acc 0.8405490517616272, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.264566929134 F1-OK:  0.910580936698
F1-score multiplied:  0.24090960215
[ 97998  95754  98742 ..., 203012 213193 125451]
Epoch 4: \Dev set LL -0.6458602120146377, Acc 0.846605658531189, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.309772587585 F1-OK:  0.913714944295
F1-score multiplied:  0.283043842609
Epoch 4: 
Test set LL -0.6751912905206581, Acc 0.8383638262748718, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.284678150499 F1-OK:  0.908887947958
F1-score multiplied:  0.258740540035
[ 82707 140148 157986 ...,  51883 106201 103473]
Epoch 5: \Dev set LL -0.7727535204092579, Acc 0.8452396988868713, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.341661824521 F1-OK:  0.912313288445
F1-score multiplied:  0.311702622665
Epoch 5: 
Test set LL -0.8151384151727492, Acc 0.8384321331977844, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.320895522388 F1-OK:  0.908308789335
F1-score multiplied:  0.291472223443
[ 96593 186600 181132 ...,  40532  34553 129546]
Epoch 6: \Dev set LL -0.781449279612514, Acc 0.8408687114715576, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.369247428262 F1-OK:  0.908948808128
F1-score multiplied:  0.335627009823
Epoch 6: 
Test set LL -0.8311216677944965, Acc 0.8320131301879883, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.33692722372 F1-OK:  0.903823598405
F1-score multiplied:  0.304522775743
[211235 108099  35584 ..., 173533 203627 154044]
Epoch 7: \Dev set LL -0.7924428930458927, Acc 0.8419615030288696, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.368794326241 F1-OK:  0.909672886252
F1-score multiplied:  0.335482199185
Epoch 7: 
Test set LL -0.8423134788193705, Acc 0.8326277136802673, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.335592301437 F1-OK:  0.904254072425
F1-score multiplied:  0.303460705249
[ 44555 140306 112838 ...,  74919 150399 165017]
Epoch 8: \Dev set LL -0.8026443748267191, Acc 0.8404589295387268, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.376734258271 F1-OK:  0.908521303258
F1-score multiplied:  0.342271099306
Epoch 8: 
Test set LL -0.8534419451166504, Acc 0.831057071685791, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.348604528699 F1-OK:  0.902942330326
F1-score multiplied:  0.314769785506
[169730  20673 186182 ...,  67812 116880 108578]
Epoch 9: \Dev set LL -0.8155513901146186, Acc 0.8366343379020691, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.387295081967 F1-OK:  0.905752561072
F1-score multiplied:  0.350793512382
Epoch 9: 
Test set LL -0.868280484473278, Acc 0.8280524611473083, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.359938993391 F1-OK:  0.900686282243
F1-score multiplied:  0.324192113792
[ 41613 123879  70949 ...,  98250 105271 211104]
Epoch 10: \Dev set LL -0.8231914324893042, Acc 0.8366343379020691, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.392893401015 F1-OK:  0.905618686869
F1-score multiplied:  0.355811605907
Epoch 10: 
Test set LL -0.877042565745267, Acc 0.8273695707321167, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.360647445625 F1-OK:  0.900213152285
F1-score multiplied:  0.324659573889
[ 79978  62975 104982 ..., 106252 110351  75418]
Epoch 11: \Dev set LL -0.827048314042897, Acc 0.8371807336807251, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.38683127572 F1-OK:  0.906126949126
F1-score multiplied:  0.350518243695
Epoch 11: 
Test set LL -0.8808464095363211, Acc 0.8280524611473083, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.358634742741 F1-OK:  0.900717609021
F1-score multiplied:  0.323028627993
[ 78595 203579  77628 ..., 171197 169790 184869]
Epoch 12: \Dev set LL -0.8288176621636452, Acc 0.8363611698150635, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.386270491803 F1-OK:  0.905594956659
F1-score multiplied:  0.349804609283
Epoch 12: 
Test set LL -0.8849724519516047, Acc 0.8277792930603027, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.358596134283 F1-OK:  0.900536362202
F1-score multiplied:  0.322928858267
[161145   8624 103629 ..., 131598  77971 145710]
Epoch 13: \Dev set LL -0.833456999537028, Acc 0.8370441198348999, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.392875318066 F1-OK:  0.905892561332
F1-score multiplied:  0.355902828167
Epoch 13: 
Test set LL -0.8888729827888241, Acc 0.8280524611473083, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.363820111167 F1-OK:  0.900592183182
F1-score multiplied:  0.327653548202
[ 88180  37890  33658 ..., 138465 165899 187250]
Epoch 14: \Dev set LL -0.8322505193875633, Acc 0.8377270698547363, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.393258426966 F1-OK:  0.906338694418
F1-score multiplied:  0.356425329266
Epoch 14: 
Test set LL -0.8887964320628302, Acc 0.8281207084655762, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.365515502899 F1-OK:  0.900596342956
F1-score multiplied:  0.329181925205
[ 95272 217111 176486 ..., 183877 161754 214087]
Epoch 15: \Dev set LL -0.8474737234796186, Acc 0.8374539017677307, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.386597938144 F1-OK:  0.906313966305
F1-score multiplied:  0.350379110685
Epoch 15: 
Test set LL -0.9023281393825263, Acc 0.8290084600448608, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.360572012257 F1-OK:  0.901308529087
F1-score multiplied:  0.324986629998
[ 43428  23203 165494 ...,  87260 127806  73850]
Epoch 16: \Dev set LL -0.8477997541181771, Acc 0.8360879421234131, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.387129724208 F1-OK:  0.905392620624
F1-score multiplied:  0.350504395523
Epoch 16: 
Test set LL -0.9052152535380127, Acc 0.8288035988807678, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.363867038823 F1-OK:  0.901092831499
F1-score multiplied:  0.327877980302
[204644  31141 103650 ...,  30090  81702 136580]
Epoch 17: \Dev set LL -0.8476163715190348, Acc 0.8355416059494019, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.394974874372 F1-OK:  0.904837179893
F1-score multiplied:  0.357387951455
Epoch 17: 
Test set LL -0.9058227376869632, Acc 0.8271647095680237, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.368670491394 F1-OK:  0.899877368567
F1-score multiplied:  0.331758231664
[151463 159307 100073 ...,  99926  64887  45399]
Epoch 18: \Dev set LL -0.8517431659452694, Acc 0.8370441198348999, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.394108684611 F1-OK:  0.905862858045
F1-score multiplied:  0.357008419422
Epoch 18: 
Test set LL -0.9117296489710632, Acc 0.8288035988807678, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.367398435529 F1-OK:  0.901006910168
F1-score multiplied:  0.331028529196
[ 77558  65635  39900 ...,  52102 175656  30661]
Epoch 19: \Dev set LL -0.8473649951645946, Acc 0.8354049921035767, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.396594892339 F1-OK:  0.904705417161
F1-score multiplied:  0.358801547517
Epoch 19: 
Test set LL -0.9082156572463885, Acc 0.826686680316925, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.36739780658 F1-OK:  0.899588542491
F1-score multiplied:  0.330506857336
[   873  43953 181616 ..., 181234   7816  90895]
Epoch 20: \Dev set LL -0.8571512846977608, Acc 0.8349952101707458, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.396603396603 F1-OK:  0.904430379747
F1-score multiplied:  0.358700160599
Epoch 20: 
Test set LL -0.9182596937803895, Acc 0.8268915414810181, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.369873228934 F1-OK:  0.899663566198
F1-score multiplied:  0.332761468184
[139636 198046 199708 ...,  32597 183833  57999]
Epoch 21: \Dev set LL -0.8564711275050088, Acc 0.8348585963249207, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.389702170621 F1-OK:  0.904509912329
F1-score multiplied:  0.352489476183
Epoch 21: 
Test set LL -0.9162071188727773, Acc 0.8281890153884888, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.367521367521 F1-OK:  0.900592651126
F1-score multiplied:  0.330987042722
[ 48250  85392  96772 ..., 159417  47120  80258]
Epoch 22: \Dev set LL -0.8614665539019747, Acc 0.8355416059494019, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.396188565697 F1-OK:  0.904807084124
F1-score multiplied:  0.358474220892
Epoch 22: 
Test set LL -0.9232147282774539, Acc 0.8269598484039307, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.365230460922 F1-OK:  0.899826059456
F1-score multiplied:  0.328643886445
[ 94976 169993 129145 ..., 212870  49486 180803]
Epoch 23: \Dev set LL -0.8631556724596497, Acc 0.834585428237915, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.395406889666 F1-OK:  0.90418545771
F1-score multiplied:  0.357521159514
Epoch 23: 
Test set LL -0.9254795121590798, Acc 0.8267549872398376, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.369376087497 F1-OK:  0.899584405304
F1-score multiplied:  0.332284968004
[188706 216106  14316 ...,  23716  33182   2019]
Epoch 24: \Dev set LL -0.8647382116054273, Acc 0.834585428237915, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.396009975062 F1-OK:  0.904170293582
F1-score multiplied:  0.358060455414
Epoch 24: 
Test set LL -0.9260399598809229, Acc 0.8265501260757446, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.369100844511 F1-OK:  0.899453724962
F1-score multiplied:  0.331989129482
[ 49555 204099 199462 ..., 142630  30997  94732]
Epoch 25: \Dev set LL -0.8672789107532626, Acc 0.8333560824394226, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.394240317776 F1-OK:  0.903389293633
F1-score multiplied:  0.356152482197
Epoch 25: 
Test set LL -0.9294212662308233, Acc 0.826481819152832, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.370883882149 F1-OK:  0.899362350984
F1-score multiplied:  0.333559000192
[ 10586  40093 110450 ..., 135123  16134  39003]
Epoch 26: \Dev set LL -0.8682645310970621, Acc 0.8339024782180786, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.396226415094 F1-OK:  0.903706050048
F1-score multiplied:  0.358072208509
Epoch 26: 
Test set LL -0.9298395820756002, Acc 0.8270281553268433, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.371308016878 F1-OK:  0.899718912071
F1-score multiplied:  0.334072844988
[124016  93299 142951 ..., 121315 130112 138434]
Epoch 27: \Dev set LL -0.8676608919518178, Acc 0.8343122601509094, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.394408387419 F1-OK:  0.904027217343
F1-score multiplied:  0.356555916975
Epoch 27: 
Test set LL -0.9306562590672847, Acc 0.826686680316925, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.368027888446 F1-OK:  0.899572649573
F1-score multiplied:  0.331067822726
[  9469 193383  39949 ..., 202484  63420  57130]
Epoch 28: \Dev set LL -0.8667432136133975, Acc 0.8329463005065918, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.396645288604 F1-OK:  0.903051922315
F1-score multiplied:  0.358191290351
Epoch 28: 
Test set LL -0.9281150791793046, Acc 0.8267549872398376, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.373734880276 F1-OK:  0.899472995998
F1-score multiplied:  0.336164432471
[ 70289  17736 199270 ...,  16586  34471 159323]
Epoch 29: \Dev set LL -0.8701280445636045, Acc 0.8341756463050842, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.400197628458 F1-OK:  0.903788239024
F1-score multiplied:  0.361693909886
Epoch 29: 
Test set LL -0.9320565179079972, Acc 0.8260038495063782, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.372104484968 F1-OK:  0.899009116132
F1-score multiplied:  0.33452532414
[ 44559 117975   7356 ..., 164492 159093  15768]
Epoch 30: \Dev set LL -0.8706620650162088, Acc 0.8347220420837402, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.39620758483 F1-OK:  0.90425700269
F1-score multiplied:  0.358273483102
Epoch 30: 
Test set LL -0.9337523765254344, Acc 0.8265501260757446, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.369414101291 F1-OK:  0.899445764054
F1-score multiplied:  0.332267948588
[109754 141570 108980 ..., 139365 215134  72258]
Epoch 31: \Dev set LL -0.8753667169893016, Acc 0.8343122601509094, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.399207528479 F1-OK:  0.903905569199
F1-score multiplied:  0.360845908259
Epoch 31: 
Test set LL -0.9388675834636383, Acc 0.8263452649116516, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.371633308624 F1-OK:  0.899251218256
F1-score multiplied:  0.334191705524
[188636 149668 181585 ...,  86043   3885 149926]
Epoch 32: \Dev set LL -0.8800449887526548, Acc 0.8326731324195862, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.399804017638 F1-OK:  0.902785493215
F1-score multiplied:  0.360937267253
Epoch 32: 
Test set LL -0.9441647573470827, Acc 0.8257306814193726, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.376039119804 F1-OK:  0.898722120803
F1-score multiplied:  0.337954675256
[118069 172817 212991 ...,   6636 158235  11185]
Epoch 33: \Dev set LL -0.8774861547837968, Acc 0.8344488739967346, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.39880952381 F1-OK:  0.904007603358
F1-score multiplied:  0.360526841815
Epoch 33: 
Test set LL -0.940298757382526, Acc 0.8267549872398376, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.370315214693 F1-OK:  0.899560552674
F1-score multiplied:  0.333120959193
[ 35430  78766   7572 ...,  21798 172876 186694]
Epoch 34: \Dev set LL -0.8861576216648462, Acc 0.8344488739967346, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.395812562313 F1-OK:  0.90408357075
F1-score multiplied:  0.357847634684
Epoch 34: 
Test set LL -0.950461629184822, Acc 0.8270964026451111, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.370775347913 F1-OK:  0.899778340722
F1-score multiplied:  0.333615627325
[ 12263 179388 192145 ...,  56465 182676  75856]
Epoch 35: \Dev set LL -0.8819330395124582, Acc 0.8341756463050842, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.39900990099 F1-OK:  0.903818729203
F1-score multiplied:  0.360632621652
Epoch 35: 
Test set LL -0.9439605549157262, Acc 0.8261404037475586, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.369801980198 F1-OK:  0.899160329531
F1-score multiplied:  0.332511270376
[159646 185256 171645 ..., 194155 131629  68601]
Epoch 36: \Dev set LL -0.8817089065788423, Acc 0.8322633504867554, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.399217221135 F1-OK:  0.902524210192
F1-score multiplied:  0.3603032072
Epoch 36: 
Test set LL -0.9454089232371605, Acc 0.8251844048500061, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.373470386686 F1-OK:  0.898420760257
F1-score multiplied:  0.33553354874
[191753 201171 122412 ...,  38797  37466 188023]
Epoch 37: \Dev set LL -0.8901016354743997, Acc 0.8329463005065918, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.394854032657 F1-OK:  0.903098011251
F1-score multiplied:  0.356591891627
Epoch 37: 
Test set LL -0.954533741437613, Acc 0.8260038495063782, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.368681863231 F1-OK:  0.89909710122
F1-score multiplied:  0.331480794503
[153734 132148 148743 ...,   8738  83102  23751]
Epoch 38: \Dev set LL -0.8911196038001308, Acc 0.833629310131073, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.395833333333 F1-OK:  0.903532393474
F1-score multiplied:  0.357648239083
Epoch 38: 
Test set LL -0.9552442111755987, Acc 0.8268915414810181, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.368932038835 F1-OK:  0.899687388706
F1-score multiplied:  0.33192350263
[ 57591  60924 105622 ..., 133844  68700  81761]
Epoch 39: \Dev set LL -0.8934875936256439, Acc 0.8329463005065918, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.398425971471 F1-OK:  0.903005789515
F1-score multiplied:  0.359780958931
Epoch 39: 
Test set LL -0.9574204254605403, Acc 0.825867235660553, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.373772102161 F1-OK:  0.898873730964
F1-score multiplied:  0.335973924
[175786  45399 125058 ...,  82247 183438 112942]
Epoch 40: \Dev set LL -0.896283829228742, Acc 0.8319901823997498, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.400584795322 F1-OK:  0.902303415409
F1-score multiplied:  0.36144902898
Epoch 40: 
Test set LL -0.9592339577098423, Acc 0.8254575133323669, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.377799415774 F1-OK:  0.898490865766
F1-score multiplied:  0.339449324165
[ 96819 206932   1785 ...,  78475 169891 180553]
Epoch 41: \Dev set LL -0.8975548260265644, Acc 0.833082914352417, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.398622047244 F1-OK:  0.903092783505
F1-score multiplied:  0.359992694212
Epoch 41: 
Test set LL -0.9605052005736014, Acc 0.826276957988739, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.374938574939 F1-OK:  0.899119676422
F1-score multiplied:  0.337114650177
[ 89635 183840 203254 ..., 106660 205319 196231]
Epoch 42: \Dev set LL -0.897016916192385, Acc 0.8319901823997498, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.39468503937 F1-OK:  0.902458366376
F1-score multiplied:  0.356186815863
Epoch 42: 
Test set LL -0.9592103628357245, Acc 0.8261404037475586, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.373215164943 F1-OK:  0.899072385634
F1-score multiplied:  0.3355474487
[110684 152431 187978 ..., 190728 153645 167144]
Epoch 43: \Dev set LL -0.8962017442401101, Acc 0.8329463005065918, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.396645288604 F1-OK:  0.903051922315
F1-score multiplied:  0.358191290351
Epoch 43: 
Test set LL -0.9578527742207346, Acc 0.826072096824646, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.370333745365 F1-OK:  0.899100740799
F1-score multiplied:  0.3329673448
[ 86065 158811  87623 ...,  10358 112387 134839]
Epoch 44: \Dev set LL -0.9007652635668739, Acc 0.833082914352417, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.398622047244 F1-OK:  0.903092783505
F1-score multiplied:  0.359992694212
Epoch 44: 
Test set LL -0.9620838771052003, Acc 0.8260038495063782, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.372413793103 F1-OK:  0.899001109878
F1-score multiplied:  0.334800413334
[157439  92985  91913 ..., 110497 154030 216302]
Epoch 45: \Dev set LL -0.9026745785473507, Acc 0.8321267366409302, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.40252795333 F1-OK:  0.902344060389
F1-score multiplied:  0.363218707828
Epoch 45: 
Test set LL -0.9637205869994984, Acc 0.8253209590911865, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.376705653021 F1-OK:  0.898427573062
F1-score multiplied:  0.338442745603
[185362 193751 196467 ...,  55673 208466 140780]
Epoch 46: \Dev set LL -0.9092373800044132, Acc 0.8323999643325806, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.39882410583 F1-OK:  0.902626775653
F1-score multiplied:  0.359989316698
Epoch 46: 
Test set LL -0.9706077603735977, Acc 0.826481819152832, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.378576669112 F1-OK:  0.899162665185
F1-score multiplied:  0.340402006776
[116178 194505  95043 ...,   1569  99101 146376]
Epoch 47: \Dev set LL -0.9107607579490159, Acc 0.8328097462654114, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.398230088496 F1-OK:  0.902918781726
F1-score multiplied:  0.359569426351
Epoch 47: 
Test set LL -0.9728565605756163, Acc 0.8254575133323669, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.373529411765 F1-OK:  0.898603617899
F1-score multiplied:  0.335654880803
[175592 162391 192422 ..., 104363 138714 190912]
Epoch 48: \Dev set LL -0.9099994927826716, Acc 0.8332195281982422, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.40234948605 F1-OK:  0.903087546631
F1-score multiplied:  0.363356810245
Epoch 48: 
Test set LL -0.9714587103401495, Acc 0.8257989883422852, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.375520195838 F1-OK:  0.898781891045
F1-score multiplied:  0.337510751741
[216202  15665  40210 ...,  93117  76944 106362]
Epoch 49: \Dev set LL -0.9175313424032169, Acc 0.8318535685539246, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.402717127608 F1-OK:  0.902154041809
F1-score multiplied:  0.363312884377
Epoch 49: 
Test set LL -0.9796902173944159, Acc 0.8239551782608032, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.374878758487 F1-OK:  0.897552058496
F1-score multiplied:  0.336473201366
[ 69791  98282 134589 ..., 145613  96047 127529]
Done!



gaussian processes - only - 72 features - matern 12 - phrase
Epoch 1: \Dev set LL -0.6080585817943559, Acc 0.8468788266181946, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.256138022561 F1-OK:  0.914655500571
F1-score multiplied:  0.234278051241
Epoch 1: 
Test set LL -0.609076118628154, Acc 0.8416416049003601, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.23842364532 F1-OK:  0.911633578478
F1-score multiplied:  0.217355000977
[ 10604 112260   2170 ...,  24253  38081  64020]
Epoch 2: \Dev set LL -0.4975710804971242, Acc 0.8470154404640198, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.252336448598 F1-OK:  0.914790018259
F1-score multiplied:  0.230834864421
Epoch 2: 
Test set LL -0.5001119249526498, Acc 0.8414367437362671, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.236686390533 F1-OK:  0.911529375905
F1-score multiplied:  0.215746597847
[ 16587 157088  51334 ...,  99848  92538  64243]
Epoch 3: \Dev set LL -0.4314410127448262, Acc 0.8467422723770142, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.243935309973 F1-OK:  0.914728682171
F1-score multiplied:  0.223134624627
Epoch 3: 
Test set LL -0.43604650115035487, Acc 0.840822160243988, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.229930624381 F1-OK:  0.911237195842
F1-score multiplied:  0.209521337399
[ 97998  95754  98742 ..., 203012 213193 125451]
Epoch 4: \Dev set LL -0.4338055352321438, Acc 0.8470154404640198, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.243243243243 F1-OK:  0.914906549157
F1-score multiplied:  0.222544836281
Epoch 4: 
Test set LL -0.44134440552188847, Acc 0.841027021408081, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.229139072848 F1-OK:  0.911375057104
F1-score multiplied:  0.208831635601
[ 82707 140148 157986 ...,  51883 106201 103473]
Epoch 5: \Dev set LL -0.4789577192917703, Acc 0.8470154404640198, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.243243243243 F1-OK:  0.914906549157
F1-score multiplied:  0.222544836281
Epoch 5: 
Test set LL -0.4899015719521964, Acc 0.8410953283309937, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.230743801653 F1-OK:  0.911396260899
F1-score multiplied:  0.210299038052
[ 96593 186600 181132 ...,  40532  34553 129546]
Epoch 6: \Dev set LL -0.5388005699764716, Acc 0.8468788266181946, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.243079000675 F1-OK:  0.914824101512
F1-score multiplied:  0.222374528389
Epoch 6: 
Test set LL -0.5535281844792258, Acc 0.8412318825721741, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.231404958678 F1-OK:  0.911472413662
F1-score multiplied:  0.210919236219
[211235 108099  35584 ..., 173533 203627 154044]
Epoch 7: \Dev set LL -0.5952410917123272, Acc 0.846605658531189, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.243771043771 F1-OK:  0.914646195941
F1-score multiplied:  0.222964257866
Epoch 7: 
Test set LL -0.6136092850067791, Acc 0.8410953283309937, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.231759656652 F1-OK:  0.911382764005
F1-score multiplied:  0.211221756465
[ 44555 140306 112838 ...,  74919 150399 165017]
Epoch 8: \Dev set LL -0.6386055363249578, Acc 0.846605658531189, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.245802552048 F1-OK:  0.914620238729
F1-score multiplied:  0.224815988835
Epoch 8: 
Test set LL -0.6601741615541501, Acc 0.8408904671669006, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.233048057933 F1-OK:  0.911238095238
F1-score multiplied:  0.21236226841
[169730  20673 186182 ...,  67812 116880 108578]
Epoch 9: \Dev set LL -0.6662196956620573, Acc 0.8472886085510254, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.254666666667 F1-OK:  0.914929234515
F1-score multiplied:  0.23300197839
Epoch 9: 
Test set LL -0.6905593744261916, Acc 0.8407539129257202, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.236910994764 F1-OK:  0.91110094541
F1-score multiplied:  0.215849831308
[ 41613 123879  70949 ...,  98250 105271 211104]
Epoch 10: \Dev set LL -0.6848728734221106, Acc 0.8476983904838562, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.262078093977 F1-OK:  0.915086436677
F1-score multiplied:  0.239824109149
Epoch 10: 
Test set LL -0.711677843256844, Acc 0.841027021408081, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.2421875 F1-OK:  0.911199267623
F1-score multiplied:  0.220681072627
[ 79978  62975 104982 ..., 106252 110351  75418]
Epoch 11: \Dev set LL -0.6984998540475604, Acc 0.8472886085510254, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.267365661861 F1-OK:  0.914760597743
F1-score multiplied:  0.24457557266
Epoch 11: 
Test set LL -0.7272555220072414, Acc 0.8413684964179993, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.251369642282 F1-OK:  0.911285086882
F1-score multiplied:  0.229069406306
[ 78595 203579  77628 ..., 171197 169790 184869]
Epoch 12: \Dev set LL -0.7067032195389228, Acc 0.846605658531189, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.273139158576 F1-OK:  0.91425517294
F1-score multiplied:  0.249718888661
Epoch 12: 
Test set LL -0.7372589122224975, Acc 0.8412318825721741, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.260260897232 F1-OK:  0.91107286288
F1-score multiplied:  0.237116640737
[161145   8624 103629 ..., 131598  77971 145710]
Epoch 13: \Dev set LL -0.7103987995488176, Acc 0.8463324904441833, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.287523749208 F1-OK:  0.913878894588
F1-score multiplied:  0.262761886094
Epoch 13: 
Test set LL -0.7426289057633779, Acc 0.8413001894950867, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.271016311167 F1-OK:  0.910957854406
F1-score multiplied:  0.24688443733
[ 88180  37890  33658 ..., 138465 165899 187250]
Epoch 14: \Dev set LL -0.7123293399075487, Acc 0.846059262752533, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.295184490306 F1-OK:  0.913593498428
F1-score multiplied:  0.269678631181
Epoch 14: 
Test set LL -0.7462673970258518, Acc 0.8413001894950867, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.281829419036 F1-OK:  0.910793797021
F1-score multiplied:  0.256688486676
[ 95272 217111 176486 ..., 183877 161754 214087]
Epoch 15: \Dev set LL -0.7090865953225155, Acc 0.8449665307998657, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.306658521686 F1-OK:  0.912725874664
F1-score multiplied:  0.279895167429
Epoch 15: 
Test set LL -0.744838795070987, Acc 0.840002715587616, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.293212669683 F1-OK:  0.909790936742
F1-score multiplied:  0.266762229416
[ 43428  23203 165494 ...,  87260 127806  73850]
Epoch 16: \Dev set LL -0.7109278820869246, Acc 0.8440103530883789, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.321046373365 F1-OK:  0.911882716049
F1-score multiplied:  0.292756638922
Epoch 16: 
Test set LL -0.748309712353746, Acc 0.8390467166900635, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.301215535132 F1-OK:  0.909048813429
F1-score multiplied:  0.273819624798
[204644  31141 103650 ...,  30090  81702 136580]
Epoch 17: \Dev set LL -0.7164574918646346, Acc 0.8438737988471985, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.33118782914 F1-OK:  0.911621433542
F1-score multiplied:  0.301917923572
Epoch 17: 
Test set LL -0.7552441481833808, Acc 0.8383638262748718, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.310113669484 F1-OK:  0.908458057779
F1-score multiplied:  0.28172526187
[151463 159307 100073 ...,  99926  64887  45399]
Epoch 18: \Dev set LL -0.7341661412258257, Acc 0.8433274030685425, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.335072463768 F1-OK:  0.911202291554
F1-score multiplied:  0.305318796822
Epoch 18: 
Test set LL -0.7742014286054725, Acc 0.8381589651107788, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.315028901734 F1-OK:  0.908239120335
F1-score multiplied:  0.286121572591
[ 77558  65635  39900 ...,  52102 175656  30661]
Epoch 19: \Dev set LL -0.7359431814846722, Acc 0.8414151072502136, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.33996588971 F1-OK:  0.909881238842
F1-score multiplied:  0.309328584893
Epoch 19: 
Test set LL -0.7763991403481839, Acc 0.836042046546936, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.320407585621 F1-OK:  0.906775383421
F1-score multiplied:  0.290537711303
[   873  43953 181616 ..., 181234   7816  90895]
Epoch 20: \Dev set LL -0.7432954705673896, Acc 0.8403223752975464, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.34214969049 F1-OK:  0.909133307423
F1-score multiplied:  0.311059679749
Epoch 20: 
Test set LL -0.7843344153763724, Acc 0.835017740726471, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.324006715165 F1-OK:  0.906043400482
F1-score multiplied:  0.293564145987
[139636 198046 199708 ...,  32597 183833  57999]
Epoch 21: \Dev set LL -0.7513232541318614, Acc 0.8397759795188904, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.345058626466 F1-OK:  0.908723056571
F1-score multiplied:  0.313562729738
Epoch 21: 
Test set LL -0.7928821124102179, Acc 0.8340617418289185, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.327614831212 F1-OK:  0.905351717691
F1-score multiplied:  0.296606650179
[ 48250  85392  96772 ..., 159417  47120  80258]
Epoch 22: \Dev set LL -0.7603558929149429, Acc 0.8395028114318848, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.34758467518 F1-OK:  0.908496223036
F1-score multiplied:  0.315779364586
Epoch 22: 
Test set LL -0.8025860322509759, Acc 0.8338568806648254, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.330674002751 F1-OK:  0.905157291545
F1-score multiplied:  0.299311984714
[ 94976 169993 129145 ..., 212870  49486 180803]
Epoch 23: \Dev set LL -0.7640535247285853, Acc 0.8386832475662231, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.350742166025 F1-OK:  0.907899867426
F1-score multiplied:  0.318438766035
Epoch 23: 
Test set LL -0.8066389940849292, Acc 0.8331056833267212, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.333696837514 F1-OK:  0.904605776737
F1-score multiplied:  0.301864086894
[188706 216106  14316 ...,  23716  33182   2019]
Epoch 24: \Dev set LL -0.7708803981891407, Acc 0.8384100794792175, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.353198469109 F1-OK:  0.907671895731
F1-score multiplied:  0.320588324025
Epoch 24: 
Test set LL -0.8140661538283306, Acc 0.8330374360084534, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.337218758471 F1-OK:  0.90448845658
F1-score multiplied:  0.305010474379
[ 49555 204099 199462 ..., 142630  30997  94732]
Epoch 25: \Dev set LL -0.7763641000642433, Acc 0.8382734656333923, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.35652173913 F1-OK:  0.907514450867
F1-score multiplied:  0.323548630309
Epoch 25: 
Test set LL -0.8199171213124344, Acc 0.8329691290855408, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.340700808625 F1-OK:  0.90437094378
F1-score multiplied:  0.308119911843
[ 10586  40093 110450 ..., 135123  16134  39003]
Epoch 26: \Dev set LL -0.780668893244684, Acc 0.8371807336807251, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.357065803668 F1-OK:  0.906787613388
F1-score multiplied:  0.32378284793
Epoch 26: 
Test set LL -0.8246678905013503, Acc 0.8326277136802673, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.343775100402 F1-OK:  0.904081712519
F1-score multiplied:  0.310800781493
[124016  93299 142951 ..., 121315 130112 138434]
Epoch 27: \Dev set LL -0.7858751685376519, Acc 0.8367709517478943, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.358561460011 F1-OK:  0.906487205572
F1-score multiplied:  0.325031375911
Epoch 27: 
Test set LL -0.8303170737258669, Acc 0.8317399621009827, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.346765641569 F1-OK:  0.903433139991
F1-score multiplied:  0.313279572404
[  9469 193383  39949 ..., 202484  63420  57130]
Epoch 28: \Dev set LL -0.7909526637257673, Acc 0.8364977240562439, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.3616 F1-OK:  0.90624265685
F1-score multiplied:  0.327697344717
Epoch 28: 
Test set LL -0.8358944480349662, Acc 0.8315351009368896, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.348904724202 F1-OK:  0.903251107887
F1-score multiplied:  0.315148578682
[ 70289  17736 199270 ...,  16586  34471 159323]
Epoch 29: \Dev set LL -0.7936178001218861, Acc 0.8366343379020691, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.367195767196 F1-OK:  0.906210790464
F1-score multiplied:  0.332756766446
Epoch 29: 
Test set LL -0.8387861323353537, Acc 0.830237627029419, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.350574712644 F1-OK:  0.902356637863
F1-score multiplied:  0.316343419021
[ 44559 117975   7356 ..., 164492 159093  15768]
Epoch 30: \Dev set LL -0.8027494924922737, Acc 0.8370441198348999, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.36643653744 F1-OK:  0.906497374402
F1-score multiplied:  0.332173759075
Epoch 30: 
Test set LL -0.8489151984309594, Acc 0.830852210521698, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.349014454665 F1-OK:  0.902797943727
F1-score multiplied:  0.315089532003
[109754 141570 108980 ..., 139365 215134  72258]
Epoch 31: \Dev set LL -0.8076322335163697, Acc 0.8371807336807251, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.369978858351 F1-OK:  0.906509803922
F1-score multiplied:  0.335389462339
Epoch 31: 
Test set LL -0.8542668391327308, Acc 0.8305791020393372, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.35137254902 F1-OK:  0.902564505361
F1-score multiplied:  0.317136390903
[188636 149668 181585 ...,  86043   3885 149926]
Epoch 32: \Dev set LL -0.8094511616259624, Acc 0.8367709517478943, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.374672946102 F1-OK:  0.906134632001
F1-score multiplied:  0.339504132136
Epoch 32: 
Test set LL -0.8559336342535928, Acc 0.8292133212089539, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.354247353473 F1-OK:  0.901593547118
F1-score multiplied:  0.319387127975
[118069 172817 212991 ...,   6636 158235  11185]
Epoch 33: \Dev set LL -0.8136431232146448, Acc 0.8364977240562439, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.37558685446 F1-OK:  0.905933202358
F1-score multiplied:  0.340256601824
Epoch 33: 
Test set LL -0.8606088900644528, Acc 0.8292133212089539, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.356573192694 F1-OK:  0.901539309476
F1-score multiplied:  0.321464749919
[ 35430  78766   7572 ...,  21798 172876 186694]
Epoch 34: \Dev set LL -0.8197827045034227, Acc 0.8364977240562439, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.374281233664 F1-OK:  0.905962762197
F1-score multiplied:  0.339084860289
Epoch 34: 
Test set LL -0.8674437707724213, Acc 0.8295547962188721, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.356701030928 F1-OK:  0.901763224181
F1-score multiplied:  0.321659871718
[ 12263 179388 192145 ...,  56465 182676  75856]
Epoch 35: \Dev set LL -0.8220786132265011, Acc 0.8366343379020691, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.378378378378 F1-OK:  0.905960056613
F1-score multiplied:  0.342795697097
Epoch 35: 
Test set LL -0.8697209038628624, Acc 0.8293498754501343, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.360706062932 F1-OK:  0.901532763308
F1-score multiplied:  0.325188333657
[159646 185256 171645 ..., 194155 131629  68601]
Epoch 36: \Dev set LL -0.8243961938456011, Acc 0.8360879421234131, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.378881987578 F1-OK:  0.905586152636
F1-score multiplied:  0.343110281433
Epoch 36: 
Test set LL -0.872227894562888, Acc 0.8292816281318665, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.363543788187 F1-OK:  0.90141955836
F1-score multiplied:  0.327705480992
[191753 201171 122412 ...,  38797  37466 188023]
Epoch 37: \Dev set LL -0.8283276476060933, Acc 0.8358147740364075, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.380412371134 F1-OK:  0.905369233192
F1-score multiplied:  0.34441365675
Epoch 37: 
Test set LL -0.8764456926076751, Acc 0.8293498754501343, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.365896980462 F1-OK:  0.901408450704
F1-score multiplied:  0.329822630275
[153734 132148 148743 ...,   8738  83102  23751]
Epoch 38: \Dev set LL -0.8344440306581189, Acc 0.8364977240562439, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.380755302638 F1-OK:  0.90581477693
F1-score multiplied:  0.344893779524
Epoch 38: 
Test set LL -0.8835456216642062, Acc 0.8294181823730469, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.361778231988 F1-OK:  0.901552770553
F1-score multiplied:  0.326162167374
[ 57591  60924 105622 ..., 133844  68700  81761]
Epoch 39: \Dev set LL -0.8358038477304198, Acc 0.8362245559692383, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.383547557841 F1-OK:  0.905568244467
F1-score multiplied:  0.347328488623
Epoch 39: 
Test set LL -0.8850244150160201, Acc 0.8292133212089539, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.364421855146 F1-OK:  0.901352897093
F1-score multiplied:  0.3284726949
[175786  45399 125058 ...,  82247 183438 112942]
Epoch 40: \Dev set LL -0.8381505147324719, Acc 0.8356781601905823, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.385283597343 F1-OK:  0.90516357903
F1-score multiplied:  0.348744679913
Epoch 40: 
Test set LL -0.8874999684123072, Acc 0.8289401531219482, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.367903103709 F1-OK:  0.901085883514
F1-score multiplied:  0.331512293254
[ 96819 206932   1785 ...,  78475 169891 180553]
Epoch 41: \Dev set LL -0.8418492025219386, Acc 0.8363611698150635, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.386270491803 F1-OK:  0.905594956659
F1-score multiplied:  0.349804609283
Epoch 41: 
Test set LL -0.8916603105922344, Acc 0.8288719058036804, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.366851945427 F1-OK:  0.901065929728
F1-score multiplied:  0.330557789279
[ 89635 183840 203254 ..., 106660 205319 196231]
Epoch 42: \Dev set LL -0.8443036595684532, Acc 0.8362245559692383, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.386700767263 F1-OK:  0.905493812564
F1-score multiplied:  0.350155152071
Epoch 42: 
Test set LL -0.8944594053848633, Acc 0.8285987377166748, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.366161616162 F1-OK:  0.900900189514
F1-score multiplied:  0.329875069393
[110684 152431 187978 ..., 190728 153645 167144]
Epoch 43: \Dev set LL -0.8484514680824821, Acc 0.8366343379020691, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.386036960986 F1-OK:  0.905782259335
F1-score multiplied:  0.349665430708
Epoch 43: 
Test set LL -0.8991176134914397, Acc 0.8285304307937622, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.365107458913 F1-OK:  0.90088027474
F1-score multiplied:  0.328918107895
[ 86065 158811  87623 ...,  10358 112387 134839]
Epoch 44: \Dev set LL -0.849748597315128, Acc 0.8362245559692383, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.387953037264 F1-OK:  0.905464006938
F1-score multiplied:  0.351277511625
Epoch 44: 
Test set LL -0.90051469627553, Acc 0.8278475999832153, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.365466901586 F1-OK:  0.900414773849
F1-score multiplied:  0.329071797541
[157439  92985  91913 ..., 110497 154030 216302]
Epoch 45: \Dev set LL -0.8505673563105066, Acc 0.8363611698150635, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.391878172589 F1-OK:  0.905460858586
F1-score multiplied:  0.354830346613
Epoch 45: 
Test set LL -0.9011450012575517, Acc 0.8268232941627502, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.365047571357 F1-OK:  0.899739068554
F1-score multiplied:  0.328447561831
[185362 193751 196467 ...,  55673 208466 140780]
Epoch 46: \Dev set LL -0.8528789889933427, Acc 0.8364977240562439, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.393309680689 F1-OK:  0.905517404689
F1-score multiplied:  0.356148761297
Epoch 46: 
Test set LL -0.9038024184927431, Acc 0.8270964026451111, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.367316341829 F1-OK:  0.899865538243
F1-score multiplied:  0.330535317645
[116178 194505  95043 ...,   1569  99101 146376]
Epoch 47: \Dev set LL -0.8557589048282545, Acc 0.8363611698150635, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.393110435664 F1-OK:  0.905431007262
F1-score multiplied:  0.355934377728
Epoch 47: 
Test set LL -0.9072134459542842, Acc 0.8270281553268433, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.366591647912 F1-OK:  0.899837874175
F1-score multiplied:  0.329873049147
[175592 162391 192422 ..., 104363 138714 190912]
Epoch 48: \Dev set LL -0.8572625030567564, Acc 0.8355416059494019, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.393756294058 F1-OK:  0.904867256637
F1-score multiplied:  0.356297177588
Epoch 48: 
Test set LL -0.90889642518922, Acc 0.8268915414810181, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.368617683686 F1-OK:  0.899695327029
F1-score multiplied:  0.331643607473
[216202  15665  40210 ...,  93117  76944 106362]
Epoch 49: \Dev set LL -0.8574439718192862, Acc 0.8352683782577515, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.397602397602 F1-OK:  0.904588607595
F1-score multiplied:  0.359666599224
Epoch 49: 
Test set LL -0.9088562727056888, Acc 0.826686680316925, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.373023715415 F1-OK:  0.899445324881
F1-score multiplied:  0.3355144369
[ 69791  98282 134589 ..., 145613  96047 127529]
Done!


our model - hybrid network - phrase level - 72 features -matern 12 - 512-256-4-gps

Epoch 1: \Dev set LL -0.40927830122164016, Acc 0.8448299169540405, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.30135301353 F1-OK:  0.912722802704
F1-score multiplied:  0.275051767113
Epoch 1: 
Test set LL -0.4202262831452409, Acc 0.8407539129257202, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.292046144505 F1-OK:  0.910286989305
F1-score multiplied:  0.26584580562
[ 10604 112260   2170 ...,  24253  38081  64020]
Epoch 2: \Dev set LL -0.584340848231078, Acc 0.8388198614120483, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.373673036093 F1-OK:  0.907509013952
F1-score multiplied:  0.339111648526
Epoch 2: 
Test set LL -0.6132472682671687, Acc 0.835017740726471, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.359830418654 F1-OK:  0.905306890335
F1-score multiplied:  0.325756957359
[ 16587 157088  51334 ...,  99848  92538  64243]
Epoch 3: \Dev set LL -0.798654893707261, Acc 0.8121841549873352, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.426844518549 F1-OK:  0.887690925427
F1-score multiplied:  0.378906005684
Epoch 3: 
Test set LL -0.8211595166000859, Acc 0.8100245594978333, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.416526845638 F1-OK:  0.886541598695
F1-score multiplied:  0.369268375631
[ 97998  95754  98742 ..., 203012 213193 125451]
Epoch 4: \Dev set LL -0.8559783497490903, Acc 0.8187406063079834, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.434115138593 F1-OK:  0.892087501017
F1-score multiplied:  0.387268689141
Epoch 4: 
Test set LL -0.8891277535258307, Acc 0.8114586472511292, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.414422057264 F1-OK:  0.887640906686
F1-score multiplied:  0.367857970661
[ 82707 140148 157986 ...,  51883 106201 103473]
Epoch 5: \Dev set LL -1.0034646418070612, Acc 0.7972954511642456, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.444610778443 F1-OK:  0.876023391813
F1-score multiplied:  0.389489442168
Epoch 5: 
Test set LL -1.016653417903947, Acc 0.7950013875961304, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.441592261905 F1-OK:  0.874456339913
F1-score multiplied:  0.386153153079
[ 96593 186600 181132 ...,  40532  34553 129546]
Epoch 6: \Dev set LL -0.9273157290019522, Acc 0.8280289769172668, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.448049101271 F1-OK:  0.898147399078
F1-score multiplied:  0.402414134966
Epoch 6: 
Test set LL -0.9645629715895957, Acc 0.8222480416297913, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.429541968004 F1-OK:  0.894721941355
F1-score multiplied:  0.384320623506
[211235 108099  35584 ..., 173533 203627 154044]
Epoch 7: \Dev set LL -1.002960124245826, Acc 0.8151891827583313, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.446172738436 F1-OK:  0.889089269612
F1-score multiplied:  0.396687394137
Epoch 7: 
Test set LL -1.0377592788768562, Acc 0.8120732307434082, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.434443074394 F1-OK:  0.887314716239
F1-score multiplied:  0.385487733278
[ 44555 140306 112838 ...,  74919 150399 165017]
Epoch 8: \Dev set LL -0.9153785365953993, Acc 0.8382734656333923, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.447245564893 F1-OK:  0.90528
F1-score multiplied:  0.404882464986
Epoch 8: 
Test set LL -0.9628158556581569, Acc 0.8319448232650757, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.420804895269 F1-OK:  0.90171332721
F1-score multiplied:  0.37944538222
[169730  20673 186182 ...,  67812 116880 108578]
Epoch 9: \Dev set LL -0.940714441370675, Acc 0.8339024782180786, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.451263537906 F1-OK:  0.902140672783
F1-score multiplied:  0.407103191689
Epoch 9: 
Test set LL -0.9952972617012005, Acc 0.8269598484039307, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.42850699143 F1-OK:  0.898044580349
F1-score multiplied:  0.384818381295
[ 41613 123879  70949 ...,  98250 105271 211104]
Epoch 10: \Dev set LL -0.9876283105019558, Acc 0.8287119269371033, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.441674087266 F1-OK:  0.898838334947
F1-score multiplied:  0.396993601188
Epoch 10: 
Test set LL -1.0280328548373676, Acc 0.8250477910041809, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.432180851064 F1-OK:  0.896593477559
F1-score multiplied:  0.38749053219
[ 79978  62975 104982 ..., 106252 110351  75418]
Epoch 11: \Dev set LL -0.9735293493011841, Acc 0.8349952101707458, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.453887884268 F1-OK:  0.902815768302
F1-score multiplied:  0.409777138958
Epoch 11: 
Test set LL -1.0304917093226962, Acc 0.8266184329986572, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.424393561551 F1-OK:  0.897937854243
F1-score multiplied:  0.381079044013
[ 78595 203579  77628 ..., 171197 169790 184869]
Epoch 12: \Dev set LL -1.0057098679241976, Acc 0.8273459672927856, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.47245409015 F1-OK:  0.896782622897
F1-score multiplied:  0.423688618163
Epoch 12: 
Test set LL -1.0884350875679605, Acc 0.8153510093688965, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.439933719967 F1-OK:  0.889452166803
F1-score multiplied:  0.391300000474
[161145   8624 103629 ..., 131598  77971 145710]
Epoch 13: \Dev set LL -0.9719091308028653, Acc 0.8370441198348999, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.445889456572 F1-OK:  0.904475938826
F1-score multiplied:  0.403296284846
Epoch 13: 
Test set LL -1.034593488583802, Acc 0.8305791020393372, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.428209264808 F1-OK:  0.900557136559
F1-score multiplied:  0.385626909363
[ 88180  37890  33658 ..., 138465 165899 187250]
Epoch 14: \Dev set LL -1.0269124385344945, Acc 0.8307608366012573, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.456340500219 F1-OK:  0.899781606406
F1-score multiplied:  0.410606788356
Epoch 14: 
Test set LL -1.0804123779471007, Acc 0.8242966532707214, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.44053055012 F1-OK:  0.895783547329
F1-score multiplied:  0.394620018893
[ 95272 217111 176486 ..., 183877 161754 214087]
Epoch 15: \Dev set LL -1.0027658310473477, Acc 0.8355416059494019, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.466312056738 F1-OK:  0.902793476506
F1-score multiplied:  0.420983482839
Epoch 15: 
Test set LL -1.0634012016627221, Acc 0.826481819152832, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.438948995363 F1-OK:  0.897370653096
F1-score multiplied:  0.393899946645
[ 43428  23203 165494 ...,  87260 127806  73850]
Epoch 16: \Dev set LL -1.0035635999807324, Acc 0.8332195281982422, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.462351387054 F1-OK:  0.901301430765
F1-score multiplied:  0.416717966668
Epoch 16: 
Test set LL -1.0618637123500476, Acc 0.8252526521682739, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.434974608081 F1-OK:  0.896643644735
F1-score multiplied:  0.390017217957
[204644  31141 103650 ...,  30090  81702 136580]
Epoch 17: \Dev set LL -1.0433055869921013, Acc 0.8269361853599548, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.464723278411 F1-OK:  0.896782077393
F1-score multiplied:  0.416755507027
Epoch 17: 
Test set LL -1.1170250188817923, Acc 0.8163753151893616, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.436150136297 F1-OK:  0.890329948203
F1-score multiplied:  0.388317528258
[151463 159307 100073 ...,  99926  64887  45399]
Epoch 18: \Dev set LL -1.0750079231587994, Acc 0.8184674382209778, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.462161068393 F1-OK:  0.890806014296
F1-score multiplied:  0.411695859298
Epoch 18: 
Test set LL -1.1243370813305518, Acc 0.8133706450462341, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.444625076204 F1-OK:  0.887840111626
F1-score multiplied:  0.394755977289
[ 77558  65635  39900 ...,  52102 175656  30661]
Epoch 19: \Dev set LL -0.989617677164392, Acc 0.8399125933647156, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.466787989081 F1-OK:  0.905818064931
F1-score multiplied:  0.422824993002
Epoch 19: 
Test set LL -1.0645392806160912, Acc 0.8273695707321167, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.426236949614 F1-OK:  0.898400450125
F1-score multiplied:  0.382931467393
[   873  43953 181616 ..., 181234   7816  90895]
Epoch 20: \Dev set LL -1.079360061912897, Acc 0.8216090798377991, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.467807660962 F1-OK:  0.892845421726
F1-score multiplied:  0.417679928338
Epoch 20: 
Test set LL -1.120886955322416, Acc 0.8156241178512573, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.449204406365 F1-OK:  0.889280734848
F1-score multiplied:  0.399468824589
[139636 198046 199708 ...,  32597 183833  57999]
Epoch 21: \Dev set LL -1.0783129366583173, Acc 0.8267996311187744, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.467674223342 F1-OK:  0.896574225122
F1-score multiplied:  0.419304654402
Epoch 21: 
Test set LL -1.1185676774004467, Acc 0.8197214007377625, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.45 F1-OK:  0.892192094087
F1-score multiplied:  0.401486442339
[ 48250  85392  96772 ..., 159417  47120  80258]
Epoch 22: \Dev set LL -1.0406309945004155, Acc 0.8319901823997498, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.461942257218 F1-OK:  0.900453221107
F1-score multiplied:  0.415957393477
Epoch 22: 
Test set LL -1.0959940211786137, Acc 0.8253892660140991, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.443525571273 F1-OK:  0.896448386182
F1-score multiplied:  0.397597782598
[ 94976 169993 129145 ..., 212870  49486 180803]
Epoch 23: \Dev set LL -1.0004723887808058, Acc 0.8416882753372192, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.471017800091 F1-OK:  0.90691510722
F1-score multiplied:  0.427173158672
Epoch 23: 
Test set LL -1.0682541227266718, Acc 0.8292133212089539, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.433008388121 F1-OK:  0.899465369619
F1-score multiplied:  0.389476049869
[188706 216106  14316 ...,  23716  33182   2019]
Epoch 24: \Dev set LL -0.9960917898419364, Acc 0.8423712849617004, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.465245597776 F1-OK:  0.907561678949
F1-score multiplied:  0.422239075841
Epoch 24: 
Test set LL -1.059439778285598, Acc 0.8324910998344421, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.43802978236 F1-OK:  0.901576856719
F1-score multiplied:  0.394917514329
[ 49555 204099 199462 ..., 142630  30997  94732]
Epoch 25: \Dev set LL -1.0370163784664275, Acc 0.835131824016571, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.462839341344 F1-OK:  0.90262202501
F1-score multiplied:  0.417768983538
Epoch 25: 
Test set LL -1.0885344529407963, Acc 0.8260038495063782, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.434280639432 F1-OK:  0.897191736604
F1-score multiplied:  0.389633001065
[ 10586  40093 110450 ..., 135123  16134  39003]
Epoch 26: \Dev set LL -1.0435825684800608, Acc 0.8326731324195862, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.469007368877 F1-OK:  0.900689096068
F1-score multiplied:  0.422429823123
Epoch 26: 
Test set LL -1.0965667241988288, Acc 0.8230674862861633, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.440751133175 F1-OK:  0.894909754614
F1-score multiplied:  0.394432488435
[124016  93299 142951 ..., 121315 130112 138434]
Epoch 27: \Dev set LL -1.081006171250204, Acc 0.8257068991661072, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.467000835422 F1-OK:  0.895819725669
F1-score multiplied:  0.418348560275
Epoch 27: 
Test set LL -1.142272691858039, Acc 0.816511869430542, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.447233079613 F1-OK:  0.889998771851
F1-score multiplied:  0.398036891587
[  9469 193383  39949 ..., 202484  63420  57130]
Epoch 28: \Dev set LL -1.0284269301183864, Acc 0.8358147740364075, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.478298611111 F1-OK:  0.902577403145
F1-score multiplied:  0.431701518344
Epoch 28: 
Test set LL -1.1108282232816844, Acc 0.821291983127594, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.441647109025 F1-OK:  0.893622210479
F1-score multiplied:  0.394665665819
[ 70289  17736 199270 ...,  16586  34471 159323]
Epoch 29: \Dev set LL -0.9939548524199278, Acc 0.842644453048706, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.458137347131 F1-OK:  0.907957813998
F1-score multiplied:  0.415969384212
Epoch 29: 
Test set LL -1.0583296957617168, Acc 0.8335154056549072, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.433550185874 F1-OK:  0.90241754723
F1-score multiplied:  0.391243295337
[ 44559 117975   7356 ..., 164492 159093  15768]
Done!



our model - hybrid network - phrase level - 72 features - matern12 - 512-256-8-gps

Epoch 1: \Dev set LL -0.4528816345199271, Acc 0.8388198614120483, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.398572884811 F1-OK:  0.906940063091
F1-score multiplied:  0.361481717297
Epoch 1: 
Test set LL -0.4728323655811652, Acc 0.8332422971725464, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.378309572301 F1-OK:  0.903706624606
F1-score multiplied:  0.341880866641
[ 10604 112260   2170 ...,  24253  38081  64020]
Epoch 2: \Dev set LL -0.6466635600642416, Acc 0.8377270698547363, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.428846153846 F1-OK:  0.905429071804
F1-score multiplied:  0.388289775024
Epoch 2: 
Test set LL -0.6815355885487778, Acc 0.8311253786087036, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.399319893126 F1-OK:  0.901752016209
F1-score multiplied:  0.360087518739
[ 16587 157088  51334 ...,  99848  92538  64243]
Epoch 3: \Dev set LL -0.7423799716083146, Acc 0.8369075059890747, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.445682451253 F1-OK:  0.904388212684
F1-score multiplied:  0.403069955514
Epoch 3: 
Test set LL -0.7929006219082544, Acc 0.8287352919578552, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.418367346939 F1-OK:  0.899583600256
F1-score multiplied:  0.376356404189
[ 97998  95754  98742 ..., 203012 213193 125451]
Epoch 4: \Dev set LL -0.8323221807636647, Acc 0.8222920298576355, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.459043659044 F1-OK:  0.893683092261
F1-score multiplied:  0.410239556697
Epoch 4: 
Test set LL -0.8715956854727127, Acc 0.8169898986816406, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.446738232865 F1-OK:  0.890361642939
F1-score multiplied:  0.397758586978
[ 82707 140148 157986 ...,  51883 106201 103473]
Epoch 5: \Dev set LL -0.8651277934554672, Acc 0.8382734656333923, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.466666666667 F1-OK:  0.904685235872
F1-score multiplied:  0.422186443407
Epoch 5: 
Test set LL -0.9259795188779278, Acc 0.8285987377166748, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.433664259928 F1-OK:  0.899018345671
F1-score multiplied:  0.389872125537
[ 96593 186600 181132 ...,  40532  34553 129546]
Epoch 6: \Dev set LL -0.8810704035521646, Acc 0.8395028114318848, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.451703219785 F1-OK:  0.905992479398
F1-score multiplied:  0.409239720046
Epoch 6: 
Test set LL -0.9395193969677212, Acc 0.8307156562805176, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.427614869545 F1-OK:  0.90066915094
F1-score multiplied:  0.385139521482
[211235 108099  35584 ..., 173533 203627 154044]
Epoch 7: \Dev set LL -0.9116260992797365, Acc 0.8303510546684265, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.472835314092 F1-OK:  0.89890932769
F1-score multiplied:  0.425036074298
Epoch 7: 
Test set LL -0.9789988224271128, Acc 0.821291983127594, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.447540637534 F1-OK:  0.893405563928
F1-score multiplied:  0.399835295657
[ 44555 140306 112838 ...,  74919 150399 165017]
Epoch 8: \Dev set LL -0.9265929735729738, Acc 0.8436006307601929, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.450312049928 F1-OK:  0.908830320885
F1-score multiplied:  0.409257244835
Epoch 8: 
Test set LL -0.9766005125037789, Acc 0.8340617418289185, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.430379746835 F1-OK:  0.902885460795
F1-score multiplied:  0.388583616038
[169730  20673 186182 ...,  67812 116880 108578]
Epoch 9: \Dev set LL -0.9546570954651987, Acc 0.8280289769172668, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.473882156289 F1-OK:  0.897216099273
F1-score multiplied:  0.425174699781
Epoch 9: 
Test set LL -1.0177450740295455, Acc 0.8199945092201233, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.450604418508 F1-OK:  0.892364230298
F1-score multiplied:  0.402103265091
[ 41613 123879  70949 ...,  98250 105271 211104]
Epoch 10: \Dev set LL -0.9857606037384931, Acc 0.8322633504867554, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.459507042254 F1-OK:  0.900727566694
F1-score multiplied:  0.413890660048
Epoch 10: 
Test set LL -1.0312223333441237, Acc 0.8248429298400879, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.442996742671 F1-OK:  0.896082323867
F1-score multiplied:  0.396961550638
[ 79978  62975 104982 ..., 106252 110351  75418]
Epoch 11: \Dev set LL -0.9912614646714131, Acc 0.8326731324195862, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.464363795365 F1-OK:  0.900849858357
F1-score multiplied:  0.418322059281
Epoch 11: 
Test set LL -1.0525936018834803, Acc 0.8255941271781921, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.438681318681 F1-OK:  0.896758024092
F1-score multiplied:  0.393390992547
[ 78595 203579  77628 ..., 171197 169790 184869]
Epoch 12: \Dev set LL -0.9697606461415318, Acc 0.8378636837005615, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.47778266608 F1-OK:  0.904034279247
F1-score multiplied:  0.431931908166
Epoch 12: 
Test set LL -1.0490209880459704, Acc 0.8273012638092041, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.450336883286 F1-OK:  0.897557418884
F1-score multiplied:  0.404203210591
[161145   8624 103629 ..., 131598  77971 145710]
Epoch 13: \Dev set LL -0.990060628365445, Acc 0.8364977240562439, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.461538461538 F1-OK:  0.903615427973
F1-score multiplied:  0.417053274449
Epoch 13: 
Test set LL -1.045152590988838, Acc 0.8297596573829651, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.441657334826 F1-OK:  0.899568948153
F1-score multiplied:  0.397301224134
[ 88180  37890  33658 ..., 138465 165899 187250]
Epoch 14: \Dev set LL -0.9836236206812295, Acc 0.8397759795188904, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.467060427079 F1-OK:  0.90571497468
F1-score multiplied:  0.423023622886
Epoch 14: 
Test set LL -1.0444922810857247, Acc 0.8301693797111511, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.442501681237 F1-OK:  0.899826801466
F1-score multiplied:  0.398174872471
[ 95272 217111 176486 ..., 183877 161754 214087]
Epoch 15: \Dev set LL -0.9914153536661141, Acc 0.8404589295387268, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.466179159049 F1-OK:  0.906214870724
F1-score multiplied:  0.422458486352
Epoch 15: 
Test set LL -1.0503923011474288, Acc 0.8317399621009827, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.442281575373 F1-OK:  0.900924809007
F1-score multiplied:  0.398462443821
[ 43428  23203 165494 ...,  87260 127806  73850]
Epoch 16: \Dev set LL -0.9804729989050904, Acc 0.8445567488670349, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.464722483537 F1-OK:  0.909076382231
F1-score multiplied:  0.422468234075
Epoch 16: 
Test set LL -1.0270862792414488, Acc 0.8351543545722961, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.446076181735 F1-OK:  0.903168872844
F1-score multiplied:  0.40288212226
[204644  31141 103650 ...,  30090  81702 136580]
Epoch 17: \Dev set LL -1.0193514073476, Acc 0.835131824016571, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.464269862406 F1-OK:  0.902574864799
F1-score multiplied:  0.419038308291
Epoch 17: 
Test set LL -1.0707186529162467, Acc 0.8279158473014832, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.450741063644 F1-OK:  0.897975708502
F1-score multiplied:  0.404754525977
[151463 159307 100073 ...,  99926  64887  45399]
Epoch 18: \Dev set LL -0.9933126848318778, Acc 0.841141939163208, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.458818054909 F1-OK:  0.906907868406
F1-score multiplied:  0.416105704164
Epoch 18: 
Test set LL -1.0479105991667623, Acc 0.8318765163421631, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.44020009095 F1-OK:  0.901084773001
F1-score multiplied:  0.396657599029
[ 77558  65635  39900 ...,  52102 175656  30661]
Epoch 19: \Dev set LL -1.0053174397691942, Acc 0.8374539017677307, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.463963963964 F1-OK:  0.904202221864
F1-score multiplied:  0.419517247081
Epoch 19: 
Test set LL -1.0600870446082924, Acc 0.8289401531219482, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.445182724252 F1-OK:  0.898881847172
F1-score multiplied:  0.400166669505
[   873  43953 181616 ..., 181234   7816  90895]
Epoch 20: \Dev set LL -1.0160000241742997, Acc 0.8356781601905823, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.456394035246 F1-OK:  0.90321023413
F1-score multiplied:  0.41221976343
Epoch 20: 
Test set LL -1.067724178545557, Acc 0.8293498754501343, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.450648494175 F1-OK:  0.898985407656
F1-score multiplied:  0.405126420245
[139636 198046 199708 ...,  32597 183833  57999]
Epoch 21: \Dev set LL -1.0098120025802912, Acc 0.8384100794792175, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.465431540895 F1-OK:  0.904819374045
F1-score multiplied:  0.421131475493
Epoch 21: 
Test set LL -1.0807246730046682, Acc 0.8283255696296692, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.439839572193 F1-OK:  0.898629032258
F1-score multiplied:  0.395252609108
[ 48250  85392  96772 ..., 159417  47120  80258]
Epoch 22: \Dev set LL -1.0126230159020448, Acc 0.8386832475662231, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.459990855053 F1-OK:  0.905178643115
F1-score multiplied:  0.416373898022
Epoch 22: 
Test set LL -1.0686147242339388, Acc 0.8283938765525818, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.437178051512 F1-OK:  0.898763243766
F1-score multiplied:  0.39291956368
[ 94976 169993 129145 ..., 212870  49486 180803]
Epoch 23: \Dev set LL -1.0419643784898727, Acc 0.8322633504867554, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.475661827498 F1-OK:  0.900162601626
F1-score multiplied:  0.428172988135
Epoch 23: 
Test set LL -1.1096282074631258, Acc 0.8220431804656982, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.451137320977 F1-OK:  0.89380603097
F1-score multiplied:  0.403229258285
[188706 216106  14316 ...,  23716  33182   2019]
Epoch 24: \Dev set LL -1.019535520308302, Acc 0.8371807336807251, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.471631205674 F1-OK:  0.903762312288
F1-score multiplied:  0.426242508987
Epoch 24: 
Test set LL -1.080587590486898, Acc 0.8273695707321167, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.446584938704 F1-OK:  0.897734627832
F1-score multiplied:  0.400914763743
[ 49555 204099 199462 ..., 142630  30997  94732]
Epoch 25: \Dev set LL -1.0613220282557827, Acc 0.831580400466919, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.465539661899 F1-OK:  0.900040535063
F1-score multiplied:  0.419004566388
Epoch 25: 
Test set LL -1.12236880894949, Acc 0.8209505677223206, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.445431472081 F1-OK:  0.893241042345
F1-score multiplied:  0.397877672415
[ 10586  40093 110450 ..., 135123  16134  39003]
Epoch 26: \Dev set LL -1.0301784231000979, Acc 0.8393661975860596, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.455050973123 F1-OK:  0.905799423262
F1-score multiplied:  0.41218490901
Epoch 26: 
Test set LL -1.0667088490230567, Acc 0.8326277136802673, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.443839346494 F1-OK:  0.901491097625
F1-score multiplied:  0.40011721964
[124016  93299 142951 ..., 121315 130112 138434]
Epoch 27: \Dev set LL -1.0104989757964487, Acc 0.841141939163208, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.448030374941 F1-OK:  0.907219784603
F1-score multiplied:  0.406462020249
Epoch 27: 
Test set LL -1.0634484004455567, Acc 0.8327642679214478, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.431390759229 F1-OK:  0.901965493775
F1-score multiplied:  0.389099579158
[  9469 193383  39949 ..., 202484  63420  57130]
Epoch 28: \Dev set LL -1.0264791700428042, Acc 0.8374539017677307, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.464446444644 F1-OK:  0.904186795491
F1-score multiplied:  0.41994634246
Epoch 28: 
Test set LL -1.0810276140601367, Acc 0.8280524611473083, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.44341290893 F1-OK:  0.898320142142
F1-score multiplied:  0.398326747378
[ 70289  17736 199270 ...,  16586  34471 159323]
Epoch 29: \Dev set LL -1.0423875138889598, Acc 0.8347220420837402, Outputs [1 1 0 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.467429577465 F1-OK:  0.902182700081
F1-score multiplied:  0.421706878295
Epoch 29: 
Test set LL -1.0896801235756488, Acc 0.8283255696296692, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.458189655172 F1-OK:  0.898003894839
F1-score multiplied:  0.41145609492
[ 44559 117975   7356 ..., 164492 159093  15768]
Done!


gaussian processes only - 144 feautres - phrases - RBF


Using TensorFlow backend.
2018-03-14 11:38:32.619355: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
[<tf.Variable 'sconvnet_kernel/W1:0' shape=(72, 512) dtype=float32_ref>, <tf.Variable 'sconvnet_kernel/b1:0' shape=(512,) dtype=float32_ref>, <tf.Variable '6/SVGP/kern/variance/unconstrained:0' shape=() dtype=float64_ref>, <tf.Variable '6/SVGP/kern/lengthscales/unconstrained:0' shape=() dtype=float64_ref>, <tf.Variable '6/SVGP/feature/Z/unconstrained:0' shape=(100, 144) dtype=float64_ref>, <tf.Variable '6/SVGP/q_mu/unconstrained:0' shape=(100, 2) dtype=float64_ref>, <tf.Variable '6/SVGP/q_sqrt/unconstrained:0' shape=(2, 5050) dtype=float64_ref>]
[157230  58090 106355 ...,  43269 152588 159419]
Epoch 1: \Dev set LL -0.5829801037740061, Acc 0.7257205247879028, Outputs [1 1 0 ..., 1 0 1]
Result from the previous epoch on dev:
F1-BAD:  0.349740932642 F1-OK:  0.82620737407
F1-score multiplied:  0.288958537563
Epoch 1: 
Test set LL -0.5843180026299908, Acc 0.7264408469200134, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.358218519705 F1-OK:  0.826173739478
F1-score multiplied:  0.295950733975
[ 10604 112260   2170 ...,  24253  38081  64020]
Epoch 2: \Dev set LL -0.48327199359415596, Acc 0.7757136821746826, Outputs [1 1 1 ..., 1 0 1]
Result from the previous epoch on dev:
F1-BAD:  0.351500789889 F1-OK:  0.86440957886
F1-score multiplied:  0.303840649757
Epoch 2: 
Test set LL -0.4877685181501961, Acc 0.7792269587516785, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.366947327198 F1-OK:  0.866299987594
F1-score multiplied:  0.317886464999
[ 16587 157088  51334 ...,  99848  92538  64243]
Epoch 3: \Dev set LL -0.4808265996989135, Acc 0.8079497218132019, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.349676225717 F1-OK:  0.88733974359
F1-score multiplied:  0.310281612467
Epoch 3: 
Test set LL -0.490903305913322, Acc 0.8106391429901123, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.374746335964 F1-OK:  0.888423932724
F1-score multiplied:  0.332933613571
[ 97998  95754  98742 ..., 203012 213193 125451]
Epoch 4: \Dev set LL -0.5239743947159541, Acc 0.8252971172332764, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.35696329814 F1-OK:  0.898917252825
F1-score multiplied:  0.320880467323
Epoch 4: 
Test set LL -0.5396641405277335, Acc 0.8251160979270935, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.367810417181 F1-OK:  0.898522011333
F1-score multiplied:  0.330485755834
[ 82707 140148 157986 ...,  51883 106201 103473]
Epoch 5: \Dev set LL -0.5702534534629937, Acc 0.8323999643325806, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.359268929504 F1-OK:  0.903590791231
F1-score multiplied:  0.324632096275
Epoch 5: 
Test set LL -0.5909027255704936, Acc 0.8318765163421631, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.360519480519 F1-OK:  0.903215661609
F1-score multiplied:  0.32562684112
[ 96593 186600 181132 ...,  40532  34553 129546]
Epoch 6: \Dev set LL -0.6286085786592275, Acc 0.833082914352417, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.360878661088 F1-OK:  0.904006284368
F1-score multiplied:  0.326236577518
Epoch 6: 
Test set LL -0.6480350401701599, Acc 0.8348128795623779, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.367250850118 F1-OK:  0.905006872177
F1-score multiplied:  0.33236454317
[211235 108099  35584 ..., 173533 203627 154044]
Epoch 7: \Dev set LL -0.6627330248031905, Acc 0.8341756463050842, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.362394957983 F1-OK:  0.904694614539
F1-score multiplied:  0.327856766824
Epoch 7: 
Test set LL -0.682730927291714, Acc 0.8342666029930115, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.362155059133 F1-OK:  0.904760036102
F1-score multiplied:  0.327663424376
[ 44555 140306 112838 ...,  74919 150399 165017]
Epoch 8: \Dev set LL -0.7009275918028103, Acc 0.8343122601509094, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.368558042686 F1-OK:  0.904645861174
F1-score multiplied:  0.333414507919
Epoch 8: 
Test set LL -0.7210862172105238, Acc 0.8345397710800171, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.364542355101 F1-OK:  0.904887144259
F1-score multiplied:  0.329869690669
[169730  20673 186182 ...,  67812 116880 108578]
Epoch 9: \Dev set LL -0.7253197002767483, Acc 0.833629310131073, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.372811534501 F1-OK:  0.904094488189
F1-score multiplied:  0.337056853475
Epoch 9: 
Test set LL -0.7436247976206934, Acc 0.8328325748443604, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.366131538063 F1-OK:  0.90372060096
F1-score multiplied:  0.330880613609
[ 41613 123879  70949 ...,  98250 105271 211104]
Epoch 10: \Dev set LL -0.7470163776356781, Acc 0.8333560824394226, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.378185524975 F1-OK:  0.903785488959
F1-score multiplied:  0.341798589606
Epoch 10: 
Test set LL -0.7652779416170302, Acc 0.8311253786087036, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.369936305732 F1-OK:  0.902495761542
F1-score multiplied:  0.333865947964
[ 79978  62975 104982 ..., 106252 110351  75418]
Epoch 11: \Dev set LL -0.7589040883725782, Acc 0.8329463005065918, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.38573581115 F1-OK:  0.903327800174
F1-score multiplied:  0.348445881735
Epoch 11: 
Test set LL -0.7838730224282988, Acc 0.8298279047012329, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.371342078708 F1-OK:  0.901595324593
F1-score multiplied:  0.334800281988
[ 78595 203579  77628 ..., 171197 169790 184869]
Epoch 12: \Dev set LL -0.7803440106737395, Acc 0.8334926962852478, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.385274836107 F1-OK:  0.903704874003
F1-score multiplied:  0.34817474722
Epoch 12: 
Test set LL -0.8077159553346649, Acc 0.8305107951164246, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.377944862155 F1-OK:  0.901889477429
F1-score multiplied:  0.340864494226
[161145   8624 103629 ..., 131598  77971 145710]
Epoch 13: \Dev set LL -0.7870816165483376, Acc 0.8328097462654114, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.390438247012 F1-OK:  0.903118568941
F1-score multiplied:  0.352612030901
Epoch 13: 
Test set LL -0.8163631022987512, Acc 0.8284621834754944, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.37975308642 F1-OK:  0.900467548934
F1-score multiplied:  0.341955330929
[ 88180  37890  33658 ..., 138465 165899 187250]
Epoch 14: \Dev set LL -0.8091525600443863, Acc 0.8332195281982422, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.398225726959 F1-OK:  0.90319511615
F1-score multiplied:  0.359675531715
Epoch 14: 
Test set LL -0.835329903500437, Acc 0.8282573223114014, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.381303813038 F1-OK:  0.900289418388
F1-score multiplied:  0.343283788069
[ 95272 217111 176486 ..., 183877 161754 214087]
Epoch 15: \Dev set LL -0.8231747367243807, Acc 0.8321267366409302, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.396068796069 F1-OK:  0.902514476085
F1-score multiplied:  0.357457821978
Epoch 15: 
Test set LL -0.8523815784430409, Acc 0.8271647095680237, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.384634087041 F1-OK:  0.899463753724
F1-score multiplied:  0.34596441974
[ 43428  23203 165494 ...,  87260 127806  73850]
Epoch 16: \Dev set LL -0.8270535279933107, Acc 0.832536518573761, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.40311587147 F1-OK:  0.90260565618
F1-score multiplied:  0.363854665685
Epoch 16: 
Test set LL -0.8591261033976252, Acc 0.8271647095680237, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.388499637594 F1-OK:  0.8993598155
F1-score multiplied:  0.349400962388
[204644  31141 103650 ...,  30090  81702 136580]
Epoch 17: \Dev set LL -0.8375377694051755, Acc 0.8328097462654114, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.401759530792 F1-OK:  0.902826294062
F1-score multiplied:  0.362719068289
Epoch 17: 
Test set LL -0.868758530313309, Acc 0.8273695707321167, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.388485728108 F1-OK:  0.899499085633
F1-score multiplied:  0.349442557215
[151463 159307 100073 ...,  99926  64887  45399]
Epoch 18: \Dev set LL -0.8456029721771318, Acc 0.8317169547080994, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.403100775194 F1-OK:  0.902051200509
F1-score multiplied:  0.36361753819
Epoch 18: 
Test set LL -0.8803185126991296, Acc 0.8259355425834656, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.391210890853 F1-OK:  0.898450260946
F1-score multiplied:  0.351483526971
[ 77558  65635  39900 ...,  52102 175656  30661]
Epoch 19: \Dev set LL -0.8485858760507397, Acc 0.8303510546684265, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.405741626794 F1-OK:  0.901051625239
F1-score multiplied:  0.36559415225
Epoch 19: 
Test set LL -0.8865030928353351, Acc 0.8253892660140991, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.39450627516 F1-OK:  0.89798523838
F1-score multiplied:  0.354260811542
[   873  43953 181616 ..., 181234   7816  90895]
Epoch 20: \Dev set LL -0.862913535575115, Acc 0.8329463005065918, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.401370533529 F1-OK:  0.902928803873
F1-score multiplied:  0.362409015749
Epoch 20: 
Test set LL -0.8991292039417308, Acc 0.826481819152832, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.388741881164 F1-OK:  0.898889817357
F1-score multiplied:  0.349436118559
[139636 198046 199708 ...,  32597 183833  57999]
Epoch 21: \Dev set LL -0.8640958328327333, Acc 0.831580400466919, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.406355320173 F1-OK:  0.901870274572
F1-score multiplied:  0.366479784179
Epoch 21: 
Test set LL -0.9007349639360741, Acc 0.8257306814193726, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.396690307329 F1-OK:  0.898156277436
F1-score multiplied:  0.356289889725
[ 48250  85392  96772 ..., 159417  47120  80258]
Epoch 22: \Dev set LL -0.8665784606075516, Acc 0.8323999643325806, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.408674698795 F1-OK:  0.902363332538
F1-score multiplied:  0.368773063129
Epoch 22: 
Test set LL -0.9048220926158516, Acc 0.826276957988739, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.393708293613 F1-OK:  0.898613103778
F1-score multiplied:  0.353791431707
[ 94976 169993 129145 ..., 212870  49486 180803]
Epoch 23: \Dev set LL -0.8700377062062966, Acc 0.8326731324195862, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.410207029369 F1-OK:  0.902506963788
F1-score multiplied:  0.370214700601
Epoch 23: 
Test set LL -0.9134896850501114, Acc 0.8253892660140991, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.395651146301 F1-OK:  0.897952667917
F1-score multiplied:  0.355276002386
[188706 216106  14316 ...,  23716  33182   2019]
Epoch 24: \Dev set LL -0.8707749041366738, Acc 0.8314437866210938, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.407869481766 F1-OK:  0.901735945214
F1-score multiplied:  0.367790572664
Epoch 24: 
Test set LL -0.9168640108942309, Acc 0.8248429298400879, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.39174768793 F1-OK:  0.897690558813
F1-score multiplied:  0.351668200891
[ 49555 204099 199462 ..., 142630  30997  94732]
Epoch 25: \Dev set LL -0.8764992272563636, Acc 0.8317169547080994, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.405405405405 F1-OK:  0.901988862371
F1-score multiplied:  0.365671160421
Epoch 25: 
Test set LL -0.9174499975640235, Acc 0.8271647095680237, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.399810291677 F1-OK:  0.899046707351
F1-score multiplied:  0.359448126297
[ 10586  40093 110450 ..., 135123  16134  39003]
Epoch 26: \Dev set LL -0.8783663166619011, Acc 0.8322633504867554, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.410182516811 F1-OK:  0.902229299363
F1-score multiplied:  0.370078684753
Epoch 26: 
Test set LL -0.9216220267588935, Acc 0.825867235660553, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.397448015123 F1-OK:  0.898227969349
F1-score multiplied:  0.356998923545
[124016  93299 142951 ..., 121315 130112 138434]
Epoch 27: \Dev set LL -0.8832342040633231, Acc 0.8303510546684265, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.410815939279 F1-OK:  0.900909526089
F1-score multiplied:  0.370107993166
Epoch 27: 
Test set LL -0.9279132327142918, Acc 0.8240234851837158, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.395496129486 F1-OK:  0.897022977023
F1-score multiplied:  0.354769115473
[  9469 193383  39949 ..., 202484  63420  57130]
Epoch 28: \Dev set LL -0.8809486302862793, Acc 0.8332195281982422, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.415509813308 F1-OK:  0.902732414562
F1-score multiplied:  0.375094177042
Epoch 28: 
Test set LL -0.9287126844631421, Acc 0.8253892660140991, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.397927949141 F1-OK:  0.897887464558
F1-score multiplied:  0.357294517331
[ 70289  17736 199270 ...,  16586  34471 159323]
Epoch 29: \Dev set LL -0.8832020531346934, Acc 0.833082914352417, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.414750957854 F1-OK:  0.902660506611
F1-score multiplied:  0.374379309734
Epoch 29: 
Test set LL -0.9308453835648028, Acc 0.8251844048500061, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.398496240602 F1-OK:  0.897730904442
F1-score multiplied:  0.357742390492
[ 44559 117975   7356 ..., 164492 159093  15768]
Epoch 30: \Dev set LL -0.8823647125774086, Acc 0.832536518573761, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.41395793499 F1-OK:  0.902310756972
F1-score multiplied:  0.373518697676
Epoch 30: 
Test set LL -0.9282437092287086, Acc 0.8251160979270935, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.399812514647 F1-OK:  0.897645977379
F1-score multiplied:  0.358890095479
[109754 141570 108980 ..., 139365 215134  72258]
Epoch 31: \Dev set LL -0.890769109413625, Acc 0.8310340046882629, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.410671748452 F1-OK:  0.901379255362
F1-score multiplied:  0.370170994817
Epoch 31: 
Test set LL -0.936787563920147, Acc 0.8257989883422852, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.403832671185 F1-OK:  0.89799672118
F1-score multiplied:  0.36264041463
[188636 149668 181585 ...,  86043   3885 149926]
Epoch 32: \Dev set LL -0.8911239098828772, Acc 0.8328097462654114, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.414354066986 F1-OK:  0.902485659656
F1-score multiplied:  0.373948603475
Epoch 32: 
Test set LL -0.9358960342605618, Acc 0.8257306814193726, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.40037593985 F1-OK:  0.898050495366
F1-score multiplied:  0.359557811115
[118069 172817 212991 ...,   6636 158235  11185]
Epoch 33: \Dev set LL -0.8887049751329645, Acc 0.8313071727752686, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.412743699477 F1-OK:  0.901507297233
F1-score multiplied:  0.372091456965
Epoch 33: 
Test set LL -0.9401001895218963, Acc 0.8245697617530823, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.401862630966 F1-OK:  0.897211219141
F1-score multiplied:  0.360555661057
[ 35430  78766   7572 ...,  21798 172876 186694]
Epoch 34: \Dev set LL -0.8907669483326504, Acc 0.832536518573761, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.41619047619 F1-OK:  0.902248445224
F1-score multiplied:  0.37550721006
Epoch 34: 
Test set LL -0.9392364664325102, Acc 0.8250477910041809, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.400841908326 F1-OK:  0.8975691668
F1-score multiplied:  0.359783337674
[ 12263 179388 192145 ...,  56465 182676  75856]
Epoch 35: \Dev set LL -0.8956419295226385, Acc 0.8318535685539246, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.409592326139 F1-OK:  0.901967030342
F1-score multiplied:  0.369438774058
Epoch 35: 
Test set LL -0.9405245165309062, Acc 0.8245015144348145, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.3984082397 F1-OK:  0.89726574992
F1-score multiplied:  0.357478067969
[159646 185256 171645 ..., 194155 131629  68601]
Epoch 36: \Dev set LL -0.8951731093557647, Acc 0.8308973908424377, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on dev:
F1-BAD:  0.411037107517 F1-OK:  0.901275917065
F1-score multiplied:  0.370457846025
Epoch 36: 
Test set LL -0.9466378280372066, Acc 0.8245015144348145, Outputs [1 1 1 ..., 1 1 1]
Result from the previous epoch on test:
F1-BAD:  0.403435468895 F1-OK:  0.897117694155
F1-score multiplied:  0.361929097596
[191753 201171 122412 ...,  38797  37466 188023]



