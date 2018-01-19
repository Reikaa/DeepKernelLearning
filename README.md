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
 
 
 
 

