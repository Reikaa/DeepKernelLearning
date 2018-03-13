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


