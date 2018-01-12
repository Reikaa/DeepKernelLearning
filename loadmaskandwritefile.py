import numpy as np
import copy
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_regression



#train_text.vectors
def rewrite(myfile, mask):
    filename = open(myfile+".features", "w")
    file = open(myfile, "r")
    count = 0
    for x in file:
        x = x.strip().split(' ')
        #print(x)
        count = 0
        for bool in mask:
            if bool==True:
                filename.write(x[count])
                filename.write(",")
            else:
                continue
            count = count + 1;
        filename.write("\n")
    filename.close()
    file.close()


file = open("mask", "r")
bools = []
for x in file:
    x = x.strip()
    if x == "True":
        bools.append(True)
    else:
        bools.append(False)
file.close()
print(bools)
rewrite("dev_text.vectors", bools)
