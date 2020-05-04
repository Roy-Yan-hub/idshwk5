from sklearn.ensemble import RandomForestClassifier
import numpy as np
import math
domainlist = []
test_domainlist = []
TEST=1
TRAIN=0
def cal_entropy(text):
    h = 0.0
    sum = 0
    letter = [0] * 26
    text = text.lower()
    for i in range(len(text)):
        if text[i].isalpha():
            letter[ord(text[i]) - ord('a')] += 1
            sum += 1
    #print('\n', letter)
    for i in range(26):
        p = 1.0 * letter[i] / sum
        if p > 0:
            h += -(p * math.log(p, 2))
    return h



class Domain:
	def __init__(self,_name,_label,_len,_num,_ent):
		self.name = _name
		self.label = _label
		self.length =_len 
		self.num = _num
		self.ent = _ent


	def returnData(self):
		return [self.length, self.num]
	def returnName(self):
		return self.name

	def returnLabel(self):
		if self.label == "notdga":
			return 0
		else:
			return 1
		
def initData(filename,Type):
	with open(filename) as f:
		for line in f:
			line = line.strip()
			if line.startswith("#") or line =="":
				continue
			label=0
			if Type == TRAIN:
				tokens = line.split(",")
				name = tokens[0]
				label = tokens[1]
			else:
				name = line
			length = len(name)
			num = 0
			for i in name:
			    if i.isdigit():
			        num=num+1
			ent = cal_entropy(name)
			if Type == TRAIN:
				domainlist.append(Domain(name,label,length,num,ent))
			else:
				test_domainlist.append(Domain(name,label,length,num,ent))


def main():
	initData("train.txt",TRAIN)
	featureMatrix = []
	test_featureMatrix = []	
	test_Name=[]
	labelList = []
	for item in domainlist:
		featureMatrix.append(item.returnData())
		labelList.append(item.returnLabel())

	clf = RandomForestClassifier(random_state=0)
	clf.fit(featureMatrix,labelList)
	



	initData("test.txt",TEST)	
	for item in test_domainlist:
		test_featureMatrix.append(item.returnData())
		test_Name.append(item.returnName())
	
	f = open("result.txt","w")
	for i in range(len(test_Name)):
		#print(test_Name[i],clf.predict([test_featureMatrix[i]]))
		if clf.predict([test_featureMatrix[i]])==0:
			f.write(test_Name[i]+",nodga\n")
		else:
			f.write(test_Name[i]+",dga\n")
		
	f.close()

if __name__ == '__main__':
	main()

