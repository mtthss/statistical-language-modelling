
####################
# IMPORT LIBRARIES #
####################

import glob
import os
import matplotlib.pyplot as plt
from pylab import linspace
import nltk.data
from nltk.tokenize import wordpunct_tokenize
from math import log10
from numpy import power


########################
# LANGUAGE MODEL CLASS #
########################

# LANGUAGE MODEL
# language model class, reads and learns from one or more documents
#
# ALGORITHMS FOR LEARNING:
# - constant model (silly baseline)
# - laplacian smoothing
# - back-off (recursion grounded on constant model)
# - back-off (recursion grounded on unigram with unknown simbol)
# - good turing smoothing
#
# ALGORITHMS FOR EVALUATION
# - evaluatePerplexity
# - plotDistribution
# - showSampleVocabularyEntries 
#
class LanguageModel:
    
    # INPUT PARAMETERS:
    # - "trainOn": path training file
    # - "testOn": path testing file
    # - "tc": token splitting choice in [nltk,my-reg-ex,python-basic]
    # - "ssc": sentence splitting choice in [nltk,my-reg-ex,python-basic]
    # - "N": n gram order
    # - "d": smoothing factor in laplacian
    # - "smooth": smoothing choice in [constantModel, laplacian, backOff, backOffWithUnknown, goodTuring]
    # - "k": kats limit
    # - "knD_p":
    #
    def __init__(self, trainOn, testOn, tc="python-basic", ssc="python-basic", N=2, d=1, smooth="laplacian", k=2, knD_p=0.75):
        print '...initializing learning system:'
        print ""
        # define
        os.chdir('C:\\Users\\Matteo\\workspace\\languageModeling')
        self.pathTR = trainOn
        self.pathTST = testOn
        # define dictionary of functions
        self.smoothing_dict = {
            'laplacian': self.laplacianSmoothing,
            'goodTuring': self.goodTuringSmoothing,
            'backOff': self.backOff,
            'backOffWithUnkown': self.backOffWithUnkown,
            'constantModel': self.constantModel
        } 
        tokenizing_dict = {
            'nltk': self.nltkTokenizer,
            'my-reg-ex': self.regexTokenizer,
            'python-basic': self.basicTokenizer
        }
        sentSplitting_dict = {
            'nltk': self.nltkSentenceSplitter,
            'my-reg-ex': self.regexSentenceSplitter,
            'python-basic': self.basicSentenceSplitter
        }
        # define execution choices, including the run time identity of the functions probability, token, sentSplit,...
        self.smoothing=smooth
        self.ngramChoice = N
        self.delta = d
        self.probability = self.smoothing_dict[smooth]
        self.token = tokenizing_dict[tc]
        self.sentSplit = sentSplitting_dict[ssc]
        self.katsLimit = k
        self.knD = knD_p
        self.minSentPPL = 1000000
        self.maxSent = '<<<>INIT<>>>'
        if(tc=='nltk'):
            self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.listOfDict = [{}]*(self.ngramChoice)
        self.N = 0
        self.NsGoodTuring = {}
        self.numFilesRead = 0
    
    
    ##### SET UP LEARNING #####
    
    # reads through the files, counting ngrams
    # performs additional postprocessing as:
    # - deriving lower order Ngram models
    # - computing utility structures for specific smoothing techniques
    def train(self, max_files):
        print '...learning the language model'    
        counter=1
        # explore directories
        for filename in glob.iglob(os.path.join(self.pathTR, '*', '*.txt')):
            if counter>max_files:
                break
            with open(filename) as f:
                self.numFilesRead=counter
                counter+=1
                # learn N gram model (N=0 -> constant uniform model)
                self.countNGram(f.read().replace('\n', ' '))
        if self.ngramChoice>1:
            self.deriveLowerOrderModels()
        elif self.ngramChoice==1 and self.smoothing=='backOffWithUnkown':
            sz = len(self.listOfDict[0])
            self.listOfDict[0][tuple(['<<<>UKN<>>>'])]=sz
        if self.smoothing=='goodTuring':
            self.computeNsForGoodTuring()
            
    
    ##### LANGUAGE MODELS #####
    
    # updates the dictionary containing the count of ngrams 
    # according to the content of the most recent sentence read from documents
    def countNGram(self, text):     
        # splitting and initilizing
        sentences = self.sentSplit(text)
        # learning of higher order model
        for sent in sentences:
            # initialize history list
            prevList=['<<<>sSs<>>>']*(self.ngramChoice-1)
            for t in self.token(sent):
                self.N+=1  
                # update N-grams count
                currTuple=tuple(prevList)+tuple([t])
                if (currTuple in self.listOfDict[self.ngramChoice-1]):
                    self.listOfDict[self.ngramChoice-1][currTuple]+=1
                else:
                    self.listOfDict[self.ngramChoice-1][currTuple]=1                
                # update history list
                if len(prevList)>0:
                    prevList.pop(0)
                    prevList.append(t)
            # manage end of sentence
            count = self.ngramChoice
            while count>1:
                prevList.append('<<<>eEe<>>>')
                endTuple=tuple(prevList)
                if (endTuple in self.listOfDict[self.ngramChoice-1]):
                    self.listOfDict[self.ngramChoice-1][endTuple]+=1
                else:
                    self.listOfDict[self.ngramChoice-1][endTuple]=1
                count-=1
                prevList.pop(0)


    ##### POST PROCESS LANGUAGE MODELS #####

    # derives n gram counts for order N-1 from those for order N
    def deriveLowerOrderModels(self):
        print '...deriving lower order models'
        # derive lower order counts from N-grams
        current = self.ngramChoice-1
        while current>=1:
            tmp={}
            ukn = 0
            if current == 1:
                ukn = 1
                tmp[tuple(['<<<>UKN<>>>'])]=0
            for key, value in self.listOfDict[current].iteritems():
                tpl = key[0:-1]
                if tpl in tmp:
                    tmp[tpl]+=value
                else:
                    tmp[tpl]=value-ukn
                    if current == 1:
                        tmp[tuple(['<<<>UKN<>>>'])]+=1
            if tuple(['<<<>sSs<>>>']*(current+1)) in self.listOfDict[current]:
                tmp[tuple(['<<<>sSs<>>>']*current)] = self.listOfDict[current][tuple(['<<<>sSs<>>>']*(current+1))]
            self.listOfDict[current-1] = tmp
            current=current-1
            
    # compute the frequency buckets for good turing
    def computeNsForGoodTuring(self):
        # go through dictionary and update Ncounts
        print '...deriving good turing buckets'
        for k, v in self.listOfDict[self.ngramChoice-1].iteritems(): 
            if v in self.NsGoodTuring:
                self.NsGoodTuring[v]+=1
            else:
                self.NsGoodTuring[v]=1
        self.NsGoodTuring[0] = len(self.listOfDict[0]) # assign count('<<<>UKN<>>>')
        
            
    # EVALUATE
    
    # implements the classic intrinsic evaluation metric for language modelling
    def evaluatePerplexity(self):
        print '...evaluating the language model'
        # initialize log-perplexity
        log_PPL = 0.0
        # initialize number of tokens in the test set
        tst_count = 0
        # walk through all files and directories
        for filename in glob.iglob(os.path.join(self.pathTST, '*', '*.txt')):
            # initialize
            currTestText = open(filename).read().replace('\n', ' ')
            sentences = self.sentSplit(currTestText)
            # loop on sentences
            for sent in sentences:
                sPPL = 0
                prevList=['<<<>sSs<>>>']*(self.ngramChoice-1)
                for t in self.token(sent):
                    tst_count+=1
                    currTuple=tuple(prevList)+tuple([t])
                    # to avoid numerical issues lets go to the log trasformed
                    sPPL = sPPL + log10(self.probability(currTuple, self.ngramChoice))
                    # update history list
                    if len(prevList)>0:
                        prevList.pop(0)
                        prevList.append(t)
                count = self.ngramChoice
                while count>1:
                    tst_count+=1
                    prevList.append('<<<>eEe<>>>')
                    endTuple=tuple(prevList)
                    log_PPL = log_PPL + log10(self.probability(endTuple, self.ngramChoice))
                    count-=1
                    prevList.pop(0)
                log_PPL = log_PPL + sPPL
                if sPPL < self.minSentPPL:
                    self.minSentPPL = sPPL
                    self.maxSent = sent
        # normalize and antitransform
        log_PPL = float(log_PPL)/tst_count
        PPL = power(10,-log_PPL)
        return PPL 
    
    
    # SMOOTHING TECHNIQUES
    
    # silly baseline
    def constantModel(self, tpl, order):
        # return the probability corresponding to a uniform distribution over word types
        #print float(1)/(len(self.listOfDict[0])+1)
        return float(1)/(len(self.listOfDict[0])+1)        
    
    # standard smoothing technique
    # the number added to all count depends on the choice of delta
    def laplacianSmoothing(self, tpl, order):    
        # useful in any case
        V = len(self.listOfDict[0])
        N = self.N
        list_tpl = list(tpl)
        list_tpl.pop()
        history = tuple(list_tpl)
        # count of <history,word>, leads to numerator=count+1
        if (tpl in self.listOfDict[order-1]):
            c = self.listOfDict[order-1][tpl]
        else:
            c = 0               
        numerator = c+self.delta
        #print numerator
        # count of <history>, leads to denominator=N+V (UNIGRAM) or denominator=count+V (NGRAM)
        if (history in self.listOfDict[order-2]) and (order>1):
            c_den = self.listOfDict[order-2][history]
        elif (order>1):
            c_den = 0
        else:
            c_den = N
        denominator = c_den+V*self.delta
        #print c_den,V
        #print denominator
        # laplacian add delta estimate 
        result = float(numerator)/denominator
        #print result
        return result
        
    # is the N gram was never found during training backs off to the N-1 gram model
    # recursion ends on the 0 order constant model
    def backOff(self, tpl, order):
        # constant model grounding
        if order==0:
            return float(1)/(len(self.listOfDict[0])+1)
        # initialization
        V = len(self.listOfDict[0])
        N = self.N
        list_tpl = list(tpl)
        list_tpl2 = list(tpl)
        list_tpl.pop()
        list_tpl2.pop(0)
        history = tuple(list_tpl)
        back_off_tpl = tuple(list_tpl2)
        # count of <history,word>, leads to numerator=count+1
        if (tpl in self.listOfDict[order-1]):
            c = self.listOfDict[order-1][tpl]
        else:
            return self.probability(back_off_tpl, order-1)    
        # count of <history>, leads to denominator=N+V (UNIGRAM) or denominator=count+V (NGRAM)
        if (history in self.listOfDict[order-2]) and (order>1):
            c_den = self.listOfDict[order-2][history]
        elif (order>1):
            c_den = 0
        else:
            c_den = N
        # laplacian add delta estimate 
        result = float(c+self.delta)/(c_den+V*self.delta)
        return result
    
    # is the N gram was never found during training backs off to the N-1 gram model
    # recursion ends on the unigram model (to whome an "unknow" symbol has been added and its probability estimated during training)
    def backOffWithUnkown(self, tpl, order):
        # useful in any case
        V = len(self.listOfDict[0])
        N = self.N
        list_tpl = list(tpl)
        list_tpl2 = list(tpl)
        list_tpl.pop()
        list_tpl2.pop(0)
        history = tuple(list_tpl)
        back_off_tpl = tuple(list_tpl2)
        # count of <history,word>, leads to numerator=count+1
        if (tpl in self.listOfDict[order-1]):
            c = self.listOfDict[order-1][tpl]
        elif order==1: 
            c = self.listOfDict[order-1][tuple(['<<<>UKN<>>>'])]
        else:
            return self.probability(back_off_tpl, order-1)    
        numerator = c+self.delta
        # count of <history>, leads to denominator=N+V (UNIGRAM) or denominator=count+V (NGRAM)
        if (history in self.listOfDict[order-2]) and (order>1):
            c_den = self.listOfDict[order-2][history]
        elif (order>1):
            c_den = 0
        else:
            c_den = N
        denominator = c_den+V*self.delta
        result = float(numerator)/denominator
        return result
    
    # get non smoothed probability and restore delta
    def noSmoothing(self, tpl, order):
        self.delta = 0
        prob = self.laplacianSmoothing(tpl, order)
        self.delta = 1
        return prob 
    
    # good turing smoothing
    # en.m.wikipedia/wiki/Good-Turing_Frequency_Estimation
    def goodTuringSmoothing(self, tpl, order):
        #initialize
        V = len(self.listOfDict[0])
        N = self.N
        k=self.katsLimit
        # non discounted count
        c = self.listOfDict[order-1][tpl] if (tpl in self.listOfDict[order-1]) else 0
        if c>k:
            return self.noSmoothing(tpl, order)
        # discounting for low counts
        N_cplus1 = float(0)
        N_c = float(0)
        N_k = float(0)
        N_1 = float(0)
        N_kplus1 = float(0)
        if (c+1) in self.NsGoodTuring:
            N_cplus1 = float(self.NsGoodTuring[c+1])
        if c in self.NsGoodTuring:
            N_c = float(self.NsGoodTuring[c])
        if k in self.NsGoodTuring:
            N_k = float(self.NsGoodTuring[k])
        if int(1) in self.NsGoodTuring:
            N_1 = float(self.NsGoodTuring[1])
        if (k+1) in self.NsGoodTuring:
            N_kplus1 = float(self.NsGoodTuring[k+1])
        num1 = (c+1)*(float(N_cplus1)/float(N_c))
        num2 = float(c*((k+1)*N_kplus1))/float(N_1)
        num = num1 - num2
        den = 1-float((k+1)*N_kplus1)/float(N_1)
        c_disc = float(num)/float(den)
        prob = c_disc / N
        return prob
        
        
    ##### SHOW RESULTS ##### 
    
    # print less frequent and more frequent vocabulary entries
    def showSampleVocabularyEntries(self,K,reverse):
        if reverse:
            # print the K less frequent
            count=1
            for w in sorted(self.listOfDict[0], key=self.listOfDict[0].get, reverse=False):
                if count > K:
                    break
                print w, self.listOfDict[0][w]
                count+=1
        else:      
            # print the K most frequent
            count=1
            for w in sorted(doc.listOfDict[0], key=doc.listOfDict[0].get, reverse=True):
                if count > K:
                    break
                print w, doc.listOfDict[0][w]
                count+=1
    
    # plot frequency distribution
    def plotDistribution(self):  
        # plot frequency count for the entire vocabulary
        threshold = 1000
        size = len(self.listOfDict[0])
        x = linspace(1, size, size)
        y = sorted(self.listOfDict[0].values(), reverse=True)
        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        axes.plot(x, y, 'r')
        axes.set_xlabel('x')
        axes.set_ylabel('y')
        axes.set_title('title');
        plt.show()
        #plot frequency count for all words with count over a given threshold (e.g. number of files read)
        threshold = self.numFilesRead
        size = len(self.listOfDict[0])
        size_relevant = sum(1 for i in self.listOfDict[0].values() if i>threshold)
        x = linspace(1, size_relevant, size_relevant)
        y = sorted([i for i in self.listOfDict[0].values() if i>threshold], reverse=True)
        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        axes.plot(x, y, 'r')
        axes.set_xlabel('x')
        axes.set_ylabel('y')
        axes.set_title('title');
        plt.show()

        
    ##### SENTENCE/TOKEN SPLITTING #####
    
    def nltkTokenizer(self, s):
        # nltk tokenizer
        return wordpunct_tokenize(s.encode('UTF-8'))
    
    def basicTokenizer(self,s):
        # my tokenizers
        return s.split()
    
    def regexTokenizer(self,s):
        # my tokenizer
        return 'return not impl'
            
    def nltkSentenceSplitter(self, currentText):
        # nltk sentence splitter
        return self.tokenizer.tokenize(currentText.decode('UTF-8'))
    
    def basicSentenceSplitter(self, currentText):
        # my sentence splitters
        return currentText.split('.')
    
    def regexSentenceSplitter(self, currentText):
        # my sentence splitter
        return 'not impl'
        
        
#############
# EXECUTION #
#############

# actual execution choices
pathTrain = '.\\ACL\ACL-TRAIN'     # '.\\ACL\ACL-TRAIN'; '.\\Train-single';    
pathTest = '.\\ACL\ACL-TEST'       # '.\\ACL\ACL-TEST'; '.\\Test-single';
tokenizerChoice='nltk'             # 'nltk'; 'my-reg-ex'; 'python-basic'
sentSplitChoice='nltk'             # 'nltk'; 'my-reg-ex'; 'python-basic'
NgramChoice=3                      # '0'; '1'; '2';
smoothing='laplacian'              # 'laplacian', 'goodTuring'
delta = float(0.001)

doc = LanguageModel(pathTrain, pathTest,tokenizerChoice,sentSplitChoice,NgramChoice, delta, smoothing)
doc.train(15000)
ppl = doc.evaluatePerplexity()
print 'perplexity:', ppl
