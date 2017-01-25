import nltk
from nltk.corpus import brown, treebank
from MaximumEntropyTagger import MaximumEntropyTagger
from nltk.classify import config_megam
import itertools
import time
import pickle
from nltk.corpus import brown, conll2000, treebank

def averageAccu(oriSents, taggedSents):
    n = 0.0
    for i in range(len(oriSents)):
        m = 0.0
        for j in range(len(oriSents[i])):
            if oriSents[i][j][1] == taggedSents[i][j][1]:
                m += 1
        n += m / len(oriSents[i])

    return n / len(oriSents)


def overallPrecision(oriSents, taggedSents):
    corS = {}
    tagS = {}
    keys = tagS.viewkeys()
    for i in range(len(oriSents)):
        for j in range(len(oriSents[i])):
            if not tagS.has_key(taggedSents[i][j][1]):
                tagS[taggedSents[i][j][1]] = 1
                corS[taggedSents[i][j][1]] = 0
            else:
                tagS[taggedSents[i][j][1]] += 1

    for i in range(len(oriSents)):
        for j in range(len(oriSents[i])):
            if oriSents[i][j][1] == taggedSents[i][j][1]:
                corS[taggedSents[i][j][1]] += 1

    # print corS
    # print tagS
    for key in keys:
        num = corS[key] / float(tagS[key])
        print("Overall precision of '" + str(key) + "': " + format(num, '.2%'))
    return 0


def overallRecall(oriSents, taggedSents):
    corS = {}
    oriS = {}
    keys = oriS.viewkeys()
    for i in range(len(oriSents)):
        for j in range(len(oriSents[i])):
            if not oriS.has_key(oriSents[i][j][1]):
                oriS[oriSents[i][j][1]] = 1
                corS[oriSents[i][j][1]] = 0
            else:
                oriS[oriSents[i][j][1]] += 1
            if oriSents[i][j][1] == taggedSents[i][j][1]:
                corS[oriSents[i][j][1]] += 1
    # print corS
    # print oriS
    for key in keys:
        num = corS[key] / float(oriS[key])
        print("Overall recall of '" + str(key) + "': " + format(num, '.2%'))
    return 0


if __name__ == '__main__':
    PATH_TO_MEGAM_EXECUTABLE = "./megam_0.92/megam"
    config_megam(PATH_TO_MEGAM_EXECUTABLE)
    a = 800
    cutoff = a * 2 / 3

    # brown_reviews = brown.tagged_sents(categories=['reviews'])
    # train = brown_reviews[:cutoff]
    # test = brown_reviews[cutoff:a]
    # brown_lore = brown.tagged_sents(categories=['lore'])
    # brown_lore_cutoff = len(brown_lore) * 2 / 3
    # brown_romance = brown.tagged_sents(categories=['romance'])
    # brown_romance_cutoff = len(brown_romance) * 2 / 3

    # brown_train = list(itertools.chain(brown_reviews[:brown_reviews_cutoff],
    #     brown_lore[:brown_lore_cutoff], brown_romance[:brown_romance_cutoff]))
    # brown_test = list(itertools.chain(brown_reviews[brown_reviews_cutoff:],
    #     brown_lore[brown_lore_cutoff:], brown_romance[brown_romance_cutoff:]))
    # brown_train = brown_reviews[:brown_reviews_cutoff]
    # brown_test = brown_reviews[brown_reviews_cutoff:]
    #
    # conll_train = conll2000.tagged_sents('train.txt')
    # conll_test = conll2000.tagged_sents('test.txt')
    # train = conll_train[:cutoff]
    # test = conll_test[cutoff:a]


    train = treebank.tagged_sents()[:cutoff]
    test = treebank.tagged_sents()[cutoff:a]

    startTime = time.time()
    me_Tagger = MaximumEntropyTagger(train)
    # test_taggedSents = me_Tagger.tag(treebank_test)
    print "Accuracy of Maximum Entropy Tagger:"
    print me_Tagger.evaluate(test)
    stoptime1 = time.time()
    print "Time used: " + str(round(stoptime1-startTime, 3))

    # Back-off tagger
    t0 = nltk.DefaultTagger('NN')
    t1 = nltk.UnigramTagger(train, backoff=t0)
    t2 = nltk.BigramTagger(train, backoff=t1)
    t3 = nltk.TrigramTagger(train, backoff=t2)
    num = t3.evaluate(test)
    print("Accuracy of Uni-Bi-Tri Backoff Tagger: ")
    print str(num)
    stoptime2 = time.time()
    print 'Time used: ' + str(round(stoptime2-stoptime1, 3))

