# coding=utf-8
import re
from collections import defaultdict
from nltk import TaggerI, MaxentClassifier, FreqDist, untag


class MaximumEntropyTagger(TaggerI):
    def __init__(self, train_sents, algorithm='megam', rare_word_cutoff=5,
                 rare_feat_cutoff=5, uppercase_letters='[A-Z]', trace=3,
                 **cutoffs):
        """
        :type train_sents: list of tuple
        :param train_sents:
        A list of tagged sents

        :type algorithm: str
        :param algorithm:
        Here default is 'megam' which is much faster than the algotirhms provided by nltk.
        However the performance is not guaranteed as well. Optional choices are 'GIS', 'IIS'.

        :type rare_word_cutoff: int
        :param rare_word_cutoff:

        :type rare_feat_cutoff: int
        :param rare_feat_cutoff:

        :type uppercase_letters: regex
        :param uppercase_letters:a regular expression that covers all
        uppercase letters of the language of your corpus (e.g. '[A-ZÄÖÜ]' for
        German)

        :type trace: int
        :param trace:The level of diagnostic tracing output to produce.
        Higher values produce more verbose output.

        :param cutoffs:
            Arguments specifying various conditions under
            which the training should be halted.  (Some of the cutoff
            conditions are not supported by some algorithms.)
            - ``max_iter=v``: Terminate after ``v`` iterations. (when using 'megam', only this is revelant)
            - ``min_ll=v``: Terminate after the negative average
              log-likelihood drops under ``v``.
            - ``min_lldelta=v``: Terminate if a single iteration improves
              log likelihood by less than ``v``.
        """
        self.uppercase_letters = uppercase_letters
        self.word_freqdist = self.gen_word_freqs(train_sents)
        self.featuresets = self.gen_featsets(train_sents, rare_word_cutoff)
        self.features_freqdist = self.gen_feat_freqs(self.featuresets)
        self.cutoff_rare_feats(self.featuresets, rare_feat_cutoff)

        self.classifier = MaxentClassifier.train(self.featuresets, algorithm, trace, **cutoffs)

    def gen_word_freqs(self, train_sents):
        word_freqdist = FreqDist()
        for tagged_sent in train_sents:
            for (word, _tag) in tagged_sent:
                word_freqdist[word] += 1
        return word_freqdist

    def gen_featsets(self, train_sents, rare_word_cutoff):
        featuresets = []
        for tagged_sent in train_sents:
            history = []
            untagged_sent = untag(tagged_sent)
            for (i, (_word, tag)) in enumerate(tagged_sent):
                featuresets.append((self.extract_feats(untagged_sent, i,
                                                    history, rare_word_cutoff), tag))
                history.append(tag)
        return featuresets

    def cutoff_rare_feats(self, featuresets, rare_feat_cutoff):
        never_cutoff_features = set(['w', 't'])

        for (feat_dict, tag) in featuresets:
            for (feature, value) in feat_dict.items():
                feat_value_tag = ((feature, value), tag)
                if self.features_freqdist[feat_value_tag] < rare_feat_cutoff:
                    if feature not in never_cutoff_features:
                        feat_dict.pop(feature)

    def gen_feat_freqs(self, featuresets):
        features_freqdist = defaultdict(int)
        for (feat_dict, tag) in featuresets:
            for (feature, value) in feat_dict.items():
                features_freqdist[((feature, value), tag)] += 1
        return features_freqdist

    def __repr__(self):
        return '<ClassifierBasedTagger: %r>' % self._classifier

    def tag(self, sentence, rare_word_cutoff=5):
        history = []
        for i in xrange(len(sentence)):
            featureset = self.extract_feats(sentence, i, history,
                                            rare_word_cutoff)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

    def extract_feats(self, sentence, i, history, rare_word_cutoff):
        features = {}
        hyphen = re.compile("-")
        number = re.compile("\d")
        uppercase = re.compile(self.uppercase_letters)

        # get features: w-1, w-2, t-1, t-2.
        # takes care of the beginning of a sentence
        if i == 0:  # first word of sentence
            features.update({"w-1": "<START>", "t-1": "<START>",
                             "w-2": "<START>", "t-2 t-1": "<START> <START>"})
        elif i == 1:  # second word of sentence
            features.update({"w-1": sentence[i - 1], "t-1": history[i - 1],
                             "w-2": "<START>",
                             "t-2 t-1": "<START> %s" % (history[i - 1])})
        else:
            features.update({"w-1": sentence[i - 1], "t-1": history[i - 1],
                             "w-2": sentence[i - 2],
                             "t-2 t-1": "%s %s" % (history[i - 2], history[i - 1])})

        # get features: w+1, w+2. takes care of the end of a sentence.
        for inc in [1, 2]:
            try:
                features["w+%i" % (inc)] = sentence[i + inc]
            except IndexError:
                features["w+%i" % (inc)] = "<END>"

        if self.word_freqdist[sentence[i]] >= rare_word_cutoff:
            # additional features for 'non-rare' words
            features["w"] = sentence[i]

        else:  # additional features for 'rare' or 'unseen' words
            features.update({"suffix(1)": sentence[i][-1:],
                             "suffix(2)": sentence[i][-2:], "suffix(3)": sentence[i][-3:],
                             "suffix(4)": sentence[i][-4:], "prefix(1)": sentence[i][:1],
                             "prefix(2)": sentence[i][:2], "prefix(3)": sentence[i][:3],
                             "prefix(4)": sentence[i][:4]})
            if hyphen.search(sentence[i]) is not None:
                # set True, if regex is found at least once
                features["contains-hyphen"] = True
            if number.search(sentence[i]) is not None:
                features["contains-number"] = True
            if uppercase.search(sentence[i]) is not None:
                features["contains-uppercase"] = True

        return features

    def classifier(self):
        """
        Return the classifier that this tagger uses to choose a tag
        for each word in a sentence.  The input for this classifier is
        generated using this tagger's feature detector.
        See ``feature_detector()``
        """
        return self.classifier
