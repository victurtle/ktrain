from ...imports import *
from ... import utils as U
import torch
import numpy as np
import time


class ZeroShotClassifier():
    """
    interface to Zero Shot Topic Classifier
    """

    def __init__(self, model_name='facebook/bart-large-mnli', device=None):
        """
        interface to BART-based text summarization using transformers library

        Args:
          model_name(str): name of BART model
          device(str): device to use (e.g., 'cuda', 'cpu')
        """
        if 'mnli' not in model_name:
            raise ValueError('ZeroShotClasifier requires an MNLI model')
        try:
            import torch
        except ImportError:
            raise Exception(
                'ZeroShotClassifier requires PyTorch to be installed.')
        self.torch_device = device
        if self.torch_device is None:
            self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        from transformers import BartForSequenceClassification, BartTokenizer
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForSequenceClassification.from_pretrained(
            model_name).to(self.torch_device)
        # turn on evaluation mode which disables dropout
        self.model.eval()

    def predict(
            self,
            docs,
            hypothesis='This text is about',
            topic_strings=[],
            max_sequence_length=5,
            include_labels=False):
        """
        zero-shot topic classification
        Args:
          docs(list): list of documents
          hypothesis(str): a guiding sentence for the zero shot learner
          topic_strings(list): a list of strings representing topics of your choice
                               Example:
                               topic_strings=['political science', 'sports', 'science']
          max_sequence_length(int): maximum allowable length of doc
        Returns:
          inferred probabilities
        """
        if topic_strings is None or len(topic_strings) == 0:
            raise ValueError('topic_strings must be a list of strings')

        hypothesis += ' {}'
        true_probs = []

        # padding the docs
        return_docs = []
        padded_docs = []
        lengths = [len(self.tokenizer.encode(doc)) for doc in docs]

        for i in range(len(lengths)):
            if lengths[i] <= max_sequence_length:
                return_docs.append(docs[i])
                padded_docs.append(
                    docs[i] + ' ' + ' '.join(['<pad>'] * (max_sequence_length - lengths[i])))

        with torch.no_grad():
            for topic_string in topic_strings:
                hypothesis_ = hypothesis.format(topic_string)
                print(hypothesis_)
                input_ids = [
                    self.tokenizer.encode(
                        doc, hypothesis_, return_tensors='pt').to(
                        self.torch_device) for doc in padded_docs]
                input_ids = torch.cat(input_ids)
                logits = [torch.reshape(x, (1, 3))
                          for x in self.model(input_ids)[0]]

                # we throw away "neutral" (dim 1) and take the probability of
                # "entailment" (2) as the probability of the label being true
                # reference: https://joeddav.github.io/blog/2020/05/29/ZSL.html
                entail_contradiction_logits = [
                    logit[:, [0, 2]] for logit in logits]
                probs = [
                    entail_contradiction_logit.softmax(
                        dim=1) for entail_contradiction_logit in entail_contradiction_logits]
                true_prob = [prob[:, 1].item() for prob in probs]
                true_probs.append(true_prob)

            # transpose the 2D list
            true_probs = np.array(true_probs).T.tolist()

        if include_labels:
            true_probs = [list(zip(topic_strings, true_prob))
                          for true_prob in true_probs]
        # TO DO: beautify the return method
        return [return_docs, true_probs]
