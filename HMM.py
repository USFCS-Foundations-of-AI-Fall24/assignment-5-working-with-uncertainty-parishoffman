

import random
import argparse
import codecs
import os
import numpy

# Sequence - represents a sequence of hidden states and corresponding
# output variables.

class Sequence:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# HMM model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities
        e.g. {'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
              'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
              'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}"""



        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""

        def load_file(dict, filename) :
            with open(filename, "r") as f:
                lines = f.readlines()
                for line in lines :
                    source, target, score = line.strip().split()
                    if source not in dict:
                        dict[source] = {}
                    dict[source][target] = float(score)

        load_file(self.transitions, f'{basename}.trans')
        load_file(self.emissions, f'{basename}.emit')


   ## you do this.
    def generate(self, n):
        """return an n-length Sequence by randomly sampling from this HMM."""
        sequence = []
        state = "#"

        for i in range(n):
            if state not in self.transitions:
                continue

            next_states = list(self.transitions[state].keys())
            next_probs = list(self.transitions[state].values())
            state = random.choices(next_states, weights=next_probs, k=1)[0]

            if state not in self.emissions:
                raise ValueError(f"No emission probabilities defined for state '{state}'")

            emissions = list(self.emissions[state].keys())
            emission_probs = list(self.emissions[state].values())
            emission = random.choices(emissions, weights=emission_probs, k=1)[0]

            sequence.append(emission)

        return sequence

    def forward(self, sequence):
        pass
    ## you do this: Implement the Viterbi algorithm. Given a Sequence with a list of emissions,
    ## determine the most likely sequence of states.






    def viterbi(self, sequence):
        pass
    ## You do this. Given a sequence with a list of emissions, fill in the most likely
    ## hidden states using the Viterbi algorithm.



def main():
    hmm = HMM()
    hmm.load('cat')
    print(hmm.generate(20))
main()



