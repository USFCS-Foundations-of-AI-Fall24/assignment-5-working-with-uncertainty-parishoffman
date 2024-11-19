

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
        observations = list(set(sequence))
        num_timesteps = len(observations)
        forward_probs = {t: {state: 0.0 for state in sequence} for t in range(1, num_timesteps + 1)}

        for state in sequence:
            transition_prob = self.transitions.get(sequence[0], {}).get(state, 0.0)
            emission_prob = self.emissions.get(state, {}).get(observations[0], 0.0)
            forward_probs[1][state] = transition_prob * emission_prob

        for time_step in range(2, num_timesteps + 1):
            for current_state in sequence:
                forward_probs[time_step][current_state] = sum(
                    forward_probs[time_step - 1][previous_state] *
                    self.transitions.get(previous_state, {}).get(current_state, 0.0) *
                    self.emissions.get(current_state, {}).get(observations[time_step - 1], 0.0)
                    for previous_state in sequence
                )

        return forward_probs
    ## you do this: Implement the Viterbi algorithm. Given a Sequence with a list of emissions,
    ## determine the most likely sequence of states.

    def viterbi(self, sequence):
        O = list(set(sequence))
        t = len(O)
        states = sequence
        T = self.transitions
        E = self.emissions

        M = {time: {state: 0 for state in states} for time in range(1, t + 1)}
        Backpointers = {time: {state: None for state in states} for time in range(1, t + 1)}

        for s in states:
            if states[0] not in T or s not in E:
                continue
            M[1][s] = (T[states[0]].get(s, 0.0) * E[s].get(O[0], 0.0))
            Backpointers[1][s] = None

        for i in range(2, t + 1):
            for s in states:
                try :
                    max_prob, best_prev_state = max(
                        ((M[i - 1][s2] * T[s2].get(s, 0.0) * E[s].get(O[i - 1], 0.0), s2)
                         for s2 in states),
                        key=lambda x: x[0]
                    )
                    M[i][s] = max_prob
                    Backpointers[i][s] = best_prev_state
                except Exception as e:
                    continue

        final_state = max(M[t], key=M[t].get)
        best_path = [final_state]

        for i in range(t, 1, -1):
            if best_path[0] not in Backpointers[i]:
                continue
            best_path.insert(0, Backpointers[i][best_path[0]])

        return best_path
    ## You do this. Given a sequence with a list of emissions, fill in the most likely
    ## hidden states using the Viterbi algorithm

def main():
    parser = argparse.ArgumentParser(description="HMM.py")
    parser.add_argument("basename")
    parser.add_argument("--generate")
    parser.add_argument("--forward")
    parser.add_argument("--viterbi")
    args = parser.parse_args()

    hmm = HMM()
    hmm.load(args.basename)

    if args.generate:
        sequence = hmm.generate(int(args.generate))
        print(" ".join(sequence))
    elif args.forward:
        final_state = hmm.forward(args.forward)
        print(final_state)
    elif args.viterbi:
        state_paths = hmm.viterbi(args.viterbi)
        print(state_paths)

main()

# code to generate lander.trans
# def generate_transitions(grid_size):
#     transitions = []
#
#     for row in range(1, grid_size + 1):
#         for col in range(1, grid_size + 1):
#             current = f"{row},{col}"
#
#             diag_down_right = (min(row + 1, grid_size), min(col + 1, grid_size))
#             right = (row, min(col + 1, grid_size))
#             down = (min(row + 1, grid_size), col)
#
#             transitions.append(f"{current} {diag_down_right[0]},{diag_down_right[1]} 0.7")
#             transitions.append(f"{current} {right[0]},{right[1]} 0.15")
#             transitions.append(f"{current} {down[0]},{down[1]} 0.15")
#
#     return transitions
#
#
# grid_size = 5
# transitions = generate_transitions(grid_size)
#
# with open("lander.trans", "w") as file:
#     for transition in transitions:
#         file.write(transition + "\n")


# code to generate lander.emit
# def generate_emissions(grid_size):
#     emissions = []
#
#     for row in range(1, grid_size + 1):
#         for col in range(1, grid_size + 1):
#             current = f"{row},{col}"
#
#             emissions.append(f"{current} {row},{col} 0.6")
#
#             neighbors = {
#                 "up": (max(row - 1, 1), col),
#                 "down": (min(row + 1, grid_size), col),
#                 "left": (row, max(col - 1, 1)),
#                 "right": (row, min(col + 1, grid_size)),
#             }
#
#             for direction, (n_row, n_col) in neighbors.items():
#                 emissions.append(f"{current} {n_row},{n_col} 0.1")
#
#     return emissions
#
# grid_size = 5
# emissions = generate_emissions(grid_size)
#
# with open("lander.emit", "w") as file:
#     for emission in emissions:
#         file.write(emission + "\n")





