from gym_subgoal_automata_multiagent.utils.subgoal_automaton import SubgoalAutomaton
from copy import deepcopy as dc
# from gym_subgoal_automata_multiagent.utils.subgoal_automaton import tb_goal_distance_to_reject_distance_ratio, tb_max_literals,tb_max_positive_literals, tb_max_shortest_distance_from_init ,tb_max_shortest_distance_to_reject ,tb_min_shortest_distance_to_goal, tb_positive_priority_max_literals

class AstarSearch:
    def __init__(self, automaton:SubgoalAutomaton):
        self.automaton = automaton
        self.initial_state = automaton.get_initial_state()
        self.accept_state = automaton.get_accept_state()
        self.reject_state = automaton.get_reject_state()
        self.current_state = automaton.get_initial_state()

        if self.accept_state == None:
            distance = float("inf")
        else:
            distance = self.automaton.get_distance(self.initial_state, self.accept_state, "min_distance")

        self.candidate_states = [(self.initial_state, distance)]
        self.distance_from_init = [(self.initial_state, 0)]
        self.best_states = []

    def get_next_automaton_states(self,observations):
        self._update_candidate_states(observations)
        self._update_distance_from_init(observations)
        return self._get_best_states()
    
    def get_next_automaton_states_without_updating(self,observations):
        copy_candidate_states = self._update_copy_candidate_states(observations)
        copy_distance_from_init = self._update_copy_distance_from_init(copy_candidate_states, observations)
        return self._get_future_best_states(copy_candidate_states, copy_distance_from_init)
    
    def get_current_automaton_states(self):
        return self._get_best_states()
    
    def get_candidate_states(self):
        return dc(self.candidate_states)
    
    def get_initial_state(self):
        return self.initial_state
    
    def _update_candidate_states(self, observations):
        discovered = []
        for (state,value) in self.candidate_states:
            satisfied_states = self.automaton.get_all_satisfying_states(state,observations)

            for next_state in satisfied_states:
                if self.accept_state == None:
                    if next_state == self.reject_state and self.reject_state != None:
                        next_value = float("inf")
                    elif self.reject_state == None:
                        next_value = - self.automaton.get_distance(self.initial_state, next_state, "min_distance")
                else:
                    next_value = self.automaton.get_distance(next_state, self.accept_state, "min_distance")
    
                if (next_state,next_value) not in self.candidate_states and (next_state,next_value) not in discovered:
                    discovered.append((next_state,next_value))
        
        self.candidate_states += discovered

    def _update_copy_candidate_states(self, observations):
        copy_candidate_states = dc(self.candidate_states)
        discovered = []
        for (state,value) in copy_candidate_states:
            satisfied_states = self.automaton.get_all_satisfying_states(state,observations)

            for next_state in satisfied_states:
                if self.accept_state == None:
                    if next_state == self.reject_state and self.reject_state != None:
                        next_value = float("inf")
                    elif self.reject_state == None:
                        next_value = - self.automaton.get_distance(self.initial_state, next_state, "min_distance")
                else:
                    next_value = self.automaton.get_distance(next_state, self.accept_state, "min_distance")
    
                if (next_state,next_value) not in copy_candidate_states and (next_state,next_value) not in discovered:
                    discovered.append((next_state,next_value))
        
        copy_candidate_states += discovered

        return copy_candidate_states

    def _update_distance_from_init(self,observations):
        for (state,value) in self.candidate_states:
            next_state = self.automaton.get_next_state(state,observations)
            next_value = self.automaton.get_distance(self.initial_state, next_state, "min_distance")

            if (next_state,next_value) not in self.distance_from_init:
                self.distance_from_init.append((next_state,next_value))

    def _update_copy_distance_from_init(self, copy_candidate_states, observations):
        copy_distance_from_init = dc(self.distance_from_init)
        for (state,value) in copy_candidate_states:
            next_state = self.automaton.get_next_state(state,observations)
            next_value = self.automaton.get_distance(self.initial_state, next_state, "min_distance")

            if (next_state,next_value) not in copy_distance_from_init:
                copy_distance_from_init.append((next_state,next_value))

        return copy_distance_from_init

    def _is_in_best_states(self,s):
        return s[0] in self.best_states 

    def _get_best_states(self):
        f = lambda x : x[1]
        self.candidate_states.sort(key=f)

        if self.candidate_states[0][0] == self.accept_state:
            return [self.accept_state]

        if self.candidate_states[-1][0] == self.reject_state:
            return [self.reject_state]

        min_dist = self.candidate_states[0][1]
        self.best_states = []

        for (state,value) in self.candidate_states:
            if value == min_dist:
                self.best_states.append(state)
            else:
                break

        distance_from_init_to_best_states = list(filter(self._is_in_best_states, self.distance_from_init))
        distance_from_init_to_best_states.sort(key=f)

        max_dist = distance_from_init_to_best_states[-1][1]
        best_states_with_max_distance = []

        for (state,value) in distance_from_init_to_best_states:
            if value == max_dist:
                best_states_with_max_distance.append(state)

        # print("DEBUG : best_states_with_max_distance :",best_states_with_max_distance)

        return best_states_with_max_distance
    
    def _get_future_best_states(self, copy_candidate_states, copy_distance_from_init):
        f = lambda x : x[1]
        copy_candidate_states.sort(key=f)

        if copy_candidate_states[0][0] == self.accept_state:
            return [self.accept_state]

        if copy_candidate_states[-1][0] == self.reject_state:
            return [self.reject_state]

        min_dist = copy_candidate_states[0][1]
        self.best_states = []

        for (state,value) in copy_candidate_states:
            if value == min_dist:
                self.best_states.append(state)
            else:
                break

        distance_from_init_to_best_states = list(filter(self._is_in_best_states, copy_distance_from_init))
        distance_from_init_to_best_states.sort(key=f)

        max_dist = distance_from_init_to_best_states[-1][1]
        best_states_with_max_distance = []

        for (state,value) in distance_from_init_to_best_states:
            if value == max_dist:
                best_states_with_max_distance.append(state)

        return best_states_with_max_distance
    
    def plot(self):
        self.automaton.plot("/tmp","next_state_wrong_pred.png")

    # def get_last_obs(self):
    #     return self.last_obs
