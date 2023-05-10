# file:///home/kavin/Documents/FYP/FYP_code/multiagent-induction-subgoal-automata-rl/src/colab_utils/a_star_for_subgoal_automaton.py {"mtime":1683631219311,"ctime":1683314044506,"size":16398,"etag":"3ahdkhijmh1u","orphaned":false,"typeId":""}
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

        self.learning_rate = 0.1

        self.target_a_star_edge_q_function = {}
        self.a_star_edge_q_function = {}

        self.target_a_star_state_q_function = {}
        self.a_star_state_q_function = {state:0 for state in self.automaton.get_states()}

        self._build_a_star_q_functions()

        if self.accept_state == None:
            distance = float("inf")
        else:
            distance = self.automaton.get_distance(self.initial_state, self.accept_state, "min_distance")

        self.discovered_state_distance_to_accept = [(self.initial_state, distance)]
        self.distance_from_init = [(self.initial_state, 0)]
        self.taken_sos_tuples = []
        self.best_states = []

    def reset_taken_sos_tuples(self):
        self.taken_sos_tuples = []

    def reset_step_count(self):
        for sos_tuple in self.a_star_edge_q_function.keys():
            self.a_star_edge_q_function[sos_tuple]["steps"] = 0

    def _build_a_star_q_functions(self):
        visited = []
        discovered = [self.initial_state]

        while discovered != []:
            state = discovered.pop()
            outgoing_edges = self.automaton.get_outgoing_edges(state)
            
            for (condition, to_state) in outgoing_edges:
                if (state, condition, to_state) not in self.a_star_edge_q_function.keys():
                    self.a_star_edge_q_function[(state, condition, to_state)] = {"steps" : 0, "q_value" : 0}
            
                if to_state not in visited and to_state not in discovered:
                    discovered.append(to_state)

            visited.append(state)

        self.target_a_star_edge_q_function = dc(self.a_star_edge_q_function)
            

    def _update_a_star_edge_q_function(self, from_state, satisfied_edges, reward, observations, a_star_edge_q_function):
        sos_tuples = []
        for (condition, to_state) in satisfied_edges:
            sos_tuples.append((from_state, condition, to_state))
            if (from_state, condition, to_state) not in self.taken_sos_tuples:
                self.taken_sos_tuples.append((from_state, condition, to_state))
                # print("DEBUG : taken sos :",self.taken_sos_tuples)
                try:
                    a_star_edge_q_function[(from_state, condition, to_state)]["steps"] = 1
                except KeyError as e:
                    print("DEBUG in a_star_edge_q_function:", a_star_edge_q_function)
                    raise e

            if to_state == self.accept_state and reward == 1.0:
                a_star_edge_q_function[(from_state, condition, to_state)]["q_value"] += 1/a_star_edge_q_function[(from_state, condition, to_state)]["steps"]
            if to_state ==self.reject_state and reward == 1.0:
                a_star_edge_q_function[(from_state, condition, to_state)]["q_value"] += 0                   #-1/a_star_edge_q_function[(from_state, condition, to_state)]["steps"]

            if to_state == self.accept_state or to_state == self.reject_state:
                continue
            else:
                all_outgoing_edges_from_destination_state = self.automaton.get_outgoing_edges(to_state)

                destination_state_sos_tuples = []
                for (next_condition, next_state) in all_outgoing_edges_from_destination_state:
                    destination_state_sos_tuples.append((to_state, next_condition, next_state))

                delta = 0

                for dsos in destination_state_sos_tuples:
                    try:
                        delta += self.target_a_star_edge_q_function[dsos]["q_value"] / (len(all_outgoing_edges_from_destination_state) * a_star_edge_q_function[dsos]["steps"])
                    except ZeroDivisionError:
                        delta += self.target_a_star_edge_q_function[dsos]["q_value"] / (len(all_outgoing_edges_from_destination_state))
                    except KeyError as e:
                        raise e

                a_star_edge_q_function[(from_state, condition, to_state)]["q_value"] += self.learning_rate * (delta - self.target_a_star_edge_q_function[(from_state, condition, to_state)]["q_value"])

        self.a_star_edge_q_function = dc(a_star_edge_q_function)


    def get_updated_a_star_edge_q_functions(self):
        return dc(self.a_star_edge_q_function)
    
    def get_updated_a_star_state_q_functions(self):
        return dc(self.a_star_state_q_function)

    def get_next_automaton_states(self, reward, observations, a_star_edge_q_function, a_star_state_q_function, terminated):
        if self.target_a_star_edge_q_function == None or self.target_a_star_edge_q_function != a_star_edge_q_function:
            self.target_a_star_edge_q_function = dc(a_star_edge_q_function)

        if self.target_a_star_state_q_function == None or self.target_a_star_state_q_function != a_star_state_q_function:
            self.target_a_star_state_q_function = dc(a_star_state_q_function)

        self._update_candidate_states_and_q_functions(reward, observations, a_star_edge_q_function, terminated)

        return self._get_best_states()
    
    def get_next_automaton_states_without_updating(self, observations):
        copy_discovered_state_distance_to_accept = self._update_copy_candidate_states(observations)
        return self._get_future_best_states(copy_discovered_state_distance_to_accept)
    

    def _update_a_star_state_q_function(self):
        all_states = self.automaton.get_states()
        for state in all_states:
            if state == self.accept_state:
                self.a_star_state_q_function[state] = 0
                continue

            incoming_sos_tuple  = []

            for sos_tuple in self.target_a_star_edge_q_function.keys():
                if sos_tuple[2] == state:
                    incoming_sos_tuple.append(sos_tuple)

            square_sum_q_value = 0
            sum_q_value = 0
            # sum_steps = 0
            for sos_tuple in incoming_sos_tuple:
                values = self.target_a_star_edge_q_function[sos_tuple]
                square_sum_q_value += values["q_value"]**2 #* values["steps"]
                sum_q_value += values["q_value"]

            try:
                self.a_star_state_q_function[state] += self.learning_rate * (square_sum_q_value/sum_q_value - self.a_star_state_q_function[state])
            except ZeroDivisionError:
                self.a_star_state_q_function[state] += self.learning_rate * (square_sum_q_value - self.a_star_state_q_function[state])


    def get_current_automaton_states(self):
        return self._get_best_states()
    
    def get_discovered_state_distance_to_accept(self):
        return dc(self.discovered_state_distance_to_accept)
    
    def get_initial_state(self):
        return self.initial_state
    
    def _update_candidate_states_and_q_functions(self, reward, observations, a_star_edge_q_function, terminated):
        if terminated:
            return
        
        discovered = []

        for sos_tuple in self.taken_sos_tuples:
            try:
                a_star_edge_q_function[sos_tuple]["steps"] += 1
            except KeyError as e:
                print("DEBUG self.taken_sos_tuples :", self.taken_sos_tuples)
                print("DEBUG a_star_edge_q_function :", a_star_edge_q_function)
                raise e

        for (state,distance) in self.discovered_state_distance_to_accept:
            satisfied_states = self.automaton.get_all_satisfying_states(state,observations)
            satisfied_edges = self.automaton.get_all_satisfying_edges(state, observations)
            # all_edges = self.automaton.get_outgoing_edges(state)

            self._update_a_star_edge_q_function(state, satisfied_edges, reward, observations, a_star_edge_q_function)

            for next_state in satisfied_states:
                if self.accept_state == None:
                    if next_state == self.reject_state and self.reject_state != None:
                        next_value = float("inf")
                    elif self.reject_state == None:
                        next_value = - self.automaton.get_distance(self.initial_state, next_state, "min_distance")
                else:
                    next_value = self.automaton.get_distance(next_state, self.accept_state, "min_distance")
    
                if (next_state,next_value) not in self.discovered_state_distance_to_accept and (next_state,next_value) not in discovered:
                    discovered.append((next_state,next_value))
        
        self.discovered_state_distance_to_accept += discovered

    def _update_copy_candidate_states(self, observations):
        copy_discovered_state_distance_to_accept = dc(self.discovered_state_distance_to_accept)
        discovered = []
        for (state,distance) in copy_discovered_state_distance_to_accept:
            satisfied_states = self.automaton.get_all_satisfying_states(state,observations)

            for next_state in satisfied_states:
                if self.accept_state == None:
                    if next_state == self.reject_state and self.reject_state != None:
                        next_value = float("inf")
                    elif self.reject_state == None:
                        next_value = - self.automaton.get_distance(self.initial_state, next_state, "min_distance")
                else:
                    next_value = self.automaton.get_distance(next_state, self.accept_state, "min_distance")
    
                if (next_state,next_value) not in copy_discovered_state_distance_to_accept and (next_state,next_value) not in discovered:
                    discovered.append((next_state,next_value))
        
        copy_discovered_state_distance_to_accept += discovered

        return copy_discovered_state_distance_to_accept

    # def _update_distance_from_init(self, observations, terminated):
    #     if terminated:
    #         return
        
    #     for (state,value) in self.discovered_state_distance_to_accept:
    #         next_state = self.automaton.get_next_state(state,observations)
    #         next_value = self.automaton.get_distance(self.initial_state, next_state, "min_distance")

    #         if (next_state,next_value) not in self.distance_from_init:
    #             self.distance_from_init.append((next_state,next_value))

    def _update_copy_distance_from_init(self, copy_discovered_state_distance_to_accept, observations):
        copy_distance_from_init = dc(self.distance_from_init)
        for (state,value) in copy_discovered_state_distance_to_accept:
            next_state = self.automaton.get_next_state(state,observations)
            next_value = self.automaton.get_distance(self.initial_state, next_state, "min_distance")

            if (next_state,next_value) not in copy_distance_from_init:
                copy_distance_from_init.append((next_state,next_value))

        return copy_distance_from_init

    def _is_in_best_states(self,s):
        return s[0] in self.best_states 

    def _get_best_states(self):
        f = lambda x : x[1]
        self.discovered_state_distance_to_accept.sort(key = f)

        if self.discovered_state_distance_to_accept[0][0] == self.accept_state:
            return [self.accept_state]

        if self.discovered_state_distance_to_accept[-1][0] == self.reject_state:
            return [self.reject_state]

        a_star_q_value_heuristic_filtered_states = []
        self.best_states = []

        q_value_heuristic_state_value_items =[]

        a_star_q_value_heuristic_state_value_items_candidates = list(self.a_star_state_q_function.items())

        cand_states_list = [c[0] for c in self.discovered_state_distance_to_accept]

        for cand in a_star_q_value_heuristic_state_value_items_candidates:
            if cand[0] in cand_states_list:
                q_value_heuristic_state_value_items.append(cand)

        q_value_heuristic_state_value_items.sort(key=f)

        max_cand_q_value = q_value_heuristic_state_value_items[-1][1]

        for item in q_value_heuristic_state_value_items:
            if item[1] == max_cand_q_value:
                a_star_q_value_heuristic_filtered_states.append(item[0])

        if len(a_star_q_value_heuristic_filtered_states)<=1:
            return a_star_q_value_heuristic_filtered_states
        
        self.best_states = a_star_q_value_heuristic_filtered_states

        distance_to_accept_to_best_states = list(filter(self._is_in_best_states, self.discovered_state_distance_to_accept))
        distance_to_accept_to_best_states.sort(key=f)

        try:
            min_dist = distance_to_accept_to_best_states[0][1]
        except IndexError as e:
            print("DEBUG ind err :", self.discovered_state_distance_to_accept, self.best_states)
        best_states_with_min_distance = []

        for (state,value) in distance_to_accept_to_best_states:
            if value == min_dist:
                best_states_with_min_distance.append(state)
            else:
                break

        return best_states_with_min_distance
    
    def _get_future_best_states(self, copy_discovered_state_distance_to_accept):
        f = lambda x : x[1]
        copy_discovered_state_distance_to_accept.sort(key=f)

        if copy_discovered_state_distance_to_accept[0][0] == self.accept_state:
            return [self.accept_state]

        if copy_discovered_state_distance_to_accept[-1][0] == self.reject_state:
            return [self.reject_state]

        a_star_q_value_heuristic_filtered_states = []
        self.best_states = []

        q_value_heuristic_state_value_items =[]

        a_star_q_value_heuristic_state_value_items_candidates = list(self.a_star_state_q_function.items())

        cand_states_list = [c[0] for c in copy_discovered_state_distance_to_accept]

        for cand in a_star_q_value_heuristic_state_value_items_candidates:
            if cand[0] in cand_states_list:
                q_value_heuristic_state_value_items.append(cand)

        q_value_heuristic_state_value_items.sort(key=f)

        max_cand_q_value = q_value_heuristic_state_value_items[-1][1]

        for item in q_value_heuristic_state_value_items:
            if item[1] == max_cand_q_value:
                a_star_q_value_heuristic_filtered_states.append(item[0])

        if len(a_star_q_value_heuristic_filtered_states)<=1:
            return a_star_q_value_heuristic_filtered_states
        
        self.best_states = a_star_q_value_heuristic_filtered_states

        distance_to_accept_to_best_states = list(filter(self._is_in_best_states, copy_discovered_state_distance_to_accept))
        distance_to_accept_to_best_states.sort(key=f)

        try:
            min_dist = distance_to_accept_to_best_states[0][1]
        except IndexError as e:
            print("DEBUG ind err :", self.discovered_state_distance_to_accept, self.best_states)
        best_states_with_min_distance = []

        for (state,value) in distance_to_accept_to_best_states:
            if value == min_dist:
                best_states_with_min_distance.append(state)
            else:
                break

        return best_states_with_min_distance
    
    def plot(self):
        self.automaton.plot("/tmp","next_state_wrong_pred.png")
