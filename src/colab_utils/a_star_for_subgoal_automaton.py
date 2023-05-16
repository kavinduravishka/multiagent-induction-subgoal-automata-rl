# file:///home/kavin/Documents/FYP/FYP_code/multiagent-induction-subgoal-automata-rl/src/colab_utils/a_star_for_subgoal_automaton.py {"mtime":1683631219311,"ctime":1683314044506,"size":16398,"etag":"3ahdkhijmh1u","orphaned":false,"typeId":""}
from gym_subgoal_automata_multiagent.utils.subgoal_automaton import SubgoalAutomaton
from gym_subgoal_automata_multiagent.utils.condition import EdgeCondition
from utils.utils import randargmax, randargmin
from collections import namedtuple
from copy import deepcopy as dc
# from gym_subgoal_automata_multiagent.utils.subgoal_automaton import tb_goal_distance_to_reject_distance_ratio, tb_max_literals,tb_max_positive_literals, tb_max_shortest_distance_from_init ,tb_max_shortest_distance_to_reject ,tb_min_shortest_distance_to_goal, tb_positive_priority_max_literals

ChildEdge = namedtuple("ChildEdge", ["condition", "state"])
ParentEdge = namedtuple("ParentEdge", ["condition", "state"])
class TransitionNode:

    def __init__(self, state):
        self.state:str = state 
        self.parent:TransitionNode = None
        self.parent_edge:ParentEdge = None
        self.children_edges:list[ChildEdge] = []
        self.children: list[TransitionNode] = []
        self.steps_after_taken_edge = {}
        # for edge in children_edges:
            

    def add_child(self, child, edge_condition):
        edge = ChildEdge(edge_condition, child.state)
        if edge not in self.children_edges:
            self.children_edges.append(edge)
            self.steps_after_taken_edge[edge_condition] = 0

        if child not in self.children:
            self.children.append(child)

        child.parent = self
        child.parent_edge = ParentEdge(edge_condition, self)

    def add_parent(self, parent, edge_condition):
        self.parent:TransitionNode = parent
        if parent != None:
            self.parent.add_child(self, edge_condition)
            self.parent_edge = ParentEdge(edge_condition, parent)

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

        self._build_a_star_edge_q_functions()

        self.discovered_states = [self.initial_state]
        if self.accept_state == None:
            distance = float("inf")
        else:
            distance = self.automaton.get_distance(self.initial_state, self.accept_state, "min_distance")
        self.discovered_state_distance_to_accept = [(self.initial_state, distance)]

        self.discovered_state_distance_from_init = [(self.initial_state, 0)]
        # self.distance_from_init = [(self.initial_state, 0)]

        self.tt_root = TransitionNode(self.initial_state)

        self.state_transition_node_mapping = {self.initial_state: [self.tt_root]}
        
        # self.discovered_sos_tuples = []
        # self.best_states = []

        self.state_transition_tree = self.tt_root

    # def reset_discovered_sos_tuples(self):
    #     self.discovered_sos_tuples = []

    def update_transition_tree(self, observations):
        newly_discovered_states = []

        for state in self.discovered_states:
            satisfied_states = self.automaton.get_all_satisfying_states(state, observations)
            satisfied_edges = self.automaton.get_all_satisfying_edges(state, observations)

            for node in self.state_transition_node_mapping[state]:
                for (condition, child_state) in satisfied_edges:
                    if (condition, child_state) in [(e.condition, e.state) for e in node.children_edges]:
                        continue
                    child_tt_node = TransitionNode(child_state)

                    child_tt_node.add_parent(node, condition)

                    try:
                        self.state_transition_node_mapping[child_state].append(child_tt_node)
                    except KeyError:
                        self.state_transition_node_mapping[child_state] = []
                        self.state_transition_node_mapping[child_state].append(child_tt_node)

            for child_state in satisfied_states:
                if child_state not in self.discovered_states and child_state not in newly_discovered_states:
                    newly_discovered_states.append(child_state) 
                    self._update_distance_to_accept(child_state)
                    self._update_distance_from_init(child_state)

        self.discovered_states += newly_discovered_states

        self._update_transistion_tree_step_count()

    def _update_transistion_tree_step_count(self):
        visited = []
        discovered = [self.state_transition_tree]

        while discovered != []:
            node = discovered.pop()
            outgoing_edges = node.children_edges

            for edge in outgoing_edges:
                codition = edge.condition
                node.steps_after_taken_edge[codition] += 1

            visited.append(node)

            for child_node in node.children:
                if child_node not in discovered and child_node not in visited:
                    discovered.append(child_node)

    def update_a_star_edge_q_functions(self, reward, is_terminal, target_a_star_edge_q_function):
        self.a_star_edge_q_function = target_a_star_edge_q_function
        self.target_a_star_edge_q_function = target_a_star_edge_q_function

        if not is_terminal or reward != 1.0:
            # print("DEBUG 1.0 : reward :", reward)
            return dc(self.a_star_edge_q_function)

        # Update edge q values up the transition tree

        leaves:list[TransitionNode] = []

        for state in self.discovered_states:
            for node in self.state_transition_node_mapping[state]:
                if node.children == [] or node.children == None: # and node.state == self.accept_state:
                    leaves.append(node)

        leaves.sort(key= lambda x : self._get_branch_depth(x))

        if self.accept_state != None:
            max_depth = self.automaton.get_distance(self.initial_state, self.accept_state, "min_distance")
        else:
            max_depth = self._get_branch_depth(leaves[-1])

        for leaf in leaves:
            # if self._get_branch_depth(leaf) != max_depth:
            #     continue

            
            parent = leaf.parent
            while parent != None:
                leaf_to_root_depth = self._get_branch_depth(leaf)

                factor = max([0, (max_depth - leaf_to_root_depth)]) + 1
                
                to_state = leaf.state
                condition = leaf.parent_edge.condition
                from_state = parent.state

                sos_tuple = (from_state, condition, to_state)

                if to_state == self.accept_state and reward == 1.0:
                    self.a_star_edge_q_function[sos_tuple] += self.learning_rate ** factor * (1 - self.target_a_star_edge_q_function[sos_tuple])/parent.steps_after_taken_edge[condition]
                elif to_state == self.reject_state:
                    pass
                else:
                    if len(leaf.children_edges) > 0:
                        child_edge_q_value_sum = 0
                        child_edge_q_value_square_sum = 0
                        child_edges = leaf.children_edges
                        for c_edge in child_edges:
                            c_sos_tuple = (leaf.state, c_edge.condition, c_edge.state)
                            child_edge_q_value_sum += self.target_a_star_edge_q_function[c_sos_tuple]
                            child_edge_q_value_square_sum += self.target_a_star_edge_q_function[c_sos_tuple] ** 2
                        try:
                            self.a_star_edge_q_function[sos_tuple] += self.learning_rate ** factor * (child_edge_q_value_square_sum/child_edge_q_value_sum - self.target_a_star_edge_q_function[sos_tuple])/parent.steps_after_taken_edge[condition]
                        except ZeroDivisionError:
                            self.a_star_edge_q_function[sos_tuple] += self.learning_rate ** factor * (0 - self.target_a_star_edge_q_function[sos_tuple])/parent.steps_after_taken_edge[condition]
                    else:
                        pass
                
                parent = leaf.parent.parent
                leaf = leaf.parent
                leaf_to_root_depth += 1

        # print("DEBUG : self.a_star_edge_q_function :",self.a_star_edge_q_function)
        return dc(self.a_star_edge_q_function)
    
    def _get_branch_depth(self, leaf):
        depth = 0
        while leaf.parent != None:
            depth += 1
            leaf = leaf.parent

        return depth

    def update_a_star_state_q_functions(self, target_a_star_state_q_function):
        self.a_star_state_q_function = target_a_star_state_q_function
        self.target_a_star_state_q_function = target_a_star_state_q_function

        # Update state q values up the transition tree

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

            for sos_tuple in incoming_sos_tuple:
                value = self.target_a_star_edge_q_function[sos_tuple]
                square_sum_q_value += value ** 2
                sum_q_value += value

            try:
                # print("DEBUG : updated state q")
                self.a_star_state_q_function[state] += self.learning_rate * (square_sum_q_value/sum_q_value - self.target_a_star_state_q_function[state])
            except ZeroDivisionError:
                self.a_star_state_q_function[state] += self.learning_rate * (square_sum_q_value - self.target_a_star_state_q_function[state])

        # if sum(self.a_star_state_q_function.values()) > 0.0:self.a_star_state_q_function.values()))
        return dc(self.a_star_state_q_function)


    def _update_distance_to_accept(self, state):
        if self.accept_state == None:
            if state == self.reject_state and self.reject_state != None:
                next_value = float("inf")
            elif self.reject_state == None:
                next_value = - self.automaton.get_distance(self.initial_state, state, "min_distance")
        else:
            next_value = self.automaton.get_distance(state, self.accept_state, "min_distance")

        if (state,next_value) not in self.discovered_state_distance_to_accept:
            self.discovered_state_distance_to_accept.append((state,next_value))
    
    def _update_distance_from_init(self, state):
        next_value = self.automaton.get_distance(self.initial_state, state, "max_distance")

        if (state,next_value) not in self.discovered_state_distance_to_accept:
            self.discovered_state_distance_from_init.append((state,next_value))

    def get_current_automaton_states(self, current_automaton_state):
        return self._get_best_states(current_automaton_state)
    
    # def get_discovered_state_distance_to_accept(self):
    #     return dc(self.discovered_state_distance_to_accept)
    
    def get_initial_state(self):
        return self.initial_state
    
    def get_updated_a_star_edge_q_functions(self):
        return dc(self.a_star_edge_q_function)
    
    def get_updated_a_star_state_q_functions(self):
        return dc(self.a_star_state_q_function)
    
    def _build_a_star_edge_q_functions(self):
        visited = []
        discovered = [self.initial_state]

        while discovered != []:
            state = discovered.pop()
            outgoing_edges = self.automaton.get_outgoing_edges(state)
            
            for (condition, to_state) in outgoing_edges:
                if (state, condition, to_state) not in self.a_star_edge_q_function.keys():
                    self.a_star_edge_q_function[(state, condition, to_state)] = 0
            
                if to_state not in visited and to_state not in discovered:
                    discovered.append(to_state)

            visited.append(state)

        self.target_a_star_edge_q_function = dc(self.a_star_edge_q_function)


    def _get_best_states(self, current_automaton_state):
        f = lambda x : x[1]
        self.discovered_state_distance_from_init.sort(key = f)
        self.discovered_state_distance_to_accept.sort(key = f)

        if self.discovered_state_distance_to_accept[0][0] == self.accept_state:
            return [self.accept_state]
        if self.discovered_state_distance_to_accept[-1][0] == self.reject_state:
            return [self.reject_state]
        
        leaves:list[TransitionNode] = []

        for state in self.discovered_states:
            for node in self.state_transition_node_mapping[state]:
                if node.children == [] or node.children == None:
                    leaves.append(node)

        max_dist_from_init = self.discovered_state_distance_from_init[-1][1]

        if (current_automaton_state, max_dist_from_init) in self.discovered_state_distance_from_init:
            return [current_automaton_state]
        elif any((child_state, max_dist_from_init) in self.discovered_state_distance_from_init for child_state in self.automaton.get_outgoing_to_states(current_automaton_state)):
            child_nodes_in_tt = []
            current_state_nodes = self.state_transition_node_mapping[current_automaton_state]
            for pn in current_state_nodes:
                for cn in pn.children:
                    if cn.state in [l.state for l in leaves] and (cn.state, max_dist_from_init) in self.discovered_state_distance_from_init and cn.state not in child_nodes_in_tt:
                        child_nodes_in_tt.append(cn.state)

            q_values_child_nodes_in_tt = [self.a_star_state_q_function[state] for state in child_nodes_in_tt]
            try:
                return [child_nodes_in_tt[randargmax(q_values_child_nodes_in_tt)]]
            except ValueError as e:
                pass

        best_states = []

        cand_states_list = list(set([leaf.state for leaf in leaves]))

        try:
            leaves_min_dist_from_init = min([distance for (state,distance) in self.discovered_state_distance_from_init if state in cand_states_list])
        except ValueError:
            return [current_automaton_state]

        for (state, distance) in self.discovered_state_distance_from_init:
            if state in cand_states_list and distance == leaves_min_dist_from_init:
                best_states.append(state)

        q_values_best_states = [self.a_star_state_q_function[state] for state in best_states]

        return [best_states[randargmax(q_values_best_states)]]


