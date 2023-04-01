from copy import deepcopy as dc

from gym_subgoal_automata_multiagent.utils.subgoal_automaton import SubgoalAutomaton
from gym_subgoal_automata_multiagent.utils.merged_subgoal_automaton import MergedSubgoalAutomaton
from gym_subgoal_automata_multiagent.utils.condition import EdgeCondition


def merge_automata(automata1:SubgoalAutomaton,automata2:SubgoalAutomaton):
    initial_1 = automata1.get_initial_state()
    initial_2 = automata2.get_initial_state()

    initial = (initial_1,initial_2)

    accept_1 = automata1.get_accept_state()
    accept_2 = automata2.get_accept_state()

    accept = (accept_1, accept_2)

    reject_1 = automata1.get_reject_state()
    reject_2 = automata2.get_reject_state()

    reject = (reject_1, reject_2)

    states_to_merge = [(initial_1, initial_2)]

    state_mapping = {(1, initial_1): initial_1,
                    #  (1, accept_1): accept_1,
                    #  (1, reject_1): reject_1,
                     (2, initial_2): initial_1,
                    #  (2, accept_2): accept_1,
                    #  (2, reject_2): reject_1,
                    }
    
    if accept_1 != None:
        state_mapping[(1, accept_1)] = accept_1
    if accept_2 != None:
        if accept_1 != None:
            state_mapping[(2, accept_2)] = accept_1
        else:
            state_mapping[(2, accept_2)] = accept_2

    if reject_1 != None:
        state_mapping[(1, reject_1)] = reject_1
    if reject_2 != None:
        if reject_1 != None:
            state_mapping[(2, reject_2)] = reject_1
        else:
            state_mapping[(2, reject_2)] = reject_2

    new_automata = SubgoalAutomaton()

    visited_1 = []
    visited_2 = []

    discovered_1 = []
    discovered_2 = []

    new_automaton_state_counter = 0

    _set_automata_IAR_states(initial , accept , reject, new_automata, state_mapping)

    while states_to_merge != []:
        pair_to_merge = states_to_merge.pop()

        s_A1 = pair_to_merge[0]
        s_A2 = pair_to_merge[1]

        s_disc_1 = automata1.get_outgoing_to_states(s_A1)
        s_disc_2 = automata2.get_outgoing_to_states(s_A2)

        visited_1.append(s_A1)
        visited_2.append(s_A2)

        if s_A1 in discovered_1:
            discovered_1.remove(s_A1)
        if s_A2 in discovered_2:
            discovered_2.remove(s_A2)

        discovered_1 += list(set(s_disc_1) - set(visited_1))
        discovered_2 += list(set(s_disc_2) - set(visited_2))

        edges_1 = automata1.get_outgoing_edges(s_A1)
        edges_2 = automata2.get_outgoing_edges(s_A2)

        edge_data_1 = []
        edge_data_2 = []

        for e1 in edges_1:
            edge_data_1.append({"from":s_A1, "to":e1[1], "pos":set(_extract_positive_literals(e1)), "neg":set(_extract_negative_literals(e1))})
        
        for e2 in edges_2:
            edge_data_2.append({"from":s_A2, "to":e2[1], "pos":set(_extract_positive_literals(e2)), "neg":set(_extract_negative_literals(e2))})

        edges_not_merged_1 = dc(edge_data_1)
        edges_not_merged_2 = dc(edge_data_2)

        # states_to_merge.append(((s_A1, s_A2), ()))

        for e1 in edge_data_1:
            for e2 in edge_data_2:
                if _mergable(e1, e2, initial , accept , reject):
                    new_automata, new_automaton_state_counter, state_mapping = _merge(e1, e2, new_automata, new_automaton_state_counter, state_mapping)
                    states_to_merge.append((e1["to"], e2["to"]))
                    try:
                        edges_not_merged_1.remove(e1)
                    except ValueError:
                        pass
                    try:
                        edges_not_merged_2.remove(e2)
                    except ValueError:
                        pass
                
        for e in edges_not_merged_1:
            new_automata, new_automaton_state_counter, state_mapping = _add_unmergable(e, 1, new_automata, new_automaton_state_counter, state_mapping)

        for e in edges_not_merged_2:
            new_automata, new_automaton_state_counter, state_mapping = _add_unmergable(e, 2, new_automata, new_automaton_state_counter, state_mapping)

    while discovered_1 != []:
        s_A1 = discovered_1.pop()
        visited_1.append(s_A1)

        s_disc_1 = automata1.get_outgoing_to_states(s_A1)

        discovered_1 += list(set(s_disc_1) - set(visited_1))

        edges_1 = automata1.get_outgoing_edges(s_A1)
        edge_data_1 = []

        for e1 in edges_1:
            edge_data_1.append({"from":s_A1, "to":e1[1], "pos":set(_extract_positive_literals(e1)), "neg":set(_extract_negative_literals(e1)) })

        edges_not_merged_1 = dc(edge_data_1)

        for e in edges_not_merged_1:
            new_automata, new_automaton_state_counter, state_mapping = _add_unmergable(e, 1, new_automata, new_automaton_state_counter, state_mapping)
    
    while discovered_2 != []:
        s_A2 = discovered_2.pop()
        visited_2.append(s_A2)

        s_disc_2 = automata2.get_outgoing_to_states(s_A2)
        discovered_2 += list(set(s_disc_2) - set(visited_2))
        edges_2 = automata2.get_outgoing_edges(s_A2)
        edge_data_2 = []

        for e2 in edges_2:
            edge_data_2.append({"from":s_A2, "to":e2[1], "pos":set(_extract_positive_literals(e2)), "neg":set(_extract_negative_literals(e2)) })

        edges_not_merged_2 = dc(edge_data_2)

        for e in edges_not_merged_2:
            new_automata, new_automaton_state_counter, state_mapping = _add_unmergable(e, 2, new_automata, new_automaton_state_counter, state_mapping)

    return new_automata

def _extract_positive_literals(edge):
    edge_condition = edge[0]
    return edge_condition.get_positive_conditions()

def _extract_negative_literals(edge):
    edge_condition = edge[0]
    return edge_condition.get_negative_conditions()

def _mergable(e1, e2, initial, accept, reject):
    if e1["pos"] != e2["pos"] or e1["neg"] != e2["neg"]:
        return False
    
    if e1["to"]==e2["to"]:
        return True
    else:
        if (e1["from"], e2["from"]) == initial or (e1["to"], e2["to"]) == accept or (e1["to"], e2["to"]) == reject:
            return True
        else:
            return False
    
def _set_automata_IAR_states(initial , accept , reject, new_automata:SubgoalAutomaton, state_mapping:dict):
    if new_automata.initial_state == None:
        assert state_mapping[(1, initial[0])] == state_mapping[(2, initial[1])]
        new_automata.set_initial_state(state_mapping[(1, initial[0])])
    
    if new_automata.accept_state == None:
        if accept[0] == None and accept[1] == None:
            pass
        elif accept[0] != None and  accept[1] == None:
            new_automata.set_accept_state(state_mapping[(1, accept[0])])
        elif accept[0] == None and  accept[1] != None:
            new_automata.set_accept_state(state_mapping[(2, accept[1])])
        else:
            assert state_mapping[(1, accept[0])] == state_mapping[(2, accept[1])]
            new_automata.set_accept_state(state_mapping[(1, accept[0])])

    if new_automata.reject_state == None:
        if reject[0] == None and reject[1] == None:
            pass
        elif reject[0] != None and  reject[1] == None:
            new_automata.set_reject_state(state_mapping[(1, reject[0])])
        elif reject[0] == None and  reject[1] != None:
            new_automata.set_reject_state(state_mapping[(2, reject[1])])
        else:
            assert state_mapping[(1, reject[0])] == state_mapping[(2, reject[1])]
            new_automata.set_reject_state(state_mapping[(1, reject[0])])

def _merge(e1, e2, new_automata:SubgoalAutomaton, new_automaton_state_counter:int, state_mapping:dict):
    e1_from = (1, e1["from"])
    e2_from = (2, e2["from"])

    e1_to = (1, e1["to"])
    e2_to = (2, e2["to"])

    if e1_from in state_mapping.keys() and e2_from not in state_mapping.keys():
        new_from = state_mapping[e1_from]
        state_mapping[e2_from] = new_from
    elif e2_from in state_mapping.keys() and e1_from not in state_mapping.keys():
        new_from = state_mapping[e2_from]
        state_mapping[e1_from] = new_from
    elif e1_from in state_mapping.keys() and e2_from in state_mapping.keys():
        if state_mapping[e1_from] != state_mapping[e2_from]:
            raise ValueError("Two states which are being merged can't have two distinct names in the new SubgoalAutomata ")
        else:
            new_from = state_mapping[e1_from]
    else:
        new_automaton_state_counter += 1
        new_from = "u"+str(new_automaton_state_counter)

        state_mapping[e1_from] = new_from
        state_mapping[e2_from] = new_from

    if e1_to in state_mapping.keys() and e2_to not in state_mapping.keys():
        new_to = state_mapping[e1_to]
        state_mapping[e2_to] = new_to
    elif e2_to in state_mapping.keys() and e1_to not in state_mapping.keys():
        new_to = state_mapping[e2_to]
        state_mapping[e1_to] = new_to
    elif e1_to in state_mapping.keys() and e2_to in state_mapping.keys():
        if state_mapping[e1_to] != state_mapping[e2_to]:
            raise ValueError("Two states which are being merged can't have two distinct names in the new SubgoalAutomata ")
        else:
            new_to = state_mapping[e1_to]
    else:
        new_automaton_state_counter += 1
        new_to = "u"+str(new_automaton_state_counter)

        state_mapping[e1_to] = new_to
        state_mapping[e2_to] = new_to

    new_pos = set.union(e1["pos"], e2["pos"])
    new_neg = set.union(e1["neg"], e2["neg"])

    new_edge_condition = _new_edge_condition(new_pos, new_neg)

    if new_from not in new_automata.get_states():
        new_automata.add_state(new_from)

    if new_to not in new_automata.get_states():
        new_automata.add_state(new_to)

    new_automata.add_edge(new_from,new_to,new_edge_condition)

    return new_automata, new_automaton_state_counter, state_mapping

def _new_edge_condition(pos, neg):
    pos_tuple = _get_positive_conditions_tuple(pos)
    neg_tuple = _get_negative_conditions_tuple(neg)
    condition_tuple = tuple(sorted(pos_tuple + neg_tuple))
    return condition_tuple

def _get_positive_conditions_tuple(pos):
    condition = pos
    return tuple(sorted(condition))

def _get_negative_conditions_tuple(neg):
    neg_tilde = ["~"+c for c in neg]
    condition = neg_tilde
    return tuple(sorted(condition))

def _add_unmergable(e, automata_id, new_automata:SubgoalAutomaton, new_automaton_state_counter:int, state_mapping:dict):
    e_from = (automata_id, e["from"])
    e_to = (automata_id, e["to"])

    if e_from in state_mapping.keys():
        new_from = state_mapping[e_from]
    else:
        new_automaton_state_counter += 1
        new_from = "u"+str(new_automaton_state_counter)

        state_mapping[e_from] = new_from

    if e_to in state_mapping.keys():
        new_to = state_mapping[e_to]
    else:
        new_automaton_state_counter += 1
        new_to = "u"+str(new_automaton_state_counter)

        state_mapping[e_to] = new_to

    new_pos = e["pos"]
    new_neg = e["neg"]

    new_edge_condition = _new_edge_condition(new_pos, new_neg)

    if new_from not in new_automata.get_states():
        new_automata.add_state(new_from)

    if new_to not in new_automata.get_states():
        new_automata.add_state(new_to)

    new_automata.add_edge(new_from,new_to,new_edge_condition)

    return new_automata, new_automaton_state_counter, state_mapping

if __name__ == "__main__":
    dfa1 = SubgoalAutomaton()
    dfa1.add_state("u0")
    dfa1.add_state("u1")
    # dfa1.add_state("u2")
    # dfa1.add_state("uA")
    # dfa1.add_state("uR")
    dfa1.set_initial_state("u0")
    dfa1.set_accept_state("uA")
    dfa1.set_reject_state("uR")

    dfa1.add_edge("u0", "u1", ["f", "~g"])
    # dfa1.add_edge("u0", "u2", ["4"])
    # dfa1.add_edge("u2", "uR", ["f", "~g"])
    # dfa1.add_edge("u0", "uA", ["f", "g"])
    dfa1.add_edge("u0", "uR", ["n", "~f", "~g"])
    dfa1.add_edge("u1", "uA", ["g"])
    dfa1.add_edge("u1", "uR", ["n", "~g"])

    dfa1.plot(".","SA1.png")

    dfa2 = SubgoalAutomaton()
    dfa2.add_state("u0")
    dfa2.add_state("u1")
    dfa2.add_state("u2")
    dfa2.add_state("uA")
    dfa2.add_state("uR")
    dfa2.set_initial_state("u0")
    dfa2.set_accept_state("uA")
    dfa2.set_reject_state("uR")

    dfa2.add_edge("u0", "u1", ["f", "~g"])
    dfa2.add_edge("u0", "uA", ["h", "g"])
    dfa2.add_edge("u0", "uR", ["h", "~f"])
    dfa2.add_edge("u1", "uA", ["g"])
    dfa2.add_edge("u1", "u2", ["h"])
    dfa2.add_edge("u2", "uA", ["g"])

    dfa2.plot(".","SA2.png")


    new_dfa = merge_automata(dfa1, dfa2)

    new_dfa.plot(".","out_new.png")
