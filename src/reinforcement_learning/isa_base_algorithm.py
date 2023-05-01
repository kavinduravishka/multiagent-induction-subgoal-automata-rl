from abc import abstractmethod
import numpy as np
import os

from gym_subgoal_automata_multiagent.utils.subgoal_automaton import SubgoalAutomaton
from colab_utils.a_star_for_subgoal_automaton import AstarSearch
from colab_utils.merge_automata import merge_automata

# from gym_subgoal_automata_multiagent.utils.merged_subgoal_automaton import MergedSubgoalAutomaton
from reinforcement_learning.learning_algorithm import LearningAlgorithm
from utils import utils
from ilasp.generator.ilasp_task_generator import generate_ilasp_task
from ilasp.parser import ilasp_solution_parser
from ilasp.solver.ilasp_solver import solve_ilasp_task



class ISAAlgorithmBase(LearningAlgorithm):
    """
    Generic class for the algorithms performing interleaving between RL and automata learning.
    """
    INITIAL_STATE_NAME = "u0"
    ACCEPTING_STATE_NAME = "u_acc"
    REJECTING_STATE_NAME = "u_rej"

    # whether to use the single state automaton (basic), load an existing solution (load) or use the target one (target)
    INITIAL_AUTOMATON_MODE = "initial_automaton"

    INTERLEAVED_FIELD = "interleaved_automaton_learning"           # whether RL is interleaved with ILASP automaton learner
    ILASP_TIMEOUT_FIELD = "ilasp_timeout"                          # time that ILASP has for finding a single automaton solution
    ILASP_VERSION_FIELD = "ilasp_version"                          # ILASP version to run
    ILASP_COMPUTE_MINIMAL = "ilasp_compute_minimal"                # whether to compute a minimal solution (an optimal is computed otherwise)
    STARTING_NUM_STATES_FIELD = "starting_num_states"              # number of states that the starting automaton has
    USE_RESTRICTED_OBSERVABLES = "use_restricted_observables"      # use the restricted set of observables (the ones that define the goal for the task)
    MAX_DISJUNCTION_SIZE = "max_disjunction_size"                  # maximum number of edges from one state to another
    MAX_BODY_LITERALS = "max_body_literals"                        # maximum number of literals that a learnt rule can have
    LEARN_ACYCLIC_GRAPH = "learn_acyclic_graph"                    # whether the target automata has cycles or not
    SYMMETRY_BREAKING_METHOD = "symmetry_breaking_method"          # which symmetry breaking method is used to break symmetries in the graph
    AVOID_LEARNING_ONLY_NEGATIVE = "avoid_learning_only_negative"  # whether to avoid learning labels made only of negative literals (e.g., ~n)
    PRIORITIZE_OPTIMAL_SOLUTIONS = "prioritize_optimal_solutions"  # prioritize some optimal solutions above others based on some weak constraints
    WAIT_FOR_GOAL_EXAMPLE = "wait_for_goal_example"                # whether an automaton is not learnt until a goal example is received

    USE_MAX_EPISODE_LENGTH_ANNEALING = "use_max_episode_length_annealing"  # whether to increase the maximum episode length as learning progresses
    INITIAL_MAX_EPISODE_LENGTH = "initial_max_episode_length"
    FINAL_MAX_EPISODE_LENGTH = "final_max_episode_length"

    USE_EXPERIENCE_REPLAY = "use_experience_replay"                  # whether to use the experience replay buffer for learning (automatically active for deep learning approach)
    EXPERIENCE_REPLAY_BUFFER_SIZE = "experience_replay_buffer_size"  # size of the ER buffer
    EXPERIENCE_REPLAY_BATCH_SIZE = "experience_replay_batch_size"    # size of the batches sampled from the ER buffer
    EXPERIENCE_REPLAY_START_SIZE = "experience_replay_start_size"    # size of the ER after which learning starts

    USE_DOUBLE_DQN = "use_double_dqn"                            # whether double DQN is used instead of simple DQN
    TARGET_NET_UPDATE_FREQUENCY = "target_net_update_frequency"  # how many steps happen between target DQN updates
    NUM_HIDDEN_LAYERS = "num_hidden_layers"                      # number of hidden layers that the network has
    NUM_NEURONS_PER_LAYER = "num_neurons_per_layer"              # number of neurons per hidden layer

    AUTOMATON_TASK_FOLDER = "automaton_tasks"  # folder where the automaton learning tasks are saved
    AUTOMATON_TASK_FILENAME = "task-%d.las"    # filename pattern of the automaton learning tasks

    AUTOMATON_SOLUTION_FOLDER = "automaton_solutions"  # folder where the solutions to automaton learning tasks are saved
    AUTOMATON_SOLUTION_FILENAME = "solution-%d.txt"    # filename pattern of the solutions to automaton learning tasks

    AUTOMATON_PLOT_FOLDER = "automaton_plots"          # folder where the graphical solutions to automaton learning tasks are saved
    AUTOMATON_PLOT_FILENAME = "plot-%d.png"            # filename pattern of the graphical solutions to automaton learning tasks

    AUTOMATON_LEARNING_EPISODES_FILENAME = "automaton_learning_episodes-agent_%d.txt"  # filename of the file containing the episodes where an automaton has been learned

    def __init__(self, tasks, num_tasks, export_folder_names, params, target_automata, binary_folder_name):
        super().__init__(tasks, num_tasks, export_folder_names, params)
        self.binary_folder_name = binary_folder_name

        self.initial_automaton_mode = utils.get_param(params, ISAAlgorithmBase.INITIAL_AUTOMATON_MODE, "basic")

        # interleaved automaton learning params
        self.interleaved_automaton_learning = utils.get_param(params, ISAAlgorithmBase.INTERLEAVED_FIELD, False)
        self.ilasp_timeout = utils.get_param(params, ISAAlgorithmBase.ILASP_TIMEOUT_FIELD, 120)
        self.ilasp_version = utils.get_param(params, ISAAlgorithmBase.ILASP_VERSION_FIELD, "2")
        self.ilasp_compute_minimal = utils.get_param(params, ISAAlgorithmBase.ILASP_COMPUTE_MINIMAL, False)
        self.num_starting_states = utils.get_param(params, ISAAlgorithmBase.STARTING_NUM_STATES_FIELD, 3)
        self.num_automaton_states = self.num_starting_states * np.ones((self.num_domains,self.num_agents), dtype=np.int)
        self.use_restricted_observables = utils.get_param(params, ISAAlgorithmBase.USE_RESTRICTED_OBSERVABLES, False)
        self.max_disjunction_size = utils.get_param(params, ISAAlgorithmBase.MAX_DISJUNCTION_SIZE, 1)
        self.max_body_literals = utils.get_param(params, ISAAlgorithmBase.MAX_BODY_LITERALS, 1)
        self.learn_acyclic_graph = utils.get_param(params, ISAAlgorithmBase.LEARN_ACYCLIC_GRAPH, False)
        self.symmetry_breaking_method = utils.get_param(params, ISAAlgorithmBase.SYMMETRY_BREAKING_METHOD, None)
        self.avoid_learning_only_negative = utils.get_param(params, ISAAlgorithmBase.AVOID_LEARNING_ONLY_NEGATIVE, False)
        self.prioritize_optimal_solutions = utils.get_param(params, ISAAlgorithmBase.PRIORITIZE_OPTIMAL_SOLUTIONS, False)
        self.wait_for_goal_example = utils.get_param(params, ISAAlgorithmBase.WAIT_FOR_GOAL_EXAMPLE, True)
        self.has_observed_goal_example = np.zeros((self.num_domains, self.num_agents), dtype=np.bool)

        # maximum episode annealing parameters
        self.use_max_episode_length_annealing = utils.get_param(params, ISAAlgorithmBase.USE_MAX_EPISODE_LENGTH_ANNEALING, False)
        self.final_max_episode_length = utils.get_param(params, ISAAlgorithmBase.FINAL_MAX_EPISODE_LENGTH, 100)
        if self.use_max_episode_length_annealing:
            self.initial_max_episode_length = utils.get_param(params, ISAAlgorithmBase.INITIAL_MAX_EPISODE_LENGTH, 100)
            self.max_episode_length = self.initial_max_episode_length
            self.max_episode_length_increase_rate = (self.final_max_episode_length - self.max_episode_length) / self.num_episodes

        # experience replay
        self.use_experience_replay = utils.get_param(params, ISAAlgorithmBase.USE_EXPERIENCE_REPLAY, False) or not self.is_tabular_case
        self.experience_replay_buffer_size = utils.get_param(params, ISAAlgorithmBase.EXPERIENCE_REPLAY_BUFFER_SIZE, 50000)
        self.experience_replay_batch_size = utils.get_param(params, ISAAlgorithmBase.EXPERIENCE_REPLAY_BATCH_SIZE, 32)
        self.experience_replay_start_size = utils.get_param(params, ISAAlgorithmBase.EXPERIENCE_REPLAY_START_SIZE, 1000)

        # deep q-learning
        self.use_double_dqn = utils.get_param(params, ISAAlgorithmBase.USE_DOUBLE_DQN, True)
        self.num_layers = utils.get_param(params, ISAAlgorithmBase.NUM_HIDDEN_LAYERS, 6)
        self.num_neurons_per_layer = utils.get_param(params, ISAAlgorithmBase.NUM_NEURONS_PER_LAYER, 64)
        self.target_net_update_frequency = utils.get_param(params, ISAAlgorithmBase.TARGET_NET_UPDATE_FREQUENCY, 100)

        # set of automata per domain
        self.automata = None
        self.merged_automata = None
        self.a_star_automata = None
        self.shared_automata = None
        self._set_automata(target_automata)
        self._set_merged_automata(target_automata)

        # sets of examples (goal, deadend and incomplete)
        self.goal_examples = None
        self.dend_examples = None
        self.inc_examples = None
        self._reset_examples()

        # keep track of the number of learnt automata per domain
        self.automaton_counters = np.zeros((self.num_domains,self.num_agents), dtype=np.int)
        self.merged_automaton_counter = 0
        self.automaton_learning_episodes = [[] for _ in range(self.num_domains)]

        if self.train_model:  # if the tasks are learnt, remove previous folders if they exist
            utils.rm_dirs(self.get_automaton_task_folders())
            utils.rm_dirs(self.get_automaton_solution_folders())
            utils.rm_dirs(self.get_automaton_plot_folders())
            utils.rm_files(self.get_automaton_learning_episodes_files())


    '''
    Learning Loop (main loop, what happens when an episode ends, changes or was not completed)
    '''
    def run(self, loaded_checkpoint=False):
        super().run(loaded_checkpoint)
        if self.interleaved_automaton_learning:
            self._write_automaton_learning_episodes()

    def _run_episode(self, domain_id, task_id):
        task = self._get_task(domain_id, task_id)  # get the task to learn

        # initialize reward and steps counters, histories and reset the task to its initial state
        total_reward = [0 for i in range(self.num_agents)]
        episode_length = [0 for i in range(self.num_agents)]
        observation_history, compressed_observation_history = [], []
        current_state = task.reset()

        # get initial observations and initialise histories
        initial_observations = self._get_task_observations(task)

        self._update_histories(observation_history, compressed_observation_history, initial_observations)

        self._initiate_A_star()

        # get actual initial automaton state (performs verification that there is only one possible initial state!)
        current_automaton_state = self._get_initial_automaton_state_successors(domain_id, initial_observations, [True for t in range(self.num_agents)])
        current_merged_automaton_state_candidates = self._get_initial_A_star_merged_automaton_state(domain_id, initial_observations, [True for t in range(self.num_agents)])

        current_merged_automaton_state = [ current_merged_automaton_state_candidates[agent_id][0] for agent_id in range(self.num_agents)]

        # update the automaton if the initial state achieves the goal and the example is not covered
        can_learn_new_automaton = self._can_learn_new_automaton(domain_id, task)

        if self.interleaved_automaton_learning and any(can_learn_new_automaton):
            updated_automaton = self._perform_interleaved_automaton_learning(task, domain_id,
                                                                             current_automaton_state,
                                                                             observation_history,
                                                                             compressed_observation_history,
                                                                             can_learn_new_automaton)

            if any(updated_automaton):  # get the actual initial state as done before
                current_automaton_state = self._get_initial_automaton_state_successors(domain_id, initial_observations, updated_automaton)
                self._share_automaton(domain_id, updated_automaton)
                updated_merged_automaton = self._merge_automaton(domain_id)
                self._initiate_A_star()
                self._on_merged_automaton_learned(domain_id)
                current_merged_automaton_state_candidates = self._get_initial_A_star_merged_automaton_state(domain_id, initial_observations, updated_merged_automaton)
                current_merged_automaton_state = [current_merged_automaton_state_candidates[agent_id][0] for agent_id in range(self.num_agents)]

        # whether the episode execution must be stopped (an automaton is learnt in the middle)
        interrupt_episode = [False]*self.num_agents

        automaton_all_agents = self.automata[domain_id]
        merged_automaton_all_agents = self.merged_automata[domain_id]

        is_terminal = task.is_terminal()

        # while not all(is_terminal) and not any(is_goal_achieved) and episode_length < self.max_episode_length and not any(interrupt_episode):
        while not all(is_terminal) and all([episode_length[i] < self.max_episode_length for i in range(self.num_agents)]) and not any(interrupt_episode):

            current_merged_automaton_state_id = [merged_automaton_all_agents[agent_id].get_state_id(current_merged_automaton_state[agent_id]) 
                                          if is_terminal[agent_id] != None else None for agent_id in range(self.num_agents)]
            
            actions = [self._choose_action(domain_id, agent_id, task_id, current_state[agent_id], merged_automaton_all_agents[agent_id], 
                                        current_merged_automaton_state_id[agent_id]) if not current_merged_automaton_state_id[agent_id] == None 
                                        else None for agent_id in range(self.num_agents)]
            
            next_state, reward, is_terminal, _ = task.step(actions)
            observations = self._get_task_observations(task)
            terminated_agents = task.get_terminated_agents()

            # whether observations have changed or not is important for QRM when using compressed traces
            observations_changed = self._update_histories(observation_history, compressed_observation_history, observations)

            if self.train_model:
                self._update_q_functions(task_id, current_state, actions, next_state, is_terminal, observations, observations_changed, terminated_agents)

            next_automaton_state = [self._get_next_automaton_state(self.automata[domain_id][agent_id], current_automaton_state[agent_id],
                                                                observations[agent_id], observations_changed[agent_id]) for agent_id in range(self.num_agents)]
            
            next_merged_automaton_state_candidates = [self._get_next_A_star_merged_automaton_state(self.a_star_automata[domain_id][agent_id], current_merged_automaton_state[agent_id],
                                                                observations[agent_id], observations_changed[agent_id]) for agent_id in range(self.num_agents)]
    
            next_merged_automaton_state = [self._get_best_candidate_state_out_of_a_star_candidate(domain_id, agent_id, task_id, current_state[agent_id],
                                                                                                  actions[agent_id], next_merged_automaton_state_candidates[agent_id])
                                                                                                    for agent_id in range(self.num_agents)]
            # episode has to be interrupted if an automaton is learnt
            can_learn_new_automaton = self._can_learn_new_automaton(domain_id, task)


            if not any(interrupt_episode) and self.interleaved_automaton_learning and any(can_learn_new_automaton):
                interrupt_episode = self._perform_interleaved_automaton_learning(task, domain_id, next_automaton_state,
                                                                                observation_history,
                                                                                compressed_observation_history,
                                                                                can_learn_new_automaton)
                
                if any(interrupt_episode):
                    self._share_automaton(domain_id, interrupt_episode)
                    updated_merged_automaton = self._merge_automaton(domain_id)
                    self._on_merged_automaton_learned(domain_id)

            if not any(interrupt_episode):
                automatons = self.merged_automata[domain_id]
                for agent_id in range(self.num_agents):
                    total_reward[agent_id] += reward[agent_id]

                self._on_performed_step(domain_id, task_id, next_state, reward, is_terminal, observations, automatons,
                                        current_merged_automaton_state, next_merged_automaton_state, episode_length)

            # update current environment and automaton states and increase episode length
            current_state = next_state
            current_automaton_state = next_automaton_state
            current_merged_automaton_state = next_merged_automaton_state
            episode_length = [episode_length[i] + 1* (not terminated_agents[i]) for i in range(self.num_agents)]

        completed_episode = not any(interrupt_episode)


        return completed_episode, total_reward, episode_length, task.is_terminal(), observation_history, compressed_observation_history

    def _on_episode_change(self, previous_episode):
        if self.use_max_episode_length_annealing:
            episode_length_increase = previous_episode * self.max_episode_length_increase_rate
            self.max_episode_length = int(min(self.initial_max_episode_length + episode_length_increase, self.final_max_episode_length))
        super()._on_episode_change(previous_episode)

    def _on_incomplete_episode(self, current_domain_id):
        # if the episode was interrupted, log the learning episode
        self.automaton_learning_episodes[current_domain_id].append(self.current_episode)

    @abstractmethod
    def _choose_action(self, domain_id, agent_id, task_id, current_state, automaton, current_automaton_state):
        pass

    @abstractmethod
    def _on_performed_step(self, domain_id, task_id, next_state, reward, is_terminal, observations, automaton, current_automaton_state, next_automaton_state, episode_length):
        pass

    @abstractmethod
    def _build_q_functions(self):
        pass

    @abstractmethod
    def _update_q_functions(self, task_id, current_state, action, next_state, is_terminal, observations, observations_changed, terminated_agents):
        pass

    @abstractmethod
    def _build_experience_replay_buffers(self):
        pass

    '''
    Greedy Policy Evaluation
    '''
    def _evaluate_greedy_policies(self):
        # we do not want automata to be learned during the evaluation of a policy
        tmp_interleaved_automaton_learning = self.interleaved_automaton_learning
        self.interleaved_automaton_learning = False
        super()._evaluate_greedy_policies()
        self.interleaved_automaton_learning = tmp_interleaved_automaton_learning

    def _set_has_observed_goal_example(self, domain_id, task):
        is_goal_achieved = task.is_goal_achieved()
        for agent_id in range(self.num_agents):
            if is_goal_achieved[agent_id] and not self.has_observed_goal_example[domain_id][agent_id]:
                self.has_observed_goal_example[domain_id][agent_id] = True

    def _can_learn_new_automaton(self, domain_id, task):
        self._set_has_observed_goal_example(domain_id, task)
        return [not self.wait_for_goal_example or self.has_observed_goal_example[domain_id][agent_id] for agent_id in range(self.num_agents)]

    '''
    Task Management Methods (getting observations)
    '''
    def _get_task_observations(self, task):
        observations = task.get_observations()
        if self.use_restricted_observables:
            return [obs_set.intersection(task.get_restricted_observables()) for obs_set in observations] 
        return observations

    '''
    Automata Sharing and Merging Methods (share, merge, setters, getters, associated rewards)
    '''

    def _share_automaton(self,domain_id, updated_automata):
        if self.shared_automata == None:
            self.shared_automata = [[{}]*self.num_agents]*self.num_domains

        for i in range(len(updated_automata)):
            if updated_automata[i]:
                for agent_id in range(self.num_agents):
                    self.shared_automata[domain_id][agent_id][i] = self._get_automaton(domain_id, i)

    def _merge_automaton(self, domain_id):
        updated_merged_automaton = []
        if self.merged_automata == None:
            self.merged_automata = [[None]*self.num_agents]*self.num_domains

        for agent_id in range(self.num_agents):
            automatas_to_combine = [v for (k,v) in self.shared_automata[domain_id][agent_id].items()]
            automata_1 = automatas_to_combine[0]
            # automata_1.plot("/tmp/before_merged_automata","before_merged-%d-%d.png" % (agent_id, self.merged_automaton_counter))
            # self.merged_automaton_counter+=1
            other_automata = automatas_to_combine[1:]
            # to_merge_counter = 0 
            for automata_2 in other_automata:
                # automata_1.plot("/tmp/automata_to_merge","1_to_be_merged-%d-%d-%d.png" % (agent_id, self.merged_automaton_counter, to_merge_counter))
                # automata_2.plot("/tmp/automata_to_merge","2_to_be_merged-%d-%d-%d.png" % (agent_id, self.merged_automaton_counter, to_merge_counter))
                automata_1 = merge_automata(automata_1, automata_2)

                # automata_1.plot("/tmp/merged_automata","merged-%d-%d.png" % (agent_id, self.merged_automaton_counter))
                # self.merged_automaton_counter+=1
                # to_merge_counter+=1
            
            # automata_1.plot("/tmp/merged_automata","final_merged-%d-%d.png" % (agent_id, self.merged_automaton_counter))
            self.merged_automata[domain_id][agent_id] = automata_1

            updated_merged_automaton.append(True)
        return updated_merged_automaton

    def _get_merged_automaton(self, domain_id,agent_id):
        return self.merged_automata[domain_id][agent_id]

    def _get_initial_merged_automaton_state_successors(self, domain_id, observations, updated_automaton):
        automaton_state_successors_all_agents = []
        for agent_id in range(self.num_agents):
            if not updated_automaton[agent_id]:
                automaton_state_successors_all_agents.append(None)
                continue
            automaton = self._get_merged_automaton(domain_id, agent_id)
            initial_state = automaton.get_initial_state()
            automaton_state_successors_all_agents.append(self._get_next_automaton_state(automaton, initial_state, observations[agent_id], True))
        return automaton_state_successors_all_agents
        
    def _set_merged_automata(self, target_automata):
        if self.initial_automaton_mode == "basic":
            self._set_basic_merged_automata()
        elif self.initial_automaton_mode == "load_solution":
            self._load_last_automata_solutions() # FIX this for merged automata
        elif self.initial_automaton_mode == "target":
            self.automata = target_automata
        else:
            raise RuntimeError("Error: The initial merged automaton mode \"%s\" is not recognised." % self.initial_automaton_mode)

    def _set_basic_merged_automata(self):
        self.merged_automata = []

        for _ in range(self.num_domains):
            # the initial automaton is an automaton that doesn't accept nor reject anything
            merged_automatas_all_agents = []
            for j in range(self.num_agents):
                # automaton = SubgoalAutomaton()
                automaton = SubgoalAutomaton()
                automaton.add_state(ISAAlgorithmBase.INITIAL_STATE_NAME)
                automaton.set_initial_state(ISAAlgorithmBase.INITIAL_STATE_NAME)
                # automaton.add_state(ISAAlgorithm.ACCEPTING_STATE_NAME)  # DO NOT UNCOMMENT!
                # automaton.add_state(ISAAlgorithm.REJECTING_STATE_NAME)
                # automaton.set_accept_state(ISAAlgorithm.ACCEPTING_STATE_NAME)
                # automaton.set_reject_state(ISAAlgorithm.REJECTING_STATE_NAME)
                merged_automatas_all_agents.append(automaton)
            self.merged_automata.append(merged_automatas_all_agents)

    '''
    Manage A star objects
    '''
    def _initiate_A_star(self):
        self.a_star_automata = [[AstarSearch(self.merged_automata[domain_id][agent_id]) for agent_id in range(self.num_agents)] for domain_id in range(self.num_domains)]

    def _reset_A_star(self):
        self._initiate_A_star()

    def _get_A_star_automaton(self, domain_id, agent_id):
        return self.a_star_automata[domain_id][agent_id]
    
    # def _get_satisfied_conditions(self, automaton, current_automaton_state, observations, observations_changed):
    #     # automaton has to be navigated with compressed traces if specified (just when a change occurs)
    #     if observations == None:
    #         return "<None : obs>"
    #     if (self.ignore_empty_observations and len(observations) == 0) or (self.use_compressed_traces and not observations_changed):
    #         return "<Empty : obs>"
    #     return automaton.get_all_satisfying_edges(current_automaton_state, observations)
    
    def _get_next_A_star_merged_automaton_state(self, a_star_automaton: AstarSearch, current_automaton_state, observations, observations_changed):
        if observations == None:
            return [None]
        if (self.ignore_empty_observations and len(observations) == 0) or (self.use_compressed_traces and not observations_changed):
            return [current_automaton_state]
        candidate_states = a_star_automaton.get_next_automaton_states(observations)
        return candidate_states
    
    # def _get_current_A_star_merged_automaton_state(self, a_star_automaton: AstarSearch, current_automaton_state, observations, observations_changed):
    #     if observations == None:
    #         return None
    #     if (self.ignore_empty_observations and len(observations) == 0) or (self.use_compressed_traces and not observations_changed):
    #         return current_automaton_state
    #     candidate_states = a_star_automaton.get_current_automaton_states()
    #     return candidate_states
    
    # def _get_future_A_star_merged_automaton_state(self, a_star_automaton: AstarSearch, current_automaton_state, observations, observations_changed):
    #     if observations == None:
    #         return None
    #     if (self.ignore_empty_observations and len(observations) == 0) or (self.use_compressed_traces and not observations_changed):
    #         return current_automaton_state
    #     candidate_states = a_star_automaton.get_next_automaton_states_without_updating(observations)
    #     return candidate_states
    
    def _get_initial_A_star_merged_automaton_state(self, domain_id, observations, updated_automaton):
        automaton_state_successor_candidates = []
        for agent_id in range(self.num_agents):
            if not updated_automaton[agent_id]:
                automaton_state_successor_candidates.append(None)
                continue
            automaton = self._get_A_star_automaton(domain_id, agent_id)
            initial_state = automaton.get_initial_state()
            automaton_state_successor_candidates.append(self._get_next_A_star_merged_automaton_state(automaton, initial_state, observations[agent_id], True))
            # automaton_state_successors_all_agents.append()
        return automaton_state_successor_candidates
    
    @abstractmethod
    def _get_best_candidate_state_out_of_a_star_candidate(self, domain_id, agent_id, task_id, current_state, action, candidate_states):
        pass

    '''
    Automata Management Methods (setters, getters, associated rewards)
    '''
    def _get_automaton(self, domain_id,agent_id):
        return self.automata[domain_id][agent_id]
    
    def _get_next_automaton_state(self, automaton, current_automaton_state, observations, observations_changed): # For both agent auotmata and merged automata
        # automaton has to be navigated with compressed traces if specified (just when a change occurs)
        if observations == None:
            return None
        if (self.ignore_empty_observations and len(observations) == 0) or (self.use_compressed_traces and not observations_changed):
            return current_automaton_state
        return automaton.get_next_state(current_automaton_state, observations)

    def _get_initial_automaton_state_successors(self, domain_id, observations, updated_automaton):
        automaton_state_successors_all_agents = []
        for agent_id in range(self.num_agents):
            if not updated_automaton[agent_id]:
                automaton_state_successors_all_agents.append(None)
                continue
            automaton = self._get_automaton(domain_id, agent_id)
            initial_state = automaton.get_initial_state()
            automaton_state_successors_all_agents.append(self._get_next_automaton_state(automaton, initial_state, observations[agent_id], True))
        return automaton_state_successors_all_agents

    def _set_automata(self, target_automata):       # don't change
        if self.initial_automaton_mode == "basic":
            self._set_basic_automata()
        elif self.initial_automaton_mode == "load_solution":
            self._load_last_automata_solutions()
        elif self.initial_automaton_mode == "target":
            self.automata = target_automata
        else:
            raise RuntimeError("Error: The initial automaton mode \"%s\" is not recognised." % self.initial_automaton_mode)

    def _load_last_automata_solutions(self):
        self.automata = []

        for i in range(self.num_domains):
            automatas_all_agents = []
            for j in range(self.num_agents):
                automaton_solution_folder = self.get_automaton_solution_folder(i,j)
                last_automaton_filename = self._get_last_solution_filename(automaton_solution_folder)
                automaton = self._parse_ilasp_solutions(last_automaton_filename)
                automaton.set_initial_state(ISAAlgorithmBase.INITIAL_STATE_NAME)
                automaton.set_accept_state(ISAAlgorithmBase.ACCEPTING_STATE_NAME)
                automaton.set_reject_state(ISAAlgorithmBase.REJECTING_STATE_NAME)
                automatas_all_agents.append(automaton)
            self.automata.append(automatas_all_agents)

    def _get_last_solution_filename(self, automaton_solution_folder):
        automaton_solutions = os.listdir(automaton_solution_folder)
        automaton_solutions.sort(key=lambda k: int(k[:-len(".txt")].split("-")[1]))
        automaton_solutions_path = [os.path.join(automaton_solution_folder, f) for f in automaton_solutions]

        if len(automaton_solutions_path) > 1 and utils.is_file_empty(automaton_solutions_path[-1]):
            return automaton_solutions_path[-2]

        return automaton_solutions_path[-1]

    def _set_basic_automata(self):
        self.automata = []

        for _ in range(self.num_domains):
            # the initial automaton is an automaton that doesn't accept nor reject anything
            automatas_all_agents = []
            for j in range(self.num_agents):
                # automaton = SubgoalAutomaton()
                automaton = SubgoalAutomaton()
                automaton.add_state(ISAAlgorithmBase.INITIAL_STATE_NAME)
                automaton.set_initial_state(ISAAlgorithmBase.INITIAL_STATE_NAME)
                # automaton.add_state(ISAAlgorithm.ACCEPTING_STATE_NAME)  # DO NOT UNCOMMENT!
                # automaton.add_state(ISAAlgorithm.REJECTING_STATE_NAME)
                # automaton.set_accept_state(ISAAlgorithm.ACCEPTING_STATE_NAME)
                # automaton.set_reject_state(ISAAlgorithm.REJECTING_STATE_NAME)
                automatas_all_agents.append(automaton)
            self.automata.append(automatas_all_agents)

    '''
    Automata Learning Methods (example update, task generation/solving/parsing)
    '''
    @abstractmethod
    def _on_automaton_learned(self, domain_id, agent_id = None):
        pass

    @abstractmethod
    def _on_merged_automaton_learned(self, domain_id, agent_id = None):
        pass

    def _perform_interleaved_automaton_learning(self, task, domain_id, current_automaton_state, observation_history,
                                                compressed_observation_history, can_learn_new_automaton):
        """Updates the set of examples based on the current observed trace. In case the set of example is updated, it
        makes a call to the automata learner. Returns True if a new automaton has been learnt, False otherwise."""
        updated_examples = self._update_examples(task, domain_id, current_automaton_state, observation_history,
                                                 compressed_observation_history, can_learn_new_automaton)
        
        out_all_agents = []

        # print("UPDATED EXAMPLES :",updated_examples)
        for agent_id in range(self.num_agents):
            if updated_examples[agent_id]:
                if self.debug:
                    if self.use_compressed_traces:
                        counterexample = str(compressed_observation_history[agent_id])
                    else:
                        counterexample = str(observation_history[agent_id])
                    print("Updating automaton " + "In Domain ID : " + str(domain_id) + " for Agent ID " + str(agent_id) + "\n\t... The counterexample is: " + counterexample)
                self._update_automaton(task, domain_id, agent_id)
                out_all_agents.append(True)  # whether a new automaton has been learnt
                continue

            out_all_agents.append(False)

        return out_all_agents

    def _reset_examples(self):
        # there is a set of examples for each domain
        self.goal_examples = [[set() for i in range(self.num_agents)]for j in range(self.num_domains)]
        self.dend_examples = [[set() for i in range(self.num_agents)]for j in range(self.num_domains)]
        self.inc_examples = [[set() for i in range(self.num_agents)]for j in range(self.num_domains)]

    def _update_examples(self, task, domain_id, current_automaton_state, observation_history, compressed_observation_history, can_learn_new_automaton):
        """Updates the set of examples. Returns True if the set of examples has been updated and False otherwise. Note
        that an update of the set of examples can be forced by setting 'current_automaton_state' to None."""
        out_all_agents = []
        for agent_id in range(self.num_agents):
            if not can_learn_new_automaton[agent_id]:
                out_all_agents.append(False)
                continue
            
            automaton = self.automata[domain_id][agent_id]

            if task.is_terminal()[agent_id]:
                if task.is_goal_achieved()[agent_id]:
                    if current_automaton_state[agent_id] is None or not automaton.is_accept_state(current_automaton_state[agent_id]):
                        self._update_example_set(self.goal_examples[domain_id][agent_id], observation_history[agent_id], compressed_observation_history[agent_id])
                        out_all_agents.append(True)
                    else:
                        out_all_agents.append(False)
                else:
                    if current_automaton_state[agent_id] is None or not automaton.is_reject_state(current_automaton_state[agent_id]):
                        self._update_example_set(self.dend_examples[domain_id][agent_id], observation_history[agent_id], compressed_observation_history[agent_id])
                        out_all_agents.append(True)
                    else:
                        out_all_agents.append(False)
            else:
                # just update incomplete examples if at least we have one goal or one deadend example (avoid overflowing the
                # set of incomplete unnecessarily)
                if current_automaton_state[agent_id] is None or automaton.is_terminal_state(current_automaton_state[agent_id]):
                    self._update_example_set(self.inc_examples[domain_id][agent_id], observation_history[agent_id], compressed_observation_history[agent_id])
                    out_all_agents.append(True)
                else:
                    out_all_agents.append(False)  # whether example sets have been updated)
        return out_all_agents

    def _update_example_set(self, example_set, observation_history, compressed_observation_history):                # don't change
        """Updates the a given example set with the corresponding history of observations depending on whether      
        compressed traces are used or not to learn the automata. An exception is thrown if a trace is readded."""
        if self.use_compressed_traces:                               # don't change
            history_tuple = tuple(compressed_observation_history)    # don't change
        else:                                                        # don't change
            history_tuple = tuple(observation_history)               # don't change

        if history_tuple not in example_set or history_tuple == (): 
            if history_tuple != ():   # don't change
                example_set.add(history_tuple)      # don't change
        else:                                   # don't change
            # print("DEBUG : example causes to fail :",history_tuple,", Example set :",example_set )
            raise RuntimeError("An example that an automaton is currently covered cannot be uncovered afterwards!") # don't change

    def _update_automaton(self, task, domain_id, agent_id):
        self.automaton_counters[domain_id][agent_id] += 1  # increment the counter of the number of aut. learnt for a domain

        self._generate_ilasp_task(task, domain_id, agent_id)  # generate the automata learning task

        solver_success = self._solve_ilasp_task(domain_id, agent_id)  # run the task solver
        if solver_success:
            ilasp_solution_filename = os.path.join(self.get_automaton_solution_folder(domain_id, agent_id),
                                                   ISAAlgorithmBase.AUTOMATON_SOLUTION_FILENAME % self.automaton_counters[domain_id][agent_id])
            candidate_automaton = self._parse_ilasp_solutions(ilasp_solution_filename)

            if candidate_automaton.get_num_states() > 0:
                # set initial, accepting and rejecting states in the automaton
                candidate_automaton.set_initial_state(ISAAlgorithmBase.INITIAL_STATE_NAME)
                candidate_automaton.set_accept_state(ISAAlgorithmBase.ACCEPTING_STATE_NAME)
                candidate_automaton.set_reject_state(ISAAlgorithmBase.REJECTING_STATE_NAME)
                self.automata[domain_id][agent_id] = candidate_automaton

                # plot the new automaton
                candidate_automaton.plot(self.get_automaton_plot_folder(domain_id, agent_id),
                                         ISAAlgorithmBase.AUTOMATON_PLOT_FILENAME % self.automaton_counters[domain_id][agent_id])

                # self._on_automaton_learned(domain_id,agent_id)
            else:
                # if the task is UNSATISFIABLE, it means the number of states is not enough to cover the examples, so
                # the number of states is incremented by 1 and try again
                self.num_automaton_states[domain_id][agent_id] += 1

                if self.debug:
                    print("The number of states in the automaton has been increased to " + str(self.num_automaton_states[domain_id][agent_id]))
                    print("Updating automaton...")

                self._update_automaton(task, domain_id, agent_id)
        else:
            raise RuntimeError("Error: Couldn't find an automaton under the specified timeout!")

    def _generate_ilasp_task(self, task, domain_id, agent_id):
        utils.mkdir(self.get_automaton_task_folder(domain_id,agent_id))

        ilasp_task_filename = ISAAlgorithmBase.AUTOMATON_TASK_FILENAME % self.automaton_counters[domain_id][agent_id]

        observables = task.get_observables()
        if self.use_restricted_observables:
            observables = task.get_restricted_observables()

        # the sets of examples are sorted to make sure that ILASP produces the same solution for the same sets (ILASP
        # can produce different hypothesis for the same set of examples but given in different order)
        generate_ilasp_task(self.num_automaton_states[domain_id][agent_id], ISAAlgorithmBase.ACCEPTING_STATE_NAME,
                            ISAAlgorithmBase.REJECTING_STATE_NAME, observables, sorted(self.goal_examples[domain_id][agent_id]),
                            sorted(self.dend_examples[domain_id][agent_id]), sorted(self.inc_examples[domain_id][agent_id]),
                            self.get_automaton_task_folder(domain_id,agent_id), ilasp_task_filename, self.symmetry_breaking_method,
                            self.max_disjunction_size, self.learn_acyclic_graph, self.use_compressed_traces,
                            self.avoid_learning_only_negative, self.prioritize_optimal_solutions, self.binary_folder_name)

    def _solve_ilasp_task(self, domain_id, agent_id):
        utils.mkdir(self.get_automaton_solution_folder(domain_id, agent_id))

        ilasp_task_filename = os.path.join(self.get_automaton_task_folder(domain_id ,agent_id),
                                           ISAAlgorithmBase.AUTOMATON_TASK_FILENAME % self.automaton_counters[domain_id][agent_id])

        ilasp_solution_filename = os.path.join(self.get_automaton_solution_folder(domain_id, agent_id),
                                               ISAAlgorithmBase.AUTOMATON_SOLUTION_FILENAME % self.automaton_counters[domain_id][agent_id])

        return solve_ilasp_task(ilasp_task_filename, ilasp_solution_filename, timeout=self.ilasp_timeout,
                                version=self.ilasp_version, max_body_literals=self.max_body_literals,
                                binary_folder_name=self.binary_folder_name, compute_minimal=self.ilasp_compute_minimal)

    def _parse_ilasp_solutions(self, last_automaton_filename):
        return ilasp_solution_parser.parse_ilasp_solutions(last_automaton_filename)

    '''
    Logging and Messaging Management Methods
    '''
    def _restore_uncheckpointed_files(self):       # don't change
        super()._restore_uncheckpointed_files()
        self._remove_uncheckpointed_files()

    def _remove_uncheckpointed_files(self):
        """Removes files which were generated after the last checkpoint."""
        for domain_id in range(self.num_domains):
            for agent_id in range(self.num_agents):
                counter = self.automaton_counters[domain_id][agent_id]
                self._remove_uncheckpointed_files_helper(self.get_automaton_task_folder(domain_id, agent_id), "task-", ".las", counter)
                self._remove_uncheckpointed_files_helper(self.get_automaton_solution_folder(domain_id, agent_id), "solution-", ".txt", counter)
                self._remove_uncheckpointed_files_helper(self.get_automaton_plot_folder(domain_id, agent_id), "plot-", ".png", counter)

    def _remove_uncheckpointed_files_helper(self, folder, prefix, extension, automaton_counter):
        if utils.path_exists(folder):
            files_to_remove = [os.path.join(folder, x) for x in os.listdir(folder)
                               if x.startswith(prefix) and int(x[len(prefix):-len(extension)]) > automaton_counter]
            utils.rm_files(files_to_remove)

    def _write_automaton_learning_episodes(self):
        for domain_id in range(self.num_domains):
            for agent_id in range(self.num_agents):
                automaton_learning_ep_folder = self.get_automaton_learning_episodes_folder(domain_id)
                utils.mkdir(automaton_learning_ep_folder)
                with open(self.get_automaton_learning_episodes_file(domain_id, agent_id), 'w') as f:
                    for episode in self.automaton_learning_episodes[domain_id]:
                        f.write(str(episode) + '\n')

    '''
    File Management Methods
    '''
    def get_automaton_task_folders(self):
        folders =[]
        for domain_id in range(self.num_domains):
            for agent_id in range(self.num_agents): 
                folders.append(self.get_automaton_task_folder(domain_id,agent_id))
        return folders

    def get_automaton_task_folder(self, domain_id,agent_id):
        return os.path.join(self.export_folder_names[domain_id], ISAAlgorithmBase.AUTOMATON_TASK_FOLDER, "agent_" + str(agent_id))

    def get_automaton_solution_folders(self):
        folders =[]
        for domain_id in range(self.num_domains):
            for agent_id in range(self.num_agents): 
                folders.append(self.get_automaton_solution_folder(domain_id,agent_id))
        return folders

    def get_automaton_solution_folder(self, domain_id, agent_id):
        return os.path.join(self.export_folder_names[domain_id], ISAAlgorithmBase.AUTOMATON_SOLUTION_FOLDER, "agent_" + str(agent_id))

    def get_automaton_plot_folders(self):
        folders =[]
        for domain_id in range(self.num_domains):
            for agent_id in range(self.num_agents): 
                folders.append(self.get_automaton_plot_folder(domain_id,agent_id))
        return folders
    
    def get_automaton_plot_folder(self, domain_id, agent_id):
        return os.path.join(self.export_folder_names[domain_id], ISAAlgorithmBase.AUTOMATON_PLOT_FOLDER, "agent_" + str(agent_id))

    def get_automaton_learning_episodes_files(self):
        files =[]
        for domain_id in range(self.num_domains):
            for agent_id in range(self.num_agents): 
                files.append(self.get_automaton_learning_episodes_file(domain_id,agent_id))
        return files
    
    def get_automaton_learning_episodes_file(self, domain_id, agent_id):
        return os.path.join(self.get_automaton_learning_episodes_folder(domain_id), ISAAlgorithmBase.AUTOMATON_LEARNING_EPISODES_FILENAME % agent_id )
    
    def get_automaton_learning_episodes_folder(self, domain_id):
        return os.path.join(self.export_folder_names[domain_id], "automaton_learning_episodes" )
