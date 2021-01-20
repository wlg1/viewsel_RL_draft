#!/usr/bin/env python
# -*- coding: utf-8 -*-

# core modules
import math, random, pickle, copy, re, csv, time, os
from functools import reduce
import itertools
import pdb, traceback, sys
import numpy as np
import networkx as nx
from pathlib import Path

class ViewselEnv():
    def __init__(self, input_file_dir, max_num_views, suffix, file, alpha):
        self.all_queries = pickle.load(open(input_file_dir/'all_queries.p', 'rb'))  #all possible queries
        self.all_query_freqs = pickle.load(open(input_file_dir/'query_freqs.p', 'rb'))
        self.reln_attr_index = pickle.load(open(input_file_dir/'reln_attr_index.p', 'rb'))
        self.RA = len(self.reln_attr_index)  #should be specific DB to DB
        self.reln_sizes = pickle.load(open(input_file_dir/'reln_sizes.p', 'rb'))
        self.join_selectivities = pickle.load(open(input_file_dir/'join_selectivities.p', 'rb'))
        self.max_num_views = max_num_views

        if os.path.exists(input_file_dir/'base_reln_update_freqs.p'):
            self.base_reln_update_freqs = pickle.load(open(input_file_dir/'base_reln_update_freqs.p', 'rb'))  
        else:
            self.base_reln_update_freqs = [1 for i in range(len(self.reln_sizes))]

        self.file = file
        self.alpha = alpha

    def new_ep(self, input_tuple):    #Reset the state of the environment with new inputs
        self.input_query_strs = input_tuple[0]
        self.num_input_qrys = len(self.input_query_strs)
        self.input_view_strs = input_tuple[1]
        self.view_sizes = input_tuple[2]
        self.view_costs = input_tuple[3]
        self.relns_of_view = input_tuple[4]
        self.relns_of_input_queries = input_tuple[5]
        self.qry_freqs = [1]*self.num_input_qrys

        # Store what the agent tried
        self.lst_relns_used_in_rewriting = [[]]*self.num_input_qrys
        self.views_sel = [] #list of actions mapped to views
        self.done = False
        self.rewrites_dict = {i:[] for i in range(self.num_input_qrys)}  #rewrite is a list of view_ids
        self.maint_cost_lst = {}
        self.init_cost, self.qry_proc_costs, x = self._get_eval_cost(self.rewrites_dict, None, 'all')
        self.init_cost = (1-self.alpha)*self.init_cost
        self.new_cost = self.init_cost
        self.reward = 0

    def step(self, file):
        action_to_rewards = {}
        action_to_results = {}
        action_to_costs = {}
        for view_id in range(len(self.input_view_strs)):
            if view_id in self.views_sel:
                continue
            temp_views_sel = copy.deepcopy(self.views_sel)  #temp copy bc will discard if not better
            temp_views_sel.append(view_id)  #need a temp bc this has the potential view candidate
            #optimally rewrite queries using selected views
            temp_rewrites_dict = copy.deepcopy(self.rewrites_dict)
            for q_ind, qry in enumerate(self.input_query_strs):
                if self.input_view_strs[view_id].issubset(qry):
                    no_sub_or_super = True
                    subviews_of_action = []
                    for sel_view in temp_rewrites_dict[q_ind]:
                        is_sub = self.input_view_strs[view_id].issubset(self.input_view_strs[sel_view]) #v3
                        is_super = self.input_view_strs[sel_view].issubset(self.input_view_strs[view_id]) #v3
                        # isdisjoint = self.input_view_strs[sel_view].isdisjoint(self.input_view_strs[view_id])
                        # if is_sub or is_super:
                        if is_sub:
                        # if is_super:
                        # if not isdisjoint:
                        # if not isdisjoint and is_sub and is_super:
                            no_sub_or_super = False  #don't add view if it's subview of existing view in rewriting
                        if is_super: 
                            subviews_of_action.append(sel_view)
                    #if new view is superview of existing views in rewriting, must remove them to add
                    #ONLY REPLACE IF IT IMPROVES COST
                    # if no_sub_or_super:
                    #     temp_rewrites_dict[q_ind].append(view_id)
                    if subviews_of_action and no_sub_or_super:                     
                        temp_temp_rewrites_dict = copy.deepcopy(temp_rewrites_dict)
                        for view in subviews_of_action:
                            temp_temp_rewrites_dict[q_ind].remove(view)
                        temp_temp_rewrites_dict[q_ind].append(view_id)
                        new_eval_cost = self._get_eval_cost(temp_temp_rewrites_dict, view_id, q_ind)
                        if self.qry_proc_costs[q_ind] > new_eval_cost:
                            for view in subviews_of_action:
                                temp_rewrites_dict[q_ind].remove(view)
                            temp_rewrites_dict[q_ind].append(view_id)      
                    elif no_sub_or_super:                        
                        temp_temp_rewrites_dict = copy.deepcopy(temp_rewrites_dict)
                        temp_temp_rewrites_dict[q_ind].append(view_id)
                        new_eval_cost = self._get_eval_cost(temp_temp_rewrites_dict, view_id, q_ind)
                        if self.qry_proc_costs[q_ind] > new_eval_cost:
                            temp_rewrites_dict[q_ind].append(view_id)
                        # if view_id == 4 and q_ind == 0:
                        #     pdb.set_trace()
                        #     self.file.write('temp_temp_rewrites_dict[q_ind]: ' + repr(temp_temp_rewrites_dict[q_ind]) + '\n')    
                        #     self.file.write('view_id: ' + repr(view_id) + '\n')
                        #     self.file.write('self.qry_proc_costs[q_ind]: ' + repr(self.qry_proc_costs[q_ind]) + '\n')
                        #     self.file.write('new_eval_cost: ' + repr(new_eval_cost) + '\n')

            #remove any views not used in rewriting
            #for each view, check if it's in at least on rewriting. if not, remove it.
            for view in temp_views_sel:
                remove_bool = True
                for rewrites_lst in temp_rewrites_dict.values():
                    if view in rewrites_lst:
                        remove_bool = False
                if remove_bool:
                    temp_views_sel.remove(view)

            #check if new rewrites of potential new state beget lower cost than before
            #(new_reward, new_cost, lst_relns_used_in_rewriting, maint_cost_lst, qry_proc_costs) 
            reward_outputs = list(self._get_reward(temp_rewrites_dict, temp_views_sel, view_id))
            reward_outputs.append(temp_views_sel)
            reward_outputs.append(temp_rewrites_dict)
            reward_outputs.append(self.maint_cost)
            reward_outputs.append(self.eval_cost)
            reward_outputs.append(self.new_cost)
            reward_outputs.append(self.scaled_maint)
            reward_outputs.append(self.scaled_eval)
            action_to_results[view_id] = copy.deepcopy(reward_outputs)
            action_to_rewards[view_id] = action_to_results[view_id][0]
            action_to_costs[view_id] = (self.scaled_maint, self.scaled_eval, self.new_cost)

        file.write('\nView : Reward '+ repr(action_to_rewards) + '\n')
        file.write('View : (M,E,O)costs '+repr(action_to_costs) + '\n')

        best_view = max(action_to_rewards, key=action_to_rewards.get)
        #after finding best view, only select view if it's used in at least ONE rewriting.
        #if it's not used in at least one view, rewritings won't change, and the reward will never be greater.
        if action_to_rewards[best_view] > self.reward:
            old_cost = self.new_cost
            self.views_sel = copy.deepcopy(action_to_results[best_view][5])
            self.rewrites_dict = copy.deepcopy(action_to_results[best_view][6])
            self.reward = copy.deepcopy(action_to_results[best_view][0])
            self.new_cost = copy.deepcopy(action_to_results[best_view][1])
            self.lst_relns_used_in_rewriting = copy.deepcopy(action_to_results[best_view][2])
            self.maint_cost_lst = copy.deepcopy(action_to_results[best_view][3])
            self.prev_qry_proc_costs = copy.deepcopy(self.qry_proc_costs)
            self.qry_proc_costs = copy.deepcopy(action_to_results[best_view][4])    
            file.write('view sel: ' + repr(best_view) + '\n')             
            file.write('maint cost: ' + repr(action_to_results[best_view][7]) + '\n')
            file.write('eval cost: ' + repr(action_to_results[best_view][8]) + '\n')
            file.write('scaled maint cost: ' + repr(action_to_results[best_view][10]) + '\n')
            file.write('scaled eval cost: ' + repr(action_to_results[best_view][11]) + '\n')
            file.write('total cost: ' + repr(action_to_results[best_view][9]) + '\n')
            file.write('prev total cost: ' + str(old_cost) + '\n')
            file.write('reward: ' + repr(self.reward) + '\n')
            self._record_choice(file)
        else:
            self.done = True
            # self._record_choice(file)
        if len(self.views_sel) == len(self.input_view_strs):
            self.done = True
            # self._record_choice(file)
        return self.done

    def _get_eval_cost(self, rewrites_dict, new_view, qry_num):
        processing_cost = 0 #cost of repr all queries
        qry_proc_costs = {}
        temp_relns_in_rewriting = copy.deepcopy(self.lst_relns_used_in_rewriting)
        if qry_num == 'all':
            qry_lst = range(self.num_input_qrys)
        else:
            qry_lst = [qry_num]
        if new_view != None:
            for q_ind in qry_lst:
                if new_view in rewrites_dict[q_ind]:
                    temp_relns_in_rewriting[q_ind] = list(set(temp_relns_in_rewriting[q_ind] + \
                        self.relns_of_view[new_view]))

        for q_ind in qry_lst:
            # covered_edges = [] #save all uncovered edges 
            rewrite_lst_views_relns = []
            rewrite_lst_view_sizes = [] #sizes of views in this query's rewrite
            for v in rewrites_dict[q_ind]:
                # covered_edges += set(self.input_view_strs[v])
                rewrite_lst_view_sizes.append(self.view_sizes[v])
                rewrite_lst_views_relns.append(self.relns_of_view[v])
            #get which relns are missing from views used in rewriting so far   
            uncovered_relns = list(set(self.relns_of_input_queries[q_ind]) - set(temp_relns_in_rewriting[q_ind]))
            for reln in uncovered_relns:
                rewrite_lst_view_sizes.append(self.reln_sizes[reln])
                rewrite_lst_views_relns.append([reln])

            # uncovered_edges = set(self.input_query_strs[q_ind]) - set(covered_edges)
            # uncovered_edges = [tuple(edge) for edge in uncovered_edges]
            iq_copy = list(copy.deepcopy(self.input_query_strs[q_ind]))
            iq_copy = [tuple(edge) for edge in iq_copy]
            if rewrite_lst_view_sizes:
                total_intermediate_sizes = 0
                joinsels_interm_qry = []
                lst_tuples = list(zip(rewrite_lst_views_relns, rewrite_lst_view_sizes, \
                    [rel[0] for rel in rewrite_lst_views_relns]))
                views_to_join = sorted( lst_tuples, key=lambda tup: (tup[1], tup[2]) ) 
                views_to_join = [x[0] for x in views_to_join]
                covered_relns_so_far = views_to_join[0]
                covered_edges_so_far = []
                del views_to_join[0]
                while views_to_join:
                    k = 0
                    if len(views_to_join) > 1:
                        joinable_view = False
                        while k < len(views_to_join):
                            #loop thru uncov edges to find edge w/ left in covered, right in views_to_join[k]
                            for edge in iq_copy:
                                left_reln = edge[0].split('.')[0]
                                right_reln = edge[1].split('.')[0]
                                if (left_reln in covered_relns_so_far) and (right_reln in views_to_join[k]):
                                    joinable_view = True
                                if (right_reln in covered_relns_so_far) and (left_reln in views_to_join[k]):
                                    joinable_view = True
                            if joinable_view:
                                break
                            else:
                                k += 1
                    covered_relns_so_far = list(set(covered_relns_so_far + views_to_join[k] ))
                    for edge in iq_copy:
                        left_reln = edge[0].split('.')[0]
                        right_reln = edge[1].split('.')[0]
                        if (left_reln in covered_relns_so_far) and (right_reln in covered_relns_so_far):
                            if edge not in covered_edges_so_far:
                                joinsels_interm_qry.append(self.join_selectivities[edge] )
                                covered_edges_so_far.append(edge)
                    covered_reln_sizes = [self.reln_sizes[reln] for reln in covered_relns_so_far]
                    total_intermediate_sizes += reduce(lambda x, y: x*y, covered_reln_sizes + joinsels_interm_qry) 
                    del views_to_join[k]
                    # print(covered_relns_so_far)
                    # print(joinsels_interm_qry)
                    # print(reduce(lambda x, y: x*y, covered_reln_sizes + joinsels_interm_qry) )
            else: #view = query
                covered_reln_sizes = [self.reln_sizes[reln] for reln in rewrite_lst_views_relns[0]]
                joinsels_interm_qry = [self.join_selectivities[edge] for edge in iq_copy]
                total_intermediate_sizes = reduce(lambda x, y: x*y, covered_reln_sizes + joinsels_interm_qry)

            qry_proc_costs[q_ind] = self.qry_freqs[q_ind] * \
                (sum(rewrite_lst_view_sizes) + total_intermediate_sizes )
            processing_cost += qry_proc_costs[q_ind]

        if qry_num == 'all':
            return processing_cost, qry_proc_costs, temp_relns_in_rewriting
        else:
            return processing_cost

    def _get_reward(self, rewrites_dict, views_sel, new_view):
        maint_cost_lst = {}
        maint_cost = 0
        for v in views_sel: 
            maint_cost += self.view_costs[v]
            maint_cost_lst[v] = self.view_costs[v]
        eval_cost, qry_proc_costs, temp_relns_in_rewriting = \
            self._get_eval_cost(rewrites_dict, new_view, 'all')
        new_cost = (self.alpha*maint_cost + (1-self.alpha)*eval_cost)
        if self.init_cost - new_cost > 0:
            reward = (self.init_cost - new_cost)/(self.init_cost)
        else:
            reward = -1
        self.maint_cost = maint_cost
        self.scaled_maint = self.alpha*maint_cost
        self.eval_cost = eval_cost
        self.scaled_eval = (1-self.alpha)*eval_cost
        self.new_cost = new_cost
        return reward, new_cost, temp_relns_in_rewriting, maint_cost_lst, qry_proc_costs

    def _record_choice(self, file):
        # file.write('\nSelected Views:\n')
        file.write('Selected Views:\n')
        for v in self.maint_cost_lst.keys():
            file.write('    View #'+str(v)+': '+repr(self.input_view_strs[v]) + ', Maint cost: ' + repr(self.maint_cost_lst[v]) +'\n')
        file.write('Rewrites:\n')
        for x in range(self.num_input_qrys):
            file.write('Query ' + repr(x)+ ', ')
            file.write('cost diff: ' + repr(self.prev_qry_proc_costs[x] - self.qry_proc_costs[x])+ ', ')
            file.write('Prev Proc cost: ' + repr(self.prev_qry_proc_costs[x])+ ', ')
            file.write('New Proc cost: ' + repr(self.qry_proc_costs[x])+ '\n')
            for v in self.rewrites_dict[x]:  #each view is a list of edges
                file.write('    View #'+str(v)+': '+repr(self.input_view_strs[v])+ '\n')

    def seed(self, seed):
        random.seed(seed)
        np.random.seed

    def get_views(self):
        return (self.input_view_strs, self.view_sizes, self.view_costs, self.relns_of_view,
                self.relns_of_input_queries, self.qry_freqs)

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def main():
    #sys.argv: [1]- k, [2]-split # : from 1 to k, [3]- 0 is train, 1 is test
    #for testing on all new queries: [1]- 0 is train, 1 is test
    dataset_name = 'DS_newqrys_5'
    # dataset_name = 'Dataset101'
    # dataset_name = 'JOB_19'
    max_num_views = 15
    alpha = 0.1

    input_file_dir = Path.cwd().parent / "datasets" / dataset_name 
    if len(sys.argv) < 3:
        begin_ind = 0
        suffix = ''
        if int(sys.argv[1]) == 0:  #train
            output_name = 'viewsel_train_greedy_newcostmodel_maxviews_'+str(max_num_views)
            samples = pickle.load(open(input_file_dir/'train_samples.p', 'rb'))   
        else: #test
            output_name = 'viewsel_greedy_newcostmodel_maxviews_'+str(max_num_views)
            samples = pickle.load(open(input_file_dir/'test_samples.p', 'rb'))
    else:
        all_samples = pickle.load(open(input_file_dir/'train_samples.p', 'rb'))
        k = int(sys.argv[1])
        split_num = int(sys.argv[2]) - 1
        if len(all_samples) % k != 0:
            print('Choose k that evenly divides # of samples')
            sys.exit()
        group_size = len(all_samples) / k
        begin_ind = int( split_num * group_size)
        end_ind = int( (split_num + 1) * group_size)
        # test samples are the smaller group, train samples are everything else
        if int(sys.argv[3]) == 0:  # train
            output_name = 'viewsel_train_greedy_newcostmodel_maxviews_'+str(max_num_views)+'_alpha_'+str(alpha)
            samples = copy.deepcopy(all_samples)
            del samples[begin_ind : end_ind]
        else: # test
            output_name = 'viewsel_greedy_newcostmodel_maxviews_'+str(max_num_views)+'_alpha_'+str(alpha)
            samples = all_samples[begin_ind : end_ind]
        suffix = '_k_'+str(k)+'_split_' + str(split_num + 1)
        output_name = output_name + suffix

    all_rewards = []
    qry_to_numviews_sel = []
    qry_to_avgLenView = []
    output_fn = output_name+'.txt'
    file = open(input_file_dir/output_fn, 'w')
    file = open(input_file_dir/output_fn, 'a')

    env = ViewselEnv(input_file_dir, max_num_views, suffix, file, alpha)

    # samples = [samples[5]]
    for num, tup in enumerate(samples):
        input_queries = tup[0]
        env.new_ep(tup)

        if len(env.input_view_strs) == 0:  
            all_rewards.append(0)
            continue
        file.write('\n'+'<'*100)
        file.write('\nSample ' + str(num) + '#\n')
        for q, qry in enumerate(input_queries):
            file.write('Input query #' + str(q) + ': ' + repr(qry) + ', ')
            file.write('Proc cost: ' + repr(env.qry_proc_costs[q])+ '\n')
        for q, qry in enumerate(env.input_view_strs):
            file.write('View #' + str(q) + ': ' + repr(qry) + '\n')
        choice_num = 1
        while True:
            print('samp', num, 'step', choice_num)
            done = env.step(file)  #run algo
            if done:
                reward = env.reward
                break
            choice_num += 1
        all_rewards.append(round(reward, 4))
        qry_to_numviews_sel.append(len(env.maint_cost_lst))
        sumToAvg = 0
        for view_id in env.views_sel:
            view = env.input_view_strs[view_id]
            sumToAvg += len(view)
        if len(env.views_sel) > 0:
            qry_to_avgLenView.append(sumToAvg / len(env.views_sel))
        else:
            qry_to_avgLenView.append(0)

    output_csvfn = output_name+'.csv'
    with open(input_file_dir/output_csvfn, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['max num views: ', max_num_views, '', 'alpha: ', alpha])
        writer.writerow([])
        writer.writerow(['Samp num', 'Reward', '# views sel', 'avgLen view'])
        rew_sum=0
        for samp_num, rew in enumerate(all_rewards):
            writer.writerow([samp_num + begin_ind, rew, qry_to_numviews_sel[samp_num], qry_to_avgLenView[samp_num]])  #bc writerow works on lists, not ints
            rew_sum += rew
        writer.writerow(['Sum', rew_sum])  #bc writerow works on lists, not ints
        writer.writerow(['Avg', '', sum(qry_to_numviews_sel)/len(qry_to_numviews_sel), sum(qry_to_avgLenView)/len(qry_to_avgLenView)]) 

if __name__ == "__main__":
  try:
    main()
  except:
    type, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)