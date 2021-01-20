#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import math, random, pickle, itertools, copy, re, os.path, time
from functools import reduce
from pathlib import Path
import pdb, traceback, sys
import numpy as np

class ViewselEnv():
    def __init__(self, input_file_dir, max_num_queries, max_num_views, suffix, alpha):
        #The following are all fixed given DB (same in every episode)
        self.all_queries = pickle.load(open(input_file_dir/'all_queries.p', 'rb'))  #all possible queries
        self.all_query_freqs = pickle.load(open(input_file_dir/'query_freqs.p', 'rb'))

        #set reln sizes + joinsel outside of making env; should be the same each time. Specific to each DB.
        self.reln_sizes = pickle.load(open(input_file_dir/'reln_sizes.p', 'rb'))
        self.join_selectivities = pickle.load(open(input_file_dir/'join_selectivities.p', 'rb'))
        self.reln_attr_index = pickle.load(open(input_file_dir/'reln_attr_index.p', 'rb'))  #keys are relations. values are which attributes those relations contain

        self.max_num_queries = max_num_queries
        self.max_num_views = max_num_views
        self.RA = len(self.reln_attr_index)  #should be specific DB to DB

        if os.path.exists(input_file_dir/'base_reln_update_freqs.p'):
            self.base_reln_update_freqs = pickle.load(open(input_file_dir/'base_reln_update_freqs.p', 'rb'))  
        else:
            self.base_reln_update_freqs = [1 for i in range(len(self.reln_sizes))]

        self.alpha = alpha

        #dummy used to define state tensor shape when init DQN
        self.input_queries = np.zeros((self.max_num_queries, self.RA, self.RA))
        self.input_views = np.zeros((self.max_num_views, self.RA, self.RA))
        self.views_selected = np.zeros((self.max_num_views, self.RA, self.RA))
        #self.views_sub_which_queries = np.zeros((self.max_num_queries, self.max_num_views))
        self.query_rewrites = np.zeros((self.max_num_queries, self.max_num_views))
        self.state = np.concatenate((self.input_queries.flatten(), self.input_views.flatten(), self.views_selected.flatten()), axis=0)
        # self.state = np.concatenate((self.input_queries.flatten(), self.input_views.flatten(), 
        #     self.views_sub_which_queries.flatten(), self.views_selected), axis=0)

    def new_ep(self, input_tuple):    #Reset the state of the environment with new inputs
        #input query reps, viewsel, query rewrites, state
        self.input_queries = np.zeros((self.max_num_queries, self.RA, self.RA))
        self.input_views = np.zeros((self.max_num_views, self.RA, self.RA))
        self.views_selected = np.zeros((self.max_num_views, self.RA, self.RA))
        self.query_rewrites = np.zeros((self.max_num_queries, self.max_num_views))
        self.current_view_int = 0
        
        self.views_selected_list = np.zeros(self.max_num_views)
        self.views_sel_as_ints = []

        self.input_query_strs = input_tuple[0]
        self.num_input_qrys = len(self.input_query_strs)
        self.input_view_strs = input_tuple[1]
        self.view_sizes = input_tuple[2]
        self.view_costs = input_tuple[3]
        self.relns_of_view = input_tuple[4]
        self.relns_of_input_queries = input_tuple[5]
        self.qry_freqs = [1]*self.num_input_qrys

        # self.views_sub_which_queries = np.zeros((self.max_num_queries, self.max_num_views))
        # for q_ind, qry in enumerate(self.input_query_strs):
        #     for v_ind, view in enumerate(self.input_view_strs):
        #         if view.issubset(qry):
        #             self.views_sub_which_queries[q_ind][v_ind] = 1

        for i, qry in enumerate(self.input_query_strs):  #represent self.input_queries in vector
            for edge in qry:
                edge = tuple(edge)
                row_ind = self.reln_attr_index.index(edge[0])
                col_ind = self.reln_attr_index.index(edge[1])
                self.input_queries[i, row_ind, col_ind] = 1
                self.input_queries[i, col_ind, row_ind] = 1  #symmetric matrix
        for i, view in enumerate(self.input_view_strs):  #represent self.input_queries in vector
            view = list(view)
            for edge in view:
                edge = tuple(edge)
                row_ind = self.reln_attr_index.index(edge[0])
                col_ind = self.reln_attr_index.index(edge[1])
                self.input_views[i, row_ind, col_ind] = 1
                self.input_views[i, col_ind, row_ind] = 1  #symmetric matrix
                
        self.state = np.concatenate((self.input_queries.flatten(), self.input_views.flatten(), self.views_selected.flatten()), axis=0)
        # self.state = np.concatenate((self.input_queries.flatten(), self.input_views.flatten(), 
        #     self.views_sub_which_queries.flatten(), self.views_selected), axis=0)
        self.state = self.state.astype(int)
        # self.state = tf.cast(self.state, tf.int32)

        # Store what the agent tried
        self.lst_relns_used_in_rewriting = [[]]*self.num_input_qrys
        self.done = False
        self.maint_cost_lst = {} #view ID : maintcost, for selected views
        self.init_cost, self.qry_proc_costs = self._get_eval_cost(None, 'all', []) #q: proc cost of curr rewrite
        self.init_cost = (1-self.alpha)*self.init_cost
        self.reward = 0
        return self.state

    def step(self, action):
        if action == 0:
            view = list(self.input_view_strs[self.current_view_int])
            for edge in view:
                edge = tuple(edge)
                row_ind = self.reln_attr_index.index(edge[0])
                col_ind = self.reln_attr_index.index(edge[1])
                self.views_selected[self.current_view_int, row_ind, col_ind] = 0
                self.views_selected[self.current_view_int, col_ind, row_ind] = 0
            if self.current_view_int in self.views_sel_as_ints:
                self.views_sel_as_ints.remove(self.current_view_int)
        else:
            self.views_selected_list[self.current_view_int] = 1
            # self.views_sel_as_ints.append(self.current_view_int)

            #choose rewritings
            for q_ind, qry in enumerate(self.input_query_strs):
                if self.input_view_strs[self.current_view_int].issubset(qry):
                    no_sub_or_super = True
                    subviews_of_currview = []
                    for sel_view, v_bool in enumerate(self.query_rewrites[q_ind]):
                        if v_bool == 1: #view used in rewriting of query q_ind
                            is_sub = self.input_view_strs[self.current_view_int].issubset(self.input_view_strs[sel_view]) #v3
                            is_super = self.input_view_strs[sel_view].issubset(self.input_view_strs[self.current_view_int]) #v3
                            # if is_sub and is_super:
                            if is_sub:
                                no_sub_or_super = False  #don't add view if it's subview of existing view in rewriting
                            if is_super:
                                subviews_of_currview.append(sel_view)
                    if no_sub_or_super: #dont add if new view is subview of existing views in rewriting
                        new_eval_cost = self._get_eval_cost(self.current_view_int, q_ind, subviews_of_currview)
                        if self.qry_proc_costs[q_ind] > new_eval_cost:
                            for v_ind in subviews_of_currview: #if new view is superview of existing views in rewriting, must remove them to add
                                self.query_rewrites[q_ind][v_ind] = 0 #replace subviews in rewriting
                            self.query_rewrites[q_ind][self.current_view_int] = 1 #add superview to rewriting

            #remove any views not used in rewriting
            #for each view, check if it's in at least on rewriting. if not, remove it.
            for v_ind, v_bool in enumerate(self.views_selected_list):
                if v_bool == 1:
                    v_col = self.query_rewrites[:,v_ind]
                    if 1 not in v_col:
                        view = list(self.input_view_strs[v_ind])
                        for edge in view:
                            edge = tuple(edge)
                            row_ind = self.reln_attr_index.index(edge[0])
                            col_ind = self.reln_attr_index.index(edge[1])
                            self.views_selected[self.current_view_int, row_ind, col_ind] = 0
                            self.views_selected[self.current_view_int, col_ind, row_ind] = 0
                        self.views_selected_list[v_ind] = 0
                        if self.current_view_int in self.views_sel_as_ints:
                            self.views_sel_as_ints.remove(self.current_view_int)

        self.reward = self._get_reward(self.current_view_int)

        self.current_view_int += 1  #get to next
        if self.current_view_int < len(self.input_view_strs):
            view = list(self.input_view_strs[self.current_view_int])
            for edge in view:
                edge = tuple(edge)
                row_ind = self.reln_attr_index.index(edge[0])
                col_ind = self.reln_attr_index.index(edge[1])
                self.views_selected[self.current_view_int, row_ind, col_ind] = 1
                self.views_selected[self.current_view_int, col_ind, row_ind] = 1  #symmetric matrix
            self.views_selected_list[self.current_view_int] = 1
            self.views_sel_as_ints.append(self.current_view_int)
        else:
            self.done = True
        self.state = np.concatenate((self.input_queries.flatten(), self.input_views.flatten(), self.views_selected.flatten()), axis=0)
        # self.state = np.concatenate((self.input_queries.flatten(), self.input_views.flatten(), 
        #             self.views_sub_which_queries.flatten(), self.views_selected), axis=0)
        self.state = self.state.astype(int)
        # self.state = tf.cast(self.state, tf.int32)
        
        return self.state, self.reward, self.done, self.views_sel_as_ints

    def _get_eval_cost(self, new_view, qry_num, subviews_of_currview):
        processing_cost = 0 #cost of repr all queries
        qry_proc_costs = {}
        temp_relns_in_rewriting = copy.deepcopy(self.lst_relns_used_in_rewriting)
        if qry_num == 'all':
            qry_lst = range(self.num_input_qrys)
            if new_view != None:
                for q_ind in qry_lst:
                    if self.query_rewrites[q_ind][new_view] == 1:
                        temp_relns_in_rewriting[q_ind] = list(set(temp_relns_in_rewriting[q_ind] + \
                            self.relns_of_view[new_view]))
        else:
            qry_lst = [qry_num]
            temp_relns_in_rewriting[qry_num] = list(set(temp_relns_in_rewriting[qry_num] + \
                self.relns_of_view[new_view]))

        for q_ind in qry_lst:
            rewrite_lst_views_relns = []
            rewrite_lst_view_sizes = [] #sizes of views in this query's rewrite
            if qry_num != 'all':
                rewrite_lst_view_sizes.append(self.view_sizes[new_view])
                rewrite_lst_views_relns.append(self.relns_of_view[new_view])
            for i, v in enumerate(self.query_rewrites[q_ind]):
                if i in subviews_of_currview:
                    continue
                if v == 1:
                    rewrite_lst_view_sizes.append(self.view_sizes[i])
                    rewrite_lst_views_relns.append(self.relns_of_view[i]) #append b/c later on do: covered_relns_so_far = list(set(covered_relns_so_far + views_to_join[k] ))
            #get which relns are missing from views used in rewriting so far   
            uncovered_relns = list(set(self.relns_of_input_queries[q_ind]) - set(temp_relns_in_rewriting[q_ind]))
            for reln in uncovered_relns:
                rewrite_lst_view_sizes.append(self.reln_sizes[reln])
                rewrite_lst_views_relns.append([reln])

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
            else: #view = query
                covered_reln_sizes = [self.reln_sizes[reln] for reln in rewrite_lst_views_relns[0]]
                joinsels_interm_qry = [self.join_selectivities[edge] for edge in iq_copy]
                total_intermediate_sizes = reduce(lambda x, y: x*y, covered_reln_sizes + joinsels_interm_qry)

            qry_proc_costs[q_ind] = self.qry_freqs[q_ind] * \
                (sum(rewrite_lst_view_sizes) + total_intermediate_sizes )
            processing_cost += qry_proc_costs[q_ind]
        if qry_num == 'all':
            self.lst_relns_used_in_rewriting = copy.deepcopy(temp_relns_in_rewriting)
            return processing_cost, qry_proc_costs
        else:
            return processing_cost

    def _get_reward(self, new_view):
        #total maint cost: add up all maint costs of views
        #maint cost of view: proccost of view 
        self.maint_cost_lst = {}
        self.maint_cost = 0
        for i, v in enumerate(self.views_selected_list[:len(self.input_view_strs)]): #get views selected
            if v == 1: #view is selected
                self.maint_cost += self.view_costs[i]
                self.maint_cost_lst[i] = self.view_costs[i]
        self.eval_cost, self.qry_proc_costs = self._get_eval_cost(new_view, 'all', [])
        self.new_cost = (self.alpha*self.maint_cost + (1-self.alpha)*self.eval_cost)
        if self.init_cost - self.new_cost > 0:
            self.reward = (self.init_cost - self.new_cost)/(self.init_cost)
        else:
            self.reward = -1
        return self.reward
        
    def seed(self, seed):
        random.seed(seed)
        np.random.seed