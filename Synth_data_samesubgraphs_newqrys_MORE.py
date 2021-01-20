import itertools, pickle, random, os, copy
import networkx as nx
from pathlib import Path
import pdb, traceback, sys
import numpy as np
from itertools import chain, repeat
import itertools
from functools import reduce

#generates variable queries + views
#tuples instead of frozenset for edges, uses lists for queries, reln-sizes are dict of reln : size
#workloads choose queries with more overlap

def gen_test_subgr(subgraphs, edges_of_training):
    while True:
        #remove edge not in 2 then add new from 2
        orig_subgr_1 = random.choice(subgraphs)
        new_subgr = set(copy.deepcopy(orig_subgr_1))
        
        edge = random.choice(list(set(orig_subgr_1)))
        new_subgr.remove(edge)
        while True:
            edge = random.choice(edges_of_training)
            if edge not in orig_subgr_1:    
                new_subgr.add(edge)
                break
                
        if frozenset(new_subgr) in subgraphs:
            continue
        
        edgelist = []
        for edge in new_subgr:
            edge = tuple(edge)
            left_reln = edge[0].split('.')[0]
            right_reln = edge[1].split('.')[0]
            edgelist.append((left_reln, right_reln))
        G = nx.Graph(edgelist)
        if nx.is_connected(G):
            break
    return frozenset(new_subgr)

def gen_train_superview(num_relns_in_DB, min_num_relns, max_num_relns, trees_or_graphs, num_small_relns):
    #ensure last reln not in view subtree, and is much bigger. this preserves join ordering
    num_relns = random.randint(min_num_relns, max_num_relns)
    relns = []
    relns_in_qry = []
    while len(relns) < num_relns:
      r = str(random.randint(1, num_small_relns))
      if r not in relns:
        relns.append(str(r))

    input_q = set()
    if trees_or_graphs == 'trees':
      while relns:
        #randomly pick a source relation from the partially generated tree
        if relns_in_qry:
          source = random.choice(relns_in_qry)
        else:
          source = random.choice(relns)
          relns_in_qry.append(source)
          relns.remove(source)  #if tree
        target = random.choice(relns) #if tree
        relns_in_qry.append(target)
        relns.remove(target)  #if tree
        source_node = source + '.1'
        target_node = target + '.1'
        edge = frozenset((source_node, target_node))
        input_q.add(edge)
    elif trees_or_graphs == 'graphs':
      if len(relns) == 2:
        num_edges = 1
      else:
        num_edges = min(10, random.randint(len(relns), len(relns)*(len(relns)-1) / 2))
      while len(input_q) < num_edges:  
        if relns_in_qry:
          source = random.choice(relns_in_qry)
        else: #no edges chosen yet
          source = random.choice(relns)
          relns_in_qry.append(source)
        relns_minus_source = copy.deepcopy(relns) 
        relns_minus_source.remove(source) 
        target = random.choice(relns_minus_source) 
        relns_in_qry.append(target)
        source_node = str(source) + '.1'
        target_node = str(target) + '.1'
        edge = frozenset((source_node, target_node))
        input_q.add(edge)
    return frozenset(input_q)

def gen_input_qry(num_relns_in_DB, trees_or_graphs, superview, max_relns_to_add_in_qry, num_small_relns):
    #ensure last reln not in view subtree, and is much bigger. this preserves join ordering
    num_new_relns = random.randint(1, max_relns_to_add_in_qry)
    new_relns = []
    while len(new_relns) < num_new_relns:
        r = str(random.randint(num_small_relns+1, num_relns_in_DB))
        if r not in new_relns:
            new_relns.append(r)
    relns_in_qry = []
    for edge in set(superview):
        edge = tuple(edge)
        left_reln = edge[0].split('.')[0]
        right_reln = edge[1].split('.')[0]
        if left_reln not in relns_in_qry:
            relns_in_qry.append(left_reln)
        if right_reln not in relns_in_qry:
            relns_in_qry.append(right_reln)

    input_q = set(superview)
    if trees_or_graphs == 'trees':
      while new_relns:
        #randomly pick a source relation from the partially generated tree
        if relns_in_qry:
          source = random.choice(relns_in_qry)
        else:
          source = random.choice(new_relns)
          relns_in_qry.append(source)
          relns.remove(source)  #if tree
        target = random.choice(new_relns) #if tree
        relns_in_qry.append(target)
        new_relns.remove(target)  #if tree
        source_node = source + '.1'
        target_node = target + '.1'
        edge = frozenset((source_node, target_node))
        input_q.add(edge)
    elif trees_or_graphs == 'graphs':
      # #randomly pick the number of edges
      # #max is # of nodes. #nodes = # reln attr pairs
      # if len(relns) == 2:
      #   num_edges = 1
      # else:
      #   num_edges = random.randint(min_num_edges_in_qry, max_num_edges_in_qry)
      # while len(input_q) < num_edges:  
      while new_relns:
        if relns_in_qry:
          source = random.choice(relns_in_qry)
        else:
          source = random.choice(new_relns)
          relns_in_qry.append(source)
        target = random.choice(new_relns) 
        relns_in_qry.append(target)
        new_relns.remove(target)  
        source_node = str(source) + '.1'
        target_node = str(target) + '.1'
        edge = frozenset((source_node, target_node))
        input_q.add(edge)
    return frozenset(input_q)

def main():
    # output_path = str(Path.cwd()) + '\\Dataset88\\'
    output_path = str(Path.cwd().parent) + "\\datasets\\" + '\\DS_newqrys_5\\'
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    else:
      while True:
        proceed_bool = input('WARNING: Dataset name already exists. Overwrite? (y/n) ')
        if proceed_bool == 'y':
          break
        elif proceed_bool == 'n':
          sys.exit()

    trees_or_graphs = 'graphs'  
    num_relns_in_DB = 9
    num_small_relns = 6
    max_relns_to_add_in_qry = 3
    min_num_queries_insamp = 10
    max_num_queries_insamp = 10
    total_train_superviews = 30
    total_test_superviews = 20
    min_num_relns_in_superview = 4
    max_num_relns_in_superview = 6
    min_superviews_insamp =  3
    max_superviews_insamp = min_superviews_insamp
    min_num_superviews_in_qry = 1
    max_num_superviews_in_qry = 2
    max_num_views = 15  #5 subviews per superview
    num_train_samples = 9000 #number of sets of input queries in training set
    num_test_samples = 1000
    num_aug_repeats = 10
    rand_view_bool = True
    #M = support_frac * len(query_set)
    # support_frac = M  / max_num_queries_insamp
    # support_frac = 0.1
    num_samples = num_train_samples + num_test_samples

    samples = []
    samples_as_sets = []
    train_samples = []  #store to make sure newly generated sample hasn't already been generated
    test_samples = []
    train_queries = []
    test_queries = []
    qry_to_samp = {}

    #generate total # of attributes in DB
    #randomly assign attributes to relns
    #each reln also has unique attr not shared by any other reln

    ##############################################################################################
    ### Generate reln sizes, joinsel
    ##############################################################################################

    # reln_sizes = {str(rel) : 100*random.randint(1,5) for rel in range(1,num_relns_in_DB + 1)}
    reln_sizes = {str(rel) : 100*random.randint(1,5) for rel in range(1,8)}
    reln_sizes_2 = {str(rel) : 100000*random.randint(1,5) for rel in range(8,num_relns_in_DB+1)}
    reln_sizes.update(reln_sizes_2)
    pickle.dump(reln_sizes, open(output_path+'reln_sizes.p', 'wb'))

    reln_attributes = {rel:tuple([1]) for rel in range(1, num_relns_in_DB + 1)}
    reln_attr_index = []  #index maps state matrix col index to (reln, attr)
    first_ind_of_reln = {}
    curr_pos = 0
    for reln, attr_tuple in reln_attributes.items():
      first_ind_of_reln[reln] = curr_pos
      curr_pos += len(attr_tuple) #doesn't work if only 1 attr in reln!
      for attr in attr_tuple:
          reln_attr_index.append(str(reln)+'.'+str(attr))
    pickle.dump(reln_attr_index, open(output_path+'reln_attr_index.p', 'wb'))

    join_selectivities = {}  #dict of join_cond (edge) to number for every possible join_cond in DB
    for row in reln_attr_index:
      for col in reln_attr_index:
          join_cond = (row, col)
          if join_cond not in join_selectivities:
              # jc_val = round(random.uniform(0.001, 0.01),3) 
              # js_bool = random.randint(0,1)
              # if js_bool == 0:
              #   jc_val = 0.001
              # else:
              #   jc_val = 0.01
              jc_val = 0.001
              join_selectivities[join_cond] = jc_val
              join_selectivities[(join_cond[1], join_cond[0])] = jc_val
    pickle.dump(join_selectivities, open(output_path+'join_selectivities.p', 'wb'))

    js_mat = np.zeros((len(reln_attr_index), len(reln_attr_index)))
    for join_cond, join_sel in join_selectivities.items():
      row_ind = reln_attr_index.index(join_cond[0])
      col_ind = reln_attr_index.index(join_cond[1])
      js_mat[row_ind][col_ind] = join_sel

    ##############################################################################################
    ### Generate training samples
    ##############################################################################################
    #use a list to avoid random order of queries when repr in state matrix bc want consistent test results
    #thus [q1,q2] != [q2,q1]. Use this to allow NN to learn multiple orderings of same result.

    #1.

    #gen training superviews
    subgraphs = []
    while len(subgraphs) < total_train_superviews:
        superview = gen_train_superview(num_relns_in_DB, min_num_relns_in_superview, max_num_relns_in_superview, trees_or_graphs, num_small_relns)
        if superview not in subgraphs:
            subgraphs.append(superview)
            
    edges_of_training = []
    for superview in subgraphs:
        for edge in superview:
            edges_of_training.append(edge)
            
    #gen test superviews
    test_subgraphs = []
    while len(test_subgraphs) < total_test_superviews:
        superview = gen_test_subgr(subgraphs, edges_of_training)
        if superview not in test_subgraphs and superview not in subgraphs:
            test_subgraphs.append(superview)
    
    subgr_views_all_samps = []
    while len(samples) < num_test_samples:
        print(len(samples), num_samples)
        input_qry_set = []
        num_input_qrys = random.randint(min_num_queries_insamp, max_num_queries_insamp)
        
        #choose rand num superviews, then rand choose superviews
        num_superviews = random.randint(min_superviews_insamp, max_superviews_insamp)
        superviews_of_samp = random.sample(test_subgraphs, num_superviews)
        
        #the superviews of each qry in samp
        superviews_of_all_qrys = []
        for i in range(num_input_qrys):
            while True:
                #decide how many superviews each query has
                num_superviews_in_qry = random.randint(min_num_superviews_in_qry, max_num_superviews_in_qry)
                superviews_in_qry = random.sample(superviews_of_samp, num_superviews_in_qry)
            
                # make sure their union forms graph
                superviews_in_qry = [set(x) for x in superviews_in_qry]
                union_superviews_in_qry = set.union(*superviews_in_qry)
                    
                edgelist = []
                for edge in union_superviews_in_qry:
                    edge = tuple(edge)
                    left_reln = edge[0].split('.')[0]
                    right_reln = edge[1].split('.')[0]
                    edgelist.append((left_reln, right_reln))
                G = nx.Graph(edgelist)
                if nx.is_connected(G):
                    break
            superviews_of_all_qrys.append(union_superviews_in_qry)

        curr_subgr_ind = 0
        while len(input_qry_set) < num_input_qrys:
            superview = superviews_of_all_qrys[curr_subgr_ind]
            input_q = gen_input_qry(num_relns_in_DB, trees_or_graphs, superview, max_relns_to_add_in_qry, num_small_relns)
            if input_q not in input_qry_set:
                input_qry_set.append(input_q)
                curr_subgr_ind += 1

        if set(input_qry_set) not in samples_as_sets:
            subgr_views_all_samps.append(superviews_of_samp)
            samples.append(input_qry_set)  
            samples_as_sets.append(set(input_qry_set))  #keep as set bc need to check if view is subset of qry
            for input_q in input_qry_set:
                if input_q not in test_queries + train_queries:
                    test_queries.append(input_q)
                    qry_to_samp[input_q] = [len(samples)-1]
                else:
                    qry_to_samp[input_q].append(len(samples)-1)

    while len(samples) < num_samples:
        print(len(samples), num_samples)
        input_qry_set = []
        num_input_qrys = random.randint(min_num_queries_insamp, max_num_queries_insamp)
        
        #choose rand num superviews, then rand choose superviews
        num_superviews = random.randint(min_superviews_insamp, max_superviews_insamp)
        superviews_of_samp = random.sample(test_subgraphs, num_superviews)
        
        #the superviews of each qry in samp
        superviews_of_all_qrys = []
        for i in range(num_input_qrys):
            while True:
                #decide how many superviews each query has
                num_superviews_in_qry = random.randint(min_num_superviews_in_qry, max_num_superviews_in_qry)
                superviews_in_qry = random.sample(superviews_of_samp, num_superviews_in_qry)
            
                # make sure their union forms graph
                superviews_in_qry = [set(x) for x in superviews_in_qry]
                union_superviews_in_qry = set.union(*superviews_in_qry)
                    
                edgelist = []
                for edge in union_superviews_in_qry:
                    edge = tuple(edge)
                    left_reln = edge[0].split('.')[0]
                    right_reln = edge[1].split('.')[0]
                    edgelist.append((left_reln, right_reln))
                G = nx.Graph(edgelist)
                if nx.is_connected(G):
                    break
            superviews_of_all_qrys.append(union_superviews_in_qry)
    
        curr_subgr_ind = 0
        while len(input_qry_set) < num_input_qrys:
            superview = superviews_of_all_qrys[curr_subgr_ind]
            input_q = gen_input_qry(num_relns_in_DB, trees_or_graphs, superview, max_relns_to_add_in_qry, num_small_relns)
            if input_q not in input_qry_set and input_q not in test_queries:
                input_qry_set.append(input_q)
                curr_subgr_ind += 1
    
        if set(input_qry_set) not in samples_as_sets:
            subgr_views_all_samps.append(superviews_of_samp)
            samples.append(input_qry_set)  
            samples_as_sets.append(set(input_qry_set))  #keep as set bc need to check if view is subset of qry
            for input_q in input_qry_set:
                if input_q not in test_queries + train_queries:
                    train_queries.append(input_q)
                    qry_to_samp[input_q] = [len(samples)-1]
                else:
                    qry_to_samp[input_q].append(len(samples)-1)

    ##############################################################################################
    ### Generate frequencies
    ##############################################################################################

    all_queries = train_queries + test_queries

    # query_freqs = [10*random.randint(1,10) for q_ind in range(len(all_queries))]
    query_freqs = [1 for q_ind in range(len(all_queries))]
    pickle.dump(query_freqs, open(output_path+'query_freqs.p', 'wb'))
    f = open(output_path+'queries.txt', 'w')  #reset old file
    file = open(output_path+'queries.txt', 'a')
    for i, qry in enumerate(all_queries):
      file.write(repr(qry)+ '\n')
      file.write('     In '+str(len(qry_to_samp[qry]))+' workload(s): ' + repr(qry_to_samp[qry])+ '\n')

    # base_reln_update_freqs = {str(rel) : 10*random.randint(1,3) for rel in range(1,num_relns_in_DB + 1)}
    base_reln_update_freqs = {str(rel) : 1 for rel in range(1,num_relns_in_DB + 1)}
    pickle.dump(base_reln_update_freqs, open(output_path+'base_reln_update_freqs.p', 'wb'))
    # f = open(output_path+'base_reln_update_freqs.txt', 'w')  #reset old file
    # file = open(output_path+'base_reln_update_freqs.txt', 'a')
    # for reln, freq in base_reln_update_freqs.items():
    #   file.write(str(freq) + ' : ' + repr(reln)+ '\n')

    ##############################################################################################
    ### Generate views
    ##############################################################################################

    def get_maint_cost(view_relns, view, reln_sizes, join_selectivities):
        processing_cost = 0
        rewrite_lst_view_sizes = []
        rewrite_lst_views_relns = []
        for reln in view_relns:
            rewrite_lst_view_sizes.append(reln_sizes[reln])
            rewrite_lst_views_relns.append([reln])

        iq_copy = list(copy.deepcopy(view))
        iq_copy = [tuple(edge) for edge in iq_copy]
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
                        joinsels_interm_qry.append(join_selectivities[edge] )
                        covered_edges_so_far.append(edge)
            covered_reln_sizes = [reln_sizes[reln] for reln in covered_relns_so_far]
            total_intermediate_sizes += reduce(lambda x, y: x*y, covered_reln_sizes + joinsels_interm_qry) 
            del views_to_join[k]

        processing_cost = 1 * (sum(rewrite_lst_view_sizes) + total_intermediate_sizes )
        return processing_cost

    def generate_views_2(input_query_strs, superviews, support_frac, max_num_views):
        #start_time = time.time()
        support = support_frac * len(input_query_strs)
        lst_views = []
        lst_views_greater_one = []
        view_to_freq = {}
        for query in input_query_strs:
            query = list(query)
            views_by_num_edges = {1:[]} # num_edges_in_view : views
            for edge in query:
                if frozenset(edge) in view_to_freq:
                    freq_val = view_to_freq[frozenset(edge)]
                else:
                    freq_val = 0
                    for qry in input_query_strs:
                        if set([edge]).issubset(qry):
                            freq_val += 1
                    view_to_freq[frozenset(edge)] = freq_val   
                if freq_val >= support:
                    views_by_num_edges[1].append(set([edge]))
            edge_to_neighbors = {}  # edge : all edges that neighbor the key edge's RELATIONS
            for edge_1 in query:
                edge_1 = tuple(edge_1)
                edge_1_relations = [edge_1[0].split('.')[0], edge_1[1].split('.')[0]]
                for reln in edge_1_relations:
                    for edge_2 in query:
                        edge_2 = tuple(edge_2)
                        edge_2_relations = [edge_2[0].split('.')[0], edge_2[1].split('.')[0]]
                        if reln in edge_2_relations and edge_1 != edge_2:
                            if frozenset(edge_1) in edge_to_neighbors:
                                edge_to_neighbors[frozenset(edge_1)].append(frozenset(edge_2))  
                            else:
                                edge_to_neighbors[frozenset(edge_1)] = [frozenset(edge_2)]

            # add only edges to new view
            if len(query) > 10:
                end_len = 10
            else:
                end_len = len(query)
            
            for num_edges in range(2, end_len):
                new_views_by_num_edges = []
                prev_lvl = views_by_num_edges[num_edges - 1]
                for ii, lower_view in enumerate(prev_lvl):
                    lower_view_neighbors = []
                    for edge in list(lower_view):
                        lower_view_neighbors += edge_to_neighbors[edge]
                    lower_view_neighbors = list(set(lower_view_neighbors))
                    for neighbor in lower_view_neighbors:
                        new_view = lower_view.union(set([neighbor]))
                        if frozenset(new_view) in view_to_freq:
                            freq_val = view_to_freq[frozenset(new_view)]
                        else:
                            freq_val = 0
                            for qry in input_query_strs:
                                if set(new_view).issubset(qry):
                                    freq_val += 1
                            view_to_freq[frozenset(new_view)] = freq_val
                        if freq_val >= support and new_view not in new_views_by_num_edges:
                            if len(new_view) == num_edges:
                                new_views_by_num_edges.append(new_view)
                if new_views_by_num_edges:
                    views_by_num_edges[num_edges] = new_views_by_num_edges[:max_num_views*10] 
                else:
                    break

            #dont use edges as candviews yet
            views_greater_one = copy.deepcopy(views_by_num_edges)
            del views_greater_one[1]
            # del views_greater_one[2]

            for v_lst in views_by_num_edges.values():
                lst_views.extend(v_lst)
            for v_lst in views_greater_one.values():
                lst_views_greater_one.extend(v_lst)
        
        pdb.set_trace()
        
        if len(lst_views) > max_num_views:
            if len(lst_views_greater_one) > max_num_views//3:
                # superviews = random.sample(lst_views_greater_one, max_num_views//3)
                # superviews = random.sample(views_by_num_edges[N], 2)
                cand_views = []
                #for every candview, split into a subview+its compl by rand rmv an edge from candview
                for view in superviews:
                    cand_views.append(view)
                    need_subview_lenTwo = True
                    for i in range(1):
                        while True:
                            if need_subview_lenTwo:
                                rmvd_edge = random.sample(view, 1)[0]
                                subview = view - {rmvd_edge}
                                rmvd_edge = random.sample(subview, 1)[0]
                                subview = subview - {rmvd_edge}
                            # else:
                            #     rmvd_edge = random.sample(view, 1)[0]
                            #     subview = view - {rmvd_edge}
                            edgelist = []
                            for edge in subview:
                                edge = tuple(edge)
                                left_reln = edge[0].split('.')[0]
                                right_reln = edge[1].split('.')[0]
                                edgelist.append((left_reln, right_reln))
                            G = nx.Graph(edgelist)
                            
                            compl = view - subview
                            compl_edgelist = []
                            for edge in compl:
                                edge = tuple(edge)
                                left_reln = edge[0].split('.')[0]
                                right_reln = edge[1].split('.')[0]
                                compl_edgelist.append((left_reln, right_reln))
                            G_compl = nx.Graph(compl_edgelist)
                            if nx.is_connected(G) and nx.is_connected(G_compl):
                                if need_subview_lenTwo:
                                    need_subview_lenTwo = False
                                break
                        if subview not in cand_views:
                            cand_views.append(subview)
                        if compl not in cand_views:
                            cand_views.append(compl)
                #gen edges as views
                k = 0
                while len(cand_views) < max_num_views:
                    # new_v = random.sample(lst_views, 1)[0]
                    new_v = random.sample(views_by_num_edges[1], 1)[0]
                    if int(tuple(tuple(new_v)[0])[0].split('.')[0]) in range(8,num_relns_in_DB+1):
                        continue
                    if int(tuple(tuple(new_v)[0])[1].split('.')[0]) in range(8,num_relns_in_DB+1):
                        continue
                    if new_v not in cand_views:
                        cand_views.append(new_v)
                    k += 1
                    if k > max_num_views*4: 
                        # cand_views = random.sample(lst_views, max_num_views)
                        break
            else:
                cand_views = random.sample(lst_views, max_num_views)
        else:
            cand_views = lst_views
        #print("--- %s seconds ---" % (time.time() - start_time))
        return cand_views


    def generate_views_3(input_query_strs, superviews, max_num_views):
        cand_views = []
        #for every superview, split into a subview+its compl by rand rmv several edges from superview
        for superview in superviews:
            cand_views.append(superview)
            for i in range(2):  #5 views per superview
                while True:
                    len_subgraph = random.randint(1, len(superview)-1)
                    rmvd_edges = random.sample(superview, len_subgraph)
                    
                    subview = superview - set(rmvd_edges)
                    edgelist = []
                    for edge in subview:
                        edge = tuple(edge)
                        left_reln = edge[0].split('.')[0]
                        right_reln = edge[1].split('.')[0]
                        edgelist.append((left_reln, right_reln))
                    G = nx.Graph(edgelist)
                    
                    compl = superview - subview
                    compl_edgelist = []
                    for edge in compl:
                        edge = tuple(edge)
                        left_reln = edge[0].split('.')[0]
                        right_reln = edge[1].split('.')[0]
                        compl_edgelist.append((left_reln, right_reln))
                    G_compl = nx.Graph(compl_edgelist)
                    
                    # if subview in cand_views or compl in cand_views:
                    #     continue
                    
                    if nx.is_connected(G) and nx.is_connected(G_compl):
                        if subview not in cand_views:
                            cand_views.append(subview)
                        if compl not in cand_views:
                            cand_views.append(compl)
                        break
        return cand_views

    # qry_to_views_path = 'qry_to_views_max_'+str(max_num_views)+'_sf_'+str(support_frac)+'.p' 
    # train_samples = samples[:num_train_samples]
    # test_samples = samples[num_train_samples:]
    # train_samples = list(chain.from_iterable(zip(*repeat(train_samples, 3))))
    # samples = train_samples + test_samples
    # samples = list(chain.from_iterable(zip(*repeat(samples, num_aug_repeats))))
    qry_to_views = []
    qry_view_pairs = []
    for snum, input_query_strs in enumerate(samples):
        print(snum, 'viewgen')
        orig_iqs = copy.deepcopy(input_query_strs)
        input_query_strs = copy.deepcopy(list(input_query_strs))
        relns_of_input_queries = [0]*len(input_query_strs)
        for i, iq in enumerate(input_query_strs):
            relns_of_query = []
            for edge in iq:
                edge = tuple(edge)
                left_reln = edge[0].split('.')[0]
                right_reln = edge[1].split('.')[0]
                if left_reln not in relns_of_query:
                    relns_of_query.append(left_reln)
                if right_reln not in relns_of_query:
                    relns_of_query.append(right_reln)
            relns_of_input_queries[i] = relns_of_query
        superviews = subgr_views_all_samps[snum]
        # input_view_strs = generate_views_2(input_query_strs, superviews, support_frac, max_num_views)  
        input_view_strs = generate_views_3(input_query_strs, superviews, max_num_views)  
        
        for i in range(num_aug_repeats):
            copy_views = copy.deepcopy(input_view_strs) 

            if rand_view_bool:
                random.shuffle(copy_views)

            view_sizes = {} #view position ID : size, for selected views
            view_costs = {}
            relns_of_views = {}
            for i, view in enumerate(copy_views):
                view_size = 1
                relns_of_view = []
                for edge in view:
                    edge = tuple(edge)
                    left_reln = edge[0].split('.')[0]
                    right_reln = edge[1].split('.')[0]
                    join_sel = join_selectivities[edge]
                    view_size *= join_sel
                    if left_reln not in relns_of_view:
                        view_size *= reln_sizes[left_reln]
                        relns_of_view.append(left_reln)
                    if right_reln not in relns_of_view:
                        view_size *= reln_sizes[right_reln]
                        relns_of_view.append(right_reln)
                u_freq = 0
                for reln in relns_of_view:
                    u_freq += base_reln_update_freqs[reln]
                u_freq = u_freq / len(relns_of_view)
                view_sizes[i] = view_size
                # pdb.set_trace()
                view_costs[i] = 3*len(view) * get_maint_cost(relns_of_view, view, reln_sizes, join_selectivities)
                relns_of_views[i] = sorted(relns_of_view)
            qry_to_views.append((orig_iqs, copy_views, view_sizes, view_costs, relns_of_views, relns_of_input_queries))

    ##############################################################################################
    ### Save to files
    ##############################################################################################
    # train_samples = qry_to_views[:(num_train_samples*num_aug_repeats)]
    # test_samples = qry_to_views[(num_train_samples*num_aug_repeats):]

    # test_samples = qry_to_views[:(num_test_samples*num_aug_repeats)]
    # train_samples = qry_to_views[(num_test_samples*num_aug_repeats):]

    train_samples = qry_to_views

    pickle.dump(train_samples, open(output_path+'train_samples.p', 'wb'))
    # pickle.dump(test_samples, open(output_path+'test_samples.p', 'wb'))
    pickle.dump(all_queries, open(output_path+'all_queries.p', 'wb'))

    f = open(output_path+'samples.txt', 'w')  #reset old file
    file = open(output_path+'samples.txt', 'a')
    for samp_num, tup in enumerate(qry_to_views):
      input_qry_set = tup[0]
      input_qry_set = [list(qry) for qry in input_qry_set] #change from frozenset for better readability
      new_input_qry_set = []
      for qry in input_qry_set:
        new_input_qry_set.append([tuple(edge) for edge in qry])
      file.write('\nSample ' + str(samp_num) + '#\n')
      for i, qry in enumerate(new_input_qry_set):
        file.write('Query '+str(i)+': '+repr(qry)+ '\n')

      input_view_set = tup[1]
      input_view_set = [list(view) for view in input_view_set] #change from frozenset for better readability
      new_input_view_set = []
      for view in input_view_set:
        new_input_view_set.append([tuple(edge) for edge in view])
      for i, view in enumerate(new_input_view_set):
        file.write('View '+str(i)+': '+repr(view)+ ', maint: '+repr(qry_to_views[samp_num][3][i]) + '\n')

    file.write('\nReln sizes:\n')
    file.write(repr(reln_sizes)+ '\n')
    file.write('\nReln attributes:\n')
    file.write(repr(reln_attr_index)+ '\n')
    file.write('\nJoinsel matrix:\n')
    file.write(repr(js_mat)+ '\n')

    # pdb.set_trace()

if __name__ == "__main__":
  try:
    main()
  except:
    type, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)

