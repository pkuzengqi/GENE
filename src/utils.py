import numpy as np
import torch
import dgl
import os
import json

from print_hook import PrintHook


type_class_dict = {
    'event': 36,
    'entity': 15,
    'event2entity': 238, #38
    'event2event':26,
    'coref': 2,
    'coref+': 2,
    'coref-': 2,
    'node': 51,
    'temp': 6
}
hetero2homo_type_dict = {
    'event': 0,
    'entity': 1,
    'event2entity': 0,
    'entity2entity': 1,
    'narr': 2,
    'coref':3,
    'eer':4
}

vocab_size_dict = {
    'train': {'event':4353,'entity':3688,'all':8041},
    'dev': {'event':494,'entity':667,'all':1161},
    'test': {'event':424,'entity':750,'all':1174}
}


def get_metapath_dict(etypes, data_path='../data/ace/'):
    # correspond to three views
    metapath_dict = {
        'entity2entity': [],
        'event2entity': [],
        'event2event': [],
        'multipath': []
    }
    with open(os.path.join(data_path,'ace.type.txt'),'r') as f:
        lines = f.readlines()
        for line in lines:
            d = line.strip().split('\t')
            i = int(d[0]) #i=0,1,2means UNK/MSK/NA
            n = d[1]
            if i > 2 and n.startswith('Event2Entity') and n in etypes:
                metapath_dict['event2entity'].append(n)
                metapath_dict['event2entity'].append('INV_'+n)
            elif i > 2 and n.startswith('Entity2Entity') and n in etypes:
                metapath_dict['entity2entity'].append(n)
            elif i > 2 and n.startswith('Event2Event') and n in etypes:
                metapath_dict['event2event'].append(n)

    return metapath_dict

def get_argroletyping_masks(data_path='./data/ace/'):
    # 238 classes --> 238 0/1

    event2arg = dict() # { 'attack':[]}
    with open(os.path.join(data_path,'ace.type.txt'),'r') as f:
        lines = f.readlines()
        for line in lines:
            d = line.strip().split('\t')
            i = int(d[0]) #i=0,1,2means UNK/MSK/NA
            n = d[1]
            eve = ':'.join(n.split(':')[1:3])
            if i > 2 and n.startswith('EventType'):
                event2arg.setdefault(eve,[]) # 'Personnel:Nominate'
            elif i > 2 and n.startswith('ArgRole'):
                event2arg[eve].append(i)

    n = type_class_dict['event2entity'] #238
    mask = np.zeros((n,n))
    for eve, arglist in event2arg.items():
        m = np.zeros(n)
        for i in arglist:
            m[i] = 1
        for i in arglist:
            mask[i] = m

    return mask



# used in evaluator
def construct_dglgraph(data_path, file_prefix='test', emb_path='./emb/',load_version='', homo=False, datainput='narr argrole temp'):

    labels = json.load(open(os.path.join(data_path, file_prefix + '.label.json'), 'r'))

    event_file = os.path.join(emb_path, load_version+'.'+file_prefix+'.eventemb.npy')
    if os.access(event_file, os.F_OK):
        entity_file = os.path.join(emb_path, load_version+'.'+file_prefix+'.entityemb.npy')
    else:
        event_file = os.path.join(data_path, file_prefix+'.eventemb.npy')
        entity_file = os.path.join(data_path, file_prefix+'.entityemb.npy')
    event_emb = np.load(event_file)
    entity_emb = np.load(entity_file)
    print('Graph Construction: loaded %s and %s' % (event_file, entity_file))


    # construct hetero graph
    dgld = dict()
    dgld[('event', 'event2entity', 'entity')] = [(t[0], t[1]) for t in labels['event2entity']]
    # dgld[('event', 'event2event', 'event')] = [(t[0], t[1]) for t in labels['event2event']]
    dgld[('event', 'narr', 'event')] = [(t[0], t[1]) for t in labels['narr']]
    dgld[('event', 'coref', 'event')] = [(t[0], t[1]) for t in labels['coref']]
    # dgld[('event', 'eer', 'event')] = [(t[0], t[1]) for t in labels['eer']]
    dgld[('event', 'argrole', 'entity')] = [(t[0], t[1]) for t in labels['argrole']]
    dgld[('event', 'temp', 'event')] = [(t[0], t[1]) for t in labels['temp']]
    dgld[('entity', 'entity2entity', 'entity')] = [(t[0], t[1]) for t in labels['entity2entity']]
    g = dgl.heterograph(dgld)
    g.nodes['event'].data['x'] = torch.tensor(event_emb)
    g.nodes['entity'].data['x'] = torch.tensor(entity_emb)
    g.nodes['event'].data['y'] = torch.tensor([t[0] for t in labels['event']])
    g.nodes['entity'].data['y'] = torch.tensor([t[0] for t in labels['entity']])
    g.edges['event2entity'].data['y'] = torch.tensor([t[2] for t in labels['event2entity']])
    g.edges['entity2entity'].data['y'] = torch.tensor([t[2] for t in labels['entity2entity']])
    #### add event/entity id ####
    event_cnt = event_emb.shape[0]
    entity_cnt = entity_emb.shape[0]
    g.nodes['event'].data['id'] = torch.tensor(list(range(event_cnt)))
    g.nodes['entity'].data['id'] = torch.tensor([i+event_cnt for i in range(entity_cnt)])


    # event-event relations cotrol by using string variable
    if 'narr' in datainput:
        g.edges['narr'].data['y'] = torch.tensor([t[2] for t in labels['narr']])
    if 'coref' in datainput:
        g.edges['coref'].data['y'] = torch.tensor([t[2] for t in labels['coref']])
    if 'eer' in datainput:
        g.edges['eer'].data['y'] = torch.tensor([t[2] for t in labels['eer']])
    if 'argrole' in datainput:
        g.edges['argrole'].data['y'] = torch.tensor([t[2] for t in labels['argrole']])
    if 'temp' in datainput:
        g.edges['temp'].data['y'] = torch.tensor([t[2] for t in labels['temp']])



    if homo:
        g.nodes['entity'].data['y'] += type_class_dict['event']
        homo_g = dgl.to_homo(g)
        g = dgl.DGLGraph()
        g.add_nodes(homo_g.number_of_nodes('_N'),homo_g.ndata)
        u,v = homo_g.all_edges(form='uv',order='eid')
        g.add_edges(u,v,homo_g.edata)
        g.readonly(True) #sampler needs readonly graph



    return g

def construct_homo_from_hetero_dglgraph(g):

    homo_g = dgl.to_homo(g) # still a heterograph in dgl

    hg = dgl.DGLGraph()
    hg.add_nodes(homo_g.number_of_nodes('_N'))
    u, v = homo_g.all_edges(form='uv', order='eid')
    hg.add_edges(u, v, homo_g.edata)
    hg.readonly(True)  # sampler needs readonly graph
    return hg

# used in training model
def construct_hetero_dglgraph(data_path, file_prefix='test', emb_path='./emb/',load_version=''):

    labels = json.load(open(os.path.join(data_path, file_prefix + '.label.json'), 'r'))

    event_file = os.path.join(emb_path, load_version+'.'+file_prefix+'.eventemb.npy')
    if os.access(event_file, os.F_OK):
        entity_file = os.path.join(emb_path, load_version+'.'+file_prefix+'.entityemb.npy')
    else:
        event_file = os.path.join(data_path, file_prefix+'.eventemb.npy')
        entity_file = os.path.join(data_path, file_prefix+'.entityemb.npy')
    event_emb = np.load(event_file)
    entity_emb = np.load(entity_file)
    print('Graph Construction: loaded %s and %s' % (event_file, entity_file))

    # get type dict
    ent2ent_typedict = []
    eve2ent_typedict = []
    eve2eve_typedict = []
    with open(os.path.join(data_path,'ace.type.txt'),'r') as f:
        lines = f.readlines()
        for line in lines:
            d = line.strip().split('\t')
            n = d[1]
            if n.startswith('Event2Entity'):
                eve2ent_typedict.append(n)
            elif n.startswith('Entity2Entity'):
                ent2ent_typedict.append(n)
            elif n.startswith('Event2Event'):
                eve2eve_typedict.append(n)


    ##### construct hetero graph with dictionary
    dgld = dict()

    for t in labels['event2entity']:
        type_id = t[2]
        type_name = eve2ent_typedict[type_id]
        dgld.setdefault(('event', type_name, 'entity'),[])
        dgld[('event', type_name, 'entity')].append((t[0],t[1]))
        dgld.setdefault(('entity', 'INV_'+type_name, 'event'), [])
        dgld[('entity', 'INV_'+type_name, 'event')].append((t[1], t[0])) # add inverse edge

    for t in labels['entity2entity']:
        type_id = t[2]
        type_name = ent2ent_typedict[type_id]
        dgld.setdefault(('entity', type_name, 'entity'),[])
        dgld[('entity', type_name, 'entity')].append((t[0],t[1]))

    # event-event relations
    dgld[('event', 'Event2Event:Narr', 'event')] = [(t[0], t[1]) for t in labels['narr']]

    g = dgl.heterograph(dgld)

    ####### add node attributes
    g.nodes['event'].data['x'] = torch.tensor(event_emb)
    g.nodes['entity'].data['x'] = torch.tensor(entity_emb)
    g.nodes['event'].data['y'] = torch.tensor([t[0] for t in labels['event']])
    g.nodes['entity'].data['y'] = torch.tensor([t[0]+type_class_dict['event'] for t in labels['entity']])
    # mark: treat entity types as later event types
    #### add event/entity id ####
    event_cnt = event_emb.shape[0]
    entity_cnt = entity_emb.shape[0]
    g.nodes['event'].data['id'] = torch.tensor(list(range(event_cnt)))
    g.nodes['entity'].data['id'] = torch.tensor([i+event_cnt for i in range(entity_cnt)])

    return g



def redirect_stdout(logfile):
    '''
    used in log file processing
    :param logfile:
    :return:
    '''
    def MyHookOut(text):
        logfile.write(text)
        logfile.flush()
        return 1, 0, text
    phOut = PrintHook()
    phOut.Start(MyHookOut)
