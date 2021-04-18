POLARITY_TYPES = ['Negative', 'Positive']
MODALITY_TYPES = ['Asserted', 'Other']
GENERICITY_TYPES = ['Generic', 'Specific']
TENSE_TYPES = ['Unspecified', 'Past', 'Future', 'Present']

# ACE-2005 Event Types/Subtypes
EVENT_TYPES = ['Business:Declare-Bankruptcy', 'Business:End-Org', 'Business:Merge-Org',
               'Business:Start-Org', 'Conflict:Attack', 'Conflict:Demonstrate', 'Contact:Meet',
               'Contact:Phone-Write', 'Justice:Acquit', 'Justice:Appeal', 'Justice:Arrest-Jail',
               'Justice:Charge-Indict', 'Justice:Convict', 'Justice:Execute', 'Justice:Extradite',
               'Justice:Fine', 'Justice:Pardon', 'Justice:Release-Parole', 'Justice:Sentence',
               'Justice:Sue', 'Justice:Trial-Hearing', 'Life:Be-Born', 'Life:Die', 'Life:Divorce',
               'Life:Injure', 'Life:Marry', 'Movement:Transport', 'Personnel:Elect',
               'Personnel:End-Position', 'Personnel:Nominate', 'Personnel:Start-Position',
               'Transaction:Transfer-Money', 'Transaction:Transfer-Ownership']

# Constants for GENE (Global Event Network Embedding)
GENE_BASE_PATH = '/shared/nas/data/m1/qizeng2/project/Event_Repr/emb_for_coref'
# Variants
TUPLE_0 = 'Tuple.0'
TUPLE_1 = 'Tuple.1'
SKG_30_1 = 'SKG.30.1'
DGI_30_2 = 'DGI.30.2'
SEM_ARC_30_3 = 'SEM_ARC.30.3'
SEM_ARC_30_4 = 'SEM_ARC.30.4'
SEM_ARC_30_5 = 'SEM_ARC.30.5'
SEM_ARC_30_6 = 'SEM_ARC.30.6'

# If not in GENE2DIM dict -> Default = 256
GENE_DIM = 256
GENE2DIM = {
    TUPLE_0: 768,
    TUPLE_1: 768
}
