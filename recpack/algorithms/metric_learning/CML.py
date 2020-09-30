from recpack.algorithms.base import Algorithm


class CML(Algorithm):
    """
    Pytorch Implementation of
    [1] Cheng-Kang Hsieh et al., Collaborative Metric Learning. WWW2017
    http://www.cs.cornell.edu/~ylongqi/paper/HsiehYCLBE17.pdf

    Version without features, referred to as CML in the paper.
    """
    pass


class CMLWithFeatures(Algorithm):
    """
    Pytorch Implementation of
    [1] Cheng-Kang Hsieh et al., Collaborative Metric Learning. WWW2017
    http://www.cs.cornell.edu/~ylongqi/paper/HsiehYCLBE17.pdf

    Version with features, referred to as CML+F in the paper.
    """
    pass


class RandomNegativeSampler:
    """
    Sample random negative samples for this user uniformly. 
    """
    pass