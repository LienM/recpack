from recpack.algorithms.base import Algorithm


class CML(Algorithm):
    """
    Pytorch Implementation of
    [1] Cheng-Kang Hsieh et al., Collaborative Metric Learning. WWW2017
    http://www.cs.cornell.edu/~ylongqi/paper/HsiehYCLBE17.pdf

    Version without features, referred to as CML in the paper.
    """

    def __init__(
        self,
        embedding_dim,
        margin,
        learning_rate,
        clip_norm,
        use_rank_weight,
        use_cov_loss,
    ):
        pass


class CMLWithFeatures(Algorithm):
    """
    Pytorch Implementation of
    [1] Cheng-Kang Hsieh et al., Collaborative Metric Learning. WWW2017
    http://www.cs.cornell.edu/~ylongqi/paper/HsiehYCLBE17.pdf

    Version with features, referred to as CML+F in the paper.
    """

    def __init__(
        self,
        embedding_dim,
        margin,
        learning_rate,
        clip_norm,
        use_rank_weight,
        use_cov_loss,
        hidden_layer_dim,
        feature_l2_reg,
        feature_proj_scaling_factor
    ):
        pass


class RandomNegativeSampler:
    """
    Sample random negative samples for this user uniformly. 
    """

    pass
