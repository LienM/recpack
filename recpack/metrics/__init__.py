from recpack.metrics.coverage import CoverageK
from recpack.metrics.dcg import DCGK, NDCGK
from recpack.metrics.diversity import IntraListDiversityK
from recpack.metrics.ips import IPSHitRateK
from recpack.metrics.precision import PrecisionK
from recpack.metrics.recall import RecallK, CalibratedRecallK
from recpack.metrics.reciprocal_rank import RRK


METRICS = {
    "Coverage": CoverageK,
    "NDCG": NDCGK,
    "DCG": DCGK,
    "IntraListDiversity": IntraListDiversityK,
    "IPSHitRate": IPSHitRateK,
    "Precision": PrecisionK,
    "Recall": RecallK,
    "CalibratedRecall": CalibratedRecallK,
    "RR": RRK
}
