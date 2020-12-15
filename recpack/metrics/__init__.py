from recpack.metrics.coverage import CoverageK
from recpack.metrics.dcg import DCGK, NDCGK
from recpack.metrics.diversity import IntraListDiversityK
from recpack.metrics.ips import IPSHitRateK
from recpack.metrics.precision import PrecisionK
from recpack.metrics.recall import RecallK, CalibratedRecallK
from recpack.metrics.reciprocal_rank import RRK

from recpack.metrics.avg_reciprocal_hit_rate import AvgReciprocalHitRateK
from recpack.metrics.auc_aman import AUCAMAN
from recpack.metrics.percentile_ranking import PercentileRanking


METRICS = {
    "Coverage": CoverageK,
    "NDCG": NDCGK,
    "DCG": DCGK,
    "IntraListDiversity": IntraListDiversityK,
    "IPSHitRate": IPSHitRateK,
    "Precision": PrecisionK,
    "Recall": RecallK,
    "CalibratedRecall": CalibratedRecallK,
    "RR": RRK,
    "ARHRK": AvgReciprocalHitRateK,
    "AUCAMAN": AUCAMAN,
    "PercentileRanking": PercentileRanking
}
