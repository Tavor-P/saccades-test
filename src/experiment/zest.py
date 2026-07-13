import math

from include.experiment.constants import (
    ZEST_BETA,
    ZEST_GRID_SIZE,
    ZEST_GUESS_RATE,
    ZEST_LAPSE_RATE,
    ZEST_LOG_CONTRAST_MIN,
    ZEST_LOG_CONTRAST_MAX,
)


class ZestStaircase:
    """Bayesian adaptive contrast staircase (King-Smith et al., 1994 ZEST
    procedure), the contrast-selection method Diamond, Ross & Morrone (2000)
    used in place of a fixed set of contrast levels: maintains a posterior
    over log-contrast detection threshold, always tests at its current mean,
    and narrows the posterior after each response via a Weibull psychometric
    function.
    """

    def __init__(
        self,
        log_contrast_min: float = ZEST_LOG_CONTRAST_MIN,
        log_contrast_max: float = ZEST_LOG_CONTRAST_MAX,
        grid_size: int = ZEST_GRID_SIZE,
        beta: float = ZEST_BETA,
        guess_rate: float = ZEST_GUESS_RATE,
        lapse_rate: float = ZEST_LAPSE_RATE,
    ) -> None:
        step = (log_contrast_max - log_contrast_min) / (grid_size - 1)
        self._grid = [log_contrast_min + i * step for i in range(grid_size)]
        self._pdf = [1.0 / grid_size] * grid_size
        self._beta = beta
        self._guess_rate = guess_rate
        self._lapse_rate = lapse_rate

    def _p_detect(self, log_contrast: float, log_threshold: float) -> float:
        ratio = 10 ** (log_contrast - log_threshold)
        weibull = 1 - math.exp(-(ratio**self._beta))
        return self._guess_rate + (1 - self._guess_rate - self._lapse_rate) * weibull

    def _posterior_mean_log(self) -> float:
        return sum(p * x for p, x in zip(self._pdf, self._grid))

    def next_contrast(self) -> float:
        return 10 ** self._posterior_mean_log()

    def update(self, contrast: float, detected: bool) -> None:
        log_contrast = math.log10(contrast)
        likelihood = [
            self._p_detect(log_contrast, lt) if detected else 1 - self._p_detect(log_contrast, lt)
            for lt in self._grid
        ]
        posterior = [p * l for p, l in zip(self._pdf, likelihood)]
        total = sum(posterior)
        self._pdf = [p / total for p in posterior]

    @property
    def threshold_estimate(self) -> float:
        return 10 ** self._posterior_mean_log()
