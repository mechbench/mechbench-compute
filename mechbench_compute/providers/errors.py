from __future__ import annotations

from mechbench_compute.errors import InterpError


class ProviderError(InterpError):
    pass


class AuthError(ProviderError):
    pass


class TransientError(ProviderError):
    retryable = True

    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


class BudgetExceeded(ProviderError):
    def __init__(self, *, cap_usd: float, spent_usd: float, estimate_usd: float,
                 provider: str = "", model: str = ""):
        where = f" ({provider}/{model})" if provider else ""
        super().__init__(
            f"budget cap ${cap_usd:.4f} would be exceeded{where}: "
            f"${spent_usd:.4f} already spent and this call's worst case is "
            f"${estimate_usd:.4f}. Raise budget_usd on the node, or lower "
            f"max_tokens.")
        self.cap_usd = cap_usd
        self.spent_usd = spent_usd
        self.estimate_usd = estimate_usd


class RateLimited(ProviderError):
    def __init__(self, message: str = "rate limited", *, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class ProviderUnavailable(ProviderError):
    def __init__(self, message: str, *, provider: str = "", seconds: float = 0.0):
        super().__init__(message)
        self.provider = provider
        self.seconds = seconds


class CapabilityUnsupported(ProviderError):
    def __init__(self, provider: str, capability: str, detail: str = ""):
        msg = f"{provider} does not support {capability}"
        if detail:
            msg += f" ({detail})"
        super().__init__(msg)
        self.provider = provider
        self.capability = capability


class CassetteMiss(ProviderError):
    def __init__(self, provider: str, request_hash: str, label: str = ""):
        super().__init__(
            f"cassette{f' {label}' if label else ''} has no recorded response "
            f"for {provider} request {request_hash[:12]}. Re-record with "
            f"mode='record' (which spends), or fix the request that drifted.")
        self.request_hash = request_hash
