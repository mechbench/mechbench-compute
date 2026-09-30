from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlsplit

from mechbench_compute.providers.errors import EndpointRefused

ALLOW_PRIVATE = "MECHBENCH_ALLOW_PRIVATE_ENDPOINTS"


def is_private_allowed() -> bool:
    return os.environ.get(ALLOW_PRIVATE, "").strip().lower() in ("1", "true", "yes")


def resolve_addresses(host: str, port: int) -> list[str]:
    infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    return sorted({str(info[4][0]) for info in infos})


def is_public_address(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def check_endpoint(url: str, *, provider: str) -> str:
    refuse = f"{provider}: the credential's base_url {url!r} is refused"
    parts = urlsplit(url)
    if parts.scheme not in ("https", "http") or not parts.hostname:
        raise EndpointRefused(f"{refuse}: it is not an https:// URL with a host")
    if is_private_allowed():
        return url
    if parts.scheme != "https":
        raise EndpointRefused(
            f"{refuse}: an endpoint that receives your key must be https://. "
            f"To reach a server on your own machine or network, set "
            f"{ALLOW_PRIVATE}=1 in the runner's environment")
    try:
        addresses = resolve_addresses(parts.hostname, parts.port or 443)
    except (OSError, UnicodeError) as e:
        raise EndpointRefused(f"{refuse}: {parts.hostname} does not resolve ({e})") from None
    private = [a for a in addresses if not is_public_address(a)]
    if private or not addresses:
        raise EndpointRefused(
            f"{refuse}: {parts.hostname} resolves to "
            f"{', '.join(private) or 'nothing'}, which is not a public address "
            f"(loopback, link-local, private, shared or reserved). To reach a "
            f"server on your own machine or network, set {ALLOW_PRIVATE}=1 in "
            f"the runner's environment")
    return url
