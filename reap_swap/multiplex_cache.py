"""LRU cache for loaded cartridges in the multiplex server.

Tracks which cartridges are currently pinned in CPU memory and evicts
the least-recently-used ones when the cache exceeds its limit.
"""

from typing import Any


def update_loaded_cartridge_order(
    loaded: list[str],
    cartridge_id: str,
    *,
    max_loaded: int = 4,
) -> dict[str, Any]:
    """Update the LRU cartridge order and return eviction instructions.

    Moves ``cartridge_id`` to the front (most recently used). If the
    cache exceeds ``max_loaded``, returns the IDs of cartridges that
    should be evicted from CPU memory.

    Args:
        loaded: Current list of loaded cartridge IDs in LRU order
                (most recent first).
        cartridge_id: The cartridge being accessed.
        max_loaded: Maximum number of cartridges to keep loaded.

    Returns:
        Dict with ``order`` (new LRU list), ``evicted`` (list of
        cartridge IDs to unload), and ``already_loaded`` (bool).
    """
    already_loaded = cartridge_id in loaded
    new_order = [cartridge_id] + [c for c in loaded if c != cartridge_id]
    evicted: list[str] = []
    while len(new_order) > max_loaded:
        evicted.append(new_order.pop())
    return {
        "order": new_order,
        "evicted": evicted,
        "already_loaded": already_loaded,
    }
