#!/usr/bin/env python3
from __future__ import annotations


def update_loaded_cartridge_order(
    current_order: list[str],
    cartridge_id: str,
    *,
    max_loaded: int,
) -> dict[str, object]:
    if max_loaded < 1:
        raise ValueError("max_loaded must be at least 1")

    order = list(current_order)
    evicted: list[str] = []
    already_loaded = cartridge_id in order

    if already_loaded:
        order.remove(cartridge_id)
        order.append(cartridge_id)
        return {
            "already_loaded": True,
            "evicted": evicted,
            "order": order,
        }

    while len(order) >= max_loaded:
        evicted.append(order.pop(0))
    order.append(cartridge_id)
    return {
        "already_loaded": False,
        "evicted": evicted,
        "order": order,
    }
