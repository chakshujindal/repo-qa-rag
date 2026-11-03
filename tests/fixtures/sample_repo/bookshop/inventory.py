"""Warehouse stock levels."""

REORDER_THRESHOLD = 5


class OutOfStockError(Exception):
    pass


def reserve_stock(stock: dict[str, int], isbn: str, quantity: int) -> None:
    """Hold copies for an order. Raises OutOfStockError if the warehouse has too few."""
    available = stock.get(isbn, 0)
    if quantity > available:
        raise OutOfStockError(f"only {available} copies of {isbn} left")
    stock[isbn] = available - quantity


def release_stock(stock: dict[str, int], isbn: str, quantity: int) -> None:
    """Put reserved copies back, e.g. when an order is cancelled."""
    stock[isbn] = stock.get(isbn, 0) + quantity


def low_stock_report(stock: dict[str, int]) -> list[str]:
    """ISBNs at or below the reorder threshold, so purchasing can restock them."""
    return sorted(isbn for isbn, count in stock.items() if count <= REORDER_THRESHOLD)
